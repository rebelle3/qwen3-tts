# coding=utf-8
"""
LOOP ITER 3: an AGE axis for custom British voices.

Age in a voice is largely vocal-tract length (formant positions) + pitch:
  - younger/child : shorter tract -> formants UP, F0 up
  - older/adult   : longer/relaxed tract -> formants DOWN, F0 down
Magnitude-scaling the x-vector didn't give a clean axis (iter 2), so instead we
manipulate the REFERENCE acoustically with the WORLD vocoder (independent
formant-warp + F0-scale), then launder the aged reference through Qwen3 ICL.

Question: does apparent age transfer through ICL while the British accent holds?

Runs on CPU. classifier = compass, user = judge.
"""
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import soundfile as sf
import librosa
import pyworld as pw

from qwen_tts import Qwen3TTSModel

QWEN_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
REF_TEXT = ("Honestly, I can't quite believe it. The weather's been rather dreadful "
            "all week, so I'd rather stay in with a cup of tea and a good book.")
SRC = "kokoro_bf.wav"                  # British female (accent source)
TARGET = "Hello, I am Claude Opus four point eight."
SR = 24000

# (name, formant_factor, f0_factor). formant>1 = shorter tract = younger.
AGE_VARIANTS = [
    ("child",   1.18, 1.30),
    ("young",   1.08, 1.12),
    ("adult",   1.00, 1.00),
    ("older",   0.92, 0.90),
    ("elderly", 0.86, 0.82),
]

GEN = dict(max_new_tokens=512, do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
           repetition_penalty=1.05, subtalker_dosample=True, subtalker_top_k=50,
           subtalker_top_p=1.0, subtalker_temperature=0.9)


def formant_f0_shift(x, sr, formant_a, f0_b):
    """WORLD analysis -> warp spectral envelope freq axis (formants) + scale F0 -> resynth."""
    x = np.ascontiguousarray(x.astype(np.float64))
    f0, t = pw.harvest(x, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    dim = sp.shape[1]
    idx = np.arange(dim)
    src = np.clip(idx / formant_a, 0, dim - 1)     # warp frequency axis
    lo = np.floor(src).astype(int); hi = np.clip(lo + 1, 0, dim - 1); frac = src - lo
    sp2 = sp[:, lo] * (1 - frac) + sp[:, hi] * frac
    y = pw.synthesize(f0 * f0_b, np.ascontiguousarray(sp2), ap, sr)
    return y.astype(np.float32)


def trim(w, sr, thr=0.02, pad=0.12):
    win = max(1, int(0.05 * sr))
    env = np.array([np.sqrt(np.mean(w[i:i+win]**2)) for i in range(0, max(1, len(w)-win), win)])
    a = np.where(env > thr)[0]
    if len(a) == 0: return w, 0.0
    s = max(0, int(a[0]*win - pad*sr)); e = min(len(w), int((a[-1]+1)*win + pad*sr))
    return w[s:e], (e-s)/sr


def f0_median(w, sr):
    try:
        f0 = librosa.yin(w.astype(np.float32), fmin=70, fmax=450, sr=sr)
        f0 = f0[np.isfinite(f0)]
        return float(np.median(f0)) if len(f0) else 0.0
    except Exception:
        return 0.0


def formant_proxy(w, sr):
    """Rough spectral centroid (Hz) as a formant/brightness proxy -> tracks tract length."""
    try:
        return float(np.mean(librosa.feature.spectral_centroid(y=w.astype(np.float32), sr=sr)))
    except Exception:
        return 0.0


class Judge:
    def __init__(self):
        from speechbrain.inference.classifiers import EncoderClassifier
        self.clf = EncoderClassifier.from_hparams(
            source="Jzuluaga/accent-id-commonaccent_ecapa", savedir="/tmp/claude-0/accent_model")
        self.lab = self.clf.hparams.label_encoder.ind2lab
        self.eng = [i for i, l in self.lab.items() if l == "england"][0]

    def score(self, w, sr):
        if sr != 16000:
            w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=16000)
        sig = torch.tensor(w, dtype=torch.float32).unsqueeze(0)
        out = self.clf.classify_batch(sig)[0].squeeze(0)
        p = torch.softmax(out, dim=-1)
        return float(p[self.eng]), self.lab[int(p.argmax())]


def main():
    torch.set_num_threads(4)
    x, sr = sf.read(SRC)
    if sr != SR:
        x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=SR); sr = SR

    print("=== Phase 1: build age-shifted references (WORLD) ===")
    refs = {}
    for name, fa, fb in AGE_VARIANTS:
        y = formant_f0_shift(x, sr, fa, fb)
        path = f"age_ref_{name}.wav"
        sf.write(path, y, sr)
        refs[name] = path
        print(f"  {name:8s} formant×{fa:.2f} f0×{fb:.2f} F0={f0_median(y,sr):3.0f} "
              f"centroid={formant_proxy(y,sr):4.0f} -> {path}")

    print(f"\n=== Phase 2: Qwen3 ICL launder each aged reference ===")
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(QWEN_PATH, device_map="cpu", dtype=torch.float32,
                                        attn_implementation="eager")
    print(f"  Qwen3 loaded in {time.time()-t0:.1f}s")
    judge = Judge()

    for name, path in refs.items():
        item = tts.create_voice_clone_prompt(ref_audio=path, ref_text=REF_TEXT,
                                             x_vector_only_mode=False)
        best = None
        for k in range(2):
            torch.manual_seed(8000 + k)
            wavs, srr = tts.generate_voice_clone(text=TARGET, language="English",
                                                 voice_clone_prompt=item, **GEN)
            w, dur = trim(np.asarray(wavs[0], dtype=np.float32), srr)
            if best is None or dur < best[1]:
                best = (w, dur, srr)
            if 1.0 <= dur <= 9.0:
                break
        w, dur, srr = best
        out = f"age_{name}.wav"
        sf.write(out, w, srr)
        pe, top = judge.score(w, srr)
        print(f"  [{name:8s}] P(eng)={pe:.3f} top={top:10s} F0={f0_median(w,srr):3.0f}Hz "
              f"centroid={formant_proxy(w,srr):4.0f} dur={dur:.2f}s -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
