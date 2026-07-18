# coding=utf-8
"""
LOOP ITER 4 (capstone): a CUSTOM-VOICE PALETTE from composable axes.

Three independent knobs, established across iters 1-3:
  ACCENT  <- ref_code   : pick the Kokoro source voice (British here)
  GENDER  <- x-vector   : source voice choice (bf/bm) or blend the x-vector
  AGE     <- WORLD shift : formant + F0 warp of the reference before ICL

This mints several NAMED British custom voices spanning gender + age, renders
your line in each, and checks the British accent (england) holds across the
whole palette -- i.e. the axes compose into a real casting toolkit.

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
TARGET = "Hello, I am Claude Opus four point eight."
SR = 24000

# name -> (source_kokoro_wav, formant_factor, f0_factor)
PALETTE = [
    ("uk_girl",        "kokoro_bf.wav", 1.18, 1.28),
    ("uk_young_woman", "kokoro_bf.wav", 1.08, 1.10),
    ("uk_woman",       "kokoro_bf.wav", 1.00, 1.00),
    ("uk_man",         "kokoro_bm.wav", 1.00, 1.00),
    ("uk_elderly_man", "kokoro_bm.wav", 0.90, 0.86),
]

GEN = dict(max_new_tokens=512, do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
           repetition_penalty=1.05, subtalker_dosample=True, subtalker_top_k=50,
           subtalker_top_p=1.0, subtalker_temperature=0.9)


def formant_f0_shift(x, sr, fa, fb):
    if fa == 1.0 and fb == 1.0:
        return x.astype(np.float32)
    x = np.ascontiguousarray(x.astype(np.float64))
    f0, t = pw.harvest(x, sr); sp = pw.cheaptrick(x, f0, t, sr); ap = pw.d4c(x, f0, t, sr)
    dim = sp.shape[1]; idx = np.arange(dim); src = np.clip(idx / fa, 0, dim - 1)
    lo = np.floor(src).astype(int); hi = np.clip(lo + 1, 0, dim - 1); frac = src - lo
    sp2 = sp[:, lo] * (1 - frac) + sp[:, hi] * frac
    return pw.synthesize(f0 * fb, np.ascontiguousarray(sp2), ap, sr).astype(np.float32)


def trim(w, sr, thr=0.02, pad=0.12):
    win = max(1, int(0.05 * sr))
    env = np.array([np.sqrt(np.mean(w[i:i+win]**2)) for i in range(0, max(1, len(w)-win), win)])
    a = np.where(env > thr)[0]
    if len(a) == 0: return w, 0.0
    s = max(0, int(a[0]*win - pad*sr)); e = min(len(w), int((a[-1]+1)*win + pad*sr))
    return w[s:e], (e-s)/sr


def f0_median(w, sr):
    f0 = librosa.yin(w.astype(np.float32), fmin=70, fmax=450, sr=sr)
    f0 = f0[np.isfinite(f0)]
    return float(np.median(f0)) if len(f0) else 0.0


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
        out = self.clf.classify_batch(torch.tensor(w).unsqueeze(0))[0].squeeze(0)
        p = torch.softmax(out, dim=-1)
        return float(p[self.eng]), self.lab[int(p.argmax())]


def main():
    torch.set_num_threads(4)
    print(f"Loading Qwen3 {QWEN_PATH}...")
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(QWEN_PATH, device_map="cpu", dtype=torch.float32,
                                        attn_implementation="eager")
    print(f"Qwen3 loaded in {time.time()-t0:.1f}s")
    judge = Judge()

    print("\n=== Custom British voice palette ===")
    for name, src, fa, fb in PALETTE:
        x, sr = sf.read(src)
        if sr != SR:
            x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=SR); sr = SR
        aged = formant_f0_shift(x, sr, fa, fb)
        rpath = f"palette_ref_{name}.wav"; sf.write(rpath, aged, sr)
        item = tts.create_voice_clone_prompt(ref_audio=rpath, ref_text=REF_TEXT,
                                             x_vector_only_mode=False)
        best = None
        for k in range(2):
            torch.manual_seed(9100 + k)
            wavs, srr = tts.generate_voice_clone(text=TARGET, language="English",
                                                 voice_clone_prompt=item, **GEN)
            w, dur = trim(np.asarray(wavs[0], dtype=np.float32), srr)
            if best is None or dur < best[1]: best = (w, dur, srr)
            if 1.0 <= dur <= 9.0: break
        w, dur, srr = best
        out = f"palette_{name}.wav"; sf.write(out, w, srr)
        pe, top = judge.score(w, srr)
        print(f"  [{name:15s}] src={src[7:9]} formant×{fa:.2f} f0×{fb:.2f} "
              f"P(eng)={pe:.3f} top={top:10s} F0={f0_median(w,srr):3.0f}Hz -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
