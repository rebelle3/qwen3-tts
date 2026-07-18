# coding=utf-8
"""
LOOP ITER 2: Can we decouple ACCENT from TIMBRE in Qwen3 ICL?

Hypothesis: in ICL mode the prompt carries TWO things --
  - ref_code          -> prosody + ACCENT (from the British reference)
  - ref_spk_embedding -> TIMBRE (gender / age / vocal character)
If they decouple, we can hold the British ref_code fixed and PAN the x-vector
to mint custom British voices (male<->female, and toward older/younger) from a
single reference -> the master key for custom voice creation.

Test: British-female ref_code (from Kokoro bf) + several x-vectors:
  - bf  : its own female x-vector          (baseline British female)
  - bm  : British-male x-vector            (British codes + male timbre)
  - 50% : lerp(bf, bm)                     (androgynous British)
  - lo  : x-vector scaled down (|v|*0.85)  (probe: does magnitude ~ age/size?)
  - hi  : x-vector scaled up   (|v|*1.15)
Measure P(england) (accent held?) and F0 (timbre moved?) for each.

Runs on CPU. The author cannot hear; classifier = compass, user = judge.
"""
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import soundfile as sf
import librosa

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

QWEN_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
REF_TEXT = ("Honestly, I can't quite believe it. The weather's been rather dreadful "
            "all week, so I'd rather stay in with a cup of tea and a good book.")
REF_BF = "kokoro_bf.wav"   # British female reference (accent source)
REF_BM = "kokoro_bm.wav"   # British male reference (timbre donor)
TARGET = "Hello, I am Claude Opus four point eight."

GEN = dict(max_new_tokens=512, do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
           repetition_penalty=1.05, subtalker_dosample=True, subtalker_top_k=50,
           subtalker_top_p=1.0, subtalker_temperature=0.9)


def trim(w, sr, thr=0.02, pad=0.12):
    win = max(1, int(0.05 * sr))
    env = np.array([np.sqrt(np.mean(w[i:i+win]**2)) for i in range(0, max(1, len(w)-win), win)])
    a = np.where(env > thr)[0]
    if len(a) == 0: return w, 0.0
    s = max(0, int(a[0]*win - pad*sr)); e = min(len(w), int((a[-1]+1)*win + pad*sr))
    return w[s:e], (e-s)/sr


def f0_median(w, sr):
    try:
        f0 = librosa.yin(w.astype(np.float32), fmin=70, fmax=400, sr=sr)
        f0 = f0[np.isfinite(f0)]
        return float(np.median(f0)) if len(f0) else 0.0
    except Exception:
        return 0.0


class Judge:
    def __init__(self):
        from speechbrain.inference.classifiers import EncoderClassifier
        self.clf = EncoderClassifier.from_hparams(
            source="Jzuluaga/accent-id-commonaccent_ecapa", savedir="/tmp/claude-0/accent_model")
        self.lab = self.clf.hparams.label_encoder.ind2lab
        self.eng = [i for i, l in self.lab.items() if l == "england"][0]
        self.us = [i for i, l in self.lab.items() if l == "us"][0]

    def score(self, w, sr):
        if sr != 16000:
            w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=16000)
        sig = torch.tensor(w, dtype=torch.float32).unsqueeze(0)
        out = self.clf.classify_batch(sig)[0].squeeze(0)
        p = torch.softmax(out, dim=-1)
        return float(p[self.eng]), float(p[self.us]), self.lab[int(p.argmax())]


def main():
    torch.set_num_threads(4)
    print(f"Loading Qwen3 {QWEN_PATH}...")
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(QWEN_PATH, device_map="cpu", dtype=torch.float32,
                                        attn_implementation="eager")
    print(f"Qwen3 loaded in {time.time()-t0:.1f}s")
    judge = Judge()

    # Extract ICL prompt (ref_code + x-vector) for the British FEMALE reference,
    # and the x-vector of the British MALE reference (timbre donor).
    bf = tts.create_voice_clone_prompt(ref_audio=REF_BF, ref_text=REF_TEXT, x_vector_only_mode=False)[0]
    bm = tts.create_voice_clone_prompt(ref_audio=REF_BM, ref_text=REF_TEXT, x_vector_only_mode=False)[0]
    code_bf = bf.ref_code                       # British-female accent codes (held fixed)
    dt = bf.ref_spk_embedding.dtype
    xbf = bf.ref_spk_embedding.detach().float().flatten()
    xbm = bm.ref_spk_embedding.detach().float().flatten()
    print(f"ref_code_bf shape={tuple(code_bf.shape)}  xvec dim={xbf.numel()}  "
          f"|xbf|={xbf.norm():.2f} |xbm|={xbm.norm():.2f}\n")

    def xvec(v):
        return v.to(dt)

    variants = [
        ("bf_own",    xbf),
        ("bm_timbre", xbm),
        ("mix50",     0.5 * xbf + 0.5 * xbm),
        ("scale_lo",  xbf * 0.85),
        ("scale_hi",  xbf * 1.15),
    ]

    print("=== British ref_code held FIXED; x-vector panned ===")
    for name, v in variants:
        item = VoiceClonePromptItem(ref_code=code_bf, ref_spk_embedding=xvec(v),
                                    x_vector_only_mode=False, icl_mode=True, ref_text=REF_TEXT)
        best = None
        for k in range(2):
            torch.manual_seed(6000 + k)
            wavs, sr = tts.generate_voice_clone(text=TARGET, language="English",
                                                voice_clone_prompt=[item], **GEN)
            w, dur = trim(np.asarray(wavs[0], dtype=np.float32), sr)
            if best is None or dur < best[1]:
                best = (w, dur, sr)
            if 1.0 <= dur <= 9.0:
                break
        w, dur, sr = best
        out = f"decouple_{name}.wav"
        sf.write(out, w, sr)
        pe, pu, top = judge.score(w, sr)
        print(f"  [{name:9s}] |v|={v.norm():6.2f} P(eng)={pe:.3f} top={top:10s} "
              f"F0={f0_median(w,sr):3.0f}Hz dur={dur:.2f}s -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
