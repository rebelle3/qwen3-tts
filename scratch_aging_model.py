# coding=utf-8
"""
LOOP ITER 5: a REAL vocal-aging model (beyond pitch).

Feedback: shifting F0/formants just sounds "deeper", not "older". The cues that
actually signal age are phonation-quality changes from vocal-fold aging
(presbyphonia), NOT pitch:
  - jitter   : cycle-to-cycle F0 instability
  - shimmer  : cycle-to-cycle amplitude instability
  - breathiness / low HNR : incomplete glottal closure -> aspiration noise
  - tremor   : slow ~5 Hz wobble in pitch + loudness
  - widened formant bandwidths : damped, less-crisp resonances
  - reduced pitch range, slower rate

We model these with the WORLD vocoder (f0 / spectral envelope / APERIODICITY =
breathiness), launder the aged reference through Qwen3 ICL, and MEASURE the
result with Praat (parselmouth): jitter / shimmer / HNR / F0 -- objective proof
the aging qualities are present, not just pitch.

Runs on CPU.
"""
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import soundfile as sf
import librosa
import pyworld as pw
import parselmouth
from parselmouth.praat import call

from qwen_tts import Qwen3TTSModel

QWEN_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
REF_TEXT = ("Honestly, I can't quite believe it. The weather's been rather dreadful "
            "all week, so I'd rather stay in with a cup of tea and a good book.")
SRC = "kokoro_bf.wav"
TARGET = "Hello, I am Claude Opus four point eight."
SR = 24000
FP = 5.0  # WORLD frame period (ms)

# age level -> aging parameters. adult = identity.
LEVELS = {
    "young":   dict(formant=1.09, f0s=1.10, f0range=1.15, jitter=0.004, shimmer=0.02,
                    trem_f0=0.0,  trem_amp=0.0,  breath=0.00, bw=0,  tilt=0.0,  rate=0.95),
    "adult":   dict(formant=1.00, f0s=1.00, f0range=1.00, jitter=0.000, shimmer=0.00,
                    trem_f0=0.0,  trem_amp=0.0,  breath=0.00, bw=0,  tilt=0.0,  rate=1.00),
    "older":   dict(formant=0.97, f0s=0.99, f0range=0.80, jitter=0.018, shimmer=0.07,
                    trem_f0=0.02, trem_amp=0.07, breath=0.28, bw=3,  tilt=-2.0, rate=1.08),
    "elderly": dict(formant=0.94, f0s=0.97, f0range=0.60, jitter=0.032, shimmer=0.12,
                    trem_f0=0.035,trem_amp=0.13, breath=0.45, bw=5,  tilt=-3.5, rate=1.18),
}
ORDER = ["young", "adult", "older", "elderly"]
TREM_RATE = 5.0  # Hz

rng = np.random.RandomState(7)


def _smooth_cols(M, win):
    if win <= 0:
        return M
    k = np.ones(2 * win + 1) / (2 * win + 1)
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, M)


def age_transform(x, sr, p):
    x = np.ascontiguousarray(x.astype(np.float64))
    f0, t = pw.harvest(x, sr, frame_period=FP)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    dim = sp.shape[1]
    v = f0 > 0

    # ---- F0: range compression, scale, jitter, tremor ----
    if v.any():
        m = f0[v].mean()
        f0[v] = m + (f0[v] - m) * p["f0range"]
        f0[v] *= p["f0s"]
        if p["jitter"] > 0:
            f0[v] *= (1.0 + p["jitter"] * rng.randn(v.sum()))
        if p["trem_f0"] > 0:
            f0[v] *= (1.0 + p["trem_f0"] * np.sin(2 * np.pi * TREM_RATE * t[v]))
        f0 = np.clip(f0, 0, None)

    # ---- Spectral envelope: formant warp, bandwidth widen, tilt ----
    idx = np.arange(dim); src = np.clip(idx / p["formant"], 0, dim - 1)
    lo = np.floor(src).astype(int); hi = np.clip(lo + 1, 0, dim - 1); frac = src - lo
    sp = sp[:, lo] * (1 - frac) + sp[:, hi] * frac
    if p["bw"] > 0:
        logsp = np.log(sp + 1e-12)
        sp = np.exp(_smooth_cols(logsp, p["bw"]))
    if p["tilt"] != 0.0:
        freqs = np.linspace(0, sr / 2, dim)
        gain = 10.0 ** ((p["tilt"] / 20.0) * (freqs / (sr / 2)))  # linear tilt, cut highs if tilt<0
        sp = sp * gain[None, :]

    # ---- Aperiodicity: breathiness (push toward 1, weighted to high bands) ----
    if p["breath"] > 0:
        w = np.linspace(0.35, 1.0, dim)[None, :]
        ap = ap + p["breath"] * w * (1.0 - ap)
        ap = np.clip(ap, 0, 1)

    # ---- Time-stretch (rate): interpolate frames ----
    if abs(p["rate"] - 1.0) > 1e-3:
        n0 = len(f0); n1 = max(2, int(round(n0 * p["rate"])))
        xi = np.linspace(0, n0 - 1, n1)
        base = np.arange(n0)
        vf = np.interp(xi, base, v.astype(float)) > 0.5
        f0i = np.interp(xi, base, f0); f0i[~vf] = 0.0
        spi = np.vstack([np.interp(xi, base, sp[:, j]) for j in range(dim)]).T
        api = np.vstack([np.interp(xi, base, ap[:, j]) for j in range(dim)]).T
        f0, sp, ap = f0i, np.ascontiguousarray(spi), np.ascontiguousarray(api)

    y = pw.synthesize(f0, np.ascontiguousarray(sp), np.ascontiguousarray(ap), sr, frame_period=FP)
    y = y.astype(np.float32)

    # ---- Waveform-level shimmer + amplitude tremor ----
    tt = np.arange(len(y)) / sr
    env = np.ones(len(y))
    if p["shimmer"] > 0:
        n = rng.randn(len(y)); k = int(sr * 0.03)
        n = np.convolve(n, np.ones(k) / k, mode="same")  # ~30ms smoothed
        env *= (1.0 + p["shimmer"] * n / (np.std(n) + 1e-9) * 0.5)
    if p["trem_amp"] > 0:
        env *= (1.0 + p["trem_amp"] * np.sin(2 * np.pi * TREM_RATE * tt))
    y = y * env
    mx = np.max(np.abs(y)) + 1e-9
    return (y / mx * 0.97).astype(np.float32)


def measure(y, sr):
    """Praat: F0, jitter(local %), shimmer(local %), HNR(dB)."""
    try:
        snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
        f0m = call(snd.to_pitch(), "Get mean", 0, 0, "Hertz")
        pp = call(snd, "To PointProcess (periodic, cc)", 70, 500)
        jit = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
        shim = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
        hnr = call(call(snd, "To Harmonicity (cc)", 0.01, 70, 0.1, 1.0), "Get mean", 0, 0)
        return f0m, jit, shim, hnr
    except Exception as e:
        return 0.0, 0.0, 0.0, 0.0


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


def trim(w, sr, thr=0.02, pad=0.12):
    win = max(1, int(0.05 * sr))
    env = np.array([np.sqrt(np.mean(w[i:i+win]**2)) for i in range(0, max(1, len(w)-win), win)])
    a = np.where(env > thr)[0]
    if len(a) == 0: return w, 0.0
    s = max(0, int(a[0]*win - pad*sr)); e = min(len(w), int((a[-1]+1)*win + pad*sr))
    return w[s:e], (e-s)/sr


def main():
    torch.set_num_threads(4)
    x, sr = sf.read(SRC)
    if sr != SR:
        x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=SR); sr = SR

    print("=== Phase 1: build aged references + measure (jitter/shimmer/HNR) ===")
    print(f"{'level':9s} {'F0':>5s} {'jit%':>5s} {'shim%':>6s} {'HNR':>6s}")
    refs = {}
    for lv in ORDER:
        y = age_transform(x, sr, LEVELS[lv])
        rp = f"aged2_ref_{lv}.wav"; sf.write(rp, y, sr); refs[lv] = rp
        f0m, jit, shim, hnr = measure(y, sr)
        print(f"{lv:9s} {f0m:5.0f} {jit:5.2f} {shim:6.2f} {hnr:6.1f}  -> {rp}")

    print(f"\n=== Phase 2: Qwen3 ICL launder + measure output ===")
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(QWEN_PATH, device_map="cpu", dtype=torch.float32,
                                        attn_implementation="eager")
    print(f"Qwen3 loaded in {time.time()-t0:.1f}s")
    judge = Judge()
    print(f"{'level':9s} {'P(eng)':>6s} {'top':>10s} {'F0':>5s} {'jit%':>5s} {'shim%':>6s} {'HNR':>6s}")
    for lv in ORDER:
        item = tts.create_voice_clone_prompt(ref_audio=refs[lv], ref_text=REF_TEXT,
                                             x_vector_only_mode=False)
        best = None
        for k in range(2):
            torch.manual_seed(9200 + k)
            wavs, srr = tts.generate_voice_clone(text=TARGET, language="English",
                                                 voice_clone_prompt=item, **GEN)
            w, dur = trim(np.asarray(wavs[0], dtype=np.float32), srr)
            if best is None or dur < best[1]: best = (w, dur, srr)
            if 1.0 <= dur <= 9.0: break
        w, dur, srr = best
        out = f"aged2_{lv}.wav"; sf.write(out, w, srr)
        pe, top = judge.score(w, srr)
        f0m, jit, shim, hnr = measure(w, srr)
        print(f"{lv:9s} {pe:6.3f} {top:>10s} {f0m:5.0f} {jit:5.2f} {shim:6.2f} {hnr:6.1f}  -> {out}")

    print("\nDone.")


GEN = dict(max_new_tokens=512, do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
           repetition_penalty=1.05, subtalker_dosample=True, subtalker_top_k=50,
           subtalker_top_p=1.0, subtalker_temperature=0.9)

if __name__ == "__main__":
    main()
