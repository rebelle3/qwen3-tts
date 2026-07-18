# coding=utf-8
"""
VoiceDesign old-age, roughness-forward wording.

Finding: 'frail/thin/breathy/wispy' -> the model renders a CLEANER voice
(HNR goes UP with age). 'gravelly/weathered/raspy' -> renders roughness = age.
So describe the elderly with ROUGHNESS/GRAVEL/HOARSE/CREAK, not frailty.

Test 70s + 80s for both genders with roughness-forward prompts; measure HNR
(should now DROP). User judges.
"""
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, soundfile as sf
import parselmouth
from parselmouth.praat import call
from qwen_tts import Qwen3TTSModel

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
TARGET = "Hello, I am Claude Opus four point eight. It's rather nice to meet you."
LANG = "English"

CASES = [
    ("m_70s", "A 72-year-old man. A deep, gravelly, hoarse voice, rough and raspy with a "
              "coarse rattle, weathered and worn. Slow, heavy, deliberate delivery with a "
              "low creaky rumble and a dry, craggy edge."),
    ("m_80s", "An 84-year-old man. An extremely gravelly, hoarse, craggy voice, very rough and "
              "raspy with heavy vocal creak and a coarse grating rattle. Very slow and labored, "
              "deep, worn and unsteady."),
    ("f_70s", "A 72-year-old woman. A husky, gravelly, raspy voice, rough and coarse with a "
              "smoky rasp, weathered and worn. Slow, deliberate delivery with dry vocal creak "
              "and a craggy edge."),
    ("f_80s", "An 84-year-old woman. A very husky, hoarse, gravelly and craggy voice, rough and "
              "raspy with heavy creak and a coarse rattle. Very slow, worn, weathered and unsteady."),
]

GEN = dict(max_new_tokens=768, do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
           repetition_penalty=1.05, subtalker_dosample=True, subtalker_top_k=50,
           subtalker_top_p=1.0, subtalker_temperature=0.9)


def trim(w, sr, thr=0.02, pad=0.12):
    win = max(1, int(0.05 * sr))
    env = np.array([np.sqrt(np.mean(w[i:i+win]**2)) for i in range(0, max(1, len(w)-win), win)])
    a = np.where(env > thr)[0]
    if len(a) == 0: return w, 0.0
    s = max(0, int(a[0]*win - pad*sr)); e = min(len(w), int((a[-1]+1)*win + pad*sr))
    return w[s:e], (e-s)/sr


def measure(y, sr):
    try:
        snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
        f0 = call(snd.to_pitch(), "Get mean", 0, 0, "Hertz")
        pp = call(snd, "To PointProcess (periodic, cc)", 50, 500)
        jit = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
        shim = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
        hnr = call(call(snd, "To Harmonicity (cc)", 0.01, 50, 0.1, 1.0), "Get mean", 0, 0)
        return f0, jit, shim, hnr
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def main():
    torch.set_num_threads(4)
    print(f"Loading VoiceDesign {MODEL}...")
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(MODEL, device_map="cpu", dtype=torch.float32,
                                        attn_implementation="eager")
    print(f"Loaded in {time.time()-t0:.1f}s")
    print(f"{'voice':8s} {'F0':>5s} {'jit%':>5s} {'shim%':>6s} {'HNR':>6s}  (dur)")
    for name, instruct in CASES:
        best = None
        for k in range(2):
            torch.manual_seed(3300 + k)
            wavs, sr = tts.generate_voice_design(text=TARGET, language=LANG,
                                                 instruct=instruct, **GEN)
            w, dur = trim(np.asarray(wavs[0], dtype=np.float32), sr)
            if best is None or dur < best[1]: best = (w, dur, sr)
            if 1.5 <= dur <= 11.0: break
        w, dur, sr = best
        out = f"vdr_{name}.wav"; sf.write(out, w, sr)
        f0, jit, shim, hnr = measure(w, sr)
        print(f"{name:8s} {f0:5.0f} {jit:5.2f} {shim:6.2f} {hnr:6.1f}  ({dur:.1f}s) -> {out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
