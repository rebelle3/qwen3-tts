# coding=utf-8
"""
VoiceDesign age range, take 2: DETAILED, SPECIFIC descriptions.

VoiceDesign responds to rich persona + concrete vocal-quality descriptions.
Each instruct names age, timbre, breath support, pacing, resonance, and
age-specific phonation (tremor, creak, breathiness, frailty).

Measure F0 / jitter / shimmer / HNR (Praat); user judges naturalness.
Runs on CPU.
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

RANGES = {
    "m": [
        ("20s", "A 22-year-old male university student. A bright, clear, resonant tenor "
                "with crisp articulation and quick, lively pacing. Full and energetic tone, "
                "strong breath support, confident and upbeat cadence."),
        ("40s", "A 45-year-old man, a seasoned professional. A warm, rounded baritone with "
                "smooth even delivery and relaxed, measured pacing. Steady breath support, "
                "clear diction, calm and assured."),
        ("60s", "A 65-year-old man, recently retired. A slightly gravelly, weathered baritone "
                "with a touch of dryness. Slower, deliberate pacing, gentle warmth, reduced "
                "projection, a soft rasp at the ends of phrases."),
        ("80s", "An 84-year-old man, frail and gentle. A thin, breathy, wavering voice with a "
                "slight tremor and low volume. Slow, halting pacing with small pauses, soft "
                "creaky phonation, weak breath support, and a fragile, papery timbre."),
    ],
    "f": [
        ("20s", "A 23-year-old woman, cheerful and lively. A bright, light soprano with clear "
                "ringing tone, crisp articulation and quick, energetic pacing. Warm and full "
                "of vitality."),
        ("40s", "A 44-year-old woman, poised and articulate. A warm, mellow mezzo with smooth "
                "even delivery and composed, measured pacing. Rich, steady tone, clear diction."),
        ("60s", "A 66-year-old woman, gentle and grandmotherly. A softer, slightly husky voice "
                "with a warm, seasoned tone, slower deliberate pacing, gentle and a little breathy."),
        ("80s", "An 82-year-old woman, delicate and frail. A thin, breathy, quavering voice with "
                "a slight tremor and low volume. Slow, halting delivery with soft pauses, light "
                "creak, and a fragile, wispy timbre."),
    ],
}

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
        pp = call(snd, "To PointProcess (periodic, cc)", 60, 500)
        jit = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
        shim = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
        hnr = call(call(snd, "To Harmonicity (cc)", 0.01, 60, 0.1, 1.0), "Get mean", 0, 0)
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

    for g, items in RANGES.items():
        for age, instruct in items:
            best = None
            for k in range(2):
                torch.manual_seed(2500 + k)
                wavs, sr = tts.generate_voice_design(text=TARGET, language=LANG,
                                                     instruct=instruct, **GEN)
                w, dur = trim(np.asarray(wavs[0], dtype=np.float32), sr)
                if best is None or dur < best[1]: best = (w, dur, sr)
                if 1.5 <= dur <= 11.0: break
            w, dur, sr = best
            out = f"vd2_{g}_{age}.wav"; sf.write(out, w, sr)
            f0, jit, shim, hnr = measure(w, sr)
            print(f"{g+'_'+age:8s} {f0:5.0f} {jit:5.2f} {shim:6.2f} {hnr:6.1f}  ({dur:.1f}s) -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
