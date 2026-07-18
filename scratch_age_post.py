# coding=utf-8
"""
LOOP ITER 6: age as a POST-processor on Qwen3 output.

Iter 5 showed Qwen3's 12Hz codec DENOISES away phonation-quality aging cues
(jitter/shimmer/breathiness) when they're injected upstream in the reference.
So we apply the same vocal-aging model DOWNSTREAM -- on Qwen3's already-rendered
British clip -- where no codec follows to strip it. Qwen3 supplies accent +
words + gender/timbre; the WORLD aging filter supplies age.

Measures jitter/shimmer/HNR with Praat to confirm the cues now SURVIVE.
Fast: pure DSP, no model load.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import soundfile as sf
import librosa

from scratch_aging_model import age_transform, measure, LEVELS, ORDER

# Clean Qwen3 British outputs to age (your line), female + male.
SOURCES = [
    ("bf", "qwen_from_bf_poc.wav"),   # British female, "Hello, I am Claude Opus 4.8"
    ("bm", "qwen_from_bm_poc.wav"),   # British male
]
SR = 24000


def main():
    for tag, path in SOURCES:
        x, sr = sf.read(path)
        if sr != SR:
            x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=SR); sr = SR
        x = x.astype(np.float32)
        print(f"\n=== POST-aging {path} ({tag}) ===")
        print(f"{'level':9s} {'F0':>5s} {'jit%':>5s} {'shim%':>6s} {'HNR':>6s}")
        f0m, jit, shim, hnr = measure(x, sr)
        print(f"{'ORIGINAL':9s} {f0m:5.0f} {jit:5.2f} {shim:6.2f} {hnr:6.1f}  (clean Qwen3)")
        for lv in ORDER:
            y = age_transform(x, sr, LEVELS[lv])
            out = f"post_{tag}_{lv}.wav"
            sf.write(out, y, sr)
            f0m, jit, shim, hnr = measure(y, sr)
            print(f"{lv:9s} {f0m:5.0f} {jit:5.2f} {shim:6.2f} {hnr:6.1f}  -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
