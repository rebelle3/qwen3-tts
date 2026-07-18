# coding=utf-8
"""
Tasteful aging: NO jitter/shimmer/tremor (those sound artifacty/wobbly).
Only the natural macro cues -- gentle breathiness, slower rate, spectral tilt,
mild formant/range -- applied downstream of Qwen3. Ceiling is "mature", not
"frail elderly" (that needs a real aged performance).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa
from scratch_aging_model import age_transform, measure

SR = 24000
# jitter=0, shimmer=0, trem=0 everywhere -> no perturbation artifacts.
VARIANTS = {
    "mature":   dict(formant=0.98, f0s=0.99, f0range=0.85, jitter=0.0, shimmer=0.0,
                     trem_f0=0.0, trem_amp=0.0, breath=0.16, bw=2, tilt=-1.5, rate=1.06),
    "senior":   dict(formant=0.95, f0s=0.98, f0range=0.70, jitter=0.0, shimmer=0.0,
                     trem_f0=0.0, trem_amp=0.0, breath=0.30, bw=3, tilt=-2.5, rate=1.12),
}
SOURCES = [("bf", "qwen_from_bf_poc.wav"), ("bm", "qwen_from_bm_poc.wav")]


def main():
    for tag, path in SOURCES:
        x, sr = sf.read(path)
        if sr != SR:
            x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=SR); sr = SR
        x = x.astype(np.float32)
        print(f"\n=== {path} ===")
        f0, jit, shim, hnr = measure(x, sr)
        print(f"{'ORIGINAL':8s} F0={f0:3.0f} jit={jit:.2f} shim={shim:.2f} HNR={hnr:.1f}")
        for name, p in VARIANTS.items():
            y = age_transform(x, sr, p)
            out = f"tasteful_{tag}_{name}.wav"; sf.write(out, y, sr)
            f0, jit, shim, hnr = measure(y, sr)
            print(f"{name:8s} F0={f0:3.0f} jit={jit:.2f} shim={shim:.2f} HNR={hnr:.1f} -> {out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
