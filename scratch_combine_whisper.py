# coding=utf-8
"""
Combine clone + a 'design' control the reliable way: DSP post-process.

'Make them whisper' is signal-definable: remove the voiced (harmonic) glottal
source and drive the SAME spectral envelope (words + timbre + accent) with noise
excitation -> a real whisper of the cloned British voice, accent preserved.

Uses WORLD: set f0=0 (unvoiced) and aperiodicity->1 (noise). Also a 'breathy'
variant (keep pitch, raise aperiodicity) for a softer effect.
Fast: pure DSP, no model.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa, pyworld as pw

SR = 24000
SOURCES = [("bf", "qwen_from_bf_poc.wav"), ("bm", "qwen_from_bm_poc.wav")]


def analyze(x, sr):
    x = np.ascontiguousarray(x.astype(np.float64))
    f0, t = pw.harvest(x, sr); sp = pw.cheaptrick(x, f0, t, sr); ap = pw.d4c(x, f0, t, sr)
    return f0, sp, ap


def whisper(x, sr):
    """Full whisper: no voiced excitation (f0=0), fully aperiodic (noise)."""
    f0, sp, ap = analyze(x, sr)
    f0w = np.zeros_like(f0)
    apw = np.ones_like(ap)
    y = pw.synthesize(f0w, np.ascontiguousarray(sp), np.ascontiguousarray(apw), sr)
    y = y.astype(np.float32)
    return y / (np.max(np.abs(y)) + 1e-9) * 0.7  # whispers sit lower


def breathy(x, sr, amt=0.7):
    """Breathy voice: keep pitch, push aperiodicity up (partial noise mix)."""
    f0, sp, ap = analyze(x, sr)
    apw = np.clip(ap + amt * (1 - ap), 0, 1)
    y = pw.synthesize(f0, np.ascontiguousarray(sp), np.ascontiguousarray(apw), sr)
    y = y.astype(np.float32)
    return y / (np.max(np.abs(y)) + 1e-9) * 0.9


def main():
    for tag, path in SOURCES:
        x, sr = sf.read(path)
        if sr != SR:
            x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=SR); sr = SR
        x = x.astype(np.float32)
        sf.write(f"combine_{tag}_whisper.wav", whisper(x, sr), sr)
        sf.write(f"combine_{tag}_breathy.wav", breathy(x, sr), sr)
        print(f"{tag}: wrote combine_{tag}_whisper.wav, combine_{tag}_breathy.wav")
    print("Done.")


if __name__ == "__main__":
    main()
