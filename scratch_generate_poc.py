# coding=utf-8
"""
Proof-of-concept: generate a single English voice line with Qwen3-TTS.

Runs on CPU (no GPU in this environment): device_map="cpu",
dtype=torch.float32, attn_implementation="eager".

Uses the 1.7B CustomVoice checkpoint with the built-in English speaker
"Ryan" (no reference audio needed). The 1.7B model is markedly higher
quality than the 0.6B; the 0.6B produced breathy, phoneme-like output.

Uses the full recommended generation params (temperature / top-k / top-p /
repetition penalty AND the sub-talker codec params) — these matter a lot
for the 12 Hz codec; omitting them yields mushy audio.

Generates a few candidates, trims silence, keeps the most compact clean take.
"""
import time
import numpy as np
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
OUT_PATH = "claude_opus_poc.wav"
SPEAKER = "Ryan"
LANGUAGE = "English"

# Listener should hear "Hello, I am Claude Opus 4.8". Spelling the number out
# reads more reliably than the digits "4.8".
CANDIDATE_TEXTS = [
    "Hello, I am Claude Opus four point eight.",
    "Hello, I am Claude Opus four point eight.",
    "Hello, I am Claude Opus 4.8.",
]

GEN_KWARGS = dict(
    max_new_tokens=512,
    do_sample=True,
    top_k=50,
    top_p=1.0,
    temperature=0.9,
    repetition_penalty=1.05,
    subtalker_dosample=True,
    subtalker_top_k=50,
    subtalker_top_p=1.0,
    subtalker_temperature=0.9,
)


def trim_silence(w, sr, thr=0.02, pad=0.15):
    win = max(1, int(0.05 * sr))
    env = np.array([np.sqrt(np.mean(w[i:i + win] ** 2))
                    for i in range(0, max(1, len(w) - win), win)])
    active = np.where(env > thr)[0]
    if len(active) == 0:
        return w, 0.0
    start = max(0, int(active[0] * win - pad * sr))
    end = min(len(w), int((active[-1] + 1) * win + pad * sr))
    return w[start:end], (end - start) / sr


def main():
    torch.set_num_threads(4)

    print(f"Loading {MODEL_PATH} on CPU (float32, eager attention)...")
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    best = None  # (dur, wav, sr, text, seed)
    for i, text in enumerate(CANDIDATE_TEXTS):
        torch.manual_seed(1234 + i)
        t0 = time.time()
        wavs, sr = tts.generate_custom_voice(
            text=text, language=LANGUAGE, speaker=SPEAKER, **GEN_KWARGS,
        )
        raw = np.asarray(wavs[0], dtype=np.float32)
        trimmed, dur = trim_silence(raw, sr)
        print(f"[cand {i}] text={text!r} raw={len(raw)/sr:.2f}s "
              f"trimmed={dur:.2f}s gen={time.time()-t0:.1f}s")
        if 1.5 <= dur <= 6.0:
            if best is None or dur < best[0]:
                best = (dur, trimmed, sr, text, 1234 + i)

    if best is None:
        torch.manual_seed(1234)
        wavs, sr = tts.generate_custom_voice(
            text=CANDIDATE_TEXTS[0], language=LANGUAGE, speaker=SPEAKER,
            **GEN_KWARGS)
        trimmed, dur = trim_silence(np.asarray(wavs[0], dtype=np.float32), sr)
        best = (dur, trimmed, sr, CANDIDATE_TEXTS[0], 1234)

    _, wav, sr, text, seed = best
    sf.write(OUT_PATH, wav, sr)
    print(f"\nSelected take: text={text!r} seed={seed} "
          f"dur={len(wav)/sr:.2f}s -> saved {OUT_PATH}")


if __name__ == "__main__":
    main()
