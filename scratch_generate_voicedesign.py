# coding=utf-8
"""
Proof-of-concept: Qwen3-TTS VoiceDesign on CPU.

VoiceDesign synthesizes a voice from a natural-language description of the
persona/style (no reference audio, no fixed speaker id). Here we describe a
British English female voice and have it read the same phrase.

Runs on CPU: device_map="cpu", dtype=torch.float32, attn_implementation="eager".
Generates a few candidates, trims silence, keeps the most compact clean take.
"""
import time
import numpy as np
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
OUT_PATH = "claude_opus_british_female.wav"
LANGUAGE = "English"

TEXT = "Hello, I am Claude Opus four point eight."

INSTRUCT = (
    "A refined British English female voice with a crisp Received Pronunciation "
    "accent. Warm, articulate and composed, speaking at a calm, measured pace "
    "with clear enunciation and a gentle, friendly tone."
)

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

    best = None  # (dur, wav, sr, seed)
    for i in range(4):
        torch.manual_seed(2024 + i)
        t0 = time.time()
        wavs, sr = tts.generate_voice_design(
            text=TEXT, language=LANGUAGE, instruct=INSTRUCT, **GEN_KWARGS,
        )
        raw = np.asarray(wavs[0], dtype=np.float32)
        trimmed, dur = trim_silence(raw, sr)
        print(f"[cand {i}] seed={2024+i} raw={len(raw)/sr:.2f}s "
              f"trimmed={dur:.2f}s gen={time.time()-t0:.1f}s")
        if 1.5 <= dur <= 6.0:
            if best is None or dur < best[0]:
                best = (dur, trimmed, sr, 2024 + i)

    if best is None:
        torch.manual_seed(2024)
        wavs, sr = tts.generate_voice_design(
            text=TEXT, language=LANGUAGE, instruct=INSTRUCT, **GEN_KWARGS)
        trimmed, dur = trim_silence(np.asarray(wavs[0], dtype=np.float32), sr)
        best = (dur, trimmed, sr, 2024)

    _, wav, sr, seed = best
    sf.write(OUT_PATH, wav, sr)
    print(f"\nSelected take: seed={seed} dur={len(wav)/sr:.2f}s -> saved {OUT_PATH}")


if __name__ == "__main__":
    main()
