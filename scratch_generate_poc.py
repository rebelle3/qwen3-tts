# coding=utf-8
"""
Proof-of-concept: generate a single English voice line with Qwen3-TTS.

Adapted to run on CPU (no GPU available in this environment):
  - device_map="cpu"
  - dtype=torch.float32
  - attn_implementation="eager" (flash-attn / sdpa not needed on CPU)

Uses the smaller 0.6B CustomVoice checkpoint (~2.5 GB) with a built-in
English speaker ("Ryan"), so no reference audio clip is required.
"""
import time
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
OUT_PATH = "claude_opus_poc.wav"
TEXT = "Hello, I am Claude opus 4.8"
SPEAKER = "Ryan"        # built-in English male voice
LANGUAGE = "English"


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

    spk = tts.get_supported_speakers()
    langs = tts.get_supported_languages()
    print(f"Supported speakers: {spk}")
    print(f"Supported languages: {langs}")

    print(f"Generating: {TEXT!r} (speaker={SPEAKER})...")
    t0 = time.time()
    wavs, sr = tts.generate_custom_voice(
        text=TEXT,
        language=LANGUAGE,
        speaker=SPEAKER,
        max_new_tokens=512,
    )
    print(f"Generated in {time.time() - t0:.1f}s, sr={sr}, n={len(wavs)}")

    sf.write(OUT_PATH, wavs[0], sr)
    dur = len(wavs[0]) / sr
    print(f"Saved {OUT_PATH} ({dur:.2f}s of audio)")


if __name__ == "__main__":
    main()
