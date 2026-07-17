# coding=utf-8
"""
Demo: instruction-driven expressive control in Qwen3-TTS.

Qwen3-TTS has NO inline markers like [laugh]/[whisper]. Expressive control is
done with a natural-language `instruct` string (supported on the 1.7B
CustomVoice and VoiceDesign models; the 0.6B CustomVoice ignores instruct).

Here we hold the voice (speaker "Ryan") and the text constant and vary only
the `instruct` emotion/style, so you can hear what the instruct mechanism does.

Runs on CPU: device_map="cpu", dtype=torch.float32, attn_implementation="eager".
"""
import time
import numpy as np
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SPEAKER = "Ryan"
LANGUAGE = "English"
TEXT = "Hello, I am Claude Opus four point eight."

# (filename-tag, instruct string)
STYLES = [
    ("neutral",  ""),  # baseline, no instruct
    ("angry",    "Say it in a furious, seething, angry tone."),
    ("whisper",  "Whisper it very softly and breathily, as if sharing a secret."),
    ("excited",  "Say it with bright, joyful, high-energy excitement."),
    ("sad",      "Say it in a slow, sombre, melancholic tone, on the verge of tears."),
    ("laughing", "Say it while laughing, amused and giddy, chuckling between words."),
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


def gen_one(tts, instruct, base_seed):
    """Generate up to 3 takes, keep the first compact one (1.2-8s)."""
    best = None
    for k in range(3):
        torch.manual_seed(base_seed + k)
        wavs, sr = tts.generate_custom_voice(
            text=TEXT, language=LANGUAGE, speaker=SPEAKER,
            instruct=(instruct or None), **GEN_KWARGS,
        )
        w, dur = trim_silence(np.asarray(wavs[0], dtype=np.float32), sr)
        if best is None or dur < best[1]:
            best = (w, dur, sr)
        if 1.2 <= dur <= 8.0:
            return w, dur, sr, k
    return best[0], best[1], best[2], -1


def main():
    torch.set_num_threads(4)
    print(f"Loading {MODEL_PATH} on CPU...")
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH, device_map="cpu", dtype=torch.float32,
        attn_implementation="eager",
    )
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    for i, (tag, instruct) in enumerate(STYLES):
        t0 = time.time()
        w, dur, sr, k = gen_one(tts, instruct, base_seed=3000 + 10 * i)
        out = f"style_{i}_{tag}.wav"
        sf.write(out, w, sr)
        print(f"[{tag:9s}] dur={dur:.2f}s take={k} gen={time.time()-t0:.1f}s "
              f"instruct={instruct!r} -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
