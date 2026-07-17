# coding=utf-8
"""
Experiment: can we push a British accent WITHOUT any recording, using only
the two self-contained levers in Qwen3-TTS Base?
  - a female x-vector (timbre)  +  orthographic respelling (pseudo-phonetic)

The model has: only an "english" language tag (no en-GB), raw-text->BPE input
(no phoneme/IPA path), American-trained acoustics. So respelling is the ONLY
in-model text lever for accent. This renders the same voice across text
treatments so a human can judge whether orthography moves the accent.

NOTE: the author (Claude) cannot hear audio -- these are for the USER to judge.
Runs on CPU.
"""
import time
import numpy as np
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
LANGUAGE = "English"
FEMALE_SRC = "claude_opus_british_female.wav"  # female timbre (accent aside)

# (tag, text). Diagnostic lines chosen to expose rhoticity + TRAP-BATH.
CASES = [
    ("poc_plain",      "Hello, I am Claude Opus four point eight."),
    ("poc_respell",    "Hulloh, I am Claude Opus faw point ait."),
    ("diag_plain",     "Can't you see the water tower over there? I'd rather dance."),
    ("diag_respell",   "Cahn't you see the wawtah tauwah ovah theah? I'd rahther dahnce."),
]

GEN_KWARGS = dict(
    max_new_tokens=512, do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
    repetition_penalty=1.05, subtalker_dosample=True, subtalker_top_k=50,
    subtalker_top_p=1.0, subtalker_temperature=0.9,
)


def trim_silence(w, sr, thr=0.02, pad=0.15):
    win = max(1, int(0.05 * sr))
    env = np.array([np.sqrt(np.mean(w[i:i + win] ** 2))
                    for i in range(0, max(1, len(w) - win), win)])
    a = np.where(env > thr)[0]
    if len(a) == 0:
        return w, 0.0
    s = max(0, int(a[0] * win - pad * sr)); e = min(len(w), int((a[-1] + 1) * win + pad * sr))
    return w[s:e], (e - s) / sr


def f0_estimate(w, sr):
    """Rough median F0 via autocorrelation -> objective gender sanity check."""
    try:
        import librosa
        f0 = librosa.yin(w.astype(np.float32), fmin=70, fmax=400, sr=sr)
        f0 = f0[np.isfinite(f0)]
        return float(np.median(f0)) if len(f0) else float("nan")
    except Exception:
        return float("nan")


def render(tts, spk_emb, text, seed_base):
    best = None
    for k in range(3):
        torch.manual_seed(seed_base + k)
        item = VoiceClonePromptItem(ref_code=None, ref_spk_embedding=spk_emb,
                                    x_vector_only_mode=True, icl_mode=False, ref_text=None)
        wavs, sr = tts.generate_voice_clone(text=text, language=LANGUAGE,
                                            voice_clone_prompt=[item], **GEN_KWARGS)
        w, dur = trim_silence(np.asarray(wavs[0], dtype=np.float32), sr)
        if best is None or dur < best[1]:
            best = (w, dur, sr)
        if 1.2 <= dur <= 9.0:
            return w, dur, sr
    return best


def main():
    torch.set_num_threads(4)
    print(f"Loading {MODEL_PATH} on CPU...")
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map="cpu",
                                        dtype=torch.float32, attn_implementation="eager")
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    item = tts.create_voice_clone_prompt(ref_audio=FEMALE_SRC, x_vector_only_mode=True)[0]
    spk = item.ref_spk_embedding
    print(f"Female x-vector from {FEMALE_SRC}: shape={tuple(spk.shape)}\n")

    for i, (tag, text) in enumerate(CASES):
        t0 = time.time()
        w, dur, sr = render(tts, spk, text, seed_base=7000 + 100 * i)
        out = f"british_{i}_{tag}.wav"
        sf.write(out, w, sr)
        print(f"[{tag:13s}] dur={dur:.2f}s F0~{f0_estimate(w,sr):.0f}Hz "
              f"gen={time.time()-t0:.1f}s text={text!r} -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
