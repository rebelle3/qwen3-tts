# coding=utf-8
"""
Explore the Qwen3-TTS speaker latent space (x-vector) with the BASE model.

Demonstrates that a voice = one continuous ~D-dim ECAPA x-vector, and that you
can navigate the space WITHOUT a fresh recording by interpolating between
existing speaker vectors and feeding them back via x_vector_only_mode.

Source voices are two clips we already generated this session:
  A = style_0_neutral.wav          (male, "Ryan")
  B = claude_opus_british_female.wav (British female, VoiceDesign)

Steps:
  1. Extract x-vector for A and B (via create_voice_clone_prompt, x-vector only).
  2. Print what the vectors look like (shape/dtype/norm/cosine sim/head).
  3. Interpolate A->B at several blend factors, hand-build VoiceClonePromptItems
     with the synthetic vectors, and render the same line at each blend.

Runs on CPU: device_map="cpu", dtype=torch.float32, attn_implementation="eager".
"""
import time
import numpy as np
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
LANGUAGE = "English"
TEXT = "Hello, I am Claude Opus four point eight."
SRC_A = "style_0_neutral.wav"            # male
SRC_B = "claude_opus_british_female.wav"  # british female
BLENDS = [0.0, 0.25, 0.5, 0.75, 1.0]      # 0 = pure A, 1 = pure B

GEN_KWARGS = dict(
    max_new_tokens=512, do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
    repetition_penalty=1.05, subtalker_dosample=True, subtalker_top_k=50,
    subtalker_top_p=1.0, subtalker_temperature=0.9,
)


def trim_silence(w, sr, thr=0.02, pad=0.15):
    win = max(1, int(0.05 * sr))
    env = np.array([np.sqrt(np.mean(w[i:i + win] ** 2))
                    for i in range(0, max(1, len(w) - win), win)])
    active = np.where(env > thr)[0]
    if len(active) == 0:
        return w, 0.0
    s = max(0, int(active[0] * win - pad * sr))
    e = min(len(w), int((active[-1] + 1) * win + pad * sr))
    return w[s:e], (e - s) / sr


def lerp_vec(v0, v1, a):
    """Linear interpolation, rescaled to the interpolated L2 norm (keeps the
    blended vector on a sensible magnitude shell rather than shrinking mid-way)."""
    v = (1 - a) * v0 + a * v1
    target_norm = (1 - a) * v0.norm() + a * v1.norm()
    cur = v.norm()
    if cur > 0:
        v = v * (target_norm / cur)
    return v


def render(tts, spk_emb, seed_base):
    best = None
    for k in range(3):
        torch.manual_seed(seed_base + k)
        item = VoiceClonePromptItem(
            ref_code=None, ref_spk_embedding=spk_emb,
            x_vector_only_mode=True, icl_mode=False, ref_text=None,
        )
        wavs, sr = tts.generate_voice_clone(
            text=TEXT, language=LANGUAGE, voice_clone_prompt=[item], **GEN_KWARGS)
        w, dur = trim_silence(np.asarray(wavs[0], dtype=np.float32), sr)
        if best is None or dur < best[1]:
            best = (w, dur, sr)
        if 1.2 <= dur <= 8.0:
            return w, dur, sr
    return best


def main():
    torch.set_num_threads(4)
    print(f"Loading {MODEL_PATH} on CPU (this downloads ~4.5GB on first run)...")
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH, device_map="cpu", dtype=torch.float32,
        attn_implementation="eager")
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # 1) Extract x-vectors (x-vector-only => no ref_text needed).
    items = tts.create_voice_clone_prompt(
        ref_audio=[SRC_A, SRC_B], x_vector_only_mode=[True, True])
    vA = items[0].ref_spk_embedding.detach().float().flatten().cpu()
    vB = items[1].ref_spk_embedding.detach().float().flatten().cpu()

    cos = torch.dot(vA, vB) / (vA.norm() * vB.norm() + 1e-9)
    print("=== x-vector inspection ===")
    print(f"A ({SRC_A}): shape={tuple(items[0].ref_spk_embedding.shape)} "
          f"dtype={items[0].ref_spk_embedding.dtype} "
          f"L2norm={vA.norm():.3f} head={np.array2string(vA[:8].numpy(), precision=3)}")
    print(f"B ({SRC_B}): shape={tuple(items[1].ref_spk_embedding.shape)} "
          f"L2norm={vB.norm():.3f} head={np.array2string(vB[:8].numpy(), precision=3)}")
    print(f"cosine(A,B) = {cos:.4f}   L2 distance = {(vA-vB).norm():.3f}\n")

    # 2) Interpolate and render.
    print("=== rendering interpolations ===")
    for i, a in enumerate(BLENDS):
        blended = lerp_vec(vA, vB, a).to(items[0].ref_spk_embedding.dtype)
        t0 = time.time()
        w, dur, sr = render(tts, blended, seed_base=5000 + 100 * i)
        out = f"latent_blend_{int(a*100):03d}.wav"
        sf.write(out, w, sr)
        print(f"[blend {a:.2f}] |v|={blended.norm():.3f} dur={dur:.2f}s "
              f"gen={time.time()-t0:.1f}s -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
