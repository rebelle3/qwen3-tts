# coding=utf-8
"""
Search the Qwen3-TTS speaker latent space for a BRITISH (england) accent,
guided by an objective English-accent classifier -- no recording, no human
judgment in the loop (the author cannot hear; the classifier is the compass,
the user is the final judge).

Fitness = P(england) from Jzuluaga/accent-id-commonaccent_ecapa on the rendered
clip. Constraints keep the candidate female (F0) + intelligible (rms/dur), and
the search stays NEAR the real-voice manifold (bounded perturbations of a real
female x-vector) to avoid adversarial vectors that fool the classifier without
sounding like speech.

Evolutionary near-manifold search:
  Phase A (explore): perturb seed at several radii.
  Phase B (refine):  perturb the best few at a small radius.
  Robustness check:  re-render finalists with a fresh seed; keep P(england) that holds.

Runs on CPU. Progress is logged line-by-line for monitoring.
"""
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import soundfile as sf
import librosa

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
from speechbrain.inference.classifiers import EncoderClassifier

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
FEMALE_SRC = "claude_opus_british_female.wav"
LANGUAGE = "English"
# Accent-rich but short (keeps render cost down): rhoticity + TRAP-BATH + LOT.
SCORE_TEXT = "I can't dance, but rather I'd drive the car to the tower."

GEN = dict(max_new_tokens=384, do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
           repetition_penalty=1.05, subtalker_dosample=True, subtalker_top_k=50,
           subtalker_top_p=1.0, subtalker_temperature=0.9)

rng = np.random.RandomState(0)  # Math.random is unavailable; use seeded numpy


def trim(w, sr, thr=0.02, pad=0.12):
    win = max(1, int(0.05 * sr))
    env = np.array([np.sqrt(np.mean(w[i:i+win]**2)) for i in range(0, max(1, len(w)-win), win)])
    a = np.where(env > thr)[0]
    if len(a) == 0: return w, 0.0
    s = max(0, int(a[0]*win - pad*sr)); e = min(len(w), int((a[-1]+1)*win + pad*sr))
    return w[s:e], (e-s)/sr


def f0_median(w, sr):
    try:
        f0 = librosa.yin(w.astype(np.float32), fmin=80, fmax=400, sr=sr)
        f0 = f0[np.isfinite(f0)]
        return float(np.median(f0)) if len(f0) else 0.0
    except Exception:
        return 0.0


class Judge:
    def __init__(self):
        self.clf = EncoderClassifier.from_hparams(
            source="Jzuluaga/accent-id-commonaccent_ecapa",
            savedir="/tmp/claude-0/accent_model")
        self.lab = self.clf.hparams.label_encoder.ind2lab
        self.eng = [i for i, l in self.lab.items() if l == "england"][0]
        self.us = [i for i, l in self.lab.items() if l == "us"][0]

    def probs(self, w, sr):
        if sr != 16000:
            w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=16000)
        sig = torch.tensor(w, dtype=torch.float32).unsqueeze(0)
        out = self.clf.classify_batch(sig)[0].squeeze(0)
        p = torch.softmax(out, dim=-1)
        return float(p[self.eng]), float(p[self.us]), self.lab[int(p.argmax())]


def render(tts, vec, seed):
    torch.manual_seed(seed)
    item = VoiceClonePromptItem(ref_code=None, ref_spk_embedding=vec,
                                x_vector_only_mode=True, icl_mode=False, ref_text=None)
    wavs, sr = tts.generate_voice_clone(text=SCORE_TEXT, language=LANGUAGE,
                                        voice_clone_prompt=[item], **GEN)
    return trim(np.asarray(wavs[0], dtype=np.float32), sr) + (sr,)


def evaluate(tts, judge, vec, seed):
    w, dur, sr = render(tts, vec, seed)
    rms = float(np.sqrt(np.mean(w**2))) if len(w) else 0.0
    if dur < 1.0 or dur > 8.0 or rms < 0.02:
        return -1.0, dict(dur=dur, rms=rms, f0=0, top="reject", p_eng=0, p_us=0)
    f0 = f0_median(w, sr)
    p_eng, p_us, top = judge.probs(w, sr)
    female = 150 <= f0 <= 320
    fit = p_eng - (0.0 if female else 0.5)  # penalize non-female timbre
    return fit, dict(dur=dur, rms=rms, f0=f0, top=top, p_eng=p_eng, p_us=p_us)


def perturb(seed_vec, radius):
    n = torch.tensor(rng.randn(seed_vec.numel()), dtype=torch.float32)
    n = n / n.norm() * (seed_vec.norm() * radius)
    return seed_vec + n


def main():
    torch.set_num_threads(4)
    print(f"Loading TTS {MODEL_PATH}...")
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map="cpu",
                                        dtype=torch.float32, attn_implementation="eager")
    print(f"TTS loaded in {time.time()-t0:.1f}s")
    judge = Judge()
    print("Accent judge ready.\n")

    item = tts.create_voice_clone_prompt(ref_audio=FEMALE_SRC, x_vector_only_mode=True)[0]
    dt = item.ref_spk_embedding.dtype
    seed_vec = item.ref_spk_embedding.detach().float().flatten().cpu()

    def as_model(v): return v.to(dt)

    # Baseline
    fit, m = evaluate(tts, judge, as_model(seed_vec), seed=100)
    print(f"[baseline] P(eng)={m['p_eng']:.3f} P(us)={m['p_us']:.3f} top={m['top']} F0={m['f0']:.0f}")

    pool = [(fit, seed_vec, m)]  # (fitness, vector, meta)
    eid = 0

    # Phase A: explore radii
    print("\n=== Phase A: explore ===")
    for radius in [0.08, 0.15, 0.25]:
        for _ in range(6):
            eid += 1
            cand = perturb(seed_vec, radius)
            fit, m = evaluate(tts, judge, as_model(cand), seed=1000 + eid)
            pool.append((fit, cand, m))
            print(f"[A r={radius:.2f} #{eid:02d}] P(eng)={m['p_eng']:.3f} "
                  f"top={m['top']:12s} F0={m['f0']:.0f} fit={fit:.3f}")

    pool.sort(key=lambda x: x[0], reverse=True)
    tops = pool[:3]
    print(f"\nTop after A: " + ", ".join(f"P(eng)={t[2]['p_eng']:.3f}({t[2]['top']})" for t in tops))

    # Phase B: refine around top-3
    print("\n=== Phase B: refine ===")
    refined = list(tops)
    for base_fit, base_vec, _ in tops:
        for _ in range(5):
            eid += 1
            cand = perturb(base_vec, 0.06)
            fit, m = evaluate(tts, judge, as_model(cand), seed=2000 + eid)
            refined.append((fit, cand, m))
            print(f"[B #{eid:02d}] P(eng)={m['p_eng']:.3f} top={m['top']:12s} "
                  f"F0={m['f0']:.0f} fit={fit:.3f}")

    refined.sort(key=lambda x: x[0], reverse=True)

    # Robustness: re-score finalists with a fresh render seed; keep min P(eng)
    print("\n=== Robustness re-check (fresh seed) ===")
    finals = []
    for i, (fit, vec, m) in enumerate(refined[:4]):
        fit2, m2 = evaluate(tts, judge, as_model(vec), seed=9000 + i)
        robust = min(m['p_eng'], m2['p_eng'])
        finals.append((robust, vec, m, m2))
        print(f"[final {i}] P(eng) run1={m['p_eng']:.3f} run2={m2['p_eng']:.3f} "
              f"robust={robust:.3f} tops=({m['top']},{m2['top']})")

    finals.sort(key=lambda x: x[0], reverse=True)
    for i, (robust, vec, m, m2) in enumerate(finals[:3]):
        np.save(f"british_xvec_{i}.npy", vec.numpy())
        print(f"SAVED british_xvec_{i}.npy robust_P(eng)={robust:.3f}")

    print(f"\nBaseline P(eng)={pool[-1] if False else ''}")
    print("Done.")


if __name__ == "__main__":
    main()
