# coding=utf-8
"""
ITER 1: Prove Kokoro -> Qwen3 ICL laundering transfers a British accent.

Pipeline (no human recording anywhere):
  1. Kokoro TTS (lang_code='b') generates British female + male reference clips.
  2. Score the Kokoro originals with the accent classifier (should be high england).
  3. Feed each as an ICL reference (ref_audio + ref_text, x_vector_only_mode=False)
     into Qwen3 Base and synthesize target lines.
  4. Score the Qwen3-laundered outputs -> does P(england) transfer?

The author cannot hear; the classifier is the objective compass, the user judges.
Runs on CPU.
"""
import time, warnings, gc
warnings.filterwarnings("ignore")
import numpy as np
import torch
import soundfile as sf
import librosa

REF_TEXT = ("Honestly, I can't quite believe it. The weather's been rather dreadful "
            "all week, so I'd rather stay in with a cup of tea and a good book.")
TARGETS = {
    "poc":  "Hello, I am Claude Opus four point eight.",
    "diag": "I can't dance, but rather I'd drive the car to the tower.",
}
VOICES = [("bf", "bf_emma", "British female"), ("bm", "bm_george", "British male")]
QWEN_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SR = 24000

GEN = dict(max_new_tokens=512, do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
           repetition_penalty=1.05, subtalker_dosample=True, subtalker_top_k=50,
           subtalker_top_p=1.0, subtalker_temperature=0.9)


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
        from speechbrain.inference.classifiers import EncoderClassifier
        self.clf = EncoderClassifier.from_hparams(
            source="Jzuluaga/accent-id-commonaccent_ecapa", savedir="/tmp/claude-0/accent_model")
        self.lab = self.clf.hparams.label_encoder.ind2lab
        self.eng = [i for i, l in self.lab.items() if l == "england"][0]
        self.us = [i for i, l in self.lab.items() if l == "us"][0]

    def score(self, w, sr):
        if sr != 16000:
            w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=16000)
        sig = torch.tensor(w, dtype=torch.float32).unsqueeze(0)
        out = self.clf.classify_batch(sig)[0].squeeze(0)
        p = torch.softmax(out, dim=-1)
        return float(p[self.eng]), float(p[self.us]), self.lab[int(p.argmax())]


def main():
    torch.set_num_threads(4)

    # ---- Phase 1: Kokoro British reference clips ----
    print("=== Phase 1: Kokoro British references ===")
    from kokoro import KPipeline
    kp = KPipeline(lang_code='b')  # British English
    refs = {}
    for tag, voice, desc in VOICES:
        audio = np.concatenate([a for _, _, a in kp(REF_TEXT, voice=voice)]).astype(np.float32)
        path = f"kokoro_{tag}.wav"
        sf.write(path, audio, SR)
        refs[tag] = (path, audio, desc)
        print(f"  {desc:14s} ({voice}) -> {path}  {len(audio)/SR:.2f}s")
    del kp; gc.collect()

    # ---- Phase 2: score Kokoro originals ----
    print("\n=== Phase 2: classifier on Kokoro originals ===")
    judge = Judge()
    for tag, (path, audio, desc) in refs.items():
        pe, pu, top = judge.score(audio, SR)
        print(f"  {desc:14s}: P(eng)={pe:.3f} P(us)={pu:.3f} top={top} F0={f0_median(audio,SR):.0f}")

    # ---- Phase 3: launder through Qwen3 ICL ----
    print(f"\n=== Phase 3: Qwen3 ICL laundering ({QWEN_PATH}) ===")
    from qwen_tts import Qwen3TTSModel
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(QWEN_PATH, device_map="cpu", dtype=torch.float32,
                                        attn_implementation="eager")
    print(f"  Qwen3 loaded in {time.time()-t0:.1f}s")

    for tag, (path, audio, desc) in refs.items():
        item = tts.create_voice_clone_prompt(ref_audio=path, ref_text=REF_TEXT,
                                             x_vector_only_mode=False)  # ICL mode
        for tkey, ttext in TARGETS.items():
            best = None
            for k in range(2):
                torch.manual_seed(4000 + k)
                wavs, sr = tts.generate_voice_clone(text=ttext, language="English",
                                                    voice_clone_prompt=item, **GEN)
                w, dur = trim(np.asarray(wavs[0], dtype=np.float32), sr)
                if best is None or dur < best[1]:
                    best = (w, dur, sr)
                if 1.0 <= dur <= 9.0:
                    break
            w, dur, sr = best
            out = f"qwen_from_{tag}_{tkey}.wav"
            sf.write(out, w, sr)
            pe, pu, top = judge.score(w, sr)
            print(f"  [{desc:13s} -> qwen {tkey:4s}] P(eng)={pe:.3f} P(us)={pu:.3f} "
                  f"top={top:10s} F0={f0_median(w,sr):.0f} dur={dur:.2f}s -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
