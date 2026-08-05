"""Tune KenLM alpha/beta on Egypt val: compute w2v-bert logprobs once, then beam-decode
with pyctcdecode + word4.arpa across a small grid. Prints WER per setting.
"""
import json
import multiprocessing
import os

import numpy as np
import soundfile as sf
import torch
from jiwer import wer
from pyctcdecode import build_ctcdecoder

from norm import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "finetune", "data")
MODELDIR = os.environ.get("MODELDIR", os.path.join(HERE, "ckpt", "checkpoint-9000"))
ARPA = os.environ.get("ARPA", os.path.join(os.path.dirname(HERE), "finetune", "word4.arpa"))
SR = 16000
D = os.environ.get("DIALECT", "Egypt")


def host_path(p):
    return p.replace("/data", DATA.replace("\\", "/"), 1)


_SPECIAL = {"|": " ", "[PAD]": "", "[UNK]": "⁇", "<s>": "➀", "</s>": "➁",
            "<pad>": "", "<unk>": "⁇"}


def build_labels(vocab):  # vocab: token -> id (full tokenizer vocab, incl specials)
    inv = {v: k for k, v in vocab.items()}
    return [_SPECIAL.get(inv[i], inv[i]) for i in range(len(inv))]


def main():
    from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor
    proc = Wav2Vec2BertProcessor.from_pretrained(MODELDIR)
    model = Wav2Vec2BertForCTC.from_pretrained(MODELDIR, torch_dtype=torch.float16).cuda().eval()
    labels = build_labels(proc.tokenizer.get_vocab())
    assert len(labels) == model.config.vocab_size, (len(labels), model.config.vocab_size)

    rows = [json.loads(l) for l in open(os.path.join(DATA, "val", f"{D}.json"), encoding="utf-8")]
    paths = [host_path(r["audio_filepath"]) for r in rows]
    refs = [normalize(r["text"]) for r in rows]

    # compute logprobs once (length-sorted for speed), keep in val order
    order = sorted(range(len(paths)), key=lambda i: os.path.getsize(paths[i]))
    logps = [None] * len(paths)
    B = 16
    for s in range(0, len(order), B):
        idx = order[s:s + B]
        wavs = [sf.read(paths[i], dtype="float32")[0] for i in idx]
        wavs = [w.mean(1) if w.ndim > 1 else w for w in wavs]
        feats = proc.feature_extractor(wavs, sampling_rate=SR, return_tensors="pt", padding=True)
        am = feats.get("attention_mask")
        feats = {k: (v.cuda().half() if v.dtype == torch.float32 else v.cuda()) for k, v in feats.items()}
        with torch.no_grad():
            logits = model(**feats).logits.float()
        lp = torch.log_softmax(logits, -1).cpu().numpy()
        for j, i in enumerate(idx):
            L = int(am[j].sum().item()) if am is not None else lp.shape[1]
            # adapter downsamples ~2x; use output length directly
            logps[i] = lp[j][: logits.shape[1]] if am is None else lp[j]
    print(f"{D}: computed logprobs for {len(paths)} clips", flush=True)

    # greedy baseline
    greedy = [normalize(proc.batch_decode(np.argmax(lp, -1)[None])[0]) for lp in logps]
    print(f"greedy WER: {wer(refs, [h or '@' for h in greedy]):.3f}", flush=True)

    grid = [(float(a), float(b)) for a in os.environ.get("ALPHAS", "0.3,0.5,0.8").split(",")
            for b in os.environ.get("BETAS", "0.5,1.5").split(",")]
    with multiprocessing.Pool(4) as pool:
        for a, b in grid:
            dec = build_ctcdecoder(labels, kenlm_model_path=ARPA, alpha=a, beta=b)
            hyps = dec.decode_batch(pool, logps, beam_width=100)
            hyps = [normalize(h) or "@" for h in hyps]
            print(f"alpha={a} beta={b}: WER={wer(refs, hyps):.3f}", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
