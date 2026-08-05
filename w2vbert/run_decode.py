"""Unified CTC+KenLM decoder for holdout (scored) or test (submission).
Env:
  MODELDIR (model weights), PROC (processor w/ vocab; default=MODELDIR),
  SRC = "holdout" | "test"        (holdout reads finetune/data/holdout/{d}.json and SCORES;
                                    test reads the HF test parquets, writes submission order)
  OUT (dir), ARPA (default big5.arpa),
  ABMAP (path to json {dialect:[alpha,beta]}; if unset uses global ALPHA/BETA),
  ALPHA/BETA/BEAM globals (defaults 0.5/1.5/150), DIALECTS (comma filter).
Writes OUT/{d}.txt (row order, trailing newline every line). For holdout prints per-dialect+weighted WER/CER.
"""
import io, json, multiprocessing, os
import soundfile as sf
import torch
from pyctcdecode import build_ctcdecoder
from norm import normalize
from tune_lm import build_labels, host_path

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "finetune", "data")
MODELDIR = os.environ["MODELDIR"]
PROC = os.environ.get("PROC", MODELDIR)
SRC = os.environ.get("SRC", "holdout")
OUT = os.environ["OUT"]
ARPA = os.environ.get("ARPA", os.path.join(os.path.dirname(HERE), "finetune", "big5.arpa"))
ABMAP = json.load(open(os.environ["ABMAP"])) if os.environ.get("ABMAP") else None
GA = float(os.environ.get("ALPHA", 0.5)); GB = float(os.environ.get("BETA", 1.5))
BEAM = int(os.environ.get("BEAM", 150)); SR = 16000
TEST_REPO = "UBC-NLP/NADI2026_subtask1.1_Robust_ASR_test"
TEST_FILES = {"Algeria":"Algeria/test-00000-of-00001-3b4e9c2476f25ec4.parquet","Egypt":"Egypt/test-00000-of-00001-2532a42b73f1a031.parquet","Jordan":"Jordan/test-00000-of-00001-7d01ca6bb8866136.parquet","Mauritania":"Mauritania/test-00000-of-00001-d92b2e83805884fa.parquet","Morocco":"Morocco/test-00000-of-00001-44dba384143ec1fb.parquet","Palestine":"Palestine/test-00000-of-00001-810cb09fe8893c33.parquet","UAE":"UAE/test-00000-of-00001-6a02ffceda1820d4.parquet","Yemen":"Yemen/test-00000-of-00001-648e4447dafc4521.parquet"}
DIALS = os.environ.get("DIALECTS", ",".join(TEST_FILES)).split(",")


def load_src(d):
    """Return (wavs, refs_or_None)."""
    if SRC == "test":
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        t = pq.read_table(hf_hub_download(TEST_REPO, TEST_FILES[d], repo_type="dataset"))
        wavs = []
        for a in t.column("audio").to_pylist():
            w, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
            wavs.append(w.mean(1) if w.ndim > 1 else w)
        return wavs, None
    rows = [json.loads(l) for l in open(os.path.join(DATA, "holdout", f"{d}.json"), encoding="utf-8")]
    wavs = []
    for r in rows:
        w, sr = sf.read(host_path(r["audio_filepath"]), dtype="float32")
        wavs.append(w.mean(1) if w.ndim > 1 else w)
    return wavs, [normalize(r["text"]) for r in rows]


def main():
    from jiwer import cer, wer
    from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor
    proc = Wav2Vec2BertProcessor.from_pretrained(PROC)
    model = Wav2Vec2BertForCTC.from_pretrained(MODELDIR, torch_dtype=torch.float16).cuda().eval()
    labels = build_labels(proc.tokenizer.get_vocab())
    assert len(labels) == model.config.vocab_size, (len(labels), model.config.vocab_size)
    os.makedirs(OUT, exist_ok=True)
    pool = multiprocessing.Pool(4)
    print(f"MODEL={MODELDIR} SRC={SRC} ABMAP={'yes' if ABMAP else f'global {GA}/{GB}'}", flush=True)
    tw = tc = tn = 0
    for d in DIALS:
        wavs, refs = load_src(d)
        n = len(wavs)
        order = sorted(range(n), key=lambda i: len(wavs[i]))
        logps = [None] * n
        B = 16
        for s in range(0, n, B):
            idx = order[s:s + B]
            feats = proc.feature_extractor([wavs[i] for i in idx], sampling_rate=SR, return_tensors="pt", padding=True)
            feats = {k: (v.cuda().half() if v.dtype == torch.float32 else v.cuda()) for k, v in feats.items()}
            with torch.no_grad():
                logits = model(**feats).logits.float()
            lp = torch.log_softmax(logits, -1).cpu().numpy()
            for j, i in enumerate(idx):
                logps[i] = lp[j]
        a, b = (ABMAP[d] if ABMAP else (GA, GB))
        decoder = build_ctcdecoder(labels, kenlm_model_path=ARPA, alpha=float(a), beta=float(b))
        hyps = [normalize(h) for h in decoder.decode_batch(pool, logps, beam_width=BEAM)]
        with open(os.path.join(OUT, f"{d}.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("".join((h or "") + "\n" for h in hyps))
        if refs is not None:
            H = [h or "@" for h in hyps]; w, c = wer(refs, H), cer(refs, H)
            print(f"{d:<12}{w:>8.3f}{c:>8.3f}{n:>7}  a={a} b={b}", flush=True)
            tw += w * n; tc += c * n; tn += n
        else:
            print(f"{d:<12} n={n} blanks={sum(1 for h in hyps if not h.strip())}", flush=True)
    pool.close()
    if tn:
        print(f"{'WEIGHTED':<12}{tw/tn:>8.3f}{tc/tn:>8.3f}{tn:>7}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
