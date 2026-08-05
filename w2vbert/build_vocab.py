"""Rebuild vocab.json (CTC charset) from the UNION of all training manifests' normalized
text, so stage-1 pretrain and stage-2 SFT share one head. Space -> '|', plus [UNK]/[PAD].
Writes w2vbert/vocab.json. The old best/ keeps its own copy, so this is safe.
"""
import json
import os

from norm import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "finetune", "data")
MANIFESTS = ["train_manifest.json", "casablanca_manifest.json", "omni_manifest.json",
             "moulsot_manifest.json", "dev_manifest.json"]


def main():
    chars = set()
    for m in MANIFESTS:
        p = os.path.join(DATA, m)
        if not os.path.exists(p):
            print(f"  (missing {m}, skipping)")
            continue
        for l in open(p, encoding="utf-8"):
            chars.update(normalize(json.loads(l)["text"]))
    chars.discard(" ")
    vocab = {c: i for i, c in enumerate(sorted(chars))}
    vocab["|"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    with open(os.path.join(HERE, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"wrote vocab.json: {len(vocab)} tokens ({len(chars)} chars + | [UNK] [PAD])")


if __name__ == "__main__":
    main()
