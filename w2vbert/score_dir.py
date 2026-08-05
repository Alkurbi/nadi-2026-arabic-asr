"""Score a prediction dir against val refs with the official norm. Usage: python score_dir.py <dir> [<dir> ...]"""
import json, os, sys
from jiwer import cer, wer
from norm import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "finetune", "data")
COUNTS = {"Algeria": 726, "Egypt": 1600, "Jordan": 1599, "Mauritania": 1600,
          "Morocco": 1600, "Palestine": 900, "UAE": 1600, "Yemen": 1183}

for dd in sys.argv[1:]:
    tw = tc = tn = 0
    print(f"\n== {os.path.basename(dd)} ==")
    for d, n in COUNTS.items():
        hyps = [normalize(x) for x in open(os.path.join(dd, f"{d}.txt"), encoding="utf-8").read().split("\n")][:n]
        refs = [normalize(json.loads(l)["text"]) for l in open(os.path.join(DATA, "val", f"{d}.json"), encoding="utf-8")]
        H = [h or "@" for h in hyps]
        w, c = wer(refs, H), cer(refs, H)
        print(f"{d:<12}{w:>8.3f}{c:>8.3f}{n:>7}")
        tw += w * n; tc += c * n; tn += n
    print(f"{'WEIGHTED':<12}{tw/tn:>8.3f}{tc/tn:>8.3f}{tn:>7}")
