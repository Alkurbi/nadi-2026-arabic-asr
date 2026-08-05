"""ROVER N prediction dirs -> OUT dir. Word-majority vote, primary=first dir, primary tiebreak.
Usage: python rover_dir.py OUT dir1 dir2 ...   (dir1 = strongest/primary)
If REFS env points to a per-dialect refs dir of {d}.json (holdout), also prints per-dialect+weighted WER/CER.
"""
import json, os, sys
from collections import Counter
from norm import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "finetune", "data")
DIALS = ["Algeria", "Egypt", "Jordan", "Mauritania", "Morocco", "Palestine", "UAE", "Yemen"]
REFS = os.environ.get("REFS")  # e.g. finetune/data/holdout  (dir of {d}.json with "text")


def align(primary, other):
    n, m = len(primary), len(other)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if primary[i - 1] == other[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    aligned = [None] * n; i, j = n, m
    while i > 0 and j > 0:
        c = 0 if primary[i - 1] == other[j - 1] else 1
        if dp[i][j] == dp[i - 1][j - 1] + c: aligned[i - 1] = other[j - 1]; i -= 1; j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1: aligned[i - 1] = None; i -= 1
        else: j -= 1
    return aligned


def rover_utt(hyps):
    prim = hyps[0].split()
    if not prim:
        alts = [h for h in hyps if h.strip()]
        return max(alts, key=len) if alts else ""
    cols = [prim] + [align(prim, h.split()) for h in hyps[1:]]
    out = []
    for i in range(len(prim)):
        votes = Counter(col[i] for col in cols)
        top = votes.most_common(2)
        word = prim[i] if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]
        if word is not None: out.append(word)
    return " ".join(out)


def main():
    out = sys.argv[1]; dirs = sys.argv[2:]
    assert len(dirs) >= 2, "need >=2 dirs"
    os.makedirs(out, exist_ok=True)
    print("ROVER:", [os.path.basename(d) for d in dirs], "-> ", os.path.basename(out))
    tw = tc = tn = 0
    from jiwer import cer, wer
    for d in DIALS:
        systems = [[normalize(x) for x in open(os.path.join(dd, f"{d}.txt"), encoding="utf-8").read().split("\n")] for dd in dirs]
        n = min(len(s) for s in systems)
        # drop trailing empty from the split (files end with newline)
        while n > 0 and all(not s[n - 1].strip() for s in systems): n -= 1
        systems = [s[:n] for s in systems]
        hyps = [rover_utt([systems[k][i] for k in range(len(systems))]) for i in range(n)]
        with open(os.path.join(out, f"{d}.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("".join((h or "") + "\n" for h in hyps))
        if REFS:
            refs = [normalize(json.loads(l)["text"]) for l in open(os.path.join(REFS, f"{d}.json"), encoding="utf-8")]
            H = [h or "@" for h in hyps]; w, c = wer(refs, H), cer(refs, H)
            print(f"{d:<12}{w:>8.3f}{c:>8.3f}{n:>7}"); tw += w * n; tc += c * n; tn += n
        else:
            print(f"{d:<12} n={n}")
    if tn: print(f"{'WEIGHTED':<12}{tw/tn:>8.3f}{tc/tn:>8.3f}{tn:>7}")


if __name__ == "__main__":
    main()
