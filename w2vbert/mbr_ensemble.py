"""Unsupervised sentence-level MBR ensemble for saved NADI hypotheses.

For every utterance, select the candidate with the smallest mean normalized word
edit distance to the other systems.  This is reference-free at inference time.
"""
import json
import os
from pathlib import Path

from jiwer import cer, wer
from rapidfuzz.distance import Levenshtein

from norm import normalize


HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("REFS", HERE.parent / "finetune" / "data" / "val"))
OUT = Path(os.environ.get("OUT", HERE / "predictions_mbr"))
DIALECTS = ("Algeria", "Egypt", "Jordan", "Mauritania", "Morocco", "Palestine", "UAE", "Yemen")
DEFAULT_SYSTEMS = (
    "predictions_big_lm",
    "predictions_biglm",
    "predictions_omni_lm",
    "predictions_omni_lm_b15",
    "predictions_stage1full_lm",
    "predictions_2stage_lm",
    "predictions_w2v_lm",
    "predictions_w2v",
)


def edit_distance(a, b) -> int:
    """Levenshtein distance over any token sequence."""
    return Levenshtein.distance(a, b)


def word_distance(left: str, right: str) -> int:
    return edit_distance(left.split(), right.split())


def normalized_distance(left: str, right: str, char_weight: float) -> float:
    words = word_distance(left, right) / max(1, len(left.split()), len(right.split()))
    chars = edit_distance(left.replace(" ", ""), right.replace(" ", "")) / max(
        1, len(left.replace(" ", "")), len(right.replace(" ", ""))
    )
    return words + char_weight * chars


def select_mbr(candidates: list[str], char_weight: float) -> str:
    """Choose the candidate closest to the ensemble; earlier systems win ties."""
    scores = []
    for candidate in candidates:
        score = sum(
            normalized_distance(candidate, other, char_weight)
            for other in candidates
        )
        scores.append(score)
    return candidates[min(range(len(candidates)), key=lambda i: (scores[i], i))]


def read_lines(path: Path) -> list[str]:
    return [normalize(line) or "@" for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    import sys

    systems = tuple(sys.argv[1:]) or DEFAULT_SYSTEMS
    char_weight = float(os.environ.get("CHAR_WEIGHT", 0))
    system_dirs = [Path(s) if Path(s).is_absolute() else HERE / s for s in systems]
    OUT.mkdir(parents=True, exist_ok=True)

    print("MBR systems:", [p.name for p in system_dirs])
    print(f"{'dialect':<12}{'WER':>8}{'CER':>8}{'n':>7}")
    all_refs, all_hyps, all_oracle, dialect_wers, oracle_errors = [], [], [], [], 0
    for dialect in DIALECTS:
        refs = [
            normalize(json.loads(line)["text"])
            for line in (DATA / f"{dialect}.json").read_text(encoding="utf-8").splitlines()
        ]
        hypotheses = [read_lines(path / f"{dialect}.txt") for path in system_dirs]
        for path, lines in zip(system_dirs, hypotheses):
            if len(lines) != len(refs):
                raise ValueError(f"{path.name}/{dialect}: {len(lines)} hypotheses != {len(refs)} references")

        selected = [select_mbr([lines[i] for lines in hypotheses], char_weight) for i in range(len(refs))]
        oracle = [
            min((lines[i] for lines in hypotheses), key=lambda candidate: word_distance(refs[i], candidate))
            for i in range(len(refs))
        ]
        (OUT / f"{dialect}.txt").write_text("\n".join(selected) + "\n", encoding="utf-8", newline="\n")
        dialect_wer, dialect_cer = wer(refs, selected), cer(refs, selected)
        dialect_wers.append(dialect_wer)
        all_refs.extend(refs)
        all_hyps.extend(selected)
        all_oracle.extend(oracle)
        oracle_errors += sum(word_distance(reference, hypothesis) for reference, hypothesis in zip(refs, oracle))
        print(f"{dialect:<12}{dialect_wer:>8.4f}{dialect_cer:>8.4f}{len(refs):>7}")

    print(f"{'MACRO':<12}{sum(dialect_wers) / len(dialect_wers):>8.4f}")
    print(f"{'POOLED':<12}{wer(all_refs, all_hyps):>8.4f}{cer(all_refs, all_hyps):>8.4f}{len(all_refs):>7}")
    reference_words = sum(len(reference.split()) for reference in all_refs)
    print(f"{'ORACLE':<12}{oracle_errors / reference_words:>8.4f}{cer(all_refs, all_oracle):>8.4f}{len(all_refs):>7}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
