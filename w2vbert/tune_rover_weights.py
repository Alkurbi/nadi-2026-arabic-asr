"""Tune dialect-specific weighted ROVER with utterance-level cross-validation.

Alignments are cached once against the first (primary) system. Integer system
weights are searched independently per dialect. Five-fold OOF WER is the honest
selection estimate; full-data weights are then written for test-time use.

Usage: python tune_rover_weights.py OUT REFS DIR [DIR ...]
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from norm import normalize
from rover_dir import align
from mbr_ensemble import word_distance


DIALECTS = tuple(os.environ.get(
    "DIALECTS", "Algeria,Egypt,Jordan,Mauritania,Morocco,Palestine,UAE,Yemen"
).split(","))
SEED = 2026
TRIALS = int(os.environ.get("TRIALS", "10000"))
MAX_WEIGHT = int(os.environ.get("MAX_WEIGHT", "4"))


def canonical(weights: tuple[int, ...]) -> tuple[int, ...]:
    divisor = 0
    for weight in weights:
        divisor = math.gcd(divisor, weight)
    return tuple(weight // max(1, divisor) for weight in weights)


def weight_candidates(system_count: int) -> list[tuple[int, ...]]:
    rng = random.Random(SEED)
    candidates = {tuple([1] * system_count)}
    for primary_weight in range(1, MAX_WEIGHT + 1):
        candidates.add((primary_weight,) + tuple([1] * (system_count - 1)))
    while len(candidates) < TRIALS:
        weights = (rng.randint(1, MAX_WEIGHT),) + tuple(
            rng.randint(0, MAX_WEIGHT) for _ in range(system_count - 1)
        )
        if sum(weight > 0 for weight in weights) >= 3:
            candidates.add(canonical(weights))
    return sorted(candidates)


def columns(hypotheses: list[str]) -> list[list[str | None]]:
    primary = hypotheses[0].split()
    if not primary:
        fallback = max((hypothesis.split() for hypothesis in hypotheses), key=len, default=[])
        primary = fallback
    return [primary] + [align(primary, hypothesis.split()) for hypothesis in hypotheses[1:]]


def vote(aligned: list[list[str | None]], weights: tuple[int, ...]) -> str:
    output = []
    for position in range(len(aligned[0])):
        scores: dict[str | None, int] = {}
        first: dict[str | None, int] = {}
        for system, weight in enumerate(weights):
            word = aligned[system][position]
            scores[word] = scores.get(word, 0) + weight
            first.setdefault(word, system)
        winner = min(scores, key=lambda word: (-scores[word], first[word]))
        if winner is not None:
            output.append(winner)
    return " ".join(output)


def main() -> None:
    if len(sys.argv) < 6:
        raise SystemExit("usage: python tune_rover_weights.py OUT REFS DIR DIR DIR [DIR ...]")
    output_dir = Path(sys.argv[1])
    refs_dir = Path(sys.argv[2])
    system_dirs = [Path(value) for value in sys.argv[3:]]
    candidates = weight_candidates(len(system_dirs))
    print("SYSTEMS:", [directory.name for directory in system_dirs])
    print(f"weight candidates={len(candidates)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    macro_oof = macro_full = 0.0
    selected_weights = {}
    for dialect in DIALECTS:
        references = [
            normalize(json.loads(line)["text"])
            for line in (refs_dir / f"{dialect}.json").read_text(encoding="utf-8").splitlines()
        ]
        systems = [
            [normalize(line) for line in (directory / f"{dialect}.txt").read_text(encoding="utf-8").splitlines()]
            for directory in system_dirs
        ]
        aligned = [columns([system[index] for system in systems]) for index in range(len(references))]
        predictions = [[vote(item, weights) for item in aligned] for weights in candidates]
        errors = [
            [word_distance(reference, hypothesis) for reference, hypothesis in zip(references, hypotheses)]
            for hypotheses in predictions
        ]

        oof_errors = 0
        for fold in range(5):
            training = [index for index in range(len(references)) if index % 5 != fold]
            validation = [index for index in range(len(references)) if index % 5 == fold]
            best = min(
                range(len(candidates)),
                key=lambda candidate: sum(errors[candidate][index] for index in training),
            )
            oof_errors += sum(errors[best][index] for index in validation)

        best = min(range(len(candidates)), key=lambda candidate: sum(errors[candidate]))
        reference_words = sum(len(reference.split()) for reference in references)
        oof_wer = oof_errors / reference_words
        full_wer = sum(errors[best]) / reference_words
        macro_oof += oof_wer
        macro_full += full_wer
        selected_weights[dialect] = candidates[best]
        (output_dir / f"{dialect}.txt").write_text(
            "".join(f"{hypothesis}\n" for hypothesis in predictions[best]),
            encoding="utf-8", newline="\n",
        )
        print(
            f"{dialect:<12} OOF={oof_wer:.4f} FULL={full_wer:.4f} "
            f"weights={candidates[best]}",
            flush=True,
        )

    (output_dir / "weights.json").write_text(
        json.dumps(selected_weights, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"MACRO OOF={macro_oof / len(DIALECTS):.4f} FULL={macro_full / len(DIALECTS):.4f}")


if __name__ == "__main__":
    main()
