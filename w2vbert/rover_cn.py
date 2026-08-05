"""Progressive confusion-network ROVER for per-dialect prediction folders.

Unlike ``rover_dir.py``, this implementation preserves insertions proposed by
non-primary systems.  Each confusion slot contains one token (or NULL) from
every system, and optional system weights control voting.

Usage: python rover_cn.py OUT DIR [DIR ...]
Environment: REFS=<directory of dialect JSONL files>, WEIGHTS=1,1,...
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from jiwer import cer, wer

from norm import normalize


DIALECTS = ("Algeria", "Egypt", "Jordan", "Mauritania", "Morocco", "Palestine", "UAE", "Yemen")
UNIT = os.environ.get("UNIT", "word")
SPELL = os.environ.get("SPELL", "word")


def tokenize(text: str) -> list[str]:
    return list(text) if UNIT == "char" else text.split()


def detokenize(tokens: list[str]) -> str:
    return normalize("".join(tokens) if UNIT == "char" else " ".join(tokens))


def edit_ops(left: list[str], right: list[str]) -> list[tuple[str, str | None]]:
    """Return deterministic edit operations transforming left into right."""
    n, m = len(left), len(right)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = left[i - 1] != right[j - 1]
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )
    reversed_ops: list[tuple[str, str | None]] = []
    i, j = n, m
    while i or j:
        if i and j:
            cost = left[i - 1] != right[j - 1]
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                reversed_ops.append(("slot", right[j - 1]))
                i -= 1
                j -= 1
                continue
        if i and dp[i][j] == dp[i - 1][j] + 1:
            reversed_ops.append(("slot", None))
            i -= 1
        else:
            reversed_ops.append(("insert", right[j - 1]))
            j -= 1
    return list(reversed(reversed_ops))


def slot_word(slot: list[str | None], weights: list[float]) -> str:
    scores: dict[str, float] = {}
    first: dict[str, int] = {}
    for index, (word, weight) in enumerate(zip(slot, weights)):
        if word is None:
            continue
        scores[word] = scores.get(word, 0.0) + weight
        first.setdefault(word, index)
    return min(scores, key=lambda word: (-scores[word], first[word]))


def add_system(slots: list[list[str | None]], hypothesis: list[str], weights: list[float]) -> list[list[str | None]]:
    consensus = [slot_word(slot, weights) for slot in slots]
    output: list[list[str | None]] = []
    slot_index = 0
    old_system_count = len(weights)
    for operation, word in edit_ops(consensus, hypothesis):
        if operation == "insert":
            output.append([None] * old_system_count + [word])
        else:
            output.append(slots[slot_index] + [word])
            slot_index += 1
    assert slot_index == len(slots)
    return output


def rover(hypotheses: list[str], weights: list[float]) -> str:
    tokenized = [tokenize(hypothesis) for hypothesis in hypotheses]
    first_nonempty = next((index for index, words in enumerate(tokenized) if words), None)
    if first_nonempty is None:
        return ""
    if first_nonempty:
        tokenized[0], tokenized[first_nonempty] = tokenized[first_nonempty], tokenized[0]
        weights = weights.copy()
        weights[0], weights[first_nonempty] = weights[first_nonempty], weights[0]
    slots: list[list[str | None]] = [[word] for word in tokenized[0]]
    processed_weights = [weights[0]]
    for words, weight in zip(tokenized[1:], weights[1:]):
        slots = add_system(slots, words, processed_weights)
        processed_weights.append(weight)
    output = []
    for slot in slots:
        scores: dict[str | None, float] = {}
        first: dict[str | None, int] = {}
        for index, (word, weight) in enumerate(zip(slot, processed_weights)):
            scores[word] = scores.get(word, 0.0) + weight
            first.setdefault(word, index)
        winner = min(scores, key=lambda word: (-scores[word], first[word]))
        if winner is not None:
            if UNIT == "word" and SPELL == "char":
                output.append(character_vote(slot, processed_weights))
            else:
                output.append(winner)
    return detokenize(output)


def character_vote(words: list[str | None], weights: list[float]) -> str:
    tokenized = [list(word or "") for word in words]
    first_nonempty = next((index for index, chars in enumerate(tokenized) if chars), None)
    if first_nonempty is None:
        return ""
    if first_nonempty:
        tokenized[0], tokenized[first_nonempty] = tokenized[first_nonempty], tokenized[0]
        weights = weights.copy()
        weights[0], weights[first_nonempty] = weights[first_nonempty], weights[0]
    slots: list[list[str | None]] = [[character] for character in tokenized[0]]
    processed = [weights[0]]
    for chars, weight in zip(tokenized[1:], weights[1:]):
        slots = add_system(slots, chars, processed)
        processed.append(weight)
    output = []
    for slot in slots:
        scores: dict[str | None, float] = {}
        first: dict[str | None, int] = {}
        for index, (character, weight) in enumerate(zip(slot, processed)):
            scores[character] = scores.get(character, 0.0) + weight
            first.setdefault(character, index)
        winner = min(scores, key=lambda character: (-scores[character], first[character]))
        if winner is not None:
            output.append(winner)
    return "".join(output)


def read_predictions(directory: Path, dialect: str) -> list[str]:
    return [normalize(line) for line in (directory / f"{dialect}.txt").read_text(encoding="utf-8").splitlines()]


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit("usage: python rover_cn.py OUT DIR DIR [DIR ...]")
    output_dir = Path(sys.argv[1])
    directories = [Path(value) for value in sys.argv[2:]]
    weights = [float(value) for value in os.environ.get("WEIGHTS", "").split(",") if value]
    if not weights:
        weights = [1.0] * len(directories)
    if len(weights) != len(directories):
        raise ValueError("WEIGHTS must have one value per prediction directory")
    refs_dir = Path(os.environ["REFS"]) if os.environ.get("REFS") else None
    output_dir.mkdir(parents=True, exist_ok=True)
    print("SYSTEMS:", [directory.name for directory in directories])
    print("WEIGHTS:", weights)
    macro_wer = macro_cer = 0.0
    for dialect in DIALECTS:
        systems = [read_predictions(directory, dialect) for directory in directories]
        lengths = {len(system) for system in systems}
        if len(lengths) != 1:
            raise ValueError(f"{dialect}: unequal prediction counts {sorted(lengths)}")
        hypotheses = [rover([system[i] for system in systems], weights) for i in range(len(systems[0]))]
        (output_dir / f"{dialect}.txt").write_text(
            "".join(f"{hypothesis}\n" for hypothesis in hypotheses), encoding="utf-8", newline="\n"
        )
        if refs_dir:
            references = [
                normalize(json.loads(line)["text"])
                for line in (refs_dir / f"{dialect}.json").read_text(encoding="utf-8").splitlines()
            ]
            dialect_wer = wer(references, [hypothesis or "@" for hypothesis in hypotheses])
            dialect_cer = cer(references, [hypothesis or "@" for hypothesis in hypotheses])
            macro_wer += dialect_wer
            macro_cer += dialect_cer
            print(f"{dialect:<12}{dialect_wer:>8.3f}{dialect_cer:>8.3f}{len(references):>7}")
    if refs_dir:
        print(f"{'MACRO':<12}{macro_wer / len(DIALECTS):>8.3f}{macro_cer / len(DIALECTS):>8.3f}")


if __name__ == "__main__":
    main()
