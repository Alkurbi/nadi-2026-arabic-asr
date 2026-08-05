"""NADI official normalization (2025 recipe, inherited 2026), shared by vocab build,
training targets, and eval so the model outputs directly in the scored space:
 (a) keep only % among specials  (b) drop diacritics  (c) hamzas/maddas -> bare alif
 (d) Eastern->Western numerals   (e) preserve Latin.
"""
import re
import unicodedata

_DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۜ۟-ۤۧ-۪ۨ-ۭـ]")
_ALEF = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ٲ": "ا", "ٳ": "ا"}
_NUM = {ord(e): ord(w) for e, w in zip("٠١٢٣٤٥٦٧٨٩", "0123456789")}
_KEEP = re.compile(r"[^ء-يA-Za-z0-9%\s]")


def normalize(t):
    if not t:
        return ""
    t = unicodedata.normalize("NFC", t)
    t = _DIAC.sub("", t)
    for k, v in _ALEF.items():
        t = t.replace(k, v)
    t = t.translate(_NUM)
    t = _KEEP.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


if __name__ == "__main__":
    assert normalize("مَرْحَبًا، يا؟ world 2!") == "مرحبا يا world 2"
    assert normalize("أإآ") == "اا ا".replace(" ", "")  # all alef
    print("norm.py OK")
