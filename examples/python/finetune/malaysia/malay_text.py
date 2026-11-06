"""Transcript cleanup for Malay and Malaysian English (Manglish). Stdlib only.

The corpora this example mixes disagree on casing, punctuation, noise tags and
chat-style shorthand. A model learns whatever spelling it is shown, so every
source goes through the same pass before training.

What this deliberately does NOT do is merge registers. ``tak`` and ``tidak``,
``nak`` and ``hendak``, ``aku`` and ``saya`` are different words that mark
street versus formal speech, and both are kept. Only shorthand that nobody
pronounces (``yg``, ``dgn``, ``utk``) is expanded.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, Mapping, Optional

# Written shorthand that is never spoken as written. Expanding it cannot change
# register: ``yg`` is always read ``yang``. Anything ambiguous (``x``, ``kt``,
# ``tu``) is left out on purpose.
SHORTHAND: Dict[str, str] = {
    "yg": "yang",
    "dgn": "dengan",
    "utk": "untuk",
    "dlm": "dalam",
    "jgn": "jangan",
    "blh": "boleh",
    "sbb": "sebab",
    "mcm": "macam",
    "org": "orang",
    "kpd": "kepada",
    "pd": "pada",
    "sdh": "sudah",
    "tgk": "tengok",
    "nk": "nak",
    "tk": "tak",
}

# Annotation markup used by conversational corpora: [LAUGHTER], <unk>, {noise}.
_TAG = re.compile(r"\[[^\]]*\]|<[^>]*>|\{[^}]*\}")
_SPACE = re.compile(r"\s+")
# Plain style keeps letters, digits, intra-word apostrophes and hyphens.
# Malay reduplication (kanak-kanak, orang-orang) needs the hyphen.
_PLAIN_DROP = re.compile(r"[^\w\s'\-]", re.UNICODE)
_LOOSE_HYPHEN = re.compile(r"(?<!\w)-|-(?!\w)")
_LOOSE_APOS = re.compile(r"(?<!\w)'|'(?!\w)")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                         "–": "-", "—": "-", " ": " "})


def is_latin_text(text: str) -> bool:
    """True when every letter is Latin script.

    The Moonshine English tokenizer spells Latin text. Rows with Chinese
    characters, Tamil, Jawi and so on are dropped rather than taught as noise.
    """
    for ch in text:
        if unicodedata.category(ch).startswith("L"):
            if not unicodedata.name(ch, "").startswith("LATIN"):
                return False
    return True


def has_digits(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def _replace_words(text: str, mapping: Mapping[str, str]) -> str:
    if not mapping:
        return text
    lowered = {k.lower(): v for k, v in mapping.items()}
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in sorted(lowered, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )

    def sub(match: "re.Match[str]") -> str:
        word = match.group(0)
        out = lowered[word.lower()]
        if word.isupper() and len(word) > 1:
            return out.upper()
        if word[0].isupper():
            return out[0].upper() + out[1:]
        return out

    return pattern.sub(sub, text)


def _join_particles(text: str, particles: Iterable[str]) -> str:
    # "okay lah" -> "okaylah", "dia lah" -> "dialah". Splitting the other way is
    # not offered: it would break sekolah, masalah, salah and every other word
    # that happens to end in the same letters.
    for particle in particles:
        text = re.sub(
            rf"(\w)\s+({re.escape(particle)})\b",
            r"\1\2",
            text,
            flags=re.IGNORECASE,
        )
    return text


def normalize(
    text: str,
    style: str = "plain",
    particles: str = "keep",
    particle_words: Iterable[str] = ("lah",),
    expand_shorthand: bool = True,
    extra_map: Optional[Mapping[str, str]] = None,
) -> str:
    """Clean one transcript. Returns "" when nothing speakable is left.

    ``style``: ``plain`` lowercases and strips punctuation (the safe choice when
    sources disagree on casing). ``cased`` keeps casing and punctuation and only
    tidies whitespace, quotes and tags.

    ``particles``: ``keep`` leaves ``lah`` as each corpus wrote it, ``join``
    attaches a detached ``lah`` to the word before it (standard Malay spelling).
    """
    if style not in ("plain", "cased"):
        raise ValueError(f"unknown style {style!r}")
    if particles not in ("keep", "join"):
        raise ValueError(f"unknown particles mode {particles!r}")
    text = unicodedata.normalize("NFC", text).translate(_QUOTES)
    text = _TAG.sub(" ", text)
    if expand_shorthand:
        text = _replace_words(text, SHORTHAND)
    if extra_map:
        text = _replace_words(text, extra_map)
    if particles == "join":
        text = _join_particles(text, particle_words)
    if style == "plain":
        text = text.lower()
        text = _PLAIN_DROP.sub(" ", text)
        text = _LOOSE_HYPHEN.sub(" ", text)
        text = _LOOSE_APOS.sub(" ", text)
    text = _SPACE.sub(" ", text).strip()
    if not any(ch.isalnum() for ch in text):
        return ""
    return text
