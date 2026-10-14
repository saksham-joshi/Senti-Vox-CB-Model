from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Script tables, generated from unicodedata so nothing is typed by hand
# --------------------------------------------------------------------------

# Code point blocks per script. Only the blocks are listed here; which
# characters inside them count as letters or digits comes from the category.
_SCRIPT_BLOCKS: Dict[str, List[Tuple[int, int]]] = {
    "latin": [(0x0030, 0x0039), (0x0041, 0x005A), (0x0061, 0x007A),
              (0x00C0, 0x024F), (0x0300, 0x036F), (0x1E00, 0x1EFF)],
    "devanagari": [(0x0900, 0x097F)],
    "bengali": [(0x0980, 0x09FF)],
    "gurmukhi": [(0x0A00, 0x0A7F)],
    "gujarati": [(0x0A80, 0x0AFF)],
    "odia": [(0x0B00, 0x0B7F)],
    "tamil": [(0x0B80, 0x0BFF)],
    "telugu": [(0x0C00, 0x0C7F)],
    "kannada": [(0x0C80, 0x0CFF)],
    "malayalam": [(0x0D00, 0x0D7F)],
    "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
               (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
}

Ranges = List[Tuple[int, int]]


def _to_ranges(codepoints: List[int]) -> Ranges:
    """Collapse a list of code points into inclusive (lo, hi) ranges."""
    out: Ranges = []
    for cp in sorted(codepoints):
        if out and cp == out[-1][1] + 1:
            out[-1] = (out[-1][0], cp)
        else:
            out.append((cp, cp))
    return out


def _build_tables() -> Dict[str, Tuple[Ranges, List[Ranges]]]:
    """script -> (letter ranges, [digit ranges, one entry per set of ten])."""
    tables = {}
    for script, blocks in _SCRIPT_BLOCKS.items():
        letters: List[int] = []
        digit_sets: Dict[int, List[int]] = {}
        for lo, hi in blocks:
            for cp in range(lo, hi + 1):
                ch = chr(cp)
                cat = unicodedata.category(ch)
                if cat[0] in "LM":
                    letters.append(cp)
                elif cat == "Nd":
                    # group by the "zero" of the set, so Arabic-Indic and
                    # Extended Arabic-Indic digits stay separate tokens
                    digit_sets.setdefault(cp - unicodedata.decimal(ch), []).append(cp)
        tables[script] = (
            _to_ranges(letters),
            [_to_ranges(d) for _, d in sorted(digit_sets.items())],
        )
    return tables


_TABLES = _build_tables()


def export_tables() -> dict:
    """Plain-data copy of the script tables, for generating the TS/Java port."""
    return {
        "unicode_version": unicodedata.unidata_version,
        "scripts": {
            name: {
                "letters": [list(r) for r in letters],
                "digit_sets": [[list(r) for r in ds] for ds in digits],
            }
            for name, (letters, digits) in _TABLES.items()
        },
    }


def _esc(cp: int) -> str:
    return f"\\u{cp:04x}" if cp <= 0xFFFF else f"\\U{cp:08x}"


def _class_body(ranges: Ranges) -> str:
    return "".join(_esc(lo) if lo == hi else f"{_esc(lo)}-{_esc(hi)}" for lo, hi in ranges)


# --------------------------------------------------------------------------
# Characters allowed inside a word
# --------------------------------------------------------------------------

_JOINER = r"[\u200c\u200d]"
_APOSTROPHE = r"[\u0027\u2019\u02bc]"

# --------------------------------------------------------------------------
# Token pattern
# --------------------------------------------------------------------------


def _word(body: str) -> str:
    # letters, optionally with ZWJ/ZWNJ between two letters
    return rf"[{body}]+(?:{_JOINER}[{body}]+)*"


def _build_token_regex() -> "re.Pattern[str]":
    """Matches words and emojis only.

    Numbers, punctuation and other symbols have no alternative here, so
    findall() steps over them and they never reach the returned list.
    """
    alts: List[str] = []
    known: List[str] = []

    # Latin first, it is the most common case
    order = ["latin"] + [s for s in _TABLES if s != "latin"]
    for script in order:
        letters, _ = _TABLES[script]
        body = _class_body(letters)
        known.append(body)
        if script == "latin":
            alts.append(rf"[{body}]+(?:{_APOSTROPHE}[{body}]+)*")
        else:
            alts.append(_word(body))

    keycap = r"[0-9#*]\ufe0f?\u20e3"
    flag = r"[\U0001f1e6-\U0001f1ff]{2}"
    emo_base = (r"[\u00a9\u00ae\u203c\u2049\u2122\u2139\u2194-\u21aa\u231a-\u23ff\u24c2"
                r"\u25aa-\u25fe\u2600-\u27bf\u2934\u2935\u2b05-\u2b55\u3030\u303d"
                r"\u3297\u3299\U0001f000-\U0001faff]")
    skin = r"[\U0001f3fb-\U0001f3ff]"
    emoji = (rf"{emo_base}\ufe0f?{skin}?"
             rf"(?:\u200d{emo_base}\ufe0f?{skin}?)*"
             r"[\U000e0020-\U000e007f]*")

    # a few plain-text emoticons: :) :( :D :P ;) :-) :'( and <3
    emoticon = (r"(?:[:;=][-']?(?:\)+|\(+|[DPp](?![A-Za-z0-9]))"
                r"|(?<![A-Za-z0-9])<3+(?![0-9]))")

    # best-effort for scripts we do not list (Cyrillic, Greek, CJK, ...):
    # a run of letters that is not one of our known scripts
    other = rf"(?:(?![{''.join(known)}])[^\W\d_])+"

    parts = alts + [keycap, flag, emoji, emoticon, other]
    return re.compile("|".join(parts))


_TOKEN = _build_token_regex()


# --------------------------------------------------------------------------
# URLs and e-mail addresses
# --------------------------------------------------------------------------

_LOCAL = r"A-Za-z0-9!#$%&'*+/=?^_`{|}~\-\u00c0-\u024f"
_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"

_EMAIL = (
    r"(?:mailto:)?"
    rf"(?<![{_LOCAL}.])"
    rf"[{_LOCAL}]+(?:\.[{_LOCAL}]+)*"
    rf"@(?:{_LABEL}\.)+[A-Za-z]{{2,}}"
)

# URL characters are ASCII only. If a Hindi word follows a URL with no space
# between them, the word is not swallowed. The price is that URLs containing
# raw non-ASCII characters are cut at the first one.
_CH = r"[^\s()<>\"'\u0080-\U0010ffff]"
_END = r"[^\s()<>\"'.,;:!?*\]}\u0080-\U0010ffff]"
_TAIL = rf"(?:{_CH}|\({_CH}*\))*(?:\({_CH}*\)|{_END})"

_URL = (
    r"(?:(?i:https?|ftps?)://|(?<![A-Za-z0-9])(?i:www)\.)"
    + _TAIL
)

# bare domains only when a path follows (bit.ly/xyz), and only for common TLDs,
# so "Mr.Smith/John" is left alone
_TLDS = (r"(?i:com|net|org|edu|gov|info|biz|io|co|in|me|ly|gl|tv|ai|app|dev|xyz"
         r"|us|uk|be|to|link|page|site|online|store)")
_BARE = (
    r"(?<![A-Za-z0-9.@_\-])"
    rf"(?:{_LABEL}\.)+{_TLDS}(?![A-Za-z0-9])"
    rf"/(?:{_TAIL})?"
)

_PATTERN = re.compile(rf"{_EMAIL}|{_URL}|{_BARE}")
_GATE = re.compile(r"[@/]|www\.", re.IGNORECASE)
_SPACES = re.compile(r"[ \t]{2,}")


def _remove_urls_and_emails(
    text: str,
    replacement: str = " ",
    collapse_spaces: bool = False,
) -> str:
    """Remove (or replace) URLs and e-mail addresses in `text`.

    The replacement defaults to a space so the words on either side of a
    removed URL never end up glued together.
    """
    if not text:
        return text

    if _GATE.search(text):
        text = _PATTERN.sub(replacement, text)

    if collapse_spaces:
        text = _SPACES.sub(" ", text).strip()
    return text


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

# curly apostrophe and modifier-letter apostrophe -> plain apostrophe
_APOS_FOLD = str.maketrans({0x2019: "'", 0x02BC: "'"})


def senti_vox_tokenizer(
    text: str,
    *,
    normalize: Optional[str] = "NFKC",
    lowercase: bool = False,
) -> List[str]:
    """Split `text` into word and emoji tokens.

    Words are letter runs in any script (with ZWJ/ZWNJ inside Indic words and
    apostrophes inside Latin ones). Emojis include ZWJ sequences, skin tones,
    flags, keycaps and a few text emoticons like :) and <3. Numbers,
    punctuation, other symbols, URLs and e-mail addresses are dropped.

    normalize  Unicode normal form applied first ("NFKC" by default, None to
               skip). It also folds curly apostrophes to a plain one.
    lowercase  Lowercase the text before tokenizing. Off by default, in case
               casing is handled on the lookup side.

    Raises TypeError for anything that is not a str (None included).
    """
    if not isinstance(text, str):
        raise TypeError(f"senti_vox_tokenizer expects str, got {type(text).__name__}")
    if not text:
        return []

    # pure ASCII text needs neither normalisation nor apostrophe folding
    if normalize and not text.isascii():
        text = unicodedata.normalize(normalize, text).translate(_APOS_FOLD)

    text = _remove_urls_and_emails(text)

    if lowercase:
        text = text.lower()

    return _TOKEN.findall(text)