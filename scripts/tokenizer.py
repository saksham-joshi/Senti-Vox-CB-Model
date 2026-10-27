"""SentiVox text cleaner and tokenizer.

Turns raw social / chat / review text into clean text for the sentiment
model, the same way for the training data and for production traffic:

* URLs, e-mails, @usernames, phone numbers, IDs (order numbers, OTPs,
  tracking numbers, promo codes, UUIDs, ...), IP addresses and platform
  boilerplate ("RT @x:", "<Media omitted>", "Read more", ...) become the
  placeholders defined below.
* Person names and religions become [NAME] and [RELIGION], from the word
  lists in values.json, so the model cannot tie sentiment to a person or a
  faith. Names that are also everyday words ("Hope", "Will") are kept.
* Everything that carries sentiment is kept: words in every script,
  negations and contractions, numbers ("10/10", "1 star"), and punctuation
  like "!", "?" and "/s".
* Emojis and emoticons become words ("face with tears of joy"). MuRIL has
  no emoji in its vocabulary and maps them, and any word they touch, to
  [UNK].
* Noise is removed or evened out: HTML tags and entities, mojibake,
  invisible characters, markdown, "#" in hashtags, elongations
  ("sooooo" -> "soo"), and runs of the same punctuation or emoji
  (at most three).

    clean_text("@united lost my bag AGAIN 😡😡😡 call 917-703-1472 https://t.co/x")
    -> "[USER] lost my bag again enraged face enraged face enraged face call [PHONE] [URL]"
"""
from __future__ import annotations

import html
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

class _Placeholders :
    URL = "[URL]"
    EMAIL = "[EMAIL]"
    USER = "[USER]"   # @usernames
    NAME = "[NAME]"   # person names, from values.json
    PHONE = "[PHONE]"
    ID = "[ID]"   # Order IDs, tracking numbers, OTPs, promo codes, reference IDs
    IP = "[IP]"
    RT = "[RT]"   # Platform boilerplate: Read more..., <Media omitted>, This message was deleted
    RELIGION = "[RELIGION]"   # religions and their followers, from values.json

_P = _Placeholders

# Every placeholder the cleaner can emit. Register them with the model's
# tokenizer (see register_placeholders) so each one is a single token.
PLACEHOLDER_TOKENS: Tuple[str, ...] = (
    _P.URL, _P.EMAIL, _P.USER, _P.NAME, _P.PHONE, _P.ID, _P.IP, _P.RT, _P.RELIGION,
)
_PLACEHOLDERS = frozenset(PLACEHOLDER_TOKENS)

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
# Emoji and emoticons
# --------------------------------------------------------------------------

# Emoji code points. Regional indicators (flags) and skin tones are left out
# on purpose: they only count as part of a sequence.
_EMO_BASE = (
    "[\u203c\u2049\u231a\u231b\u2328\u23cf\u23e9-\u23f3\u23f8-\u23fa\u24c2"
    "\u25aa\u25ab\u25b6\u25c0\u25fb-\u25fe\u2600-\u2775\u2795-\u2797\u27a1"
    "\u27b0\u27bf\u2934\u2935\u2b05-\u2b07\u2b1b\u2b1c\u2b50\u2b55\u3030"
    "\u303d\u3297\u3299\U0001f000-\U0001f1e5\U0001f200-\U0001f3fa"
    "\U0001f400-\U0001faff]"
)
_SKIN = "[\U0001f3fb-\U0001f3ff]"
_KEYCAP = "[0-9#*]\ufe0f?\u20e3"
_FLAG = "[\U0001f1e6-\U0001f1ff]{2}"
_EMOJI_SEQ = (
    rf"{_EMO_BASE}[\ufe0e\ufe0f]?{_SKIN}?"
    rf"(?:\u200d{_EMO_BASE}[\ufe0e\ufe0f]?{_SKIN}?)*"
    "[\U000e0020-\U000e007f]*"
)
_EMOJI = rf"(?:{_KEYCAP}|{_FLAG}|{_EMOJI_SEQ})"
_EMOJI_RE = re.compile(_EMOJI)
# a run of emoji, possibly with spaces between them
_EMOJI_RUN = re.compile(rf"{_EMOJI}(?:\s*{_EMOJI})*")

# Short names for emoji whose Unicode name is long or misleading
# ("HEAVY BLACK HEART" for a red heart, "POUTING FACE" for an angry one).
# All other emoji use their Unicode name, lowercased.
_EMOJI_NAMES: Dict[str, str] = {
    "\u2639": "frowning face",                      # ☹
    "\u263a": "smiling face",                       # ☺
    "\u2665": "heart",                              # ♥
    "\u2763": "heart exclamation",                  # ❣
    "\u2764": "red heart",                          # ❤
    "\u2605": "star",                               # ★
    "\u2606": "empty star",                         # ☆
    "\u2b50": "star",                               # ⭐
    "\u2705": "check mark",                         # ✅
    "\u2713": "check mark",                         # ✓
    "\u2714": "check mark",                         # ✔
    "\u2717": "cross mark",                         # ✗
    "\u2718": "cross mark",                         # ✘
    "\U0001f44b": "waving hand",                    # 👋
    "\U0001f44c": "ok hand",                        # 👌
    "\U0001f44d": "thumbs up",                      # 👍
    "\U0001f44e": "thumbs down",                    # 👎
    "\U0001f44f": "clapping hands",                 # 👏
    "\U0001f595": "middle finger",                  # 🖕
    "\U0001f64c": "raising hands",                  # 🙌
    "\U0001f64f": "folded hands",                   # 🙏
    "\U0001f91e": "crossed fingers",                # 🤞
    "\U0001f4af": "hundred points",                 # 💯
    "\U0001f4a5": "collision",                      # 💥
    "\U0001f4a4": "zzz",                            # 💤
    "\U0001f47f": "angry face with horns",          # 👿
    "\U0001f601": "beaming face with smiling eyes", # 😁
    "\U0001f603": "grinning face with big eyes",    # 😃
    "\U0001f604": "grinning face with smiling eyes",# 😄
    "\U0001f605": "grinning face with sweat",       # 😅
    "\U0001f606": "grinning squinting face",        # 😆
    "\U0001f60b": "face savoring food",             # 😋
    "\U0001f60d": "smiling face with heart eyes",   # 😍
    "\U0001f613": "downcast face with sweat",       # 😓
    "\U0001f61b": "face with tongue",               # 😛
    "\U0001f61c": "winking face with tongue",       # 😜
    "\U0001f61d": "squinting face with tongue",     # 😝
    "\U0001f621": "enraged face",                   # 😡
    "\U0001f624": "face with steam from nose",      # 😤
    "\U0001f625": "sad but relieved face",          # 😥
    "\U0001f630": "anxious face with sweat",        # 😰
    "\U0001f926": "facepalm",                       # 🤦
    "\U0001f928": "face with raised eyebrow",       # 🤨
    "\U0001f929": "star struck",                    # 🤩
    "\U0001f92a": "zany face",                      # 🤪
    "\U0001f92b": "shushing face",                  # 🤫
    "\U0001f92c": "face with symbols on mouth",     # 🤬
    "\U0001f92d": "face with hand over mouth",      # 🤭
    "\U0001f92e": "face vomiting",                  # 🤮
    "\U0001f92f": "exploding head",                 # 🤯
    "\U0001f970": "smiling face with hearts",       # 🥰
    "\U0001f973": "partying face",                  # 🥳
    "\U0001f975": "hot face",                       # 🥵
    "\U0001f976": "cold face",                      # 🥶
    "\U0001f97a": "pleading face",                  # 🥺
}

# parts of a sequence that add nothing to its name
_EMOJI_SKIP = frozenset("\u200d\ufe0e\ufe0f\u20e3")
_GENDER = frozenset("\u2640\u2642\u26a7")
_NAME_PREFIX = re.compile(r"^(?:(?:black|white|heavy) )+")
_NAME_SUFFIX = re.compile(r" (?:sign|symbol)$")


def _char_name(ch: str) -> str:
    name = _EMOJI_NAMES.get(ch)
    if name is not None:
        return name
    name = unicodedata.name(ch, "").lower()
    if ord(ch) < 0x1F000:
        # in the old symbol blocks "black"/"white" mean filled/hollow
        name = _NAME_PREFIX.sub("", name)
    return _NAME_SUFFIX.sub("", name).replace("-", " ")


@lru_cache(maxsize=4096)
def _emoji_name(seq: str) -> str:
    """Plain-word name of one emoji sequence ("" if it has none)."""
    if seq.endswith("\u20e3"):
        return seq[0]                          # keycap: 1️⃣ -> 1
    first = ord(seq[0])
    if 0x1F1E6 <= first <= 0x1F1FF or (first == 0x1F3F4 and len(seq) > 1):
        return "flag"                          # 🇮🇳, 🏴󠁧󠁢󠁥󠁮󠁧󠁿
    zwj = "\u200d" in seq
    parts: List[str] = []
    for ch in seq:
        cp = ord(ch)
        if ch in _EMOJI_SKIP or 0x1F3FB <= cp <= 0x1F3FF or cp >= 0xE0000:
            continue
        if zwj and ch in _GENDER:
            continue                           # 🤦‍♂️ -> facepalm
        name = _char_name(ch)
        if name and name not in parts:
            parts.append(name)
    return " ".join(parts)


def _emoji_run_to_words(m: "re.Match[str]") -> str:
    out: List[str] = []
    prev, count = None, 0
    for unit in _EMOJI_RE.findall(m.group()):
        count = count + 1 if unit == prev else 1
        prev = unit
        if count <= 3:                         # 😂😂😂😂😂 -> three times
            name = _emoji_name(unit)
            if name:
                out.append(name)
    return f" {' '.join(out)} " if out else " "


# Emoticons, converted to the matching emoji when emojis become words, so
# ":)" and "🙂" end up as the same words.
_EMOTICONS: List[Tuple[str, str]] = [
    (r">:-?\(", "\U0001f620"),                                      # >:(  angry
    (r":'-?\(", "\U0001f622"),                                      # :'(  crying
    (r":'-?\)", "\U0001f972"),                                      # :')  happy tears
    (r"[:=]-?\)+", "\U0001f642"),                                   # :)
    (r"[:=]-?\(+", "\U0001f641"),                                   # :(
    (r";-?\)+", "\U0001f609"),                                      # ;)
    (r"[:=]-?D+(?![A-Za-z0-9])", "\U0001f603"),                     # :D
    (r"[:=]-?[Pp]+(?![A-Za-z0-9])", "\U0001f61b"),                  # :P
    (r"(?<![A-Za-z0-9])[xX]D+(?![A-Za-z0-9])", "\U0001f606"),       # XD
    (r"(?<![0-9])[:=]-?[oO](?![A-Za-z0-9])", "\U0001f62e"),         # :O
    (r"[:=]-?[/\\](?![/\\\w])", "\U0001f615"),                      # :/
    (r"[:=]-?\|", "\U0001f610"),                                    # :|
    (r"[:=]-?\*", "\U0001f617"),                                    # :*
    (r"</3", "\U0001f494"),                                         # </3
    (r"(?<![<A-Za-z])<3+(?![0-9])", "\u2764"),                      # <3
    (r"\^[_.-]\^", "\U0001f60a"),                                   # ^_^
    (r"(?<![\w-])-[_.]-(?![\w-])", "\U0001f611"),                   # -_-
    (r"(?<!\w)(?:T[_.]T|;_;)(?!\w)", "\U0001f62d"),                 # T_T
    (r"(?<!\w)[oO][_.][oO](?!\w)", "\U0001f928"),                   # o_O
]
_EMOTICON = "|".join(f"(?P<e{i}>{p})" for i, (p, _) in enumerate(_EMOTICONS))
_EMOTICON_RE = re.compile(_EMOTICON)


def _emoticon_to_emoji(m: "re.Match[str]") -> str:
    return f" {_EMOTICONS[int(m.lastgroup[1:])][1]} "


# star ratings, before the emoji run would cut them to three stars:
# ★★★★☆ -> 4/5 stars, ⭐⭐⭐⭐⭐ -> 5 stars
_STARS = re.compile("[\u2605\u2606\u2b50]\ufe0f?(?:[ \t]?[\u2605\u2606\u2b50]\ufe0f?)+")


def _stars_repl(m: "re.Match[str]") -> str:
    s = m.group()
    full, empty = s.count("\u2605") + s.count("\u2b50"), s.count("\u2606")
    return f" {full}/{full + empty} stars " if empty else f" {full} stars "


# ¯\_(ツ)_/¯ has to be caught before NFKC, which breaks the ¯ apart
_SHRUG = re.compile(r"\u00af\\*_*\(\u30c4\)_*/*\u00af")

# --------------------------------------------------------------------------
# Unicode clean-up
# --------------------------------------------------------------------------

# UTF-8 text that was decoded as cp1252/latin-1 ("donâ€™t", "ðŸ˜‚"):
# lead bytes each followed by continuation bytes, as cp1252 characters. The
# two classes do not overlap, so a whole run is taken in one linear pass.
_MOJIBAKE = re.compile(
    "(?:[\u00c2-\u00f4][\u0080-\u00bf\u0152\u0153\u0160\u0161\u0178\u017d\u017e"
    "\u0192\u02c6\u02dc\u2013\u2014\u2018-\u201a\u201c-\u201e\u2020-\u2022"
    "\u2026\u2030\u2039\u203a\u20ac\u2122]+)+"
)


def _unmojibake(m: "re.Match[str]") -> str:
    s = m.group()
    raw = bytearray()
    for ch in s:
        try:
            raw += ch.encode("cp1252")
        except UnicodeEncodeError:
            raw.append(ord(ch))                # C1 control, never above 0xFF here
    try:
        fixed = raw.decode("utf-8")
    except UnicodeDecodeError:
        return s                               # not mojibake after all
    # a lone combining mark means the bytes were never UTF-8 ("YouÛªve")
    return s if unicodedata.category(fixed[0])[0] == "M" else fixed


# only entities with a closing ";" ("&notice" must not become "¬ice")
_ENTITY = re.compile(r"&(?:#[0-9]{1,7}|#[xX][0-9A-Fa-f]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")

# typographic apostrophes, quotes and dashes -> ASCII. Runs before NFKC,
# which would turn the acute accent in "can´t" into a space and a mark.
_TYPO_FOLD = str.maketrans({
    0x2018: "'", 0x2019: "'", 0x201B: "'", 0x02BC: "'", 0x02BB: "'",
    0x2032: "'", 0x00B4: "'", 0x0060: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"', 0x2033: '"',
    0x00AB: '"', 0x00BB: '"',
    0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: "-",
    0x2015: "-", 0x2212: "-",
})

# zero-width space separates words; the rest of these are simply dropped
_ZW_SPACE = re.compile("[\u200b\u2028\u2029\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_INVISIBLE = re.compile("[\u00ad\u034f\u061c\u180e\u200e\u200f\u202a-\u202e"
                        "\u2060-\u2064\u2066-\u206f\ufeff\ufff9-\ufffb]")

# --------------------------------------------------------------------------
# Platform boilerplate -> [RT]
# --------------------------------------------------------------------------

_DEVICES = (r"(?:iphone|ipad|android|phone|mobile|device|smartphone|tablet|galaxy"
            r"|blackberry|samsung|pixel|oneplus|xiaomi|redmi|nokia|huawei|motorola)")

_BOILERPLATE = re.compile("|".join([
    # chat exports (WhatsApp, Telegram, Signal)
    r"(?i:<\s*(?:media|image|video|audio|gif|sticker|document|contact card|file)s?\s+omitted\s*>)",
    r"(?i:\b(?:image|video|audio|gif|sticker|document|contact card)\s+omitted\b)",
    r"(?i:<\s*attached:[^<>\n]{0,200}>)",
    r"(?i:<?\s*\bthis message was (?:deleted|edited)\b\s*>?)",
    r"(?i:\byou deleted this message\b)",
    r"(?i:\bwaiting for this message\b\.?(?:\s*this may take a while\.?)?)",
    r"(?i:\bmessages (?:and calls )?are end-to-end encrypted\b\.?"
    r"(?:\s*no one outside of this chat[^.\n]{0,120}\.)?(?:\s*tap (?:to|for) (?:learn )?more(?: info)?\.?)?)",
    r"(?i:\bmissed (?:group )?(?:voice|video) call\b)",
    # social media
    r"(?<![\w@\[])RT(?=[\s_~*^|]*[@:])",
    r"(?<![\w@\[])MT(?=[\s_~*^|]*@)",
    r"(?i:\breplying to\b)(?=\s*@)",
    # "... Read more", also with dropped decoration between ("... _ Read more")
    r"(?<!\.)(?<!\.\s)(?i:\.(?:\s?\.)+[\s_~*^|]*(?:read|see|show|view)\s+(?:more|less)\b)",
    r"(?:(?<=[.!?\n])|^)\s*(?:Read|See|Show|View) (?:more|More|MORE|less|Less)\b\s*\.{0,3}\s*$",
    r"(?i:\bclick to expand\b\.*)",
    r"(?i:\bshow this thread\b)",
    r"(?i:\bsee translation\b)",
    r"(?i:\(?\btranslated by google\b\)?)",
    r"\(Original\)",
    r"(?i:\[(?:deleted|removed)\])",
    r"(?i:\(edited\)|\[edited\])",
    # e-mail and phone signatures
    rf"(?i:\bsent from my (?:\w+ ){{0,3}}?{_DEVICES}\w*)",
    r"(?i:\bsent from (?:yahoo mail|mail for windows(?: 1[01])?|outlook|gmail|proton ?mail)"
    r"(?: for (?:android|ios|iphone|ipad))?\b)",
    r"(?i:\bget outlook for (?:ios|android)\b)",
    # review sites
    r"(?i:\bthe media could not be loaded\b\.?)",
    r"(?i:\b(?:one|[0-9][0-9,]*) (?:person|people) found this helpful\b\.?)",
    r"(?i:\bwas this review helpful(?: to you)?\??)",
    r"\bReport abuse\b",
    r"\bVerified Purchase\b",
]))

# --------------------------------------------------------------------------
# HTML and markdown
# --------------------------------------------------------------------------

_HTML_TAG = re.compile(
    r"</?(?i:a|abbr|b|big|blockquote|br|center|code|del|div|em|font|h[1-6]|hr|i|img"
    r"|ins|li|ol|p|pre|q|s|small|span|strike|strong|sub|sup|table|tbody|td|th"
    r"|thead|tr|tt|u|ul)\b[^<>]{0,500}?/?>"
)
# "<unk>" is left in some datasets by an earlier tokenizer
_UNK = re.compile(r"(?i:<unk>)")
# "{link}" is how some tweet datasets anonymised their URLs
_LINK_STUB = re.compile(r"\{link\}")
# [text](https://...) -> text https://...
_MD_LINK = re.compile(
    r"\[([^\[\]\n]{1,300})\]\(\s*((?:(?i:https?|ftps?)://|(?i:www)\.)[^\s()]{1,2000})\s*\)"
)

_MARKDOWN: List[Tuple["re.Pattern[str]", str]] = [
    (re.compile(r"(?<![\w*])\*{3}(?=[^\s*])([^*\n]{1,300}?)(?<=[^\s*])\*{3}(?![\w*])"), r"\1"),
    (re.compile(r"(?<![\w*])\*\*(?=[^\s*])([^*\n]{1,300}?)(?<=[^\s*])\*\*(?![\w*])"), r"\1"),
    (re.compile(r"(?<![\w_])__(?=[^\s_])([^_\n]{1,300}?)(?<=[^\s_])__(?![\w_])"), r"\1"),
    (re.compile(r"(?<![\w*])\*(?=[^\s*])([^*\n]{1,300}?)(?<=[^\s*])\*(?![\w*])"), r"\1"),
    (re.compile(r"~~(?=\S)([^~\n]{1,300}?)(?<=\S)~~"), r"\1"),
    # "> quoted" at the start of a line
    (re.compile(r"(?m)^[ \t]*(?:>[ \t]?)+"), ""),
    # separator lines ("-----", "=====", "*****")
    (re.compile(r"(?<!\w)([-=_~*#+])\1{2,}(?!\w)"), " "),
    # arrows (">>>", "-->", "=>")
    (re.compile(r"(?<![<>])(?:<{2,}|>{2,}|-+>|<-+|=+>)"), " "),
]

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

# bare domains when a path follows (bit.ly/xyz, t.co/abc, dlvr.it/x), only
# for common TLDs, so "Mr.Smith/John" is left alone
_TLDS = (r"(?i:com|net|org|edu|gov|info|biz|io|co|in|me|ly|gl|tv|ai|app|dev|xyz"
         r"|us|uk|be|to|it|tt|gy|gd|cc|ws|fm|gg|sh|link|page|site|online|store|shop|news)")
_BARE = (
    r"(?<![A-Za-z0-9.@_\-])"
    rf"(?:{_LABEL}\.)+{_TLDS}(?![A-Za-z0-9])"
    # "pic.twitter.com / zM9BeW24H4": spaced-out paths, as in some datasets
    rf"(?:/(?:{_TAIL})?|(?: / [A-Za-z0-9_.~%-]+)+)"
)
# bare domains with no path, only for TLDs that are never ordinary words, and
# only in lowercase, so "movie.In the end" and "ASP.NET" stay text
_SOLO_TLDS = (r"(?:com|net|org|edu|gov|io|info|biz|app|dev|xyz|online|store|shop|site"
              r"|tech|blog|news|live|(?:co|com|gov|ac|org|net|edu|nic)\.[a-z]{2})")
_SOLO = (
    r"(?<![A-Za-z0-9.@_\-])"
    rf"(?:{_LABEL}\.)+{_SOLO_TLDS}(?![A-Za-z0-9\-]|\.[A-Za-z0-9])"
)

_WEB = re.compile(rf"(?P<url>{_URL})|(?P<email>{_EMAIL})|(?P<bare>{_BARE}|{_SOLO})")
_WEB_GATE = re.compile(r"[@/]|\.[A-Za-z]")
# "john [at] gmail [dot] com"
_EMAIL_OBFUSCATED = re.compile(
    r"(?<![\w.+-])[\w.+-]+\s*[\[(]\s*at\s*[\])]\s*[\w-]+(?:\s*[\[(]\s*dot\s*[\])]\s*[A-Za-z]{2,}){1,3}",
    re.IGNORECASE,
)


def _web_repl(m: "re.Match[str]") -> str:
    return f" {_P.EMAIL} " if m.lastgroup == "email" else f" {_P.URL} "


# --------------------------------------------------------------------------
# IP addresses
# --------------------------------------------------------------------------

_OCTET = r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"
_IPV4 = re.compile(
    rf"(?<![A-Za-z0-9])(?<![0-9]\.)(?:{_OCTET}\.){{3}}{_OCTET}(?::[0-9]{{1,5}})?(?:/[0-9]{{1,2}})?"
    r"(?![A-Za-z0-9]|\.[0-9])"
)
_HEX4 = r"[0-9A-Fa-f]{1,4}"
_IPV6 = re.compile(
    r"(?<![A-Za-z0-9:])(?<![0-9]\.)(?:"
    rf"(?:{_HEX4}:){{7}}{_HEX4}"
    rf"|(?:{_HEX4}:){{1,7}}:(?:{_HEX4}(?::{_HEX4}){{0,5}})?"
    rf"|::{_HEX4}(?::{_HEX4}){{0,6}}"
    r")(?![A-Za-z0-9:])"
)


def _ipv6_repl(m: "re.Match[str]") -> str:
    s = m.group()
    groups = [g for g in s.split(":") if g]
    # needs a digit ("bad::" is not an address), two groups ("::3" is a
    # face) and at most 8 groups
    if not any(c.isdigit() for c in s) or not 2 <= len(groups) <= (7 if "::" in s else 8):
        return s
    return f" {_P.IP} "


# --------------------------------------------------------------------------
# Usernames
# --------------------------------------------------------------------------

_USER = re.compile(
    # @handle: not inside a word ("sh@t"), not a time ("@5pm"), with a letter
    r"(?<![A-Za-z0-9@])@(?![0-9]{1,2}(?::[0-9]{2})?\s?[aApP]\.?[mM]\b)"
    r"(?=[A-Za-z0-9_.]*[A-Za-z_])[A-Za-z0-9_](?:[A-Za-z0-9_.]{0,28}[A-Za-z0-9_])?(?![\w@])"
    # "@ Borderlands": a space after the @, which some tweet datasets have
    r"|(?<!@)@[ \t_~*^|]*[ \t](?=[A-Z])(?![xX]D+(?![A-Za-z0-9]))[A-Za-z0-9_]{3,15}(?![A-Za-z0-9])"
    # Reddit users
    r"|(?<![\w/])/?u/[A-Za-z0-9_-]{3,20}(?![\w-])"
)

# --------------------------------------------------------------------------
# IDs and phone numbers
# --------------------------------------------------------------------------

# UUIDs, MAC addresses and 12/16 digit card or Aadhaar-style numbers are
# always IDs
_ID_FIXED = re.compile(
    r"(?<![\w-])(?:"
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
    r"|[0-9A-Fa-f]{2}([:-])(?:[0-9A-Fa-f]{2}\1){4}[0-9A-Fa-f]{2}"
    r"|[0-9]{4}([ -])[0-9]{4}(?:\2[0-9]{4}){1,2}"
    r")(?![\w-])"
)

# a value named by the words before it: "order #12345", "OTP is 4821",
# "use code SAVE20", "PNR: 2345678901"
_ID_CONTEXT = re.compile(
    r"(?<![A-Za-z0-9\[])(?i:(?:order|otp|pin|code|coupon|promo|voucher|tracking|awb|ref|reference"
    r"|ticket|case|complaint|booking|pnr|invoice|receipt|transaction|txn|utr|policy"
    r"|serial|imei|incident|claim|confirmation|account|acct|customer|membership|id)(?![A-Za-z0-9]))"
    r"((?:\s*(?i:(?:number|num|no|code|id|is)(?![A-Za-z0-9])\.?|[#:=-])){0,3}\s*)"
    r"([A-Za-z0-9](?:[A-Za-z0-9_/-]{0,40}[A-Za-z0-9])?)(?![A-Za-z0-9])"
)
# "482913 is your OTP"
_OTP_FIRST = re.compile(
    r"(?<![A-Za-z0-9])[0-9]{4,8}(?=\s+is\s+(?:your|the)\s+(?i:otp|one[ -]time|verification|login"
    r"|security|confirmation|pin|code)\b)",
    re.IGNORECASE,
)

# a phone number named by the words before it, any shape: "call me at 555-1234"
_PHONE_CONTEXT = re.compile(
    r"(?<![A-Za-z0-9\[])(?i:(?:call|text|sms|phone|mobile|mob|cell|contact|tel|telephone|whatsapp"
    r"|helpline|hotline|fax|number|ph)(?![A-Za-z0-9]))"
    r"((?:\s*(?i:(?:number|num|no|them|him|her|me|us|at|on|is)(?![A-Za-z0-9])\.?|[#:-])){0,3}\s*)"
    r"(\+?\(?[0-9]{1,13}\)?(?:[ .-]\(?[0-9]{1,13}\)?){0,5})(?![A-Za-z0-9])"
)
_YEAR_RANGE = re.compile(r"(?:19|20)[0-9]{2}[ .-](?:19|20)[0-9]{2}")

# phone shapes that are safe without context. The boundaries only look at
# what survives tokenization (ASCII letters and digits, which stay in one
# token, and digits joined into one number by . , : /), so a second pass
# over clean text changes nothing.
_PHONE = re.compile(
    r"(?<![A-Za-z0-9+])(?<![0-9][.,/:])(?:"
    r"(?P<intl>\+[0-9]{1,3}(?:[ .-]?\([0-9]{1,4}\))?(?:[ .-]?[0-9]{2,10}){1,4})"
    r"|\([0-9]{2,5}\)[ .-]?[0-9]{3,4}[ .-]?[0-9]{3,5}"                        # (555) 123-4567
    r"|1(?P<s1>[ .-])[0-9]{3}(?P=s1)[0-9]{3}(?P=s1)[0-9]{4}"                  # 1-800-123-4567
    r"|[0-9]{3}(?P<s2>[ .-])[0-9]{3,4}(?P=s2)[0-9]{4}"                        # 555-123-4567, 020 7946 0958
    r"|[0-9]{4}(?P<s3>[ .-])[0-9]{3}(?P=s3)[0-9]{4}"                          # 1800 123 4567
    r"|[6-9][0-9]{4}[ -][0-9]{5}"                                             # 98765 43210
    r"|0[0-9]{2,4}-[0-9]{6,8}"                                                # 011-23456789
    r"|(?P<bare>[0-9]{10,12})"
    r")(?![A-Za-z0-9]|[.,/:][0-9])"
)


def _digits(s: str) -> int:
    return sum(c.isdigit() for c in s)


_PHONE_GROUP = re.compile(r"\(?[0-9]+\)?")


def _phone_repl(m: "re.Match[str]") -> str:
    s = m.group()
    if m.group("intl"):
        # "+91 98765 43210 555-123" ran into the next number: keep the
        # longest run of whole groups with at most 13 digits (country
        # code and a ten-digit number)
        cut, count = 0, 0
        for g in _PHONE_GROUP.finditer(s):
            count += _digits(g.group())
            if count > 13:
                break
            cut = g.end()
        return f" {_P.PHONE} " + s[cut:] if _digits(s[:cut]) >= 8 else s
    if m.group("bare") and not (
        (len(s) == 10 and s[0] != "0" and s[0] != "1")             # 98765 43210, (555) 123 4567
        or (len(s) == 11 and s[0] in "01" and s[1] in "23456789")  # 0 / 1 + ten digits
        or (len(s) == 12 and s.startswith("91") and s[2] in "6789")
    ):
        return s
    return f" {_P.PHONE} "


def _phone_context_repl(m: "re.Match[str]") -> str:
    value = m.group(2)
    if not 7 <= _digits(value) <= 15 or _YEAR_RANGE.fullmatch(value):
        return m.group()
    return m.group()[: m.start(2) - m.start()] + f" {_P.PHONE} "


# a number with a unit is a quantity, not an ID ("10000mAh", "1080p", "21st",
# "1920x1080")
_UNIT = re.compile(
    r"[0-9]+(?:\.[0-9]+)?(?:mah|mbps|gbps|mb|gb|tb|kb|mp|px|hz|khz|mhz|ghz|kg|mg|km|cm|mm"
    r"|ml|kw|kwh|hrs?|mins?|secs?|ms|fps|rpm|cc|bhp|nm|ft|lbs?|oz|rs|st|nd|rd|th"
    r"|[kmbxpswvgl])|[0-9]+(?:x[0-9]+)+p?",
    re.IGNORECASE,
)
_YEAR = re.compile(r"(?:19|20)[0-9]{2}")
_ID_SEP = re.compile(r"[-_/]")


def _id_context_repl(m: "re.Match[str]") -> str:
    value = m.group(2)
    digits = _digits(value)
    if any(c.isalpha() for c in value):
        # SAVE20, F6DK04; not a quantity ("case 10000mAh")
        is_id = digits >= 2 and len(value) >= 4 and not _UNIT.fullmatch(value)
    elif value.isdigit():
        is_id = digits >= 4 and not _YEAR.fullmatch(value)  # 4821, not "the claim is 2018"
    else:
        # "150202-000419", not "case 10/10" or "order 2019-2020"
        is_id = max(map(len, _ID_SEP.split(value))) >= 5
    if is_id:
        # "order #12345" -> "order [ID]"
        return m.group()[: m.start(2) - m.start()].rstrip("# \t") + f" {_P.ID} "
    return m.group()


# letters+digits tokens and long numbers, judged by _id_repl. Boundaries as
# for _PHONE.
_ID_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?<![0-9][.,/:])#?(?=[A-Za-z0-9_/-]*[0-9])[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*"
    r"(?![A-Za-z0-9])"
)
_ROUND = re.compile(r"[0-9]{1,3}0{6,}")       # 10000000000: an amount, not an ID


def _is_id_part(part: str) -> bool:
    """One run of letters and digits with no separators in it."""
    digits = _digits(part)
    if digits == 0:
        return False
    if digits == len(part):
        return digits >= 9 and not _ROUND.fullmatch(part)
    if _UNIT.fullmatch(part):
        return False
    if digits >= 5:
        return True                            # ORD123456, TXN8273648, B07XJ8C8F5
    # short codes need letters and digits mixed up: AB12CD34, x7k2p9
    switches = sum(a.isdigit() != b.isdigit() for a, b in zip(part, part[1:]))
    return len(part) >= 6 and switches >= 3


def _id_repl(m: "re.Match[str]") -> str:
    tok = m.group()
    # a hashtag body is judged like any other token ("#8Q6MFD" is a booking
    # code), since the "#" is gone by the time a second pass would see it
    core = tok[1:] if tok[0] == "#" else tok
    parts = _ID_SEP.split(core)
    numbers = [len(p) for p in parts if p.isdigit()]
    # "150202-000419" and "ORD-2023-11-998877" are IDs, "1985-2012-ish"
    # and "150-600mm" are not
    is_id = any(map(_is_id_part, parts)) or (
        len(parts) > 1 and bool(numbers) and max(numbers) >= 5 and _digits(core) >= 8
    ) or (tok[0] == "#" and core.isdigit() and len(core) >= 5)   # "#12345", not "#2020"
    return f" {_P.ID} " if is_id else tok


# --------------------------------------------------------------------------
# Hashtags
# --------------------------------------------------------------------------

_HASHTAG = re.compile(r"(?<![\w&#])#([A-Za-z0-9_]*[A-Za-z][A-Za-z0-9_]*)(?!\w)")
_HASH_OTHER = re.compile(r"(?<![\w&#])#(?=[^\W\d_])")       # #भारत
# split CamelCase: NeverAgain, SXSWParty, iPhone6Plus (but not iPhone, eBay)
_CAMEL = re.compile(r"(?<=[a-z])(?<!\b[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[0-9])(?=[A-Z][a-z])")


def _hashtag_repl(m: "re.Match[str]") -> str:
    return " " + _CAMEL.sub(" ", m.group(1).replace("_", " ")) + " "


# --------------------------------------------------------------------------
# Token pattern
# --------------------------------------------------------------------------

_JOINER = r"[\u200c\u200d]"
_APOSTROPHE = r"[\u0027\u2019\u02bc]"


def _word(body: str) -> str:
    # letters, optionally with ZWJ/ZWNJ between two letters
    return rf"[{body}]+(?:{_JOINER}[{body}]+)*"


def _build_token_regex() -> "re.Pattern[str]":
    """Matches placeholders, emoji, emoticons, numbers, words and punctuation.

    Whitespace, underscores and anything else no alternative accepts fall
    between matches and never reach the token list.
    """
    words: List[str] = []
    known: List[str] = []

    # Latin first, it is the most common case
    order = ["latin"] + [s for s in _TABLES if s != "latin"]
    for script in order:
        letters, _ = _TABLES[script]
        body = _class_body(letters)
        known.append(body)
        if script == "latin":
            # ASCII digits belong to Latin words ("covid19", "ps5"), and so do
            # the asterisks of censored words ("f**k", "sh**")
            words.append(rf"[0-9{body}]+(?:(?:{_APOSTROPHE}|\*+)[0-9{body}]+)*(?:\*{{2,}})?")
        else:
            words.append(_word(body))

    # best-effort for scripts we do not list (Cyrillic, Greek, CJK, ...):
    # a run of letters that is not one of our known scripts
    words.append(rf"(?:(?![{''.join(known)}])[^\W\d_])+")

    placeholder = "|".join(re.escape(p) for p in PLACEHOLDER_TOKENS)
    # a number, with any letters glued to it: 4.5/5, 10:30pm, 2nd, 1080p
    number = r"\d+(?:[.,:/]\d+)*[0-9A-Za-z]*"
    punct = r"[!?]+|\.+|(?P<pc>[^\w\s])(?P=pc)*"

    return re.compile(
        f"(?P<ph>{placeholder})"
        f"|(?P<emo>{'|'.join(f'(?:{p})' for p, _ in _EMOTICONS)})"
        f"|(?P<emoji>{_EMOJI})"
        f"|(?P<num>{number})"
        f"|(?P<word>{'|'.join(words)})"
        f"|(?P<punct>{punct})"
    )


_TOKEN = _build_token_regex()

# --------------------------------------------------------------------------
# Token post-processing
# --------------------------------------------------------------------------

# case-insensitive, so "Sooooo" and "AAAa" are caught the first time
_ELONGATED = re.compile(r"([^\W\d_])\1{2,}", re.IGNORECASE)          # sooooo -> soo
_REPEATED_PAIR = re.compile(r"([^\W\d_]{2})\1{3,}", re.IGNORECASE)   # hahahahaha -> hahaha
# valid Roman numerals, any case: "World War III", "DCCC" stay as they are
_ROMAN = re.compile(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})", re.IGNORECASE)
_PUNCT_RUN = re.compile(r"(.)\1{3,}")
# punctuation that is only noise to the model; a lone "*" is a footnote,
# a correction ("*their") or a bullet
_DROP_PUNCT = frozenset("*\\^`|~_\u2022\u00b7\u2023\u25e6\u2043")
_PUNCT_OK: Dict[str, bool] = {}


def _keep_2(m: "re.Match[str]") -> str:
    return m.group()[:2]


def _keep_6(m: "re.Match[str]") -> str:
    return m.group()[:6]


def _keep_punct(ch: str) -> bool:
    ok = _PUNCT_OK.get(ch)
    if ok is None:
        cat = unicodedata.category(ch)
        ok = _PUNCT_OK[ch] = ch not in _DROP_PUNCT and (cat[0] == "P" or cat in ("Sm", "Sc"))
    return ok


# --------------------------------------------------------------------------
# Names and religions -> [NAME], [RELIGION]
# --------------------------------------------------------------------------

# The word lists. Ship this file with tokenizer.py: production has to mask
# the same words as training did.
VALUES_FILE = Path(__file__).with_name("values.json")

# lists in values.json that are not things to mask on their own
_NOT_ENTITY_LISTS = frozenset(("names", "religions", "name_exceptions"))


def _emoji_words() -> FrozenSet[str]:
    """Every word an emoji can be written as ("fox face", "tulip", "om")."""
    words = {"flag", "stars"}
    for lo, hi in ((0x2000, 0x3300), (0x1F000, 0x1FB00)):
        for cp in range(lo, hi):
            if _EMOJI_RE.fullmatch(chr(cp)):
                words.update(_char_name(chr(cp)).split())
    return frozenset(words)


def _load_entity_words(path: Path) -> Dict[str, str]:
    """Lowercase word -> [NAME] or [RELIGION].

    A name stays a word when it is also an everyday word ("name_exceptions",
    made by build_name_exceptions.py), a religion ("Jain"), a word in any
    other list ("May" is a month, "Jordan" a country), or part of an emoji's
    name ("🦊" is "fox face", so "Fox" stays too).
    """
    values = json.loads(path.read_text(encoding="utf-8"))
    norm = lambda word: unicodedata.normalize("NFKC", word).lower()
    religions = set(map(norm, values["religions"]))
    keep = set(map(norm, values.get("name_exceptions", ()))) | religions | _emoji_words()
    for key, words in values.items():
        if key not in _NOT_ENTITY_LISTS:
            keep.update(map(norm, words))
    entities = {word: _P.NAME for word in map(norm, values["names"]) if word not in keep}
    entities.update(dict.fromkeys(religions, _P.RELIGION))
    return entities


_ENTITY_WORDS = _load_entity_words(VALUES_FILE)


def _entity(word: str) -> Optional[str]:
    """The placeholder for a name or religion word, also possessive ("Rahul's")."""
    key = word.lower()
    found = _ENTITY_WORDS.get(key)
    if found is None and len(key) > 2 and key[-1] == "s" and key[-2] in "'\u2019\u02bc":
        found = _ENTITY_WORDS.get(key[:-2])
    return found


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


class SentiVoxTokenizer:
    """Cleans text for the SentiVox model. Use the same settings for the
    training data and in production, or the model sees two different inputs.

    lowercase  Lowercase every word (placeholders stay uppercase). With
               False, normal casing is kept but words written entirely in
               capitals are still lowercased, because a cased WordPiece
               vocabulary splits them into fragments ("HATE" -> "H ##ATE").
    demojize   Write emojis and emoticons as words ("😂" -> "face with tears
               of joy", ":)" -> "slightly smiling face"). Keep this on for
               MuRIL, which has no emoji in its vocabulary. With False,
               emojis and emoticons are kept as they are.
    normalize  Unicode normal form ("NFKC" by default, None to skip). It also
               folds curly quotes, apostrophes and dashes to ASCII.
    """

    def __init__(
        self,
        *,
        lowercase: bool = True,
        demojize: bool = True,
        normalize: Optional[str] = "NFKC",
    ) -> None:
        self.lowercase = lowercase
        self.demojize = demojize
        self.normalize = normalize

    def __call__(self, text: str) -> str:
        return self.clean(text)

    def clean(self, text: str) -> str:
        """The cleaned text, ready for the model's own tokenizer."""
        return " ".join(self.tokenize(text))

    def tokenize(self, text: str) -> List[str]:
        """The cleaned text as a list of tokens. Joining them with spaces
        gives exactly what clean() returns."""
        if not isinstance(text, str):
            raise TypeError(f"SentiVoxTokenizer expects str, got {type(text).__name__}")
        if not text or text.isspace():
            return []
        return self._tokens(self._prepare(text))

    # -- stage 1: text -> text --------------------------------------------

    def _prepare(self, text: str) -> str:
        if not text.isascii():
            text = _MOJIBAKE.sub(_unmojibake, text)
            text = _SHRUG.sub(" \U0001f937 ", text)
        if "&" in text:
            # twice, for double-escaped text like "&amp;lt;3"
            text = _ENTITY.sub(lambda m: html.unescape(m.group()), text)
            text = _ENTITY.sub(lambda m: html.unescape(m.group()), text)
        if self.normalize and not text.isascii():
            text = unicodedata.normalize(self.normalize, text.translate(_TYPO_FOLD))
        text = _ZW_SPACE.sub(" ", text)
        text = _INVISIBLE.sub("", text)

        # boilerplate first: "<Media omitted>" would otherwise go as an HTML tag
        text = _BOILERPLATE.sub(f" {_P.RT} ", text)
        if "<" in text:
            text = _UNK.sub(" ", _HTML_TAG.sub(" ", text))
        if "{" in text:
            text = _LINK_STUB.sub(f" {_P.URL} ", text)
        if "](" in text:
            text = _MD_LINK.sub(r"\1 \2", text)

        # hashtags before masking, so the words they hold ("#CaseClosed")
        # are seen by the context rules on the first pass
        if "#" in text:
            text = _HASH_OTHER.sub("", _HASHTAG.sub(_hashtag_repl, text))
        text = self._mask(text)
        if self.demojize:
            text = _EMOTICON_RE.sub(_emoticon_to_emoji, text)
        for pattern, repl in _MARKDOWN:
            text = pattern.sub(repl, text)
        if self.demojize and not text.isascii():
            text = _STARS.sub(_stars_repl, text)
            text = _EMOJI_RUN.sub(_emoji_run_to_words, text)
        return text

    @staticmethod
    def _mask(text: str) -> str:
        """Replace the entities with placeholders. The order matters: URLs
        hold @ and digits, e-mails hold @, and handles can hold digits."""
        if _WEB_GATE.search(text):
            text = _WEB.sub(_web_repl, text)
        if "[" in text or "(" in text:
            text = _EMAIL_OBFUSCATED.sub(f" {_P.EMAIL} ", text)
        if ":" in text:
            text = _IPV6.sub(_ipv6_repl, text)
        if "." in text:
            text = _IPV4.sub(f" {_P.IP} ", text)
        if "@" in text or "u/" in text:
            text = _USER.sub(f" {_P.USER} ", text)
        if any(c.isdigit() for c in text):
            text = _ID_FIXED.sub(f" {_P.ID} ", text)
            text = _ID_CONTEXT.sub(_id_context_repl, text)
            text = _OTP_FIRST.sub(f" {_P.ID} ", text)
            text = _PHONE_CONTEXT.sub(_phone_context_repl, text)
            text = _PHONE.sub(_phone_repl, _PHONE.sub(_phone_repl, text))
            text = _ID_TOKEN.sub(_id_repl, text)
        return text

    # -- stage 2: text -> tokens ------------------------------------------

    def _tokens(self, text: str) -> List[str]:
        out: List[str] = []
        append, lowercase = out.append, self.lowercase
        prev, repeats = None, 0
        for m in _TOKEN.finditer(text):
            kind, tok = m.lastgroup, m.group()

            if kind == "word":
                entity = _entity(tok)
                if entity is not None:
                    # a religion right after a name is a surname: "Rahul Jain"
                    tok = _P.NAME if prev == _P.NAME else entity
                else:
                    if (len(tok) > 2 and (_ELONGATED.search(tok) or _REPEATED_PAIR.search(tok))
                            and not _ROMAN.fullmatch(tok)):
                        tok = _REPEATED_PAIR.sub(_keep_6, _ELONGATED.sub(_keep_2, tok))
                    if lowercase or (tok.isupper() and sum(c.isalpha() for c in tok) > 1):
                        tok = tok.lower()
            elif kind == "num":
                if lowercase:
                    tok = tok.lower()
            elif kind == "punct":
                if not _keep_punct(tok[0]) or (len(tok) > 2 and tok[0] in "-=#+"):
                    continue                   # noise, or a separator like "---"
                # "!!!!!!" -> "!!!", "!!!!???" -> "!!!???"
                tok = _PUNCT_RUN.sub(r"\1\1\1", tok)[:6] if len(tok) > 3 else tok
                if not tok.strip(".") and out and not out[-1].strip("."):
                    out[-1] = prev = (out[-1] + tok)[:3]   # ". . ." -> "..."
                    continue

            if tok == prev:
                repeats += 1
                # placeholders once, anything else at most three times
                if tok in _PLACEHOLDERS or repeats > 3:
                    continue
            else:
                prev, repeats = tok, 1
            append(tok)
        # a ">" left at the very start is a quote marker, as at the start of a line
        while out and not out[0].strip(">"):
            del out[0]
        return out


_DEFAULT = SentiVoxTokenizer()


def _tokenizer_for(normalize: Optional[str], lowercase: bool, demojize: bool) -> SentiVoxTokenizer:
    if normalize == "NFKC" and lowercase and demojize:
        return _DEFAULT
    return SentiVoxTokenizer(lowercase=lowercase, demojize=demojize, normalize=normalize)


def senti_vox_tokenizer(
    text: str,
    *,
    normalize: Optional[str] = "NFKC",
    lowercase: bool = True,
    demojize: bool = True,
) -> List[str]:
    """Clean `text` and return it as tokens. See SentiVoxTokenizer for the
    options. " ".join() of the result is the cleaned text.

    Raises TypeError for anything that is not a str (None included).
    """
    return _tokenizer_for(normalize, lowercase, demojize).tokenize(text)


def clean_text(
    text: str,
    *,
    normalize: Optional[str] = "NFKC",
    lowercase: bool = True,
    demojize: bool = True,
) -> str:
    """Clean `text` and return it as a single string, ready for the model's
    tokenizer. Same options as senti_vox_tokenizer."""
    return _tokenizer_for(normalize, lowercase, demojize).clean(text)


# --------------------------------------------------------------------------
# Hugging Face integration
# --------------------------------------------------------------------------

# a word whose sub-word embeddings seed each new placeholder embedding
_PLACEHOLDER_SEEDS = {
    _P.URL: "link", _P.EMAIL: "email", _P.USER: "user", _P.NAME: "name",
    _P.PHONE: "phone", _P.ID: "number", _P.IP: "address", _P.RT: "repost",
    _P.RELIGION: "religion",
}


def register_placeholders(hf_tokenizer, model=None) -> int:
    """Add the placeholders to a Hugging Face tokenizer as whole tokens.

    Without this, MuRIL splits them up ("[PHONE]" becomes "[", "PH", "##ON",
    "##E", "]"). Pass the model as well to grow its embedding matrix; each
    new row starts as the average embedding of a matching word ("[URL]" from
    "link"), which learns faster than a random start. Call it once before
    training, then save the tokenizer next to the model so production loads
    the same vocabulary. Returns the number of tokens added.
    """
    added = hf_tokenizer.add_tokens(list(PLACEHOLDER_TOKENS), special_tokens=True)
    if model is None or not added:
        return added

    import torch

    model.resize_token_embeddings(len(hf_tokenizer))
    weights = model.get_input_embeddings().weight
    with torch.no_grad():
        for token, seed in _PLACEHOLDER_SEEDS.items():
            seed_ids = hf_tokenizer(seed, add_special_tokens=False)["input_ids"]
            if seed_ids:
                weights[hf_tokenizer.convert_tokens_to_ids(token)] = weights[seed_ids].mean(dim=0)
    return added
