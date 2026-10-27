"""Rebuilds "name_exceptions" in values.json: names that are also everyday words.

The tokenizer masks every word in the "names" list as [NAME], in any case.
Many names are ordinary words too ("love", "hope", "will", "white", "in"),
and masking those would erase exactly the words the model needs. They are
found by how raw, cased text writes them: a name is written "Johnson" in
the middle of a sentence, a word is written "love" and an acronym "AI".
The few that case gets wrong ("Hindi", "Christmas") are listed by hand below.

Hinglish words ("kiya", "khushi", "abhi") never show up in this English
text; they are kept safe by the "hinglish" list in values.json instead, as
the tokenizer never masks a word that any other list holds.

Run it again after changing the names list:

    python scripts/build_name_exceptions.py dataset/uncleaned
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

VALUES = Path(__file__).with_name("values.json")

# raw, cased text: file -> (text column, encoding). twitter_training.csv is
# left out, it repeats each tweet several times with small edits.
SOURCES = {
    "IMDB-Dataset.csv": ("review", "utf-8"),
    "goemotions_1.csv": ("text", "utf-8"),
    "goemotions_2.csv": ("text", "utf-8"),
    "goemotions_3.csv": ("text", "utf-8"),
    "Airline-Sentiment-2-w-AA.csv": ("text", "latin-1"),
    "Apple-Twitter-Sentiment-DFE.csv": ("text", "latin-1"),
    "judge-1377884607_tweet_product_company.csv": ("tweet_text", "latin-1"),
}

# a word is written in lowercase at least MIN_USES times, and in at least
# SHARE of its uses inside a sentence. An acronym is written in capitals at
# least MIN_USES times, and at least as often as Capitalized.
MIN_USES = 5
SHARE = 0.2

# words the counts get wrong: languages, peoples and festivals, which English
# always capitalizes; words the film reviews' cast lists make look like names
# ("Dick Van Dyke", "Mad Max"); and slang and news words that are newer than
# most of the text ("stan", "corona")
NOT_NAMES = {
    "arab", "bihari", "danish", "english", "french", "german", "hindi", "irish",
    "malay", "roman", "sanskrit", "tamil", "thai", "welsh",
    "baisakhi", "christmas", "deepavali", "rakhi", "vaisakhi", "valentine",
    "dick", "dyke", "max",
    "corona", "stan", "tory",
}

# links, @handles, #hashtags and HTML end a sentence, so the word after them
# counts as the first word
_NOISE = re.compile(r"https?://\S+|www\.\S+|[@#][ _]?\w+|<[^>]*>")
_WORD = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]+)(?:'[A-Za-z]+)?(?![A-Za-z0-9])")
_MID = re.compile(r"[A-Za-z0-9,]\s*$")


def case_counts(texts, names):
    """name -> uses written lowercase, Capitalized inside a sentence, and in
    capitals (not counting texts that are mostly capitals)."""
    lower, cap, upper = Counter(), Counter(), Counter()
    for text in texts:
        text = _NOISE.sub("\n", text)
        shouting = sum(c.isupper() for c in text) * 2 > sum(c.isalpha() for c in text)
        for m in _WORD.finditer(text):
            word = m.group(1)
            key = word.lower()
            if key not in names:
                continue
            if word.islower():
                lower[key] += 1
            elif word.isupper() and len(word) > 1:
                upper[key] += not shouting
            elif word[1:].islower() and _MID.search(text, max(0, m.start() - 20), m.start()):
                cap[key] += 1
    return lower, cap, upper


def main(raw_dir: str) -> None:
    values = json.loads(VALUES.read_text(encoding="utf-8"))
    names = set(values["names"])

    texts = []
    for file, (column, encoding) in SOURCES.items():
        path = Path(raw_dir) / file
        if not path.exists():
            print(f"skipping {file}: not found")
            continue
        texts += pd.read_csv(path, encoding=encoding)[column].dropna().astype(str).drop_duplicates().tolist()
    print(f"{len(texts)} texts")

    lower, cap, upper = case_counts(texts, names)
    words = {n for n in names if lower[n] >= MIN_USES and lower[n] >= SHARE * (lower[n] + cap[n])}
    acronyms = {n for n in names if upper[n] >= MIN_USES and upper[n] >= cap[n]}
    values["name_exceptions"] = sorted(words | acronyms | (NOT_NAMES & names))
    print(f"{len(values['name_exceptions'])} of {len(names)} names are everyday words")

    # values.json is kept with CRLF line endings
    out = json.dumps(values, indent=2, ensure_ascii=False).replace("\n", "\r\n") + "\r\n"
    VALUES.write_bytes(out.encode("utf-8"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dataset/uncleaned")
