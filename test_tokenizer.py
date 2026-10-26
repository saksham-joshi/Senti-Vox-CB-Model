"""Runs test_vectors.json against the tokenizer, plus a few Python-only checks.

    python test_tokenizer.py
"""
import json
import random
import re
import time
import unicodedata
import unittest
from pathlib import Path

from scripts.tokenizer import (
    PLACEHOLDER_TOKENS,
    SentiVoxTokenizer,
    clean_text,
    export_tables,
    register_placeholders,
    senti_vox_tokenizer as tokenize,
)

_TESTFILE = Path(__file__).parent / "tests" / "test_vectors.json"

print(f"LOADING TEST FILES FROM '{_TESTFILE}'...")

VECTORS = json.loads(_TESTFILE.read_text(encoding="utf-8"))

# text that exercises every stage at once
_MESSY = [
    "@united lost my bag AGAIN \U0001f621\U0001f621\U0001f621\U0001f621 call 917-703-1472 https://t.co/x",
    "RT @user: Amazon and Goldman partner. Perfect strategy! . . thefinancialbrand.com/92681/marcus",
    "So I spent a few hours making something for fun. . . pic.twitter.com / zM9BeW24H4",
    "[NAME] was nowhere near them, he was by the Falcon. /s",
    ">not a cure-all. **Fuck no why u do this?** f**ked *and* ~~this~~",
    "#IDidNotVoteForThisClown! #1 pick #8Q6MFD order #12345 OTP is 482913",
    "Mail comes from 153.69.214.203 but fe80::1 too, write to a.b@c.co.in",
    "Sooooo goooood!!!!!! hahahahahaha lol lol lol lol \u2605\u2605\u2605\u2606\u2606",
    "\u092e\u0948\u0902 \u092c\u0939\u0941\u0924 \u0916\u0941\u0936 \u0939\u0942\u0901 \U0001f60a #\u092d\u093e\u0930\u0924",
    "<Media omitted> This message was deleted. Great product\u2026 Read more",
    "<br /><br />This movie was <i>awful</i>.&amp; don\u00e2\u20ac\u2122t \u00f0\u0178\u02dc\u201a",
    "\u00af\\_(\u30c4)_/\u00af whatever :P :-) ;) :'( </3 -_- T_T XD",
]


class SharedVectors(unittest.TestCase):
    def test_vectors(self):
        for case in VECTORS["cases"]:
            with self.subTest(case["name"]):
                tokens = tokenize(case["input"], **case["options"])
                self.assertEqual(tokens, case["expected"])
                self.assertEqual(clean_text(case["input"], **case["options"]), " ".join(tokens))


class PythonOnly(unittest.TestCase):
    def test_rejects_non_str(self):
        for bad in (None, 123, b"bytes", ["a"]):
            with self.assertRaises(TypeError):
                tokenize(bad)
            with self.assertRaises(TypeError):
                clean_text(bad)

    def test_class_matches_functions(self):
        for options in ({}, {"lowercase": False}, {"demojize": False}, {"normalize": None}):
            tok = SentiVoxTokenizer(**options)
            for text in _MESSY:
                self.assertEqual(tok.tokenize(text), tokenize(text, **options))
                self.assertEqual(tok(text), clean_text(text, **options))

    def test_tokens_never_contain_whitespace(self):
        pool = list("ab \u0967\u0968 \u0939\u0948\u0902\u0964\u200d\u200b\r\n\t \U0001f600\U0001f44d\U0001f3fd"
                    "\u09b0\u09cd\u09af@/.:)#*!") + ["https://x.com", "a@b.com", "\ufe0f", "9876543210"]
        rng = random.Random(7)
        for _ in range(3000):
            text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 40)))
            for options in ({}, {"demojize": False}):
                for tok in tokenize(text, **options):
                    self.assertTrue(tok)
                    self.assertFalse(any(c in " \t\r\n\u00a0\u200b" for c in tok), (text, tok))

    def test_every_token_is_a_known_kind(self):
        # a placeholder, a word or number, an emoji, an emoticon, or punctuation
        pool = list("abcXYZ019 \u0939\u0948\u0902\u0964\u0967\u0968\u09e9\u0995 \u0b67\u0b0f \u06cc\u06d4\u0662 \u0416\u65e5"
                    "\U0001f600\U0001f44d\U0001f3fd\U0001f1ee\U0001f1f3\u2764 !?,.;:()'#*-$\u20b9%\n\t\r@/")
        is_word_char = lambda c: unicodedata.category(c)[0] in "LMN" or c in "'*\u200c\u200d"
        number = re.compile(r"\d+(?:[.,:/]\d+)*[0-9A-Za-z]*")
        is_punct = lambda c: unicodedata.category(c)[0] == "P" or unicodedata.category(c) in ("Sm", "Sc")
        is_emoji = lambda c: ord(c) >= 0x1F000 or c in "\u2764\ufe0f\u200d"
        emoticon = re.compile(r"[:=;>][-']?(?:\)+|\(+|[DPp]+|[oO]|\||\*|/)|:'-?[()]|<3+|</3|[xX]D+"
                              r"|\^[_.-]\^|-[_.]-|T[_.]T|;_;|[oO][_.][oO]")
        rng = random.Random(11)
        for _ in range(3000):
            text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 60)))
            for options in ({}, {"demojize": False}):
                for tok in tokenize(text, **options):
                    ok = (tok in PLACEHOLDER_TOKENS or all(map(is_word_char, tok)) or number.fullmatch(tok)
                          or all(map(is_punct, tok)) or all(map(is_emoji, tok)) or emoticon.fullmatch(tok))
                    self.assertTrue(ok, (text, tok))

    def test_cleaning_twice_changes_nothing(self):
        # realistic text: exact, in every configuration
        texts = _MESSY + [c["input"] for c in VECTORS["cases"]]
        for options in ({}, {"lowercase": False}, {"demojize": False}, {"normalize": None}):
            for text in texts:
                once = clean_text(text, **options)
                self.assertEqual(clean_text(once, **options), once, (text, options))

    def test_cleaning_twice_rarely_changes_random_fragments(self):
        # random fragments can put dropped noise ("~", "**") between the parts
        # of a pattern, which only the second pass sees through; keep it rare
        rng = random.Random(3)
        pool = list("aAbB1290 .!?:)(-#@/*_~<>&;\u00e2\u20ac\u2122\U0001f602\u2605") + [
            "http://x.co", "www.a.com", "a@b.com", "@user", "u/name", "9876543210", "555-123-4567",
            "ORD123456", "#tag", "#CamelCase", "RT", "Read more", "<br>", "&amp;", "[NAME]", "\n", "...",
            "sooo", "III", "10/10", "10000mAh", "call me at", "order #", "This message was deleted"]
        changed = 0
        for _ in range(5000):
            text = " ".join(rng.choice(pool) for _ in range(rng.randint(0, 30)))
            once = clean_text(text)
            changed += clean_text(once) != once
        self.assertLess(changed, 10)

    def test_placeholders_stay_uppercase(self):
        text = "see https://x.com, mail a@b.com, ping @bob or 9876543210, order ORD123456, ip 1.2.3.4"
        for options in ({}, {"lowercase": False}):
            self.assertEqual(tokenize(text, **options),
                             ["see", "[URL]", ",", "mail", "[EMAIL]", ",", "ping", "[USER]", "or", "[PHONE]",
                              ",", "order", "[ID]", ",", "ip", "[IP]"])

    def test_adversarial_inputs_stay_fast(self):
        # guards against catastrophic backtracking: a quadratic regex takes
        # minutes here, a linear one well under a second
        n = 100_000
        probes = {
            "long local part": "a" * n + "@",
            "dotted local part": "a." * (n // 2) + "@",
            "many @": "a@" * (n // 2),
            "many labels": "x@" + "a." * (n // 2),
            "www spam": "www." * (n // 4),
            "www then punctuation": "www." + "!" * n,
            "http then dots": "http://" + "." * n,
            "open parens": "http://x.com/" + "(a" * (n // 2),
            "bare domain run": "a." * (n // 2),
            "bare domain slash": "bit.ly/" * (n // 7),
            "spaced path": "a.com" + " / a" * (n // 4),
            "joiners": "\u0915" + "\u200d" * n,
            "apostrophes": "a'" * (n // 2),
            "censored": "a*" * (n // 2),
            "emoji joiners": "\U0001f468" + "\u200d\U0001f469" * (n // 2),
            "emoji run": "\U0001f602" * n,
            "stars run": "\u2605" * n,
            "emoticon run": ":" * n,
            "digits": "1" * n,
            "digit groups": "123 " * (n // 4),
            "digit dashes": "12-" * (n // 3),
            "plus digits": "+1 " * (n // 3),
            "alnum dashes": "a1-" * (n // 3),
            "colons hex": "a:" * (n // 2),
            "keywords": "order " * (n // 6),
            "keyword then digits": "order # " + "1 " * (n // 2),
            "call then digits": "call " + "1 " * (n // 2),
            "hashtags": "#a" * (n // 2),
            "hash digits": "#" + "1" * n,
            "markdown stars": "**a " * (n // 4),
            "quotes": ">" * n,
            "html open": "<a " + "x" * n,
            "entities": "&a" * (n // 2),
            "mojibake": "\u00c3\u00a9" * (n // 2),
            "read more": "... read more " * (n // 14),
            "spaced dots": ". " * (n // 2),
            "elongation": "a" * n,
            "pairs": "ha" * (n // 2),
            "reddit users": "u/" * (n // 2),
        }
        for name, text in probes.items():
            with self.subTest(name):
                t = time.perf_counter()
                clean_text(text)
                self.assertLess(time.perf_counter() - t, 1.0)

    def test_register_placeholders(self):
        class StubTokenizer:
            def __init__(self):
                self.added = []

            def add_tokens(self, tokens, special_tokens=False):
                self.assertion = special_tokens
                new = [t for t in tokens if t not in self.added]
                self.added += new
                return len(new)

        stub = StubTokenizer()
        self.assertEqual(register_placeholders(stub), len(PLACEHOLDER_TOKENS))
        self.assertEqual(stub.added, list(PLACEHOLDER_TOKENS))
        self.assertTrue(stub.assertion)
        self.assertEqual(register_placeholders(stub), 0)

    def test_export_tables(self):
        tables = export_tables()
        self.assertIn("devanagari", tables["scripts"])
        self.assertTrue(tables["scripts"]["latin"]["letters"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
