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

from scripts.tokenizer import senti_vox_tokenizer as tokenize, _remove_urls_and_emails

_TESTFILE = Path(__file__).parent / "tests" / "test_vectors.json"

print(f"LOADING TEST FILES FROM '{_TESTFILE}'...")

VECTORS = json.loads(_TESTFILE.read_text(encoding="utf-8"))

class SharedVectors(unittest.TestCase):
    def test_vectors(self):
        for case in VECTORS["cases"]:
            with self.subTest(case["name"]):
                self.assertEqual(tokenize(case["input"], **case["options"]), case["expected"])


class PythonOnly(unittest.TestCase):
    def test_rejects_non_str(self):
        for bad in (None, 123, b"bytes", ["a"]):
            with self.assertRaises(TypeError):
                tokenize(bad)

    def test_tokens_never_contain_whitespace(self):
        pool = list("ab १२ हैं।\u200d\u200b\r\n\t 😀👍🏽র্য@/.:)") + ["https://x.com", "a@b.com", "\ufe0f"]
        rng = random.Random(7)
        for _ in range(3000):
            text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 40)))
            for tok in tokenize(text):
                self.assertTrue(tok)
                self.assertFalse(any(c in " \t\r\n\u00a0\u200b" for c in tok), (text, tok))

    def test_only_words_and_emojis_and_no_letter_lost(self):
        # every token is a word or an emoji, and every letter of the text
        # survives, when there is nothing to strip (no URL triggers, no
        # joiners, no variation selectors)
        pool = list("abcXYZ019 हैं।१२৩ক ୧ଏ ی۔٢ Ж日 😀👍🏽🇮🇳❤ !?,.;:()'#*-$₹%\n\t\r")
        is_letter = lambda c: unicodedata.category(c)[0] in "LM"
        is_emoji = lambda c: ord(c) >= 0x1F000 or c == "❤"
        rng = random.Random(11)
        for _ in range(3000):
            text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 60)))
            tokens = tokenize(text, normalize=None)
            for tok in tokens:
                is_word = is_letter(tok[0]) and all(is_letter(c) or c == "'" for c in tok)
                is_emoticon = re.fullmatch(r"[:;=][-']?(?:\)+|\(+|[DPp])|<3+", tok)
                self.assertTrue(is_word or is_emoticon or all(map(is_emoji, tok)), (text, tok))
            self.assertEqual("".join(c for t in tokens for c in t if is_letter(c)),
                             "".join(c for c in text if is_letter(c)), text)

    def test_adversarial_inputs_stay_fast(self):
        n = 200_000
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
            "joiners": "क" + "\u200d" * n,
            "apostrophes": "a'" * (n // 2),
            "emoji joiners": "👨" + "\u200d👩" * (n // 2),
            "emoticon run": ":" * n,
        }
        for name, text in probes.items():
            with self.subTest(name):
                t = time.perf_counter()
                tokenize(text)
                self.assertLess(time.perf_counter() - t, 1.0)

    def test_url_helper_defaults(self):
        self.assertEqual(_remove_urls_and_emails("a http://x.com b"), "a   b")


if __name__ == "__main__":
    unittest.main(verbosity=1)