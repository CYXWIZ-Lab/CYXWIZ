"""Generate the text computation-truth fixtures and the backend's text tables.

The word rule, the stop words, the Porter stemmer and VADER of
cyxwiz/text_processing.h are checked against Python on real text: the
mental-health statements Data Studio's Text tab is shown on, the Amazon
reviews, and hand-written cases for every VADER rule.

Writes, next to the backend sources (cyxwiz-backend/src/algorithms):
  text_unicode_tables.inc  Python's str.isalnum / isupper / islower / lower
                           (this Python's Unicode database), as code point
                           ranges, so the C++ reads letters, digits and case
                           exactly as Python does;
  text_stop_words.inc      sklearn's ENGLISH_STOP_WORDS plus their
                           contractions;
  vader_lexicon.inc        vaderSentiment's vader_lexicon.txt and the
                           one-code-point keys of emoji_utf8_lexicon.txt
                           (MIT, C.J. Hutto; the licence is in the file).
and the fixture fixtures/text_truth.json: hand-written
cases for every rule, sentences generated from VADER's own lexicon, boosters
and negations, mixed-script strings, and the 6,000 commonest words of the
statements for Porter (single words only: no row of a dataset is committed).

With --local OUT it also writes OUT from the real rows (2,000 statements, 600
reviews), which is not committed; test_computation_truth_text checks it too
when CYXWIZ_TEXT_TRUTH_LOCAL names it.

Run: python generate_text_fixtures.py [statements.csv] [reviews.csv] [--local OUT]
(defaults: D:/tmp/gui129/nbproj/sentiment_mental_health.csv and
reviews_20k.csv). Needs scikit-learn, vaderSentiment 3.3.2 and nltk.
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
import unicodedata

import sklearn
import nltk
import vaderSentiment
import vaderSentiment.vaderSentiment as vs_module
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

HERE = os.path.dirname(os.path.abspath(__file__))
ALGORITHMS = os.path.join(HERE, "..", "..", "..", "..", "cyxwiz-backend", "src", "algorithms")
FIXTURE = os.path.join(HERE, "..", "fixtures", "text_truth.json")
ARGS = [a for a in sys.argv[1:]]
LOCAL = None
if "--local" in ARGS:
    i = ARGS.index("--local")
    LOCAL = ARGS[i + 1]
    del ARGS[i:i + 2]
STATEMENTS = ARGS[0] if len(ARGS) > 0 else "D:/tmp/gui129/nbproj/sentiment_mental_health.csv"
REVIEWS = ARGS[1] if len(ARGS) > 1 else "D:/tmp/gui129/nbproj/reviews_20k.csv"

# The contractions of the stop words: with the apostrophe kept, "don't" is a
# word of its own and would otherwise top every count.
CONTRACTIONS = [
    "ain't", "aren't", "can't", "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't",
    "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "isn't", "it'd", "it'll",
    "it's", "let's", "mightn't", "mustn't", "needn't", "shan't", "she'd", "she'll", "she's", "shouldn't",
    "that's", "there's", "they'd", "they'll", "they're", "they've", "wasn't", "we'd", "we'll", "we're",
    "we've", "weren't", "what's", "where's", "who's", "won't", "wouldn't", "you'd", "you'll", "you're",
    "you've", "y'all",
]
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | set(CONTRACTIONS))

WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*")


def words(text: str) -> list[str]:
    """The word rule of TextProcessing::Words."""
    text = text.replace("\u2019", "'")
    out = []
    for m in WORD.finditer(text):
        w = "".join(lower_char(c) for c in m.group(0))
        if len(w) >= 2:
            out.append(w)
    return out


def lower_char(c: str) -> str:
    """str.lower of one code point, kept when Python lowers it to several."""
    low = c.lower()
    return low if len(low) == 1 else c


def ranges(pred) -> list[tuple[int, int]]:
    out = []
    start = None
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            ok = False
        else:
            ok = pred(chr(cp))
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


def c_string(s: str) -> str:
    out = []
    for b in s.encode("utf-8"):
        if b == 0x22:
            out.append('\\"')
        elif b == 0x5C:
            out.append("\\\\")
        elif 0x20 <= b < 0x7F and b != 0x3F:  # '?' escaped against trigraphs
            out.append(chr(b))
        else:
            out.append("\\x%02x" % b)
    # A hex escape swallows following hex digits: split the literal after one.
    text = "".join(out)
    text = re.sub(r"(\\x[0-9a-f]{2})(?=[0-9a-fA-F])", r'\1""', text)
    return '"' + text + '"'


def write_unicode_tables() -> None:
    alnum = ranges(lambda c: c.isalnum())
    upper = ranges(lambda c: c.isupper())
    # Title case of one character is its category (Lt): str.istitle of a
    # single capital is True, which is not what str.isupper tests.
    lower_or_title = ranges(lambda c: c.islower() or unicodedata.category(c) == "Lt")
    space = ranges(lambda c: c.isspace())
    lowers = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        low = c.lower()
        if len(low) == 1 and low != c:
            lowers.append((cp, ord(low)))
    path = os.path.join(ALGORITHMS, "text_unicode_tables.inc")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"// Generated by cyxwiz-engine/tests/computation_truth/reference/generate_text_fixtures.py from\n")
        f.write(f"// Python {sys.version.split()[0]}'s Unicode database ({unicodedata.unidata_version}); do not edit.\n")
        f.write("// clang-format off\n")
        for name, table in (("kAlnum", alnum), ("kUpper", upper), ("kLowerOrTitle", lower_or_title), ("kSpace", space)):
            f.write(f"constexpr CodePointRange {name}[] = {{\n")
            for i in range(0, len(table), 8):
                f.write("    " + " ".join(f"{{0x{a:X}, 0x{b:X}}}," for a, b in table[i:i + 8]) + "\n")
            f.write("};\n")
        f.write("constexpr LowerPair kLower[] = {\n")
        for i in range(0, len(lowers), 8):
            f.write("    " + " ".join(f"{{0x{a:X}, 0x{b:X}}}," for a, b in lowers[i:i + 8]) + "\n")
        f.write("};\n// clang-format on\n")


def write_stop_words() -> None:
    path = os.path.join(ALGORITHMS, "text_stop_words.inc")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"// Generated by cyxwiz-engine/tests/computation_truth/reference/generate_text_fixtures.py from\n")
        f.write(f"// scikit-learn {sklearn.__version__}'s ENGLISH_STOP_WORDS ({len(ENGLISH_STOP_WORDS)}) plus {len(CONTRACTIONS)} contractions; do not edit.\n")
        f.write("// clang-format off\n")
        for i in range(0, len(STOP_WORDS), 10):
            f.write("    " + " ".join(c_string(w) + "," for w in STOP_WORDS[i:i + 10]) + "\n")
        f.write("// clang-format on\n")


def write_vader_lexicon(sia: SentimentIntensityAnalyzer) -> None:
    licence = open(os.path.join(os.path.dirname(os.path.dirname(vaderSentiment.__file__)),
                                "vaderSentiment-3.3.2.dist-info", "LICENSE.txt"), encoding="utf-8").read().strip()
    path = os.path.join(ALGORITHMS, "vader_lexicon.inc")
    lex_lines = [ln for ln in sia.lexicon_full_filepath.rstrip("\n").split("\n") if ln]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("// Generated by cyxwiz-engine/tests/computation_truth/reference/generate_text_fixtures.py from\n")
        f.write("// vaderSentiment 3.3.2 (vader_lexicon.txt, emoji_utf8_lexicon.txt); do not edit.\n//\n")
        for ln in licence.splitlines():
            f.write(("// " + ln).rstrip() + "\n")
        f.write("// clang-format off\n#ifdef CYXWIZ_VADER_WORDS\n")
        for ln in lex_lines:
            word, measure = ln.strip().split("\t")[0:2]
            f.write(f"{{{c_string(word)}, {measure}}},\n")
        f.write("#endif\n#ifdef CYXWIZ_VADER_EMOJIS\n")
        for emoji, description in sia.emojis.items():
            if len(emoji) == 1:
                f.write(f"{{0x{ord(emoji):X}, {c_string(description)}}},\n")
        f.write("#endif\n// clang-format on\n")


def read_column(path: str, column: str, limit: int | None = None) -> list[str]:
    csv.field_size_limit(10 ** 9)
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out.append(row.get(column) or "")
            if limit and len(out) >= limit:
                break
    return out


VADER_CASES = [
    "VADER is smart, handsome, and funny.", "VADER is smart, handsome, and funny!",
    "VADER is very smart, handsome, and funny.", "VADER is VERY SMART, handsome, and FUNNY.",
    "VADER is VERY SMART, handsome, and FUNNY!!!", "VADER is VERY SMART, uber handsome, and FRIGGIN FUNNY!!!",
    "VADER is not smart, handsome, nor funny.", "The book was good.", "At least it isn't a horrible book.",
    "The book was only kind of good.", "The plot was good, but the characters are uncompelling and the dialog is not great.",
    "Today SUX!", "Today only kinda sux! But I'll get by, lol", "Make sure you :) or :D today!",
    "Catch utf-8 emoji such as \U0001F498 and \U0001F48B and \U0001F601", "Not bad at all",
    "Sentiment analysis has never been good.", "Sentiment analysis has never been this good!",
    "Most automated sentiment analysis tools are shit.", "With VADER, sentiment analysis is the shit!",
    "Other sentiment analysis tools can be quite bad.", "On the other hand, VADER is quite bad ass",
    "VADER is such a badass!", "Without a doubt, excellent idea.",
    "Roger Dodger is one of the most compelling variations on this theme.",
    "Roger Dodger is at least compelling as a variation on the theme.",
    "Roger Dodger is one of the least compelling variations on this theme.", "Not such a badass after all.",
    "Without a doubt, an excellent idea.", "no good", "no no good", "no, or bad", "no nor bad thing",
    "good good bad bad but good", "but", "", "   ", "!!!", "????", "good???", "bad?? really?",
    "the food was good", "the food was not good", "the food was very good", "the food was GOOD!!!",
    "I don't hate it", "never so happy", "never this sad", "without doubt good", "Kind of happy", "kind of",
    "sort of happy, the shit", "a bus stop is fine", "yeah right, great", "to die for", "kiss of death indeed",
    "I am SO HAPPY", "ALL CAPS ARE HAPPY", "happy SAD", "\u00c9T\u00c9 happy", "caf\u00e9 is lovely",
    "she's not   un-happy", "\u2764 love", "\U0001F600\U0001F600 fine", "text\u00a0with\u2003spaces good",
]

WORD_CASES = [
    "Don't stop, I've got it!", "rock'n'roll 'tis o'clock", "Caf\u00e9 \u00fcber na\u00efve r\u00e9sum\u00e9",
    "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440", "\u4f60\u597d\u4e16\u754c abc", "a I x 1 22 333",
    "snake_case and_more", "it\u2019s typographic", "hello--world...again", "HTML &quot;quoted&quot; text",
    "\u00b2 superscript x\u00b2", "\U0001F600 emoji \U0001F600", "", "   ", "''", "a'b 'ab' ab'",
]

PORTER_EXTRA = [
    "caresses", "ponies", "ties", "caress", "cats", "feed", "agreed", "plastered", "bled", "motoring", "sing",
    "conflated", "troubled", "sized", "hopping", "tanned", "falling", "hissing", "fizzed", "failing", "filing",
    "happy", "sky", "relational", "conditional", "rational", "valenci", "hesitanci", "digitizer", "conformabli",
    "radicalli", "differentli", "vileli", "analogousli", "vietnamization", "predication", "operator", "feudalism",
    "decisiveness", "hopefulness", "callousness", "formaliti", "sensitiviti", "sensibiliti", "triplicate",
    "formative", "formalize", "electriciti", "electrical", "hopeful", "goodness", "revival", "allowance",
    "inference", "airliner", "gyroscopic", "adjustable", "defensible", "irritant", "replacement", "adjustment",
    "dependent", "adoption", "homologou", "communism", "activate", "angulariti", "homologous", "effective",
    "bowdlerize", "probate", "rate", "cease", "controll", "roll", "a", "is", "y", "yy", "syzygy", "toy",
    "business", "feeling", "anxiety", "anxious", "worried", "generously", "happiness", "running", "flies",
    "dying", "lying", "skies", "news", "caf\u00e9s", "na\u00efvety",
]


def generated_sentences(sia: SentimentIntensityAnalyzer, rng: random.Random, n: int) -> list[str]:
    """Sentences that exercise every VADER rule: lexicon words, boosters and
    dampeners (in capitals too), negations, no / least / never / without
    doubt, but, kind of, the special idioms, emoticons, emojis, ! and ?."""
    lexicon = sorted(sia.lexicon)
    plain = [w for w in lexicon if w.isalpha()]
    boosters = sorted(vs_module.BOOSTER_DICT)
    negations = vs_module.NEGATE
    idioms = sorted(vs_module.SPECIAL_CASES)
    emojis = [e for e in sia.emojis if len(e) == 1]
    fillers = ["the", "a", "it", "was", "is", "and", "this", "so", "at", "least", "very", "no", "or", "nor",
               "never", "without", "doubt", "kind", "of", "but", "BUT", "I", "you", "really", "movie", "day"]
    out = []
    for _ in range(n):
        k = rng.randint(1, 18)
        parts = []
        for _ in range(k):
            r = rng.random()
            if r < 0.30:
                w = rng.choice(plain)
            elif r < 0.38:
                w = rng.choice(lexicon)
            elif r < 0.50:
                w = rng.choice(boosters)
            elif r < 0.58:
                w = rng.choice(negations)
            elif r < 0.62:
                w = rng.choice(idioms)
            elif r < 0.66:
                w = rng.choice(emojis)
            else:
                w = rng.choice(fillers)
            if rng.random() < 0.15:
                w = w.upper()
            elif rng.random() < 0.1:
                w = w.capitalize()
            if rng.random() < 0.15:
                w += rng.choice([",", ".", "!", "?", "!!", "...", ";", ":)"])
            parts.append(w)
        out.append(" ".join(parts) + rng.choice(["", "", ".", "!", "!!!", "?", "??", "?!", "!!!!!", "????"]))
    return out


def generated_texts(rng: random.Random, vocabulary: list[str], n: int) -> list[str]:
    """Mixed-script strings for the word rule: words, accents, Cyrillic,
    Greek, CJK, digits, underscores, apostrophes, punctuation, emojis."""
    pieces = vocabulary[:2000] + ["caf\u00e9", "\u00fcber", "na\u00efve", "\u00c9COLE", "\u0391\u03b8\u03ae\u03bd\u03b1",
              "\u041c\u043e\u0441\u043a\u0432\u0430", "\u6771\u4eac", "\ud55c\uad6d\uc5b4", "\u0627\u0644\u0639\u0631\u0628\u064a\u0629", "x\u00b2",
              "42", "3.14", "snake_case", "don't", "it\u2019s", "'quoted'", "o'clock", "--", "...", "&quot;",
              "\U0001f600", "\u2764", "#tag", "@user", "http://x.y/z", "A", "I", "e.g.", "U.S.A."]
    out = []
    for _ in range(n):
        k = rng.randint(1, 14)
        sep = [" ", " ", " ", ",", ", ", "\t", "\n", "-", "/", "\u00a0"]
        out.append("".join(rng.choice(pieces) + rng.choice(sep) for _ in range(k)).strip())
    return out


def main() -> None:
    sia = SentimentIntensityAnalyzer()
    write_unicode_tables()
    write_stop_words()
    write_vader_lexicon(sia)

    # VADER unrounded: polarity_scores rounds through the module's round().
    vs_module.round = lambda x, n=None: x
    porter = PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM)

    statements = [s for s in read_column(STATEMENTS, "statement") if s.strip()]
    rng = random.Random(129)
    vocab = {}
    for t in statements:
        for w in words(t):
            vocab[w] = vocab.get(w, 0) + 1
    common = sorted(vocab, key=lambda w: (-vocab[w], w))[:6000]

    word_cases = [{"text": t, "words": words(t)} for t in WORD_CASES + generated_texts(rng, common, 1500)]

    def vader_case(t: str) -> dict[str, object]:
        s = sia.polarity_scores(t)
        return {"text": t, "neg": s["neg"], "neu": s["neu"], "pos": s["pos"], "compound": s["compound"]}

    vader_cases = [vader_case(t) for t in VADER_CASES + generated_sentences(sia, rng, 3000)]
    stems = [{"word": w, "stem": porter.stem(w)} for w in PORTER_EXTRA + common]

    if LOCAL:
        non_ascii = [s for s in statements if any(ord(c) > 127 for c in s)]
        sample = rng.sample(statements, 1500) + rng.sample(non_ascii, min(500, len(non_ascii)))
        reviews = read_column(REVIEWS, "review_text", 600)
        local = {"source": {"statements": STATEMENTS, "reviews": REVIEWS},
                 "words": [{"text": t, "words": words(t)} for t in sample],
                 "vader": [vader_case(t) for t in sample + reviews],
                 "porter": [{"word": w, "stem": porter.stem(w)} for w in sorted(vocab)]}
        with open(LOCAL, "w", encoding="utf-8", newline="\n") as f:
            json.dump(local, f, ensure_ascii=False)
        print(f"local: words {len(local['words'])}, vader {len(local['vader'])}, porter {len(local['porter'])}")

    stop_check = sorted(STOP_WORDS)
    fixture = {
        "generator": "generate_text_fixtures.py",
        "versions": {"python": sys.version.split()[0], "unicode": unicodedata.unidata_version, "sklearn": sklearn.__version__,
                     "vaderSentiment": "3.3.2", "nltk": nltk.__version__},
        "vocabulary_from": os.path.basename(STATEMENTS),
        "stop_words": stop_check,
        "words": word_cases,
        "vader": vader_cases,
        "porter": stems,
    }
    with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
        json.dump(fixture, f, ensure_ascii=False, indent=0)
    print(f"words {len(word_cases)}, vader {len(vader_cases)}, porter {len(stems)}, stop words {len(stop_check)}")


if __name__ == "__main__":
    main()
