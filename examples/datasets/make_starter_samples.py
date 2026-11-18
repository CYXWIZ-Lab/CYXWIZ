"""Build the small samples the starter examples train on.

The Engine's empty canvas and start page offer five starter graphs
(gui/starter_graphs.cpp). Each one reads a small, licence-clean sample that
lives in this folder, so the graph trains right after it opens, offline:

  airline_passengers.csv   already here (time-series forecasting)
  digits_8x8.csv           UCI optical handwritten digits, 8x8 (multiclass)
  review_sentences.csv     UCI Sentiment Labelled Sentences (text)
  spoken_digits/           Free Spoken Digit Dataset subset (audio)
  cats_dogs_sample/        Wikimedia Commons photos of cats and dogs (images)

Sources and licences are in README.md beside this script. Run it again to
rebuild a sample (an existing file is kept; delete it to fetch it again):

    python examples/datasets/make_starter_samples.py [output folder]

Needs scikit-learn (digits) and Pillow (images); the rest is the standard
library. The text, audio and image samples are downloaded.
"""

import csv
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

USER_AGENT = "cyxwiz-example-data/1.0 (starter sample builder)"
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))


def fetch(url, pause=0.0):
    """GET `url`; waits `pause` seconds first, and backs off when the server
    says too many requests (Wikimedia Commons does at about one per second)."""
    for attempt in range(6):
        time.sleep(pause if attempt == 0 else min(60, 5 * 2 ** attempt))
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 5:
                raise
    raise RuntimeError("unreachable")


def digits():
    """1,797 handwritten digits as 8x8 grey levels 0..16, label column 'class'."""
    path = os.path.join(OUT, "digits_8x8.csv")
    if os.path.exists(path):
        return
    from sklearn.datasets import load_digits  # ships the UCI data, no download

    data = load_digits()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"pixel_{i}" for i in range(64)] + ["class"])
        for row, label in zip(data.data.astype(int), data.target):
            writer.writerow(list(row) + [int(label)])
    print("digits_8x8.csv:", len(data.target), "rows")


def sentences():
    """3,000 review sentences (Amazon, IMDb, Yelp), each positive or negative."""
    path = os.path.join(OUT, "review_sentences.csv")
    if os.path.exists(path):
        return
    archive = zipfile.ZipFile(io.BytesIO(fetch(
        "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip")))
    rows = []
    for name in sorted(archive.namelist()):
        base = os.path.basename(name)
        if not base.endswith("_labelled.txt") or base.startswith("."):
            continue
        source = base.split("_")[0]
        for line in archive.read(name).decode("utf-8", errors="replace").splitlines():
            if "\t" not in line:
                continue
            sentence, label = line.rsplit("\t", 1)
            if label.strip() in ("0", "1"):
                rows.append((sentence.strip(), "positive" if label.strip() == "1" else "negative", source))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "sentiment", "source"])
        writer.writerows(rows)
    print("review_sentences.csv:", len(rows), "rows")


def spoken_digits():
    """Spoken digits 0..9, four speakers, ten takes each: one folder per digit."""
    folder = os.path.join(OUT, "spoken_digits")
    base = "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/"
    count = 0
    for digit in range(10):
        os.makedirs(os.path.join(folder, str(digit)), exist_ok=True)
        for speaker in ("george", "jackson", "lucas", "theo"):
            for take in range(10):
                name = f"{digit}_{speaker}_{take}.wav"
                path = os.path.join(folder, str(digit), name)
                if not os.path.exists(path):
                    with open(path, "wb") as f:
                        f.write(fetch(base + name))
                count += 1
    print("spoken_digits:", count, "clips")


def commons_files(category, limit):
    """File titles in a Wikimedia Commons category, sorted, at most `limit`."""
    query = urllib.parse.urlencode({
        "action": "query", "list": "categorymembers", "cmtitle": "Category:" + category,
        "cmtype": "file", "cmlimit": "500", "format": "json"})
    members = json.loads(fetch("https://commons.wikimedia.org/w/api.php?" + query, pause=1.0))["query"]["categorymembers"]
    titles = sorted(m["title"] for m in members if m["title"].lower().endswith((".jpg", ".jpeg", ".png")))
    return titles[:limit]


def commons_info(title):
    query = urllib.parse.urlencode({
        "action": "query", "titles": title, "prop": "imageinfo", "iiprop": "url|extmetadata",
        "iiurlwidth": "192", "format": "json"})
    page = next(iter(json.loads(fetch("https://commons.wikimedia.org/w/api.php?" + query, pause=1.0))["query"]["pages"].values()))
    info = page["imageinfo"][0]
    meta = info.get("extmetadata", {})
    plain = lambda key: _strip_tags(meta.get(key, {}).get("value", ""))
    return info["thumburl"], info["descriptionurl"], plain("Artist"), plain("LicenseShortName")


def _strip_tags(text):
    out, inside = [], False
    for ch in text:
        if ch == "<":
            inside = True
        elif ch == ">":
            inside = False
        elif not inside:
            out.append(ch)
    return " ".join("".join(out).split())


# Pictures in those categories that do not show one clear cat or dog at
# 96x96 (a stone cat, a crowd, a cat and a dog together, a tiny dog far off).
NOT_CLEAR = ("File:Ganjali Khan Complex", "File:Jakarta Indonesia Merdeka-Square",
             "File:Blossom and Laysha", "File:Caminata por los perros y animales Maracaibo",
             "File:Contre-jour photograph of a standing puppy", "File:DLRG", "File:Delta del Danubio",
             "File:Digging dog Delaware seashore")


def cats_dogs():
    """Photos of cats and dogs, 96x96, one folder per class, with their credits."""
    from PIL import Image

    folder = os.path.join(OUT, "cats_dogs_sample")
    credits_path = os.path.join(folder, "ATTRIBUTION.csv")
    if os.path.exists(credits_path):
        return
    free = ("cc0", "cc by", "cc-by", "public domain", "pd")
    credits = []
    for label, categories in (("cat", ["Quality_images_of_cats", "Featured_pictures_of_cats"]),
                              ("dog", ["Quality_images_of_dogs", "Featured_pictures_of_dogs"])):
        os.makedirs(os.path.join(folder, label), exist_ok=True)
        titles = []
        for category in categories:
            titles += [t for t in commons_files(category, 200) if t not in titles and not t.startswith(NOT_CLEAR)]
        kept = 0
        for title in titles:
            if kept == 30:
                break
            thumb, page, author, licence = commons_info(title)
            if not licence.lower().startswith(free):
                continue
            image = Image.open(io.BytesIO(fetch(thumb, pause=1.0))).convert("RGB")
            side = min(image.size)
            left, top = (image.width - side) // 2, (image.height - side) // 2
            image = image.crop((left, top, left + side, top + side)).resize((96, 96), Image.LANCZOS)
            name = f"{label}_{kept:02d}.jpg"
            image.save(os.path.join(folder, label, name), quality=88)
            credits.append((f"{label}/{name}", title, page, author, licence))
            kept += 1
        print(f"cats_dogs_sample/{label}:", kept, "photos")
    with open(credits_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "commons_title", "source", "author", "licence"])
        writer.writerows(credits)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    digits()
    sentences()
    spoken_digits()
    cats_dogs()
