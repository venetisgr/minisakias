#!/usr/bin/env python3
"""Build the law corpus the service answers from, out of https://github.com/lexifina/greece-law.

    python tools/sync_laws.py                      # sparse-clones the codes and the Constitution, then builds
    python tools/sync_laws.py --source ../greece-law   # or reads a checkout you already have

Writes two files under data/laws/:

    articles.jsonl   one line per article: id, code, label («ΠΚ 308»), number, heading, chapter, text
    codes.json       one entry per code: its name, abbreviation, enacting act, source PDF and — the part that matters —
                     which version of the text this is (as enacted / amended until a date / consolidated)

The upstream repository is ~10 GB (most of it Diavgeia decisions), so the clone is sparse and blobless: only
`codes/` and `constitution/` are fetched. Standard library only; needs `git` on the PATH for the clone.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

REPO = "https://github.com/lexifina/greece-law"
OUT = Path(__file__).resolve().parent.parent / "data" / "laws"

# folder → (abbreviation used in citations, short Greek name). The abbreviation is what the model cites and the
# website turns into a link, so it has to be unique and stable.
CODES: dict[str, tuple[str, str]] = {
    "constitution": ("Σ", "Σύνταγμα"),
    "penal-code": ("ΠΚ", "Ποινικός Κώδικας"),
    "criminal-procedure-code": ("ΚΠΔ", "Κώδικας Ποινικής Δικονομίας"),
    "civil-code": ("ΑΚ", "Αστικός Κώδικας"),
    "civil-procedure-code": ("ΚΠολΔ", "Κώδικας Πολιτικής Δικονομίας"),
    "labour-code": ("ΚΑΕΔ", "Κώδικας Ατομικού Εργατικού Δικαίου"),
    "road-traffic-code": ("ΚΟΚ", "Κώδικας Οδικής Κυκλοφορίας"),
    "administrative-procedure-code": ("ΚΔΔιαδ", "Κώδικας Διοικητικής Διαδικασίας"),
    "code-of-administrative-procedure-courts": ("ΚΔΔ", "Κώδικας Διοικητικής Δικονομίας"),
    "civil-servants-code": ("ΥΚ", "Υπαλληλικός Κώδικας"),
    "municipal-employees-code": ("ΚΚΔΚΥ", "Κώδικας Δημοτικών και Κοινοτικών Υπαλλήλων"),
    "municipalities-code": ("ΚΔΚ", "Κώδικας Δήμων και Κοινοτήτων"),
    "courts-organisation-code": ("ΚΟΔΚΔΛ", "Κώδικας Οργανισμού Δικαστηρίων"),
    "lawyers-code": ("ΚωδΔικ", "Κώδικας Δικηγόρων"),
    "public-revenue-collection-code": ("ΚΕΔΕ", "Κώδικας Είσπραξης Δημοσίων Εσόδων"),
    "tax-procedure-code": ("ΚΦΔ", "Κώδικας Φορολογικής Διαδικασίας"),
    "income-tax-code": ("ΚΦΕ", "Κώδικας Φορολογίας Εισοδήματος"),
    "vat-code": ("ΚΦΠΑ", "Κώδικας ΦΠΑ"),
    "property-tax-code": ("ΚΦΠ", "Κώδικας Φορολογίας Περιουσίας"),
    "indirect-transaction-taxes-code": ("ΚΕΦΣ", "Κώδικας Εμμέσων Φόρων επί των Συναλλαγών"),
    "customs-code": ("ΕΤΚ", "Εθνικός Τελωνειακός Κώδικας"),
}
# the introductory laws published with two of the codes get an abbreviation of their own
INTRO_LAWS = {"civil-code": ("ΕισΝΑΚ", "Εισαγωγικός Νόμος Αστικού Κώδικα"),
              "civil-procedure-code": ("ΕισΝΚΠολΔ", "Εισαγωγικός Νόμος Κώδικα Πολιτικής Δικονομίας")}

ARTICLE_FILE = re.compile(r"^(intro-law-)?art-(\d+.*)\.md$")
TITLE = re.compile(r"^#?\s*Άρθρ[οα]\s+(.+?)\s*$")
# what follows «Άρθρο»: 308 · 5Α · 3 - 4 · «52. - Δικαίωμα προσφυγής» (the ΚΠΔ prints the heading on the title line)
TITLE_NUMBER = re.compile(r"^(\d+)\s*([Α-Ωα-ωA-Za-z]{1,2})?(?:\s*[-–]\s*(\d+)\s*([Α-Ωα-ω]{1,2})?)?\s*(?:\.\s*[-–]?\s*(.+))?$")
# «Έννοια. Όποιος ζημιώσει …» — the 1984/1985 codes print the heading as a first short sentence
SENTENCE_HEADING = re.compile(r"^([Α-ΩΆΈΉΊΌΎΏ][^.\n\d]{2,90})\.\s+(?=[Α-ΩΆΈΉΊΌΎΏ«0-9])")
SENTENCE_HEADING_CODES = {"civil-code", "civil-procedure-code"}
COMMENT = re.compile(r"<!--.*?-->", re.S)
CHAPTER = re.compile(r"^_(.+)_$")
# «Σωματική βλάβη 1. Όποιος …» — a heading printed on the same line as paragraph 1
INLINE_HEADING = re.compile(r"^([^.\n]{3,140}?)\s+1\.\s+\S")
# OCR keeps the line-break hyphen: «οινο-- πνεύματος»
OCR_HYPHEN = re.compile(r"([α-ωάέήίόύώϊϋΐΰ])-{1,2}\s+([α-ωάέήίόύώϊϋΐΰ])")
LATIN_SUFFIX = {"a": "Α", "b": "Β", "c": "Γ", "d": "Δ", "e": "Ε", "f": "ΣΤ", "g": "Ζ", "h": "Η"}


def iso_to_greek(d: str) -> str:
    """2019-11-18 → 18.11.2019"""
    y, m, dd = d.split("-")
    return f"{int(dd)}.{int(m)}.{y}"


def version_of(source: dict) -> dict:
    """Which text this is, in words a citizen can read. The record keeps most codes as first published, so an answer
    must be able to say «this is the 1984 text» instead of passing it off as today's law."""
    history = (source.get("history") or "").lower()
    ocr = str(source.get("text") or "").lower().startswith("ocr")      # not «PDF text layer (no OCR)»
    fek = source.get("fek")
    if source.get("consolidated_to"):
        upto = (source["consolidated_to"].replace("as in force until", "όπως ισχύει έως")
                .replace("updated up to", "ενημερωμένο έως"))
        status, as_of = "consolidated", None
        note = f"Κωδικοποιημένο κείμενο, {upto}."
    elif source.get("text_as_of"):
        status, as_of = "amended_until", source["text_as_of"]
        note = (f"Κείμενο με τις τροποποιήσεις έως {iso_to_greek(as_of)}. "
                "Μεταγενέστερες τροποποιήσεις δεν έχουν ενσωματωθεί.")
    elif "enacted" in history:
        status, as_of = "as_enacted", source.get("in_force") or source.get("published")
        where = f" (ΦΕΚ Α΄ {fek})" if fek else (f" ({iso_to_greek(as_of)})" if as_of else "")
        note = f"Κείμενο όπως αρχικά δημοσιεύθηκε{where}. Μεταγενέστερες τροποποιήσεις δεν έχουν ενσωματωθεί."
    else:
        status, as_of, note = "unknown", None, "Η έκδοση του κειμένου δεν είναι γνωστή."
    if ocr:
        note += " Το κείμενο προέρχεται από οπτική αναγνώριση (OCR) του ΦΕΚ και μπορεί να έχει τυπογραφικά λάθη."
    return {"status": status, "as_of": as_of, "ocr": ocr, "note": note}


def article_number(title: str, stem: str) -> tuple[str | None, str]:
    """(number, heading printed on the title line): «Άρθρο 5Α» → 5Α · «Άρθρα 3 - 4» → 3-4 · «Άρθρο 52. - Δικαίωμα
    προσφυγής» → 52 with its heading. Falls back to the file name (art-005-a → 5Α)."""
    m = TITLE.match(title)
    if m and (n := TITLE_NUMBER.match(m.group(1))):
        number = n.group(1) + (n.group(2) or "").upper()
        if n.group(3):
            number += f"-{n.group(3)}{(n.group(4) or '').upper()}"
        return number, (n.group(5) or "").strip()
    m = re.match(r"^0*(\d+)-0*(\d+)$", stem)
    if m:
        return f"{m.group(1)}-{m.group(2)}", ""
    m = re.match(r"^0*(\d+)(?:-?([a-zα-ω]+))?$", stem)
    if m:
        suffix = m.group(2) or ""
        return m.group(1) + (LATIN_SUFFIX.get(suffix, suffix.upper()) if suffix else ""), ""
    return None, ""


def parse_article(raw: str, stem: str, ocr: bool, sentence_heading: bool = False) -> dict | None:
    """One article file → number, heading, chapter, text. None for a file that holds no article text."""
    lines = COMMENT.sub("", raw).strip().splitlines()
    if not lines:
        return None
    number, heading = article_number(lines[0], stem)
    if number is None:
        return None
    body, chapter = [], ""
    for line in lines[1:]:
        m = CHAPTER.match(line.strip())
        if m and not any(b.strip() for b in body):
            chapter = re.sub(r"\s*·\s*", " ", m.group(1)).strip()
            continue
        body.append(line.rstrip())
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(body).strip())
    if ocr:
        text = OCR_HYPHEN.sub(r"\1\2", text)
    if not text:
        return None
    first, _, rest = text.partition("\n")
    if heading:
        # the heading began on the title line and may run on into the body, up to paragraph 1
        m = re.match(r"^([^\n]{1,160}?)\.?\s+(?=1\.\s)", text)
        if m and not re.match(r"^\d", m.group(1)):
            heading, text = f"{heading} {m.group(1).strip()}", text[m.end():].strip()
    elif (rest.strip() and len(first) <= 140 and not first.rstrip().endswith((".", ":", ";", ","))
          and not re.match(r"^(\d+\.|[α-ω]\))", first)):
        heading, text = first.strip(), rest.strip()          # a heading on a line of its own
    elif (m := INLINE_HEADING.match(text)) and not re.match(r"^\d", m.group(1)):
        heading, text = m.group(1).strip(), text[len(m.group(1)):].strip()
    elif sentence_heading and (m := SENTENCE_HEADING.match(text)) and len(m.group(1).split()) <= 7:
        heading, text = m.group(1).strip(), text[m.end():].strip()
    if not text:
        text, heading = heading, ""
    return {"number": number, "heading": heading.rstrip("."), "chapter": chapter, "text": split_paragraphs(text)}


# «παρ. 2. Η …»: the number belongs to the abbreviation before it, it does not open a paragraph
ABBREVIATION_BEFORE = re.compile(r"(?:παρ|παραγρ|άρθρ|αρθρ|αρ|περ|εδ|στοιχ|αριθ|αριθμ|υπ|ν|π\.δ)\.$", re.I)


def split_paragraphs(text: str) -> str:
    """«1. … 2. … 3. …» printed as one line → one paragraph per number. Only numbers that follow in sequence after a
    sentence end count, so «του άρθρου 308 παρ. 2. Η …» inside a paragraph is not taken for a new one."""
    if "\n" in text or not text.startswith("1. "):
        return text
    n, at = 2, 0
    while True:
        pattern = re.compile(rf"(?<=[.·:;»)]) {n}\. (?=[Α-ΩΆΈΉΊΌΎΏ«(])")
        m = pattern.search(text, at)
        while m and ABBREVIATION_BEFORE.search(text[: m.start()]):
            m = pattern.search(text, m.end())
        if not m:
            return text
        text = text[: m.start()] + "\n\n" + text[m.start() + 1:]
        n, at = n + 1, m.start() + 2


def number_key(number: str) -> tuple:
    m = re.match(r"^(\d+)(.*)$", number)
    return (int(m.group(1)), m.group(2)) if m else (10**9, number)


def build(source: Path, out: Path) -> dict:
    articles, codes = [], []
    for folder, (abbr, name) in CODES.items():
        root = source / folder if folder == "constitution" else source / "codes" / folder
        if not root.is_dir():
            print(f"  ! {folder}: not in the checkout, skipped", file=sys.stderr)
            continue
        meta_file = root / "source.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        if folder == "constitution":
            version = {"status": "consolidated", "as_of": None, "ocr": False,
                       "note": "Ισχύον κείμενο του Συντάγματος (με την αναθεώρηση του 2019)."}
        else:
            version = version_of(meta)
        counts = {abbr: 0}
        for f in sorted(root.iterdir()):                      # top level only: sub-folders hold older versions
            m = ARTICLE_FILE.match(f.name)
            if not m or not f.is_file():
                continue
            intro = bool(m.group(1))
            art_abbr, art_name = (INTRO_LAWS[folder] if intro and folder in INTRO_LAWS else (abbr, name))
            parsed = parse_article(f.read_text(encoding="utf-8"), m.group(2), version["ocr"],
                                   sentence_heading=folder in SENTENCE_HEADING_CODES)
            if not parsed:
                continue
            articles.append({"id": f"{art_abbr}-{parsed['number']}", "code": art_abbr, "folder": folder,
                             "label": f"{art_abbr} {parsed['number']}", **parsed})
            counts[art_abbr] = counts.get(art_abbr, 0) + 1
        act = meta.get("act") or meta.get("base_act") or ""
        if not act and (m := re.match(r"^nomos-(\d{4})-(\d+)$", meta.get("act_id") or "")):
            act = f"ν. {m.group(2)}/{m.group(1)}"
        base = {"folder": folder, "act": act, "fek": meta.get("fek") or "", "source": meta.get("source") or "",
                "in_force": meta.get("in_force") or "", **version}
        codes.append({"abbr": abbr, "name": name, "articles": counts[abbr], **base})
        if folder in INTRO_LAWS and counts.get(INTRO_LAWS[folder][0]):
            ia, iname = INTRO_LAWS[folder]
            codes.append({"abbr": ia, "name": iname, "articles": counts[ia], **base})

    seen: dict[str, int] = {}
    for a in articles:                                          # an id must be unique: the website links by it
        seen[a["id"]] = seen.get(a["id"], 0) + 1
    dupes = [k for k, v in seen.items() if v > 1]
    if dupes:
        raise SystemExit(f"duplicate article ids: {dupes[:10]}")
    order = {abbr: i for i, abbr in enumerate(c["abbr"] for c in codes)}
    articles.sort(key=lambda a: (order[a["code"]], number_key(a["number"])))

    out.mkdir(parents=True, exist_ok=True)
    with (out / "articles.jsonl").open("w", encoding="utf-8") as fh:
        for a in articles:
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")
    commit = ""
    try:
        commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    info = {"repository": REPO, "commit": commit, "built": date.today().isoformat(), "articles": len(articles),
            "codes": codes}
    (out / "codes.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return info


def sparse_clone(dest: Path) -> None:
    def git(*args: str, cwd: Path | None = None) -> None:
        subprocess.run(["git", *args], cwd=cwd, check=True)
    git("clone", "--depth", "1", "--filter=tree:0", "--sparse", "--no-checkout", REPO, str(dest))
    git("sparse-checkout", "set", "--cone", "codes", "constitution", cwd=dest)
    git("checkout", cwd=dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=Path, help="an existing checkout of lexifina/greece-law (default: sparse-clone it)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.source:
        info = build(args.source, args.out)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "greece-law"
            print(f"cloning {REPO} (codes/ and constitution/ only) …", file=sys.stderr)
            sparse_clone(dest)
            info = build(dest, args.out)
    print(f"{info['articles']} articles in {len(info['codes'])} codes → {args.out}")
    for c in info["codes"]:
        print(f"  {c['abbr']:<10} {c['articles']:>5}  {c['status']:<14} {c['name']}")


if __name__ == "__main__":
    main()
