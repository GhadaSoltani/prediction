#!/usr/bin/env python3
"""
entity_linkage.py -- Step 1 of the Ooredoo PU-learning lead-scoring pipeline.

Reads the two Excel purchase exports + the scraped Google Maps CSV directly
(no separate merge step), fuzzy-matches purchase businesses against Maps
businesses, and labels the Maps dataset with is_customer = 1 for the matches.

Outputs:
  - maps_labeled.csv : full Maps data + is_customer column  (your positives)
  - match_review.csv : every purchase's best match + score, for eyeballing

Usage:
    pip install pandas rapidfuzz unidecode tqdm openpyxl
    python entity_linkage.py
"""

from __future__ import annotations
import re
import sys
import pandas as pd
from rapidfuzz import process, fuzz

try:
    from unidecode import unidecode
except ImportError:
    import unicodedata
    def unidecode(s: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFKD", s)
                       if not unicodedata.combining(c))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x

# ============================ CONFIG ====================================
PURCHASE_FILE_1 = "purchases_1.xlsx"   # offers export
PURCHASE_FILE_2 = "purchases_2.xlsx"   # accounts export
SHEET_1 = "OUT_1_SAMPLE" 
SHEET_2 = 0
MAPS_FILE = "maps.csv"                 # ~29k scraped Google Maps businesses

PURCHASE_NAME_COL = "account_contact_name"   # business name in BOTH excel files
MAPS_NAME_COL     = "name"                    # business name in the maps csv

SCORE_THRESHOLD = 85
REVIEW_LOW      = 75
REVIEW_HIGH     = 92

# Which rapidfuzz scorer to use. token_sort_ratio is strict (safe default).
# token_set_ratio is more forgiving when one name is a subset of the other
# (e.g. "COTES" vs "COTES Industries") -- switch to it if recall looks low.
SCORER_NAME = "token_set_ratio"   # "token_sort_ratio" | "token_set_ratio" | "WRatio"

OUT_LABELED = "maps_labeled.csv"
OUT_REVIEW  = "match_review.csv"
# ========================================================================

_SCORERS = {
    "token_sort_ratio": fuzz.token_sort_ratio,
    "token_set_ratio":  fuzz.token_set_ratio,
    "WRatio":           fuzz.WRatio,
}

LEGAL_SUFFIXES = {
    "sarl", "suarl", "sarlu", "sa", "snc", "scs", "sca", "spa", "suar",
    "ste", "societe", "soc",
    "ets", "etablissement", "etablissements", "entreprise", "entreprises",
    "groupe", "group", "cie", "co", "company",
    "llc", "ltd", "inc", "gmbh",
}

_punct = re.compile(r"[^\w\s]", flags=re.UNICODE)
_ws    = re.compile(r"\s+")


def normalize(name) -> str:
    """lowercase, strip accents/punctuation, drop legal-form tokens,
    embedded codes (digit-containing tokens like 1482105VAM000), and
    single-letter debris from dotted abbreviations (S.A.R.L. -> s a r l)."""
    if not isinstance(name, str):
        return ""
    s = unidecode(name).lower()
    s = _punct.sub(" ", s)
    s = _ws.sub(" ", s).strip()
    raw = [t for t in s.split(" ") if t]
    tokens = [t for t in raw
              if t not in LEGAL_SUFFIXES
              and not any(ch.isdigit() for ch in t)       # drop codes/years
              and not (len(t) == 1 and t.isalpha())]       # drop S.A.R.L. debris
    if not tokens:
        # name was all initials (e.g. "M.W.S") -> collapse letters into one token
        letters = [t for t in raw if t.isalpha() and len(t) == 1]
        collapsed = "".join(letters)
        if collapsed and collapsed not in LEGAL_SUFFIXES:
            tokens = [collapsed]
    return " ".join(tokens)


def read_purchase_excel(path: str, sheet) -> pd.DataFrame:
    try:
        for skip in range(0, 4):
            df = pd.read_excel(path, sheet_name=sheet, dtype=str,
                               engine="openpyxl", skiprows=skip)
            if PURCHASE_NAME_COL in df.columns:
                note = "" if skip == 0 else f" (skipped {skip} title row{'s' if skip>1 else ''})"
                print(f"  {path}: {df.shape[0]} rows, {df.shape[1]} cols{note}")
                return df
        preview = pd.read_excel(path, sheet_name=sheet, header=None,
                                nrows=5, engine="openpyxl")
        sys.exit(f"[ERROR] '{PURCHASE_NAME_COL}' not found in {path} "
                 f"within first 4 rows.\nFirst rows seen:\n{preview}")
    except FileNotFoundError:
        sys.exit(f"[ERROR] File not found: {path}")
    except ImportError:
        sys.exit("[ERROR] openpyxl not installed. Run: pip install openpyxl")


def load_purchases() -> pd.DataFrame:
    print("Reading purchase files...")
    df1 = read_purchase_excel(PURCHASE_FILE_1, SHEET_1)
    df2 = read_purchase_excel(PURCHASE_FILE_2, SHEET_2)

    d1 = df1[[PURCHASE_NAME_COL]].copy(); d1["source_file"] = "file_1"
    d2 = df2[[PURCHASE_NAME_COL]].copy(); d2["source_file"] = "file_2"

    merged = pd.concat([d1, d2], ignore_index=True)
    merged = merged.replace({"#N/A": pd.NA, "N/A": pd.NA, "": pd.NA})

    before = len(merged)
    merged = merged.dropna(subset=[PURCHASE_NAME_COL])
    merged = merged[merged[PURCHASE_NAME_COL].str.strip() != ""]
    blank = before - len(merged)

    before = len(merged)
    merged = merged.drop_duplicates(subset=[PURCHASE_NAME_COL], keep="first")
    dupes = before - len(merged)

    print(f"  -> {len(df1)} + {len(df2)} rows; dropped {blank} blank, "
          f"{dupes} duplicate  =>  {len(merged):,} unique businesses\n")
    return merged.reset_index(drop=True)


def load_maps() -> pd.DataFrame:
    try:
        df = pd.read_csv(MAPS_FILE)
    except FileNotFoundError:
        sys.exit(f"[ERROR] Maps file not found: {MAPS_FILE}")
    if MAPS_NAME_COL not in df.columns:
        sys.exit(f"[ERROR] Column '{MAPS_NAME_COL}' not in {MAPS_FILE}.\n"
                 f"        Available: {list(df.columns)}")
    return df


def main() -> None:
    purch = load_purchases()
    maps  = load_maps()
    print(f"Maps businesses  : {len(maps):,}")
    print(f"Purchase records : {len(purch):,}\n")

    maps["_norm"]  = maps[MAPS_NAME_COL].map(normalize)
    purch["_norm"] = purch[PURCHASE_NAME_COL].map(normalize)

    cand_idx   = [i for i, n in enumerate(maps["_norm"]) if n]
    cand_names = [maps["_norm"].iloc[i] for i in cand_idx]
    if not cand_names:
        sys.exit("[ERROR] No usable Maps names after normalization.")

    scorer = _SCORERS[SCORER_NAME]
    print(f"Matching with scorer = {SCORER_NAME}, threshold = {SCORE_THRESHOLD}")

    rows = []
    matched_map_rows: set[int] = set()

    for _, prow in tqdm(purch.iterrows(), total=len(purch), desc="matching"):
        q = prow["_norm"]
        if not q:
            rows.append({"purchase_name": prow[PURCHASE_NAME_COL],
                         "purchase_norm": "", "map_name": None, "map_norm": None,
                         "score": 0.0, "map_row_index": None})
            continue

        match_name, score, pos = process.extractOne(
            q, cand_names, scorer=scorer)
        map_row = cand_idx[pos]

        rows.append({"purchase_name": prow[PURCHASE_NAME_COL], "purchase_norm": q,
                     "map_name": maps.at[map_row, MAPS_NAME_COL], "map_norm": match_name,
                     "score": score, "map_row_index": map_row})
        if score >= SCORE_THRESHOLD:
            matched_map_rows.add(map_row)

    maps["is_customer"] = 0
    if matched_map_rows:
        maps.loc[sorted(matched_map_rows), "is_customer"] = 1
    maps.drop(columns=["_norm"]).to_csv(OUT_LABELED, index=False)

    review = pd.DataFrame(rows)
    review["review_needed"] = review["score"].between(REVIEW_LOW, REVIEW_HIGH)
    review.sort_values("score", ascending=False).to_csv(OUT_REVIEW, index=False)

    n_pos       = int(maps["is_customer"].sum())
    n_matched_p = int((review["score"] >= SCORE_THRESHOLD).sum())
    n_review    = int(review["review_needed"].sum())
    n_nomatch   = int((review["score"] < REVIEW_LOW).sum())

    print("\n================ SUMMARY ================")
    print(f"Purchases matched (score >= {SCORE_THRESHOLD}) : {n_matched_p:,}")
    print(f"Unique Maps businesses labeled 1     : {n_pos:,}")
    print(f"In review band [{REVIEW_LOW}-{REVIEW_HIGH}]           : {n_review:,}  -> eyeball these")
    print(f"Purchases with weak/no match (<{REVIEW_LOW})  : {n_nomatch:,}")
    print(f"\nWrote: {OUT_LABELED}  and  {OUT_REVIEW}")


if __name__ == "__main__":
    main()
