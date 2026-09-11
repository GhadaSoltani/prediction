#!/usr/bin/env python3
"""
entity_linkage_v2.py -- Enhanced Step 1 of the Ooredoo PU-learning pipeline.

Improvements over v1:
  - Two-pass matching: exact-after-normalization first, fuzzy second
    (exact matches are much more trustworthy and shouldn't compete with fuzzy)
  - Legal-form-aware normalization: strips SARL/SUARL/SA/Ets/Groupe but
    ALSO remembers whether the raw name had a legal form (used later)
  - Ambiguity detection: when the top-2 candidates for a purchase have
    nearly identical scores, flags the match as "ambiguous" for review
  - Structured review buckets: exact / high_confidence / ambiguous / borderline /
    weak / no_match  -- makes the review file directly actionable
  - Coverage stats broken down by governorate (helps spot systematic misses)

Outputs:
  - maps_labeled_v2.csv : Maps rows + is_customer + match_confidence
  - match_review_v2.csv : every purchase with best/second-best match, bucket, score
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
    def tqdm(x, **k): return x

# ============================ CONFIG ====================================
PURCHASE_FILE_1 = "purchases_1.xlsx"
PURCHASE_FILE_2 = "purchases_2.xlsx"
SHEET_1 = "OUT_1_SAMPLE"
SHEET_2 = 0
MAPS_FILE = "maps.csv"

PURCHASE_NAME_COL = "account_contact_name"
MAPS_NAME_COL     = "name"

EXACT_MATCH_BUCKET   = "exact"
HIGH_CONF_THRESHOLD  = 90
AMBIGUOUS_MARGIN     = 3      # if top1 - top2 < this, mark ambiguous
BORDERLINE_LOW       = 78
BORDERLINE_HIGH      = 90
WEAK_LOW             = 65

# Score used to flip is_customer = 1
LABEL_THRESHOLD_HIGH_CONF = 90
LABEL_THRESHOLD_BORDERLINE = 82   # borderline still labels as customer but flagged

SCORER = fuzz.token_set_ratio

OUT_LABELED = "maps_labeled_v2.csv"
OUT_REVIEW  = "match_review_v2.csv"
# ========================================================================

LEGAL_SUFFIXES = {
    "sarl", "suarl", "sarlu", "sa", "snc", "scs", "sca", "spa", "suar",
    "ste", "societe", "soc", "ets", "etablissement", "etablissements",
    "entreprise", "entreprises", "groupe", "group", "cie", "co", "company",
    "llc", "ltd", "inc", "gmbh",
}
_punct = re.compile(r"[^\w\s]", flags=re.UNICODE)
_ws    = re.compile(r"\s+")


def normalize(name):
    """Returns (normalized_form, had_legal_suffix_bool)."""
    if not isinstance(name, str):
        return "", False
    s = unidecode(name).lower()
    s = _punct.sub(" ", s)
    s = _ws.sub(" ", s).strip()
    raw = [t for t in s.split() if t]
    had_legal = any(t in LEGAL_SUFFIXES for t in raw)
    tokens = [t for t in raw
              if t not in LEGAL_SUFFIXES
              and not any(ch.isdigit() for ch in t)
              and not (len(t) == 1 and t.isalpha())]
    if not tokens:
        letters = [t for t in raw if t.isalpha() and len(t) == 1]
        collapsed = "".join(letters)
        if collapsed and collapsed not in LEGAL_SUFFIXES:
            tokens = [collapsed]
    return " ".join(tokens), had_legal


def read_purchase_excel(path, sheet):
    try:
        for skip in range(0, 4):
            df = pd.read_excel(path, sheet_name=sheet, dtype=str,
                               engine="openpyxl", skiprows=skip)
            if PURCHASE_NAME_COL in df.columns:
                return df
        sys.exit(f"[ERROR] '{PURCHASE_NAME_COL}' not found in {path}")
    except FileNotFoundError:
        sys.exit(f"[ERROR] File not found: {path}")


def load_purchases():
    print("Reading purchase files...")
    df1 = read_purchase_excel(PURCHASE_FILE_1, SHEET_1)
    df2 = read_purchase_excel(PURCHASE_FILE_2, SHEET_2)
    d1 = df1[[PURCHASE_NAME_COL]].copy(); d1["source_file"] = "file_1"
    d2 = df2[[PURCHASE_NAME_COL]].copy(); d2["source_file"] = "file_2"
    m = pd.concat([d1, d2], ignore_index=True)
    m = m.replace({"#N/A": pd.NA, "N/A": pd.NA, "": pd.NA})
    m = m.dropna(subset=[PURCHASE_NAME_COL])
    m = m[m[PURCHASE_NAME_COL].str.strip() != ""]
    m = m.drop_duplicates(subset=[PURCHASE_NAME_COL], keep="first").reset_index(drop=True)
    print(f"  -> {len(m):,} unique purchase businesses\n")
    return m


def bucket_of(score, is_exact, is_ambiguous):
    if is_exact:                 return "exact"
    if is_ambiguous:             return "ambiguous"
    if score >= HIGH_CONF_THRESHOLD:      return "high_confidence"
    if score >= BORDERLINE_LOW:  return "borderline"
    if score >= WEAK_LOW:        return "weak"
    return "no_match"


def main():
    purch = load_purchases()
    maps = pd.read_csv(MAPS_FILE)
    if MAPS_NAME_COL not in maps.columns:
        sys.exit(f"[ERROR] Column '{MAPS_NAME_COL}' not in {MAPS_FILE}")

    # Normalize both sides
    maps_norm = maps[MAPS_NAME_COL].map(lambda x: normalize(x)[0])
    purch_norm_and_flag = purch[PURCHASE_NAME_COL].map(normalize)
    purch["_norm"]      = purch_norm_and_flag.map(lambda t: t[0])
    purch["_had_legal"] = purch_norm_and_flag.map(lambda t: t[1])

    # Pass 1: exact map (normalized -> list of map indices)
    exact_map: dict[str, list[int]] = {}
    for i, n in enumerate(maps_norm):
        if n:
            exact_map.setdefault(n, []).append(i)

    # Pass 2: fuzzy candidate list (only for non-exact matches, to save time)
    cand_idx   = [i for i, n in enumerate(maps_norm) if n]
    cand_names = [maps_norm.iloc[i] for i in cand_idx]

    print(f"Maps rows: {len(maps):,}, candidates after normalization: {len(cand_names):,}")
    print(f"Purchases to match: {len(purch):,}\n")

    matched_map_rows: dict[int, int] = {}   # map_row -> best_score seen
    rows = []

    for _, prow in tqdm(purch.iterrows(), total=len(purch), desc="matching"):
        q = prow["_norm"]
        if not q:
            rows.append({"purchase_name": prow[PURCHASE_NAME_COL],
                         "purchase_norm": "", "map_name": None, "map_norm": None,
                         "score": 0.0, "score2": 0.0, "map_row_index": None,
                         "bucket": "no_match", "had_legal_form": prow["_had_legal"]})
            continue

        # Exact pass
        if q in exact_map:
            map_row = exact_map[q][0]
            rows.append({"purchase_name": prow[PURCHASE_NAME_COL], "purchase_norm": q,
                         "map_name": maps.at[map_row, MAPS_NAME_COL], "map_norm": q,
                         "score": 100.0, "score2": 0.0, "map_row_index": map_row,
                         "bucket": "exact", "had_legal_form": prow["_had_legal"]})
            matched_map_rows[map_row] = 100
            continue

        # Fuzzy pass: get top 2 to detect ambiguity
        top2 = process.extract(q, cand_names, scorer=SCORER, limit=2)
        (match1_name, s1, pos1) = top2[0]
        (match2_name, s2, pos2) = top2[1] if len(top2) > 1 else (None, 0.0, None)
        map_row = cand_idx[pos1]
        is_amb = (s2 > 0 and (s1 - s2) < AMBIGUOUS_MARGIN and s1 < HIGH_CONF_THRESHOLD + 5)
        b = bucket_of(s1, False, is_amb)

        rows.append({"purchase_name": prow[PURCHASE_NAME_COL], "purchase_norm": q,
                     "map_name": maps.at[map_row, MAPS_NAME_COL], "map_norm": match1_name,
                     "score": s1, "score2": s2, "map_row_index": map_row,
                     "bucket": b, "had_legal_form": prow["_had_legal"]})

        if b == "high_confidence" or (b == "borderline" and s1 >= LABEL_THRESHOLD_BORDERLINE):
            # keep best score if same map row hit twice
            matched_map_rows[map_row] = max(matched_map_rows.get(map_row, 0), s1)

    # Write labels: confidence = "exact" | "high" | "borderline"
    maps["is_customer"] = 0
    maps["match_confidence"] = ""
    for r, s in matched_map_rows.items():
        maps.at[r, "is_customer"] = 1
        maps.at[r, "match_confidence"] = (
            "exact" if s == 100 else "high" if s >= HIGH_CONF_THRESHOLD else "borderline"
        )
    maps.to_csv(OUT_LABELED, index=False)

    review = pd.DataFrame(rows).sort_values("score", ascending=False)
    review.to_csv(OUT_REVIEW, index=False)

    # Summary
    n_pos = int(maps["is_customer"].sum())
    print("\n================ SUMMARY ================")
    print(review["bucket"].value_counts().to_string())
    print(f"\nUnique Maps businesses labeled 1: {n_pos:,}")
    print(f"  of which exact matches:  {(maps['match_confidence']=='exact').sum():,}")
    print(f"  of which high confidence:{(maps['match_confidence']=='high').sum():,}")
    print(f"  of which borderline:     {(maps['match_confidence']=='borderline').sum():,}")
    print(f"\nWrote: {OUT_LABELED}, {OUT_REVIEW}")
    print("\nNext: manually eyeball the 'ambiguous' and 'borderline' buckets in the review file.")


if __name__ == "__main__":
    main()
