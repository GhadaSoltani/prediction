#!/usr/bin/env python3
"""
entity_linkage_v3.py -- Step 1 of the pipeline, v3.

Extends v2 with multiple matching axes, in order of reliability:

  1. PHONE MATCH      -- exact after normalization (e.g. "+216 93 472 427"
                          == "0093472427" == "93472427"). Highest precision.
  2. WEBSITE MATCH    -- exact domain match after stripping protocol/www.
  3. RNE MATCH        -- exact match on Tunisian business registry code.
  4. NAME MATCH (v2)  -- two-pass exact-then-fuzzy, with token_set_ratio.

If ANY strong signal (phone/website/RNE) matches, we mark it high confidence
regardless of the name fuzzy score -- these signals are far more reliable
than name similarity in Tunisia's B2B space where names have many variants.

CONFIG NOTE:
  The current Ooredoo purchase files do NOT have phone, website, or RNE-in-
  Maps-form columns. This script degrades gracefully -- if a column doesn't
  exist, that axis is skipped and we fall back to name matching. When you can
  extract phones/websites from Ooredoo's CRM, just set the column names in
  the CONFIG block below and re-run.

Outputs (same shape as v2 for downstream compatibility):
  - maps_labeled_v3.csv : Maps + is_customer + match_confidence + match_reason
  - match_review_v3.csv : per-purchase best-match log with which axis fired
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
    def unidecode(s):
        return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

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

# Name columns (required)
PURCHASE_NAME_COL = "account_contact_name"
MAPS_NAME_COL     = "name"

# Optional strong-signal columns. Set to None if the column doesn't exist
# in your file, and that axis will be skipped.
PURCHASE_PHONE_COL   = None    # e.g. "phone" or "gsm" -- set when you have it
PURCHASE_WEBSITE_COL = None    # e.g. "website"
PURCHASE_RNE_COL     = "RNE"   # exists in purchases_1.xlsx (may be empty)
MAPS_PHONE_COL       = "phone"
MAPS_WEBSITE_COL     = "website"
MAPS_RNE_COL         = None    # Maps doesn't have RNE natively; set if enriched

# Fuzzy name matching config (unchanged from v2)
HIGH_CONF_THRESHOLD = 90
AMBIGUOUS_MARGIN    = 3
BORDERLINE_LOW      = 78
BORDERLINE_HIGH     = 90
WEAK_LOW            = 65
LABEL_THRESHOLD_BORDERLINE = 82

SCORER = fuzz.token_set_ratio

OUT_LABELED = "maps_labeled_v3.csv"
OUT_REVIEW  = "match_review_v3.csv"
# ========================================================================

LEGAL_SUFFIXES = {
    "sarl","suarl","sarlu","sa","snc","scs","sca","spa","suar",
    "ste","societe","soc","ets","etablissement","etablissements",
    "entreprise","entreprises","groupe","group","cie","co","company",
    "llc","ltd","inc","gmbh",
}
_punct = re.compile(r"[^\w\s]", flags=re.UNICODE)
_ws    = re.compile(r"\s+")


# ============================================================
# NORMALIZERS -- one per matching axis
# ============================================================
def normalize_name(name):
    if not isinstance(name, str): return "", False
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


def normalize_phone(phone):
    """
    Turn any Tunisian phone into a canonical 8-digit local form.
      "+216 93 472 427"  -> "93472427"
      "0093472427"       -> "93472427"
      "216-93-472-427"   -> "93472427"
      "93 472 427"       -> "93472427"
    Anything not 8-9 digits after cleanup is treated as invalid (empty).
    """
    if not isinstance(phone, str): return ""
    digits = re.sub(r"\D", "", phone)
    if digits.startswith("00216"):
        digits = digits[5:]
    elif digits.startswith("216") and len(digits) >= 11:
        digits = digits[3:]
    if len(digits) == 8 and digits[0] in "234579":  # valid TN mobile/landline
        return digits
    return ""


def normalize_website(url):
    """
    Extract the registrable domain.
      "http://www.tunisair.com/"          -> "tunisair.com"
      "https://facebook.com/somebrand"    -> "facebook.com"
      "somebrand.tn"                      -> "somebrand.tn"
    Social-media pages are still returned but the caller can decide to skip.
    """
    if not isinstance(url, str) or not url.strip():
        return ""
    u = url.strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    u = u.split("/")[0].split("?")[0].strip()
    return u


def normalize_rne(rne):
    """RNE codes are alphanumeric; strip whitespace and lowercase."""
    if not isinstance(rne, str): return ""
    return re.sub(r"\s+", "", rne).lower()


# ============================================================
# LOADING
# ============================================================
def read_purchase_excel(path, sheet):
    for skip in range(4):
        try:
            df = pd.read_excel(path, sheet_name=sheet, dtype=str,
                               engine="openpyxl", skiprows=skip)
            if PURCHASE_NAME_COL in df.columns:
                return df
        except FileNotFoundError:
            sys.exit(f"[ERROR] File not found: {path}")
    sys.exit(f"[ERROR] '{PURCHASE_NAME_COL}' not found in {path}")


def load_purchases():
    print("Reading purchase files...")
    df1 = read_purchase_excel(PURCHASE_FILE_1, SHEET_1)
    df2 = read_purchase_excel(PURCHASE_FILE_2, SHEET_2)

    def _grab(df, source):
        cols = [PURCHASE_NAME_COL]
        for c in [PURCHASE_PHONE_COL, PURCHASE_WEBSITE_COL, PURCHASE_RNE_COL]:
            if c and c in df.columns:
                cols.append(c)
        sub = df[cols].copy()
        sub["source_file"] = source
        return sub

    m = pd.concat([_grab(df1, "file_1"), _grab(df2, "file_2")],
                  ignore_index=True, sort=False)
    m = m.replace({"#N/A": pd.NA, "N/A": pd.NA, "": pd.NA})
    m = m.dropna(subset=[PURCHASE_NAME_COL])
    m = m[m[PURCHASE_NAME_COL].str.strip() != ""]
    m = m.drop_duplicates(subset=[PURCHASE_NAME_COL], keep="first").reset_index(drop=True)
    print(f"  -> {len(m):,} unique purchase businesses\n")
    return m


# ============================================================
# BUCKET LOGIC
# ============================================================
def bucket_of(score, is_strong_signal, is_ambiguous):
    if is_strong_signal:                            return "strong_signal"
    if is_ambiguous:                                return "ambiguous"
    if score >= HIGH_CONF_THRESHOLD:                return "high_confidence"
    if score >= BORDERLINE_LOW:                     return "borderline"
    if score >= WEAK_LOW:                           return "weak"
    return "no_match"


def main():
    purch = load_purchases()
    maps = pd.read_csv(MAPS_FILE)

    # ---- Which axes are actually available? ----
    axes_used = ["name"]
    phone_available   = (PURCHASE_PHONE_COL   in purch.columns and MAPS_PHONE_COL   in maps.columns)
    website_available = (PURCHASE_WEBSITE_COL in purch.columns and MAPS_WEBSITE_COL in maps.columns)
    rne_available     = (PURCHASE_RNE_COL     in purch.columns and MAPS_RNE_COL     in maps.columns)
    if phone_available:   axes_used.append("phone")
    if website_available: axes_used.append("website")
    if rne_available:     axes_used.append("rne")
    print(f"Matching axes enabled: {axes_used}\n")

    if not phone_available:
        print("  [note] Phone matching skipped: no phone column on purchase side.")
        print(f"         To enable: add a phone column and set PURCHASE_PHONE_COL.\n")
    if not website_available:
        print("  [note] Website matching skipped: no website column on purchase side.\n")
    if not rne_available:
        print("  [note] RNE matching skipped: Maps side has no RNE (would need enrichment).\n")

    # ---- Build normalized lookup tables on the Maps side ----
    maps_norm_name = maps[MAPS_NAME_COL].map(lambda x: normalize_name(x)[0])
    exact_name_lookup = {}
    for i, n in enumerate(maps_norm_name):
        if n: exact_name_lookup.setdefault(n, []).append(i)

    phone_lookup = {}
    if phone_available:
        for i, p in enumerate(maps[MAPS_PHONE_COL]):
            pn = normalize_phone(p)
            if pn: phone_lookup.setdefault(pn, []).append(i)

    website_lookup = {}
    if website_available:
        SOCIAL_DOMAINS = {"facebook.com", "instagram.com", "linkedin.com",
                          "twitter.com", "youtube.com", "tiktok.com"}
        for i, w in enumerate(maps[MAPS_WEBSITE_COL]):
            wn = normalize_website(w)
            # Skip social pages -- one Facebook page can be a business's site,
            # but a purchase-side Facebook URL matching a Maps-side different
            # Facebook page would create false positives.
            if wn and wn not in SOCIAL_DOMAINS:
                website_lookup.setdefault(wn, []).append(i)

    rne_lookup = {}
    if rne_available:
        for i, r in enumerate(maps[MAPS_RNE_COL]):
            rn = normalize_rne(r)
            if rn: rne_lookup.setdefault(rn, []).append(i)

    # Fuzzy candidate list (only names)
    cand_idx   = [i for i, n in enumerate(maps_norm_name) if n]
    cand_names = [maps_norm_name.iloc[i] for i in cand_idx]

    # ---- Match each purchase ----
    print(f"Matching {len(purch):,} purchases against {len(maps):,} Maps rows...")
    rows = []
    matched_map_rows = {}   # map_row -> (best_score, reason)

    for _, prow in tqdm(purch.iterrows(), total=len(purch), desc="matching"):
        purchase_name = prow[PURCHASE_NAME_COL]
        purchase_norm, _had_legal = normalize_name(purchase_name)

        record = {"purchase_name": purchase_name, "purchase_norm": purchase_norm,
                  "map_name": None, "score": 0.0, "score2": 0.0,
                  "map_row_index": None, "bucket": "no_match",
                  "match_reason": ""}

        # ---- AXIS 1: Phone ----
        if phone_available and pd.notna(prow.get(PURCHASE_PHONE_COL)):
            pn = normalize_phone(prow[PURCHASE_PHONE_COL])
            if pn and pn in phone_lookup:
                map_row = phone_lookup[pn][0]
                record.update({
                    "map_name": maps.at[map_row, MAPS_NAME_COL],
                    "score": 100.0, "map_row_index": map_row,
                    "bucket": "strong_signal", "match_reason": "phone"
                })
                if map_row not in matched_map_rows or matched_map_rows[map_row][0] < 100:
                    matched_map_rows[map_row] = (100, "phone")
                rows.append(record); continue

        # ---- AXIS 2: Website ----
        if website_available and pd.notna(prow.get(PURCHASE_WEBSITE_COL)):
            wn = normalize_website(prow[PURCHASE_WEBSITE_COL])
            if wn and wn in website_lookup:
                map_row = website_lookup[wn][0]
                record.update({
                    "map_name": maps.at[map_row, MAPS_NAME_COL],
                    "score": 100.0, "map_row_index": map_row,
                    "bucket": "strong_signal", "match_reason": "website"
                })
                if map_row not in matched_map_rows or matched_map_rows[map_row][0] < 100:
                    matched_map_rows[map_row] = (100, "website")
                rows.append(record); continue

        # ---- AXIS 3: RNE ----
        if rne_available and pd.notna(prow.get(PURCHASE_RNE_COL)):
            rn = normalize_rne(prow[PURCHASE_RNE_COL])
            if rn and rn in rne_lookup:
                map_row = rne_lookup[rn][0]
                record.update({
                    "map_name": maps.at[map_row, MAPS_NAME_COL],
                    "score": 100.0, "map_row_index": map_row,
                    "bucket": "strong_signal", "match_reason": "rne"
                })
                if map_row not in matched_map_rows or matched_map_rows[map_row][0] < 100:
                    matched_map_rows[map_row] = (100, "rne")
                rows.append(record); continue

        # ---- AXIS 4: Name (exact + fuzzy) ----
        if not purchase_norm:
            rows.append(record); continue

        # Exact name after normalization
        if purchase_norm in exact_name_lookup:
            map_row = exact_name_lookup[purchase_norm][0]
            record.update({
                "map_name": maps.at[map_row, MAPS_NAME_COL], "score": 100.0,
                "map_row_index": map_row, "bucket": "exact",
                "match_reason": "exact_name"
            })
            if map_row not in matched_map_rows or matched_map_rows[map_row][0] < 100:
                matched_map_rows[map_row] = (100, "exact_name")
            rows.append(record); continue

        # Fuzzy name (top 2 for ambiguity)
        top2 = process.extract(purchase_norm, cand_names, scorer=SCORER, limit=2)
        if not top2:
            rows.append(record); continue
        (m1, s1, p1) = top2[0]
        (m2, s2, p2) = top2[1] if len(top2) > 1 else (None, 0.0, None)
        map_row = cand_idx[p1]
        is_amb = (s2 > 0 and (s1 - s2) < AMBIGUOUS_MARGIN and s1 < HIGH_CONF_THRESHOLD + 5)
        b = bucket_of(s1, False, is_amb)
        record.update({
            "map_name": maps.at[map_row, MAPS_NAME_COL], "score": s1, "score2": s2,
            "map_row_index": map_row, "bucket": b, "match_reason": f"fuzzy_name({b})"
        })
        rows.append(record)

        if b == "high_confidence" or (b == "borderline" and s1 >= LABEL_THRESHOLD_BORDERLINE):
            existing = matched_map_rows.get(map_row, (0, ""))
            if s1 > existing[0]:
                matched_map_rows[map_row] = (s1, f"fuzzy_{b}")

    # ---- Write labels ----
    maps["is_customer"] = 0
    maps["match_confidence"] = ""
    maps["match_reason"]     = ""
    for r, (s, reason) in matched_map_rows.items():
        maps.at[r, "is_customer"] = 1
        maps.at[r, "match_reason"] = reason
        if reason in ("phone", "website", "rne", "exact_name"):
            maps.at[r, "match_confidence"] = "exact"
        elif s >= HIGH_CONF_THRESHOLD:
            maps.at[r, "match_confidence"] = "high"
        else:
            maps.at[r, "match_confidence"] = "borderline"
    maps.to_csv(OUT_LABELED, index=False)

    review = pd.DataFrame(rows).sort_values("score", ascending=False)
    review.to_csv(OUT_REVIEW, index=False)

    # ---- Summary ----
    n_pos = int(maps["is_customer"].sum())
    print("\n================ SUMMARY ================")
    print(review["bucket"].value_counts().to_string())
    print(f"\nUnique Maps businesses labeled 1: {n_pos:,}")
    print(f"  by match_reason:")
    print(maps[maps["is_customer"]==1]["match_reason"].value_counts().to_string())
    print(f"\nWrote: {OUT_LABELED}, {OUT_REVIEW}")


if __name__ == "__main__":
    main()
