#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict.py
==========
Score new businesses with the trained, calibrated best model.

Usage:
    python predict.py --input new_businesses.csv --output scored_new.csv
    python predict.py --input new_businesses.csv          # -> new_businesses_scored.csv

The input CSV must have the same *raw* columns as the training data
(name, category, governorate, rating, reviews, website, email, ...). Missing
feature columns are added as NaN and imputed by the pipeline; extra columns are
ignored. Output adds:
    prob_client       - P(client) in [0, 1] from the calibrated model
    predicted_client  - 1 if prob_client >= 0.5 else 0
"""
import argparse
import os
import sys
import pandas as pd

# Reuse the exact preprocessing/scoring logic from the training module.
from supervised_models import (
    load_bundle, predict_proba_new, engineer_presence_flags, THRESHOLD,
)


def main():
    ap = argparse.ArgumentParser(description="Score new businesses for Ooredoo B2B lead scoring")
    ap.add_argument("--input", required=True, help="CSV of new businesses (raw columns)")
    ap.add_argument("--output", default=None, help="output CSV path")
    ap.add_argument("--model", default="best_model.joblib", help="path to saved model bundle")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"[FATAL] Input not found: {args.input}")
    if not os.path.exists(args.model):
        sys.exit(f"[FATAL] Model bundle not found: {args.model} "
                 f"(run supervised_models.py first).")

    out_path = args.output or (os.path.splitext(args.input)[0] + "_scored.csv")

    bundle = load_bundle(args.model)
    df = pd.read_csv(args.input)
    df = engineer_presence_flags(df)          # recreate has_website/has_email/has_phone

    proba = predict_proba_new(df, bundle=bundle)
    df["prob_client"] = proba
    df["predicted_client"] = (proba >= THRESHOLD).astype(int)
    df.to_csv(out_path, index=False)

    n_pos = int(df["predicted_client"].sum())
    print(f"[predict] model      : {bundle.get('best_model_name','?')} "
          f"({bundle.get('calibration','?')}-calibrated)")
    print(f"[predict] scored rows: {len(df):,}")
    print(f"[predict] predicted client (prob>=0.5): {n_pos:,} ({n_pos/len(df):.1%})")
    print(f"[predict] wrote      : {out_path}")


if __name__ == "__main__":
    main()
