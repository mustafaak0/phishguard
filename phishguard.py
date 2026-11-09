#!/usr/bin/env python3
"""
PhishGuard — ML-powered phishing URL detector (educational/defensive tool).

Features are purely lexical (no network calls), so the classifier is fast,
deterministic and privacy-friendly. Trains on data/dataset.csv (synthetic
demo data — swap with PhishTank / Tranco feeds for real-world use).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

DATA = Path(__file__).parent / "data" / "dataset.csv"

SUSPICIOUS_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "buzz", "click", "loan", "work", "fit"}
PHISH_KEYWORDS = (
    "login", "verify", "secure", "account", "update", "confirm",
    "paypal", "bank", "invoice", "billing", "password", "signin",
    "webscr", "recover", "unlock", "suspend", "bonus", "free", "gift",
)


def shannon_entropy(text: str) -> float:
    """Random-looking (DGA) domains have higher entropy."""
    if not text:
        return 0.0
    freq = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def extract_features(url: str) -> list[float]:
    """16 lexical/structural features for a URL."""
    url = url.strip()
    parsed = urlparse(url if "//" in url else f"http://{url}")
    host = parsed.hostname or ""
    host_no_tld = host.rsplit(".", 1)[0] if "." in host else host
    tld = host.rsplit(".", 1)[-1] if "." in host else ""

    digits = sum(ch.isdigit() for ch in url)
    subdomains = host_no_tld.split(".") if "." in host_no_tld else [host_no_tld]
    path = parsed.path or ""
    query = parsed.query or ""

    features = [
        float(len(url)),                                        # 1 length
        digits / max(len(url), 1),                              # 2 digit ratio
        float("@" in url),                                      # 3 @ decoy
        float(host.count("-")),                                 # 4 hyphens in host
        float(len([s for s in subdomains if s])),               # 5 subdomain depth
        float(bool(re.fullmatch(r"[\d.]+", host))),             # 6 IP literal
        float(tld.lower() in SUSPICIOUS_TLDS),                  # 7 suspicious TLD
        shannon_entropy(host),                                  # 8 entropy
        float(not parsed.scheme or parsed.scheme == "http"),    # 9 no TLS
        float(host.count(".") >= 3),                            # 10 excessive dots
        float(len(path) > 40),                                  # 11 bloated path
        float(any(k in path.lower() for k in PHISH_KEYWORDS)),  # 12 keywords in path
        float(any(k in query.lower() for k in PHISH_KEYWORDS)), # 13 keywords in query
        float(any(k in host.lower() for k in PHISH_KEYWORDS)),  # 14 brand in host
        float(bool(re.search(r"\d+[a-z]|\w{20,}", host, re.I))),# 15 mangled host
        float(url.lower().count("http") > 1),                   # 16 embedded http
    ]
    return features


def load_dataset(path: Path = DATA) -> tuple[list[list[float]], list[int]]:
    X, y = [], []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            X.append(extract_features(row["url"]))
            y.append(int(row["label"]))
    return X, y


def train_report() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    X, y = load_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_tr, y_tr)

    print(f"[*] Dataset: {len(X)} URLs ({sum(y)} phishing / {len(y)-sum(y)} legitimate)")
    print(f"[*] Train/test split: {len(X_tr)}/{len(X_te)}")
    print("\n=== Holdout evaluation ===")
    print(classification_report(y_te, model.predict(X_te), target_names=["legit", "phishing"]))


def predict(urls: list[str]) -> None:
    from sklearn.linear_model import LogisticRegression

    X, y = load_dataset()
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X, y)

    for url in urls:
        feats = extract_features(url)
        proba = model.predict_proba([feats])[0, 1]
        verdict = "⚠ PHISHING" if proba >= 0.5 else "LEGITIMATE"
        print(f"\n[*] {url}")
        print(f"    → {verdict} (confidence {max(proba, 1-proba)*100:.1f}%)  risk={proba:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description="PhishGuard — ML phishing URL detector")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="train + evaluate on the demo dataset")
    pred = sub.add_parser("predict", help="classify URLs")
    pred.add_argument("urls", nargs="+", help="URL(s) to classify")
    args = p.parse_args()

    try:
        if args.cmd == "train":
            train_report()
        else:
            predict(args.urls)
    except FileNotFoundError:
        sys.exit(f"[!] Dataset not found: {DATA}")
    except ImportError:
        sys.exit("[!] Missing deps — run: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
