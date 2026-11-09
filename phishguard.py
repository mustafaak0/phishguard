#!/usr/bin/env python3
"""
PhishGuard — ML-powered phishing URL detector (educational/defensive tool).

Features are purely lexical (no network calls), so the classifier is fast,
deterministic and privacy-friendly. Trains on data/dataset.csv (synthetic
demo data — swap with PhishTank / Tranco feeds for real-world use).

Backends:
  - scikit-learn LogisticRegression when available
  - built-in pure-NumPy logistic regression otherwise (e.g. restrictive
    Windows policies such as Smart App Control blocking SciPy DLLs)
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


# ----------------------------------------------------------------------------
# NumPy fallback backend (used when scikit-learn cannot be imported)
# ----------------------------------------------------------------------------

class NumPyLogisticRegression:
    """Batch gradient-descent logistic regression with feature standardization."""

    def __init__(self, lr: float = 0.15, epochs: int = 4000, class_weight=None):
        self.lr = lr
        self.epochs = epochs
        self.class_weight = class_weight

    def fit(self, X, y):
        import numpy as np
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        Z = np.hstack([(X - self.mu_) / self.sd_, np.ones((len(X), 1))])  # bias term

        if self.class_weight == "balanced":
            n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
            w_pos = len(y) / (2 * n_pos) if n_pos else 1.0
            w_neg = len(y) / (2 * n_neg) if n_neg else 1.0
            sw = np.where(y == 1, w_pos, w_neg)
        else:
            sw = np.ones_like(y)
        denom = sw.sum()

        w = np.zeros(Z.shape[1])
        for _ in range(self.epochs):
            p = 1.0 / (1.0 + np.exp(-(Z @ w)))
            grad = Z.T @ (sw * (p - y)) / denom
            w -= self.lr * grad
        self.w_ = w
        return self

    def predict_proba(self, X):
        import numpy as np
        X = np.asarray(X, dtype=float)
        Z = np.hstack([(X - self.mu_) / self.sd_, np.ones((len(X), 1))])
        p1 = 1.0 / (1.0 + np.exp(-(Z @ self.w_)))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_model():
    """Return (model, backend_name) — scikit-learn if importable, else NumPy."""
    try:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, class_weight="balanced"), "scikit-learn"
    except ImportError:
        return NumPyLogisticRegression(class_weight="balanced"), "pure-NumPy fallback"


def split_stratified(X, y, test_size: float = 0.25, seed: int = 42):
    try:
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    except ImportError:
        import numpy as np
        rng = np.random.default_rng(seed)
        y_arr = np.asarray(y)
        idx0, idx1 = np.flatnonzero(y_arr == 0), np.flatnonzero(y_arr == 1)
        rng.shuffle(idx0), rng.shuffle(idx1)
        n0 = max(1, int(round(len(idx0) * test_size)))
        n1 = max(1, int(round(len(idx1) * test_size)))
        te = np.concatenate([idx0[:n0], idx1[:n1]])
        tr = np.concatenate([idx0[n0:], idx1[n1:]])
        X_arr = np.asarray(X, dtype=float)
        return X_arr[tr], X_arr[te], y_arr[tr], y_arr[te]


def plain_report(y_true, y_pred, labels=("legit", "phishing")) -> str:
    """Minimal classification report when sklearn is unavailable."""
    y_true, y_pred = list(y_true), list(y_pred)
    lines = [f"{'':>12}{'precision':>10}{'recall':>9}{'f1-score':>10}{'support':>9}"]
    for cls_id, name in enumerate(labels):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls_id and p == cls_id)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls_id and p == cls_id)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls_id and p != cls_id)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        lines.append(f"{name:>12}{prec:>10.2f}{rec:>9.2f}{f1:>10.2f}{tp + fn:>9}")
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    lines.append(f"\n{'accuracy':>30}{acc:>10.2f}{len(y_true):>9}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------

def train_report() -> None:
    X, y = load_dataset()
    X_tr, X_te, y_tr, y_te = split_stratified(X, y, test_size=0.25, seed=42)

    model, backend = make_model()
    model.fit(X_tr, y_tr)

    print(f"[*] Dataset: {len(X)} URLs ({sum(y)} phishing / {len(y)-sum(y)} legitimate)")
    print(f"[*] Train/test split: {len(X_tr)}/{len(X_te)}")
    print(f"[*] Backend: {backend}")
    print("\n=== Holdout evaluation ===")
    y_pred = model.predict(X_te)
    try:
        from sklearn.metrics import classification_report
        print(classification_report(y_te, y_pred, target_names=["legit", "phishing"]))
    except ImportError:
        print(plain_report(y_te, y_pred))


def predict(urls: list[str]) -> None:
    X, y = load_dataset()
    model, backend = make_model()
    model.fit(X, y)
    print(f"[*] Backend: {backend}\n")

    for url in urls:
        feats = extract_features(url)
        proba = float(model.predict_proba([feats])[0, 1])
        verdict = "⚠ PHISHING" if proba >= 0.5 else "LEGITIMATE"
        print(f"[*] {url}")
        print(f"    → {verdict} (confidence {max(proba, 1-proba)*100:.1f}%)  risk={proba:.3f}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="PhishGuard — ML phishing URL detector")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="train + evaluate on the demo dataset")
    pred = sub.add_parser("predict", help="classify URLs")
    pred.add_argument("urls", nargs="+", help="URL(s) to classify")
    args = p.parse_args()

    try:
        import numpy  # noqa: F401  (hard dependency of the fallback backend)
    except ImportError:
        sys.exit("[!] NumPy missing — run: pip install numpy")

    try:
        if args.cmd == "train":
            train_report()
        else:
            predict(args.urls)
    except FileNotFoundError:
        sys.exit(f"[!] Dataset not found: {DATA}")


if __name__ == "__main__":
    main()
