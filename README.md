# 🛡️ PhishGuard

> Machine-learning powered phishing URL detector — because "hmm, this link looks fishy" should be a metric, not a vibe.

[![Python](https://img.shields.io/badge/Python-3.9%2B-0D1117?style=for-the-badge&logo=python&logoColor=3776AB)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-00E676?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-0D1117?style=for-the-badge&labelColor=00E676)](LICENSE)
[![Author](https://img.shields.io/badge/Author-mustafaak0-0D1117?style=for-the-badge&logo=github&logoColor=00E676)](https://github.com/mustafaak0)

## 🧠 How It Works

PhishGuard extracts **16 lexical & structural features** from a URL and classifies it with logistic regression:

| # | Feature | Intuition |
|---|---------|-----------|
| 1 | URL length | Phishing URLs are often bloated |
| 2 | Digit ratio | Fake domains stuff digits (`paypa1-0nline`) |
| 3 | `@` symbol | Everything before `@` is a decoy |
| 4 | Hyphen count | `secure-login-verify-account.com` |
| 5 | Subdomain count | Long chains hide the real host |
| 6 | IP-literal host | `http://185.234.219.11/...` |
| 7 | Suspicious TLD | `.tk .xyz .top .buzz ...` |
| 8 | Shannon entropy | Random-looking DGA domains |
| 9–16 | Brand/keyword hits | `login`, `verify`, `secure`, `paypal`, `bank`... |

## 🚀 Usage

```bash
pip install -r requirements.txt

# Train + evaluate
python phishguard.py train

# Classify URLs
python phishguard.py predict "https://github.com/mustafaak0" "http://paypa1-secure-verify.tk/login"
```

```
[*] https://github.com/mustafaak0
    → LEGITIMATE (confidence 97.4%)  risk=0.026

[*] http://paypa1-secure-verify.tk/login
    → ⚠ PHISHING (confidence 93.1%)  risk=0.931
```

## ⚠️ Dataset Note

The bundled `data/dataset.csv` is a **small synthetic demo set** for education.
For real-world use, feed it public feeds:
- [PhishTank](https://phishtank.org) · [OpenPhish](https://openphish.com) — phishing
- [Tranco](https://tranco-list.eu) / [Cisco Umbrella Top 1M](https://s3-us-west-1.amazonaws.com/umbrella-static/index.html) — legitimate

## 🔭 Roadmap

- [ ] RandomForest / XGBoost comparison
- [ ] WHOIS + domain-age features
- [ ] TLS certificate sanity features
- [ ] Batch mode (CSV of URLs)
- [ ] REST API wrapper (FastAPI)

## ⚖️ Disclaimer

PhishGuard is a **defensive** tool for research, education and SOC triage support.
Use responsibly.

---

<div align="center">

*Defense in depth starts with a single suspicious link.*

</div>
