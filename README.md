---
title: Probabilistic Forecasting Engine
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# Probabilistic Forecasting Engine

**Production-grade uncertainty quantification for energy load forecasting.**
Calibrated prediction intervals with provable coverage guarantees — no distributional assumptions.

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-orange)](https://huggingface.co/spaces/foyie/prob-forecasting-engine)
[![GitHub](https://img.shields.io/badge/GitHub-foyie-black?logo=github)](https://github.com/foyie/prob-forecasting-engine)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## Results

> Trained on 10 years of PJM Interconnection energy load data (PJME_MW, hourly).

| Metric | Value | Notes |
|--------|-------|-------|
| **RMSE** | 92.4 MW | Point forecast error |
| **MAE** | 50.1 MW | Mean absolute error |
| **MAPE** | 0.19% | Near-perfect point accuracy |
| **Coverage @ 80%** | 80.2% | Target >= 80% |
| **Coverage @ 95%** | 90.1% | Slight undercoverage — see findings |
| **Winkler Score (80%)** | 271.2 | Interval sharpness + accuracy |
| **ECE** | 0.0146 | 0 = perfect calibration |

![Evaluation Results](assets/results_plot.png)

### Coverage by Horizon

| Horizon | PICP (80%) | Winkler | Width |
|---------|-----------|---------|-------|
| h = 1h | 80.2% | 403 | 139 MW |
| h = 6h | 79.0% | 446 | 139 MW |
| h = 24h | 81.0% | 342 | 139 MW |
| h = 168h | 73.1% | 318 | 139 MW |

Long-horizon (168h) undercoverage is a known limitation of fixed-width conformal on non-stationary series. See [Findings](#findings-failures--learnings) for root cause and planned fix.

---

## What This Project Solves

Most forecasting systems return point estimates: *"demand will be 25,000 MW."*

Real operational decisions require calibrated uncertainty: *"demand will be between 24,500–25,500 MW with 80% confidence."*

This system delivers those intervals using **conformal prediction** — a framework that provides finite-sample coverage guarantees without assuming any data distribution.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Ensemble Layer                         │
│                                                         │
│  NeuralProphet ──┐                                      │
│  (trend +        ├──► Stacking meta-learner ──► point   │
│   seasonality)   │    (Ridge, val-optimised)    forecast │
│                  │                                      │
│  LSTM + MC  ─────┤                                      │
│  Dropout         │                                      │
│  (nonlinear)     │                                      │
│                  │                                      │
│  LightGBM ───────┘                                      │
│  Quantile (q05/q10/q50/q90/q95)                         │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Conformal Prediction Layer                  │
│                                                         │
│  EnbPI (rolling calibration)                            │
│  + KS-test drift detection                              │
│  + Adaptive window sizing                               │
│  → Distribution-free 80%/95% intervals                  │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│               Production Layer                          │
│                                                         │
│  FastAPI · Docker · Hugging Face Spaces                 │
│  JSONL monitoring · Slack alerts · /docs endpoint       │
└─────────────────────────────────────────────────────────┘
```

---

## Why These Techniques

### Conformal Prediction (not Gaussian intervals)

Standard approach: fit model → compute residual std dev → assume normality → `[μ ± 1.96σ]`.

Problem: energy demand is **right-skewed**. It cannot go negative but can spike 40%+ above mean during heatwaves. A Gaussian assumption systematically undercovers the upper tail.

Conformal prediction makes no distributional assumption. It calibrates intervals empirically from held-out residuals. The coverage guarantee is:

```
P(Y_{n+1} ∈ Ĉ(X_{n+1})) ≥ 1 − α
```

for any distribution, provably, in finite samples.

### Adaptive EnbPI (not static conformal)

Standard split conformal breaks for time series because it assumes **exchangeability** (i.i.d. observations). Residuals in a time series are temporally correlated.

[Xu & Xie, ICML 2021](https://arxiv.org/abs/2006.05703) introduced EnbPI — a rolling calibration window that recalibrates as new data arrives. This implementation adds:

- **KS-test drift detection** via `scipy.stats.ks_2samp`
- **Adaptive window sizing** — shrinks 15% on drift, expands 2% when stable
- Window bounds: 240h min, 1440h max

### Winkler Score (not RMSE)

RMSE only evaluates point accuracy. Two models with identical RMSE can have radically different prediction intervals. Winkler score jointly optimizes:

```
W = interval_width + (2/α) × penalty_for_misses
```

It cannot be gamed by inflating interval width — wider intervals are penalized directly.

### MC Dropout (not deterministic LSTM)

[Gal & Ghahramani, ICML 2016](https://arxiv.org/abs/1506.02142) showed that dropout at test time approximates variational Bayesian inference. This model runs T=100 stochastic forward passes and aggregates the resulting distribution — giving epistemic uncertainty from the neural network without full Bayesian training.

---

## Findings, Failures & Learnings

### What Failed

**Attempt 1 — Gaussian confidence intervals**

- Built: LSTM point forecast → `[μ ± 1.96σ]`
- Result: ~71% empirical coverage at 80% nominal
- Root cause: Right-skewed demand distribution violates normality. Upper tail was consistently underestimated.
- Fix: Replaced with quantile regression + conformal calibration.

**Attempt 2 — Fixed conformal window**

- Built: Split conformal, calibrate once on val set, apply statically to test
- Result: Coverage degraded 8% by week 2 of the test period
- Root cause: Exchangeability violated. Calibration residuals became stale as the test distribution drifted.
- Fix: Rolling EnbPI with KS-test drift detection.

**Attempt 3 — Symmetric intervals**

- Built: Center intervals at point forecast, expand equally both directions
- Result: Many misses above (demand spikes), few below
- Root cause: Demand spikes are asymmetric. Symmetric intervals waste coverage budget on the low side.
- Fix: Asymmetric quantile regression — q10/q90 learned separately.

**Attempt 4 — Single model**

- Built: LightGBM quantile only
- Result: 85% coverage at h=1h, 78% at h=168h — inconsistent across horizons
- Root cause: Short- and long-horizon patterns require different feature interactions.
- Fix: Ensemble of NeuralProphet + LSTM + LightGBM, weighted on validation Winkler.

### What Currently Needs Work

**95% interval undercoverage (90.1% vs 95% target) and h=168h at 73.1%**

The conformal calibration window (720h default) assumes recent residuals are representative of future residuals. At long horizons, forecasts drift and the residual distribution widens — but the calibration window doesn't respond fast enough.

Planned fix: Horizon-stratified calibration — separate EnbPI calibrators per horizon bucket (1–6h, 7–24h, 25–168h), preventing short-horizon residuals from diluting long-horizon calibration.

This is left in the README because a 95% interval achieving 90.1% coverage is a real failure. Knowing the root cause and fix is more valuable than hiding it.

---

## Feature Engineering

39 features across 5 categories:

| Category | Features |
|----------|----------|
| Lag features | 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h lags |
| Rolling statistics | 3h, 6h, 12h, 24h, 48h, 168h mean and std |
| Calendar | Hour, day-of-week, month, year, is_weekend |
| Cyclical encoding | sin/cos of hour and day-of-week |
| Holiday flags | US federal holidays via `holidays` library |

Cyclical encoding matters — without it, hour 23 and hour 0 are maximally distant in feature space despite being adjacent in time.

---

## Project Structure

```
prob-forecasting-engine/
├── src/
│   ├── models/
│   │   ├── neuralprophet_model.py   # Trend + seasonality
│   │   ├── lstm_model.py            # MC Dropout uncertainty
│   │   └── lgbm_quantile.py         # Quantile regression
│   ├── evaluation/
│   │   ├── conformal.py             # Split conformal baseline
│   │   ├── adaptive_conformal.py    # EnbPI + KS-test drift detection
│   │   └── metrics.py               # Winkler, PICP, ECE
│   ├── serving/
│   │   ├── api.py                   # FastAPI
│   │   └── monitoring.py            # JSONL logging + Slack alerts
│   └── utils/
│       ├── features.py              # Feature engineering
│       └── data_validation.py       # Data quality checks
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── download_data.py
├── dashboard/
│   └── index.html                   # Interactive monitoring UI
├── configs/config.yaml
├── Dockerfile
└── requirements.txt
```

---

## Running Locally

```bash
git clone https://github.com/foyie/prob-forecasting-engine
cd prob-forecasting-engine

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py
python scripts/train.py --config configs/config.yaml
python scripts/evaluate.py

uvicorn src.serving.api:app --reload --port 8000
open dashboard/index.html
```

---

## API

```bash
BASE=https://foyie-prob-forecasting-engine.hf.space

curl $BASE/health
curl -X POST $BASE/forecast \
  -H "Content-Type: application/json" \
  -d '{"horizon": 24, "coverage": 0.80}'
curl $BASE/monitoring/health

# Interactive docs
open $BASE/docs
```

---

## References

1. Xu & Xie (2021). *Conformal Prediction Interval for Dynamic Time-Series.* ICML. https://arxiv.org/abs/2006.05703
2. Gal & Ghahramani (2016). *Dropout as a Bayesian Approximation.* ICML. https://arxiv.org/abs/1506.02142
3. Angelopoulos & Bates (2022). *A Gentle Introduction to Conformal Prediction.* https://arxiv.org/abs/2107.03541
4. Winkler (1972). *A Decision-Theoretic Approach to Interval Estimation.* JASA.
5. Koenker & Bassett (1978). *Regression Quantiles.* Econometrica.

---

## License

MIT
## Author

**CHANDRIMA DAS**

*MS DS , UC SAN DIEGO*

[LinkedIn](https://linkedin.com/in/foyie) · [Portfolio](https://foyie.github.io/foyie/) · [Email](mailto:chdas@ucsd.edu)

**Last Updated:** May 2024
