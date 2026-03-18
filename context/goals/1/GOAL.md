/goal

Use the below program to evolve a system S that discovers profitable trading strategies for BTC directional prediction on a 15-minute horizon.

## Primary Objective

Build and continuously improve a trading system that predicts whether **BTC will go UP or DOWN over the next 15 minutes**.

The task is strictly:

- **Asset:** BTC

- **Prediction type:** binary directional classification (**UP / DOWN**)

- **Forecast horizon:** **15 minutes**

- **Trading style:** high-frequency directional decision making

- **Primary success metric:** **directional accuracy >= 65%**

- **Secondary success metric:** **more than 90 bets per month**

The system must optimize for both:

1. **Prediction quality**: at least **65% directional accuracy**

2. **Trading activity**: at least **90 valid bets per month**

A model that is accurate but generates too few trades is not acceptable.  

A model that generates many trades but fails to reach 65% directional accuracy is also not acceptable.

---

## Initial State (build first)

S₀ = online continuous adaptive trading system which:

- uses **online market data** for training and updating

- uses **evolutionary model search** (mutation, crossover, replacement of weak models)

- uses **strict walk-forward validation**

- uses **online Sharpe filtering**

- uses **horizon ensembles** H = {1h, 4h, 8h, 12h, 24h} as contextual signals

- uses **consensus gating** for signal generation

- uses **time-series foundation models** and sequence models

- is centered on the **main execution target = BTC up/down over the next 15 minutes**

---

## Core Prediction Task

At each decision point, the system must output one of:

- **UP** = BTC is expected to be higher in 15 minutes

- **DOWN** = BTC is expected to be lower in 15 minutes

- optional: **NO TRADE** only if confidence is too low

The system should be designed to maximize profitable directional decisions while maintaining enough signal frequency to exceed **90 bets per month**.

---

## Optimization Targets

The system must optimize for the following target profile:

- **Directional accuracy >= 65%**

- **Bets per month > 90**

- **Positive net profitability**

- **Strong Sharpe ratio**

- **Controlled drawdown**

- **Robustness across market regimes**

- **Stable out-of-sample performance**

---

## Fitness Priorities

Fitness should prioritize:

1. **Directional accuracy on 15m BTC prediction**

2. **Monthly bet count**

3. **PnL**

4. **Sharpe ratio**

5. **Drawdown control**

6. **Regime robustness**

7. **Consistency across walk-forward folds**

Any candidate that fails either:

- **accuracy < 65%**

- **bets/month <= 90**

should be considered below target even if other metrics look good.

---

## Data and Modeling Requirements

The system should:

- learn from **BTC market structure**

- exploit **short-term directional patterns**

- use **multi-timescale context** while keeping the target fixed at 15m

- adapt online as conditions change

- discover alpha from:

  - price action

  - volatility

  - momentum

  - order-flow proxies

  - volume

  - derivatives context if available

  - regime features

  - cross-horizon agreement

Longer horizons {1h, 4h, 8h, 12h, 24h} should be used as **contextual ensemble inputs**, not as replacements for the core 15m target.

---

## Validation Rules

The system must use:

- strict **temporal splits**

- strict **walk-forward validation**

- no lookahead bias

- no leakage

- no random split for final evaluation

- evaluation on realistic trading frequency

- explicit measurement of:

  - accuracy

  - number of bets

  - PnL

  - Sharpe

  - drawdown

  - regime-specific performance

A candidate is only truly good if it remains strong in out-of-sample walk-forward testing.

---

## Evolution Strategy

Run this loop continuously:

loop t = 1..∞

    S_t = design_or_modify(S_{t-1})   # implement the current design

    O_t = run(S_t)                    # train, validate, and trade / paper trade

    P_t = measure(O_t)                # evaluate accuracy, bets/month, Sharpe, PnL, drawdown, regime behavior

    Δ_t = reflect(S_t, P_t)           # identify weaknesses and bottlenecks

    S_{t+1} = improve(S_t, Δ_t)       # produce a better design

end

---

## Reflection Requirements

During reflection, explicitly diagnose:

- why accuracy is below 65% if target is missed

- why signal frequency is below 90 bets/month if target is missed

- whether the consensus gate is too strict

- whether the model is overfitting to specific regimes

- whether the ensemble is improving or diluting edge

- whether the system is sacrificing too much frequency for precision

- whether the system is producing too many low-quality trades

- whether thresholding should be tightened or loosened

The next design must directly address the identified weakness.

---

## Design Guidance

The system should search for the best balance between:

- **high directional precision**

- **high enough trade frequency**

- **strong risk-adjusted return**

- **robustness through time**

This is not a low-frequency swing system.  

This is not a generic forecasting system.  

It is a **BTC 15-minute directional trading system** that must be both:

- **accurate enough** (>= 65%)

- **active enough** (> 90 bets/month)

---

## True Success Condition

The process should continue until the system consistently achieves:

- **BTC 15m directional accuracy >= 65%**

- **more than 90 bets per month**

- **positive and robust trading performance in walk-forward validation**

---

## One-Line Mission

Evolve a continuous adaptive system that predicts **BTC UP or DOWN over the next 15 minutes** with **at least 65% directional accuracy** and **more than 90 bets per month**, while maintaining strong out-of-sample profitability and robustness.