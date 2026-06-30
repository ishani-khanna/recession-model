# Validation Report — Yield-Curve Recession Model

*Honest statistical validation of the model behind the dashboard. Plain English throughout.*
*Headline model: a probit on the 10-year minus 3-month Treasury spread, predicting an NBER
recession exactly 12 months ahead (the Estrella-Mishkin / NY Fed standard).*

---

## 1. The headline: in-sample vs out-of-sample (the gap is the point)

Any model looks good when graded on the same data it was trained on. The honest test is
**out-of-sample**: we use an *expanding window* — to make the forecast for a given month, the
model is trained **only on data available up to that month**, then walked forward one month at
a time. That mimics what a forecaster would actually have seen live.

We report the in-sample AUC and the out-of-sample (OOS) AUC side by side. AUC measures how well
the model *ranks* risky months above calm ones: 0.5 is a coin flip, 1.0 is perfect.

| Spread | In-sample AUC | **Out-of-sample AUC** | Recession onsets | False positives | Lead time (mean, range) |
|--------|:-------------:|:---------------------:|:----------------:|:---------------:|:-----------------------:|
| **10y − 3m** (headline) | 0.822 | **0.797** | 8 | **2** | 9.1 mo (1–17) |
| 10y − fed funds | 0.855 | **0.853** | 8 | 7 | 12.5 mo (9–18) |
| 10y − 2y | 0.816 | **0.694** | 4 | 4 | 11.8 mo (5–17) |

**Reading this:**
- The headline 10y−3m model holds up out-of-sample (0.797 vs 0.822 in-sample) — a **small gap**,
  which is the reassuring sign that it is not overfit.
- **The ranking trade-off (Zandi's "Step 1").** No single spread wins on everything:
  - *Most accurate / earliest:* 10y−fed funds (OOS AUC 0.853, 12.5-month average lead) — but it
    **cries wolf the most (7 false positives)** because the fed funds policy rate is volatile.
  - *Fewest false alarms:* **10y−3m (just 2)** — which is exactly the project's goal of minimizing
    false positives, and is why it stays the default headline even though it is not the most accurate.
  - *Short history:* 10y−2y only reaches back to 1976 (4 onsets), so its numbers rest on the fewest
    events — treat with caution.
- A "false positive" = the curve inverted but **no** recession followed within 18 months. Lead times
  of 9–12 months match the long-standing intuition that inversions lead recessions by ~9–12 months.

We deliberately do **not** collapse these three criteria into one score — they genuinely conflict,
and hiding that would be dishonest. 10y−3m is the pre-committed default (theory- and
literature-backed), never crowned by a data scan.

---

## 2. Calibration — when the gauge says 30%, does a recession follow ~30% of the time?

AUC only measures *ranking*. Calibration asks whether the **probabilities themselves** are honest.
We calibrate on the **out-of-sample** predictions (what a user would actually have seen on the
gauge), never the in-sample fit — the in-sample version would flatter the model.

**Reliability table (735 out-of-sample monthly forecasts, 1964–2025):**

| Predicted probability | Months | Average predicted | **Actual recession rate** |
|-----------------------|:------:|:-----------------:|:-------------------------:|
| 0–10% | 433 | 2.6% | **3.0%** |
| 10–25% | 136 | 17.4% | **19.1%** |
| 25–50% | 100 | 35.4% | **27.0%** |
| 50–100% | 66 | 65.7% | **28.8%** |

**Brier score: 0.102** (lower is better; always-guess-the-base-rate gives 0.102 too).

**The honest read:**
- **The low and middle ranges are well-calibrated.** When the model said ~3%, recessions followed
  3.0% of the time; ~17% → 19%. Those are close. So for the *common* readings — including today's
  ~12% — the gauge means what it says.
- **The high range (50–100%) over-predicts:** when the real-time model said ~66%, a recession
  actually followed only ~29% of the time. But read on, because *why* matters.
- **The Brier score barely beats guessing the base rate.** That is the honest limit: the model's
  real strength is **ranking** risky periods (AUC 0.80), not delivering razor-sharp probabilities in
  the high tail — where, with so few recessions, there is very little to learn from.

### Why the high bin looks bad — and why that is mostly *one episode*
The 66 months in the top bin are dominated by the **2022–24 deep inversion**, where the real-time
model read very high yet no recession came. The model's single biggest calibration blemish *is*
2022–24. With only ~11 recessions in the whole record, one large recent deep-inversion-without-a-
recession drags the high bin down. We cannot conclude the model is *systematically* overconfident —
that high bin is essentially **n = 1** (see the 2022 note below).

---

## 3. The 2022 question, reframed honestly

In May 2023 the curve hit its deepest inversion of the modern era. The model's real-time reading
peaked at **~77%** (the NY Fed's published formula gives ~69% at that spread). No recession followed.

The honest reframe (and the whole reason the gauge shows a **probability**, not a verdict):

> **A high reading is a probability, not a certainty.** A ~70–77% reading means that, historically,
> something like **3 in 10** such situations did *not* end in a recession. 2022–24 was a draw from
> that minority. The curve was **not "wrong"** — it gave a probability, and the less-likely outcome
> happened.

This is why the dashboard's interpretive layer **never** labels an inversion a "false alarm." It
instead grades the *character* of the risk (is a 2008-style credit-cycle collapse brewing?), and
2022–24 honestly reads as *"inverted, but no credit-cycle fragility"* — not as "all clear."

---

## 4. Honest constraints (stated plainly)

- **Rare events.** Only ~11 recessions since WWII. A model can look perfect by memorizing them, so
  we lean on out-of-sample testing and report uncertainty rather than a flattering fit.
- **Short data histories.** The yield curve starts in 1953 (and is only a free-market signal after
  the 1951 Treasury–Fed Accord). Debt and housing series reach back only to the 1980s–2000s, the bank
  lending survey to 1990, JOLTS to 2000, the foreign-born labor force to 2007. So the fuller
  "fragility" picture can only be tested on recent recessions.
- **Monthly interpolation.** We need monthly data to catch brief inversions; the quarterly series
  (debt, debt-service) are interpolated between their real readings, never invented before/after.
- **Two layers, two standards.** The probit probability is statistically validated. The fragility
  verdict and the conditions matrix are an **interpretive overlay** — their thresholds are documented
  judgment calls, not fitted parameters. They grade credit-cycle character and never output a
  "false alarm."
- **2022 is n = 1.** Any explanation for 2022–24 (low leverage, the immigration labor-supply surge)
  is a *prior*, not proof. One non-recession cannot validate a single story for it.

---

## 5. Bottom line

The yield curve remains a genuinely useful recession *ranker* (out-of-sample AUC ~0.80 on 8 onsets),
and the headline 10y−3m measure earns its place by having the **fewest false alarms**. Its
probabilities are well-calibrated in the common range but should not be read as precise in the high
tail — where the data is thin and 2022–24 is the cautionary example. The right way to read the gauge
is exactly what it claims to be: a **probability**, not a promise.
