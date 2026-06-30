# Recession-Prediction Model - Project Brief for Claude Code
# >>> RENAME THIS FILE TO  CLAUDE.md  AND PLACE IT IN THE PROJECT ROOT <<<
# (Claude Code auto-reads CLAUDE.md at the start of every session.)

Standing instructions. Read at the start of every session. Build in phases, one at a time, confirm before moving on. Explain in plain English (the user is non-technical). Validate against Section 6. This brief is the source of truth and reflects everything Mark Zandi specified (see Sections 4A and 9).

## 1. Goal (Zandi)
Two goals, in his words: (a) establish the BEST measure of the yield curve for predicting US recessions since World War II, and (b) understand WHY the 2022-2024 inversion did not produce a recession. End products: a self-updating, INTERACTIVE dashboard and a research paper (Substack).

## 2. What we are building
A monthly model that reads FRED data, computes a recession probability (from the curve) and a fragility read on the CHARACTER of the risk, presented as an INTERACTIVE dashboard (live gauge + character banner + fragility lights + trajectory chart + scenario lab + conditions matrix), updating weekly and drafting a Substack post.

## 3. Data (all FRED; user has an API key). Make everything MONTHLY (interpolate quarterly; forward-fill latest so the live read never breaks - but NEVER let pct_change forward-fill, see 6C).
Use EVERY series below - all pulled deliberately, all must be put to work (probability models, verdict layer, scenario lab, conditions matrix, or historical comparison). None should sit unused.
- Yields/rates (monthly): GS10, TB3MS, GS2, FEDFUNDS.
- Recession dates (monthly): USREC (Zandi: use FRED's official recession dates to pin the exact monthly dates to test against).
- Credit (monthly): BAA, AAA. Spreads: Baa-Aaa = BAA-AAA; Baa-10yr Treasury = BAA-GS10. (For the paper's narrative, a high-yield spread is useful: it peaked ~6% in late 2022 vs ~10%/11%/20% in 2001/2020/2008.)
- Inflation (monthly): CPIAUCSL (deflator for real house prices).
- Housing (monthly): CSUSHPINSA. Nominal YoY AND real YoY (= CSUSHPINSA/CPIAUCSL, YoY).
- Debt levels (quarterly, interpolate): CMDEBT (household), BCNSDODNS (nonfinancial corporate), DPI, GDP. Ratios: HH debt/income = CMDEBT/DPI; HH debt/GDP = CMDEBT/GDP; corp debt/GDP = BCNSDODNS/GDP. (Zandi named all three: household debt/GDP, nonfinancial corporate debt/GDP, household debt relative to income.)
- Debt GROWTH / credit impulse (per Zandi): YoY % growth of CMDEBT and of BCNSDODNS - "growth in debt outstanding as a proxy for credit availability."
- Household debt-service ratio (quarterly, from 2005): TDSP.
- Credit availability (Zandi: the Fed's quarterly Senior Loan Officer Opinion Survey, whether banks are tightening or easing): DRTSCILM (SLOOS net % tightening C&I), from 1990.
- Labor / migration (Zandi Theory 2): UNRATE (monthly); foreign-born civilian labor force LNU01073395 (monthly, from 2007, BLS household survey on FRED) and its YoY growth as the immigration / labor-supply proxy; job openings JTSJOL (monthly, from 2000).

## 4. The model
### Three yield-spread models - all FIRST-CLASS (Zandi Step 1: experiment with all three)
The three measures Zandi named: 10y-Fed funds, 10y-3m, 10y-2y. Build and SHOW all three as comparable probability models. Confirmed results (12-month horizon):
- 10y-3m: in-AUC 0.822, OOS 0.797, 8 onsets, current ~12.4% - the pre-committed DEFAULT headline.
- 10y-fed funds: in 0.855, OOS 0.853 (highest), 8 onsets, current ~8.5% - leans on policy rate (Wright 2006). Historically often treated as most accurate. Show, do not crown.
- 10y-2y: in 0.816, OOS 0.694, 4 onsets (short history from 1976), current ~9.9%. The "increasingly favored" measure in commentary - let the analysis judge, do not assume.
The lab lets the user SELECT which spread drives the gauge; show the comparison table. 10y-3m stays default (rule 7).

## 4A. Define "most accurate" and RANK the three spreads (Zandi Step 1) - DONE (Phase 4D)
Per Zandi, define "most accurate" before ranking, then rank-order the three measures. Decided: keep his three named criteria (do NOT collapse into one composite - they genuinely conflict), report all three rankings, hit-rate as supporting detail. Confirmed scorecard:
1. ACCURACY (OOS AUC): 10y-ff (0.853) > 10y-3m (0.797) > 10y-2y (0.694).
2. FEWEST FALSE POSITIVES (inverts/crosses without a recession within ~18 mo): 10y-3m (2) > 10y-2y (4) > 10y-ff (7).
3. LONGEST LEAD TIME (mean months signal-to-onset): 10y-ff (12.5) > 10y-2y (11.8) > 10y-3m (9.1). Lead times of 9-12 mo match Zandi's ~9-12 intuition (external validation).
Result: real trade-off, no single winner. 10y-ff is most accurate and earliest but cries wolf most (7 false positives); 10y-3m has the FEWEST false positives (2), which is exactly Goal 1(b) and independently justifies it as the default headline (reinforces rule 7). The ranking is reported in the dashboard, never used to silently re-crown the headline.

### Probability engine = probit on the chosen spread, SPREAD ONLY
Probit -> probability of an NBER recession at the 12-month-ahead month (Estrella-Mishkin / NY Fed standard). Zandi Step 2: recession probability on the left, the curve on the right; try transformations of the curve (we test 6/12/18-month horizons). Phase 4 confirmed no added variable beats the spread OOS, so the engine stays spread-only. Report a LINEAR PROBABILITY MODEL beside it.

### Where the other variables live (NOT the probit - Phase 4; rule 10)
All remaining variables power the VERDICT/character layer, the SCENARIO LAB, the CONDITIONS MATRIX (Section 10), and a "most resembles [year]" comparison. Each may be tested as a probit add-on under rule 9, but expect (per Phase 4) they inform character, not the probability.
- Fragility / confirming: real house price YoY, credit spread (Baa-10yr), SLOOS, household debt-to-income, debt growth.
- Context / severity: household & corporate debt-to-GDP, fed funds level, debt-service ratio (TDSP), unemployment.
- Labor / migration: foreign-born labor force growth, job openings.

### Verdict logic - grades CHARACTER, never flips to "false alarm"
An inverted curve ALWAYS warns via the probability. The fragility layer grades TYPE/severity; never downgrade an inversion to "false alarm." Readings: inverted + fragility (high debt, OR real house prices falling AND credit spreads spiking) -> "GENUINE WARNING: credit-cycle (2008-type) risk"; inverted, no fragility -> "INVERTED, no credit-cycle fragility - a 2008-style collapse looks unlikely; a recession from other causes is still possible"; not inverted -> "ALL CLEAR." (The no-fragility bucket held 2022 AND real recessions 1980/2000/2019 - "false alarm" was wrong 3 of 4 times; never use it.) Fragility thresholds live in the shared config (Section 10).

## 4B. Inference rules (STANDING)
1. Out-of-sample = expanding window only; report in-sample vs OOS AUC side by side (the gap is the headline).
2. Overlapping labels -> Newey-West (HAC) errors, maxlags = horizon minus 1 (11 for 12-month).
3. Vintage: spread-only is clean; lag Case-Shiller ~2 months for real-time; never let pct_change forward-fill (6C); note we train on revised values (limitation).
4. The 2022 non-event is n=1: a prior, not evidence. Earn a place only by improving full-record OOS AUC, never by "explaining 2022."
5. Validation vs own fit are separate. NY Fed published coefficients = a math check only; own fitted coefficients differ by sample (expected).
6. Recession label = point-in-time (in recession at exactly the 12-month-ahead month), matching the NY Fed.
7. No specification-search inflation: 10y-3m@12mo is the pre-committed default headline; show the other two clearly, never crown a scan winner.
8. Always report the onset count behind each OOS AUC.
9. Same-sample comparison: re-evaluate the baseline on the exact shortened window when testing a short-history variable; it earns a place only by beating the baseline on the SAME period (most won't, per Phase 4 - fine).
10. Two layers, two standards. The probit (spread only) is the validated PROBABILITY. The fragility/verdict layer, the scenario lab, and the conditions matrix are an INTERPRETIVE OVERLAY - thresholds are documented judgment calls, not fitted. Grades credit-cycle CHARACTER; never outputs "false alarm."

## 5. Build phases (one at a time, confirm before next)
1-4. [DONE] Data pulled & monthly; three spreads + real house prices; probit + linear across three spreads x 6/12/18 lags (10y-3m@12mo primary, OOS ~0.80, 8 onsets); fragility variables tested - none beat the spread OOS, moved to the verdict/lab.
4C. [DONE] Expanded to the FULL Section-3 set (TDSP, foreign-born LF + growth, job openings, debt growth). Built all three spread models as first-class with the comparison table.
4D. [DONE] Zandi Step-1 scorecard (Section 4A): per-model accuracy + false-positive count + average lead time, with explicit rankings, surfaced as a table in the dashboard.
5. [DONE] Verdict/character layer (rule 10), three readings above; grades 2008-type credit-cycle risk; honestly under-calls policy/sector/exogenous recessions (the probability still warns via the inversion).
6. [DONE] Static dashboard: probability gauge headline; character banner (never "false alarm"); fragility lights with values + thresholds; trajectory chart with BOTH lines (real-time prominent, fitted dashed) + recessions shaded; real-time AUC as headline performance.
6B. [DONE] Interactive scenario lab: three-model selector; four yield sliders (fed funds, 3m, 2y, 10y) that draw the curve and compute all three spreads live; probability recomputes from the selected model's fitted probit; fragility sliders move only the character verdict; "your scenario most resembles [year]" on the full standardized vector (curve + fragility).
6C. [DONE] Fixed a real data bug: pandas pct_change() was silently forward-filling missing months, fabricating house-price/debt/migration YoY readings for months with no underlying data. Fixed across all YoY features (honors rule 3). Reconciled the headline vs lab house-price value (both now -2.5% from the same as-of month). Added the lab note clarifying which sliders drive the verdict.
6D. [TODO] Recession conditions matrix panel (pattern-tracker style), built from LIVE data and refreshed each run - see Section 10 for the full spec.
7. [AFTER 6D] Self-updating WEEKLY. A scheduled GitHub Action re-pulls FRED, recomputes, and republishes to GitHub Pages once a week (Zandi suggested daily; we settled on weekly - the spread moves slowly at a 12-month horizon and most fragility series are quarterly, so weekly is plenty). FRED key as a GitHub secret, never in code.
8. Validate + write up. Expanding-window OOS, AUC + onset counts + lead times. ADD a CALIBRATION / reliability check on the headline gauge (do readings near 12% actually precede recessions ~12% of the time? bin predictions and plot observed vs predicted) - this validates the number the whole dashboard hangs on and directly supports the 2022 framing (a high reading is a probability, not a certainty). Draft a Substack post each run.

## 6. Validation targets
- NY Fed formula: prob = std-normal-CDF of (-0.5333 - 0.6629 * (10y-3m spread, pp)). Reliable check: 50% at spread = -0.80; 10%/90% at +1.13/-2.74. The old "10% at +0.76, 90% at -2.40" is from the 1996 table (different estimation) - won't match at the tails (source, not bug).
- Deepest-inversion 10y-3m: ~-0.5 (2001), -0.4 (2008), -0.3 (Aug 2019), -1.5 (mid-2023). KEY PATTERN (workbook): Dec 1980 had the deepest inversion overall (-2.65) and a recession followed; May 2023 (-1.57) was the deepest inversion that did NOT - the difference is leverage (debt-service 10.6% in 2023 vs 15.5% in 2007; HH debt/GDP ~72% vs ~95%). Headline evidence for the low-leverage theory.
- 2022-24 fragility was absent: real house prices did NOT decline (Case-Shiller dipped only ~5% then hit new highs, vs ~-20% before 2008); credit spreads did NOT spike to recessionary levels. Confirms the "inverted, no credit-cycle fragility" verdict for 2023.
- HH debt-to-income: ~132-134% before 2008, ~95% in 2023. Real house prices: ~-12% (2008), ~-0.7% (2022).
- Foreign-born labor force: ~23.8M (2007), ~28M (2019), ~31M (2023). [confirmed 23,751 / 28,153 / 31,249]
- NBER recession start years: 1980, 1981, 1990, 2001, 2007, 2020.

## 7. Honest constraints (state in the paper)
- Rare events: ~11 recessions; perfect fit risks overfitting.
- Data history: debt/housing 1980s-2000s, SLOOS 1990, JOLTS 2000, foreign-born LF 2007 - the full set is testable mainly on recent recessions (rule 9 governs comparisons).
- Monthly needed for brief inversions; quarterly series interpolated.
- The verdict / lab / matrix layer is interpretive (rule 10); grades 2008-type credit-cycle risk; cannot distinguish 2022 from 2000/2019 (n=1).

## 8. How to work with this user
Non-technical: explain in plain English. One phase at a time; show results; wait. Commit to git each session. Simple, well-commented code.

## 9. The economic narrative (Zandi) - what the paper must explain, and what the verdict/lab/matrix encodes
### Why the yield curve predicts recessions (intuition)
An inversion is when short-term rates rise above long-term rates, and it tends to LEAD recessions by ~3-18 months (average ~9-12). The mechanism runs through bond investors: when the Fed jacks up short rates to fight inflation, investors bet the Fed has overdone it and that a recession (and lower future inflation) is coming, so they buy long-term bonds, driving long yields DOWN. Fed pushing short up + investors pulling long down = the inversion. Because investors put real money behind their read, the signal is informative. (This is exactly what the lab's four yield sliders dramatize.)
The un-inversion is different and is NOT a leading signal: once a recession hits, unemployment rises, the Fed cuts, short rates fall, and the curve re-steepens. By then you are already in/on top of the recession, so steepening is coincident, not a warning.

### The classic credit cycle (Theory 1 - why 2022-24 was a false positive: low leverage)
The normal post-WWII cycle: in booms banks extend credit aggressively, leverage and total debt rise sharply; the economy overheats, inflation rises, the Fed hikes and the curve inverts; the cost of leverage becomes onerous; delinquency, default and bankruptcy rise; recession follows. In 2022-24 the chain BROKE because leverage going in was low - a legacy of 2008 and Dodd-Frank (higher bank capital and liquidity requirements, tighter scrutiny), so banks stayed cautious and never extended the usual boom-time credit. With less leverage, the Fed's hikes did not trigger the usual delinquency/default wave. Even the 2023 banking stress (SVB, First Republic) fits: those banks blew up on their bond portfolios, not on household/corporate credit losses, so it did not cascade. Capture: HH debt/GDP, nonfinancial corporate debt/GDP, HH debt/income, SLOOS.

### The immigration / labor-supply surge (Theory 2)
2022-24 saw a large immigration surge. The added labor supply relieved a very tight post-pandemic labor market, cooled wage growth, and meant the Fed did not have to raise rates as high as it otherwise would - the extra labor did some of the Fed's heavy lifting on inflation, so tightening did not tip the economy into recession. Proxy: the foreign-born civilian labor force (LNU01073395), which rose significantly over the period.

## 10. Phase 6D spec - "Recession conditions matrix" panel (build from LIVE data, update each weekly run)
Purpose: a pattern-tracker-style matrix showing, for each historical yield-curve inversion, which recession ingredients were in place and whether a recession followed. It answers the project's core question VISUALLY: compare 2007 vs 2023 - both inverted, both with falling real house prices, but household leverage was 133% of income vs 95%. That leverage gap is the cleanest single piece of evidence for why 2022-24 inverted yet produced no recession. Place it in the dashboard directly below or beside the scenario lab.

Rows (reuse the lab's episode reference set so the two panels stay in sync; one row per episode at its deepest-inversion / reference month per the workbook): 1980, 1989, 1998, 2000, 2004, 2007, 2019, 2023. PLUS a highlighted live "Today" row computed from the latest monthly data, so the panel refreshes with the model on every run.

Columns - grouped into two LABELED blocks with a visual divider:
- LEADING conditions (present BEFORE a recession):
  1. Curve inverted - 10y-3m spread < 0 pp.
  2. Leverage high - household debt-to-income > 110%.
  3. House prices falling - real (CPI-deflated) house price YoY < -2%.
- COINCIDENT confirmers (these widen DURING/AFTER the downturn, so they read "calm" at the inversion - e.g. 2007's Baa-10yr spread was only 1.71 and banks were still easing (SLOOS -3); both must show GREEN for 2007, which is the point of labeling this block):
  4. Credit stress - Baa-10yr Treasury spread > 2.5 pp.
  5. Banks tightening - SLOOS net % tightening C&I > 20.
- Outcome column: historical rows = "Recession <year> (<type>)" or "No recession"; the Today row = the current character verdict.

Cell rendering: show the ACTUAL value (e.g. "133%", "-3.0%", "2.78"), colored RED when it crosses into the danger zone, GREEN when benign, and a gray dash when the FRED series does not cover that date (TDSP/debt-to-GDP pre-2005, SLOOS pre-1990, foreign-born pre-2007). Outcome cell: red = recession, green = no recession.

Thresholds = documented JUDGMENT CALLS (rule 10), NOT fitted. CRITICAL: store them in ONE shared config object consumed by the verdict banner, the headline fragility lights, AND this matrix, so they can never drift apart (a drift like that caused the 6C mismatch). Exact thresholds:
  inverted: 10y-3m < 0 ; leverage: HH debt/income > 110% ; house: real YoY < -2% ; credit: Baa-10yr > 2.5 ; banks: SLOOS > 20.
  (Changes from the mockup: house moved 0 -> -2% so a near-flat reading like 2022's -0.7% is not flagged as a decline; banks moved 15 -> 20 because SLOOS is very volatile and 15 is roughly its mean. Apply the house change consistently in the verdict layer and lights, since the config is shared.)

Honesty notes to encode on the panel:
- One line under the coincident block: "these widen during the downturn, so they can read calm at the inversion (2007 is the example)."
- The matrix DESCRIBES conditions and states what history did; it does NOT predict, and it never labels an episode a "false alarm" (rule 4, rule 10).
- Takeaway caption: there is no single recipe - the full credit-cycle stack lined up only in 2007; other recessions had other triggers (policy 1980, tech bust 2000, COVID 2020); the curve is the one ingredient present ahead of most of them; the 2007-vs-2023 leverage gap is the cleanest evidence for the low-leverage explanation of 2022-24.

Optional (nice-to-have): clicking a historical row loads that episode's curve + fragility into the scenario lab, tying the two panels together.

Validation: historical cells must reproduce the episode reference vectors / workbook - 2007 leverage 133% and real house -3.0%; 2023 leverage 95% and real house -4.3%; 2000 credit 2.78 and SLOOS +54 (both red); 2007 credit 1.71 and SLOOS -3 (both GREEN, demonstrating the coincident-timing point).
