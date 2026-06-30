# The Recession That Never Came: What the Yield Curve Got Right in 2022

*The most reliable recession alarm in modern economics went off louder than it had in
40 years — and no recession came. Here's what actually happened, and why the alarm
wasn't broken.*

---

## The puzzle

For decades, one chart has had an almost spooky track record of predicting U.S. recessions:
the **yield curve**. When it "inverts" — when short-term interest rates climb above long-term
ones — a recession has tended to follow within a year or so. It called 2001. It called 2008.
It even flashed before the 2020 downturn.

In 2022 and 2023, it inverted more deeply than at any time since the early 1980s. By the
standard models, the probability of a recession shot up toward **70%**.

And then… nothing. Growth held up. Unemployment stayed low. No recession arrived.

So what happened? Did the world's most trusted recession signal finally break? I built a model
to find out — and the answer turns out to be more interesting, and more honest, than "the curve
was wrong."

---

## First, why the curve predicts recessions at all

An inverted curve isn't magic. It's a bet, made with real money, by bond investors.

Two forces push the curve toward inversion at the same time:
1. **The Fed pushes short-term rates up.** When inflation runs hot, the Federal Reserve raises
   its short-term policy rate to cool the economy.
2. **Investors pull long-term rates down.** If those same investors believe the Fed has tightened
   too much and a slowdown is coming, they rush to lock in today's yields by buying long-term
   bonds — which pushes long-term rates *down*.

Short rates up, long rates down: the curve inverts. Because investors are putting real money
behind the view that a downturn is coming, the signal carries information. Historically, it leads
recessions by about **9 to 12 months**. *(This mechanism has been studied for decades — by Campbell
Harvey, by Arturo Estrella and Frederic Mishkin, and it underpins the New York Fed's published
recession model.)*

*[Visual: the interactive scenario lab — drag the short-term yield up and the long-term yield
down, and watch the curve invert and the probability climb in real time.]*

---

## The model: keep it honest, keep it simple

I built the recession probability the standard way — a "probit" model that turns the size of the
curve's inversion into a probability of recession 12 months out. This is the same family of model
the New York Fed publishes.

There are three popular versions of the "spread," and rather than crown a favorite, I tested all
three and reported the trade-off honestly:

- **10-year minus fed funds** is the *most accurate* and gives the *earliest* warning — but it
  also raises the most false alarms (it's twitchy, because the Fed's rate moves around a lot).
- **10-year minus 3-month** raises the *fewest* false alarms — which is why I made it the default.
- **10-year minus 2-year** has the shortest track record, so I trust its numbers the least.

There's no single winner; they genuinely trade off. Saying so is part of being honest.

*[Visual: the model's headline gauge — today's reading.]*

---

## The honest result nobody likes to admit

Here's where most "we cracked the 2022 mystery" stories go wrong. The tempting move is to find
some extra variable — debt, house prices, bank lending — that "explains" why 2022 was different,
and declare victory.

I tested those variables. Carefully. On a fair, like-for-like basis. **None of them reliably
improved the forecast** beyond the curve itself.

That's not a failure — it's the intellectually honest finding. The yield curve is *hard to beat*,
and a single odd year (2022) is not enough evidence to crown a new explanation. So instead of
pretending I found "the variable that explains 2022," I built something more defensible: a model
that's upfront about what it does and doesn't know.

---

## Two layers: a probability, and a character read

The dashboard has two distinct parts, held to two different standards:

1. **The probability engine** — the statistically validated number. It always warns when the
   curve inverts. It never says "relax."
2. **A "character" read** — an interpretive layer that asks a different question: *if* a recession
   came, would it look like 2008? It grades whether the classic **credit-cycle** ingredients are
   present (high household debt, falling real house prices, spiking credit spreads).

Crucially, this character layer **never declares a "false alarm."** That phrasing would be
dangerous — because the same "inverted, but calm" reading also preceded the real recessions of
1980, 2000, and 2019. An inverted curve always deserves to be taken seriously. The character
layer only tells you *what kind* of risk you're looking at.

---

## The answer, in one picture: 2007 vs 2023

The clearest way to see why 2022–24 was different is to line the episodes up side by side.

*[Visual: the conditions matrix — each historical inversion as a row, with red/green cells for
each "recession ingredient."]*

Compare **2007** (which led to the financial crisis) and **2023** (which led to nothing):

| | Curve inverted? | Household leverage | Real house prices |
|---|:---:|:---:|:---:|
| **2007** | Yes | **133% of income** 🔴 | Falling 🔴 |
| **2023** | Yes (deeper) | **95% of income** 🟢 | Falling 🔴 |

Both had an inverted curve. Both had falling real house prices. The decisive difference is
**leverage**: households went into 2007 buried in debt (133% of income) and into 2023 far less so
(95%). That single gap is the cleanest evidence for what economists think actually happened.

### Two reasons the alarm didn't end in a fire

- **Low leverage (the Dodd-Frank legacy).** After 2008, tighter bank rules meant the boom of the
  2010s never piled up the dangerous debt that turns a slowdown into a crisis. When the Fed hiked,
  there simply wasn't the leverage to trigger the usual cascade of defaults. Even the 2023 banking
  scare (Silicon Valley Bank, First Republic) fits: those banks blew up on their *bond portfolios*,
  not on bad loans to households and businesses — so it didn't spread.
- **An immigration-driven labor surge.** A large rise in the labor supply (visible in the
  foreign-born labor force) cooled an overheated job market and helped bring down inflation —
  meaning the Fed didn't have to tighten as hard as it otherwise would have. The extra workers did
  some of the Fed's job for it.

---

## So was the curve "wrong"? No — it gave a probability

This is the part worth sitting with. The model's reading in 2023 wasn't "recession: yes." It was a
**probability** — around 70%.

A 70% chance means that, historically, roughly **3 in 10** such situations *don't* end in
recession. 2022–24 was a draw from that minority. The curve didn't malfunction; it gave you the
odds, and the less-likely outcome happened.

I checked this directly (it's called a calibration test): across the model's whole history, when it
said "low risk," low risk is what followed. Its biggest miss in the *high* range is — you guessed
it — 2022–24 itself. One unusual episode. Which is exactly why a careful model reports a
probability and refuses to promise a certainty.

---

## Honest limitations

- There have only been about **11 recessions** since World War II — a tiny sample to learn from.
- The richest data (debt, housing, lending) only goes back a few decades, so the full picture can
  only be tested on recent cycles.
- The "character" layer is interpretive judgment, not a validated prediction — and one strange year
  (2022) can't, by itself, prove any single explanation.

None of that makes the curve useless. It makes it honest: a genuinely good *odds-maker* for
recessions, not a crystal ball.

---

## See it yourself

The full interactive dashboard — live probability gauge, the model's real-time track record, the
2007-vs-2023 conditions matrix, and a scenario lab where you can build your own yield curve and
economy — lives here:

**[ dashboard URL — coming once it's published ]**

---

*Methods note (for the curious): The probability is a probit on the 10-year minus 3-month Treasury
spread, trained to predict an NBER recession 12 months ahead, evaluated with an expanding-window
(real-time) backtest. Out-of-sample AUC ≈ 0.80 across 8 recession onsets. Data is monthly, from
FRED. The fragility/character layer uses documented threshold judgments, not fitted parameters, and
never outputs a "false alarm."*
