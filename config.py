"""
config.py  -  the SINGLE source of truth for the verdict/matrix thresholds.

These thresholds are documented JUDGMENT CALLS (rule 10), not fitted parameters. They are
imported by build_verdict.py (the character banner), build_dashboard_data.py (the headline
fragility lights, the scenario-lab sliders, AND the conditions matrix) so the same numbers
drive every panel and can NEVER drift apart - a drift like that caused the 6C mismatch.

Direction: "above" means the danger zone is values ABOVE the threshold; "below" means below.
"""

THRESHOLDS = {
    # key            value   direction  human label                         where it appears
    "inverted":  {"value": 0.0,   "dir": "below", "label": "Curve inverted (10y-3m < 0)"},
    "leverage":  {"value": 110.0, "dir": "above", "label": "Household debt-to-income > 110%"},
    "house":     {"value": -2.0,  "dir": "below", "label": "Real house prices YoY < -2%"},
    "credit":    {"value": 2.5,   "dir": "above", "label": "Credit spread Baa-10yr > 2.5pp"},
    "banks":     {"value": 20.0,  "dir": "above", "label": "Banks tightening, SLOOS > 20"},
}

# Note on the two changes from the original mockup (applied consistently everywhere via this file):
#  - house moved 0 -> -2% so a near-flat dip like 2022's -0.7% is NOT flagged as a real decline.
#  - banks moved 15 -> 20 because SLOOS is very volatile and ~15 is roughly its long-run mean.


def is_danger(key, value):
    """True if `value` is in the danger zone for `key`. None/NaN -> False (no data)."""
    if value is None:
        return False
    try:
        if value != value:   # NaN
            return False
    except TypeError:
        return False
    t = THRESHOLDS[key]
    return value > t["value"] if t["dir"] == "above" else value < t["value"]
