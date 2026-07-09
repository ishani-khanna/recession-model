"""Signal-2 credit-feed fallback — WIRING ONLY, not part of the authored engine.

Prefer the authored Moody's Baa-Aa (DataBuffet ``IRAACM.IUSA``). When the DataBuffet
keys are absent -- the current deployed / weekly configuration -- fall back to the FRED
Aaa substitute (Baa-Aaa) so the full three-signal framing still holds, exactly as the
site behaves today. Import this module for its side effects BEFORE calling
``reporting.two_clock_dashboard`` or ``validation.write_report``.

The authored engine files (credit_spread / reporting / validation) are untouched; this
only rebinds the ``load_baa_aa`` name they look up at call time. The real Baa-Aa path
reactivates automatically once the DataBuffet keys are added -- no code change needed.
"""

from yield_curve import reporting
from yield_curve.models import credit_spread

# Capture the authored function first so the fallback can call it without recursing.
_orig_load_baa_aa = credit_spread.load_baa_aa


def load_baa_aa_with_fallback():
    try:
        return _orig_load_baa_aa()               # Moody's Seasoned Aa via DataBuffet (keys present)
    except Exception:
        return credit_spread.load_baa_aaa()      # FRED Aaa substitute (Baa - Aaa), DataBuffet off


# validation.write_report calls ``credit_spread.load_baa_aa`` (module attribute);
# reporting.two_clock_dashboard uses its own imported binding. Patch both.
credit_spread.load_baa_aa = load_baa_aa_with_fallback
reporting.load_baa_aa = load_baa_aa_with_fallback
