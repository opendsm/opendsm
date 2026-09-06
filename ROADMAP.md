Roadmap
=======

Potential and planned improvements to OpenDSM. This is the one document allowed to
reference prior states; completed items move out once shipped.

## Comparison groups

* Unify daily and billing per-point prediction uncertainty with the hourly path's
  ASHRAE aggregate-share construction, so a daily/billing period's `savings_unc`
  reconstructs the aggregate band exactly instead of quadrature-summing t-scaled
  prediction-interval bands. This is an eemeter-side change (the daily/billing
  models produce the per-point band that the comparison-groups correction and
  savings layers combine).
* Empirical coverage validation: check correction and savings uncertainty bands
  against held-out reporting observations to quantify how far the heuristic
  bands sit from actual coverage.
* Per-meter coverage window: the reporting-coverage prune measures every meter
  against the treatment population's group window. A meter's own reporting span
  would be the more natural denominator for a program whose meters enroll on
  different dates, and would change which meters survive the prune.
