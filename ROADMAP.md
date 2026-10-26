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
* Per-meter baseline coverage window: selection's baseline-coverage check
  measures every treatment meter against one window, by default the union of the
  treatment meters' baseline spans. In a program whose meters enroll on
  different dates, a meter whose baseline year differs from its peers' can fail
  that check; judging each meter over its own baseline span would remove the
  coupling and change which meters survive selection.
