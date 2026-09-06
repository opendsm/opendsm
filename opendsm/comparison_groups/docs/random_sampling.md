# Random sampling

Random sampling draws a comparison group from the pool without reference to the
treatment meters. It is the simplest selection method. No distance, no
stratification, no clustering. A fixed number of pool meters is drawn at random,
and that single sample serves as the shared comparison group for every treatment
meter.

Random sampling uses the modeled-load basis, but the basis matters little here
because the sample does not depend on any loadshape comparison.

## How many meters are drawn

Two settings control the sample size, and exactly one of them must be set.
`n_meters_total` fixes the total number of pool meters to draw, independent of
how many treatment meters there are. `n_meters_per_treatment`, currently 4,
instead scales the sample with the treatment group, drawing that many meters per
treatment meter. With eight treatment meters and `n_meters_per_treatment` of
four, the sample is $8 \times 4 = 32$ pool meters. Setting both at once raises,
and so does setting neither, because the sample size would be undefined.

The draw is a uniform random sample of pool meters without replacement, so a
sampled meter appears once. Every drawn meter is placed in a single cluster and
given equal weight, which is what makes the sample a single shared comparison
group rather than a per-treatment one.

## Seeding

The `seed` setting fixes the random draw. With a seed set, the same pool produces
the same sample on every run, which is what makes a random comparison group
reproducible and lets a selection be serialized and reused. Leaving the seed
unset draws fresh entropy each run, so the sample changes from run to run. A seed
should be set whenever a run needs to be repeated or audited.

## When a random group is appropriate

Random sampling is appropriate in two situations. The first is as a baseline for
comparison. A random comparison group carries no selection structure, so
comparing a structured method against a random one shows how much the structure
is actually buying. If a clustered or matched comparison group barely improves on
a random draw, the added complexity is not earning its keep on that data.

The second is a very homogeneous pool. When the pool meters resemble each other
closely, a targeted selection has little to target, because any subset looks much
like any other. A random draw then produces a comparison group about as
representative as a matched one at a fraction of the effort. The more the pool
varies, the more a structured method earns over a random draw, and the weaker the
case for random sampling becomes.

## Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `n_meters_total` | `None` | total pool meters to draw (mutually exclusive with the next) |
| `n_meters_per_treatment` | 4 | pool meters to draw per treatment meter |
| `seed` | `None` | sampling seed (`None` draws fresh entropy) |
