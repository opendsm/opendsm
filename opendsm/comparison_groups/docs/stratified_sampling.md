# Stratified sampling

Stratified sampling draws a pool sample balanced against the treatment
distribution over supplied stratification features. It does not match meters
individually or cluster them. It divides both populations into bins along a few
features, then samples the pool bin by bin so the sampled comparison group
mirrors the treatment group's shape along those features. The result is one
shared comparison group used for every treatment meter.

Stratified sampling defaults to the modeled-load basis when a loadshape is
needed, and unlike the other methods it requires a feature frame on each
population beyond the loadshape.

## Stratification features and the feature frame

Stratification runs on a feature frame, not on the loadshape. Each population
must carry a `features` frame, one row per meter, and each stratification column
names a column in that frame. The default columns are `summer_usage` and
`winter_usage`, so a meter is placed by its warm-season and cold-season usage.
Selection raises if stratified sampling is requested without a feature frame on
the population, because there is nothing to stratify on. At most three
stratification columns may be used.

Each stratification column carries its own binning controls. `min_value_allowed`
and `max_value_allowed`, currently 3000 and 6000, bound the treatment values
used to construct the bins, so extreme treatment meters do not stretch the bin
edges. `is_fixed_width` chooses between fixed-width bins, equal spans across the
value range, and fixed-proportion bins, equal counts per bin. Fixed-proportion
bins are the default, so each bin holds a comparable number of treatment meters
rather than a comparable value span.

## Fixed-bin and equivalence-driven variants

The method has two variants, distinguished by whether the bin counts are fixed in
advance or searched for.

The fixed-bin variant sets a bin count per column directly, `n_bins`, currently
8 per column. No search is run. The treatment group is fit into the bins and the
pool is sampled to match the resulting bin proportions. This is the variant to
use when the right binning is known.

The equivalence-driven variant, distance-stratified sampling, is the default for
this selection method. It does not fix the bin counts. Instead it searches over
bin-count configurations and keeps the one that makes the sampled comparison
group most equivalent to the treatment group. Equivalence is measured on the
loadshapes rather than the stratification features, so the search balances the
sampled group against the treatment group's actual usage behavior, not only the
few features the bins are drawn on. The bin-count search runs a grid over each
column's bin count between `min_n_bins` and `max_n_bins`, currently 1 and 8, and
selects the configuration with the smallest equivalence distance.

## The equivalence distance

Equivalence between two groups is measured quantile by quantile. Both groups are
cut into quantiles, with the number of quantiles chosen so each treatment
quantile holds about `equivalence_quantile` meters, currently 25. Within each
quantile the mean of each feature is computed, and the per-quantile means of the
two groups are compared. The per-quantile distances are summed across all
quantiles and features.

Two distance metrics are available. The Euclidean distance sums the squared
differences of the quantile means. The chi-square distance, the default for
distance-stratified sampling, weights each quantile's squared difference by the
combined magnitude:

$$
d_{\chi^2} = \sum_{q} \frac{\left( \bar{x}_q - \bar{y}_q \right)^2}{\bar{x}_q + \bar{y}_q}
$$

where $\bar{x}_q$ is the treatment group's mean in quantile $q$, $\bar{y}_q$ is
the comparison group's mean in quantile $q$, and the sum runs over quantiles and
features. Quantiles where both means are zero contribute nothing, which avoids a
zero-over-zero term. The chi-square form makes a fixed absolute gap count for
more in a low-magnitude quantile than in a high-magnitude one, so the balance is
judged in relative terms. As a worked example, a treatment group of 1000 meters
with an `equivalence_quantile` of 100 is cut into ten quantiles, ten per-quantile
distances are computed per feature, and their sum is the equivalence distance
the bin search minimizes.

## Sampling ratios

Two controls govern how much of the pool is sampled. `n_samples_approx` is the
approximate total number of pool meters to sample, approximate because the count
is adjusted per bin to hold the target proportions. It defaults to 5000 for the
distance-stratified variant and is unset for the fixed-bin variant, where an
unset value takes as many samples as the bins allow.
`min_n_sampled_to_n_treatment_ratio` sets a floor on how many sampled meters each
bin must carry per treatment meter in that bin, currently 0.25 for the
distance-stratified variant and 4 for the fixed-bin variant.

`relax_n_samples_approx_constraint` decides what happens when the pool cannot
supply the requested sample. When it is true, the requested count is treated as
an upper bound and the method takes what is available. When it is false, an
insufficient pool raises rather than returning a thin sample. The
distance-stratified variant relaxes the constraint by default, and during the
bin-count search a configuration whose sampled-to-treatment ratio falls below the
floor is disqualified along with every configuration that uses at least as many
bins in every column.

## Settings

The distance-stratified variant is the default. Its fields and defaults are
below.

| Setting | Default | Meaning |
| --- | --- | --- |
| `stratification_column` | `[summer_usage, winter_usage]` | feature-frame columns to stratify on (max 3) |
| `equivalence_method` | `chisquare` | quantile distance metric (`euclidean` or `chisquare`) |
| `equivalence_quantile` | 25 | target treatment meters per equivalence quantile |
| `min_n_bins` | 1 | smallest per-column bin count in the search |
| `max_n_bins` | 8 | largest per-column bin count in the search |
| `n_samples_approx` | 5000 | approximate total pool meters to sample |
| `relax_n_samples_approx_constraint` | `True` | treat the sample count as an upper bound rather than raising |
| `min_n_sampled_to_n_treatment_ratio` | 0.25 | floor on sampled meters per treatment meter per bin |
| `min_n_treatment_per_bin` | 0 | treatment meters a fixed-width bin needs to count as non-outlier |
| `seed` | `None` | sampling seed (`None` draws fresh entropy) |

The fixed-bin variant replaces the equivalence controls with a per-column
`n_bins`, currently 8, sets `n_samples_approx` unset, and raises to a
`min_n_sampled_to_n_treatment_ratio` of 4.
