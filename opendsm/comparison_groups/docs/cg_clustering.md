# CG clustering

CG clustering is the only selection method that gives every treatment meter its
own comparison group. It groups the comparison pool into clusters by loadshape,
then fits each treatment meter a set of weights over those clusters. The
correction later draws on the clusters a treatment meter weights most heavily.
This page describes the loadshape basis the method uses, the feature pipeline
that turns raw loadshapes into cluster labels, and the per-treatment weight fit
that follows.

## The error basis and why

Every selection method builds a loadshape per meter, but the quantity that
loadshape represents differs. CG clustering defaults to the modeling-error
basis, the residual between a meter's model and its observed baseline usage. The
other three methods default to the modeled load. The default is set per method
in the selection layer and can be overridden, but the error basis is the natural
choice here for a mechanism-level reason.

The correction operates on the comparison group's own model-versus-observed
residual over the reporting period. It subtracts a scaled version of that
residual from the treatment counterfactual. The correction is therefore only as
good as the match between the comparison group's residual structure and the
treatment's. Grouping meters by how their models err, rather than by how much
they use, produces comparison groups whose residuals move together with the
treatment's. A pool meter that uses twice the energy of a treatment meter but
misfits its model in the same pattern is a better correction source than one
that matches the treatment's usage level but misfits differently. The error
basis targets that pattern directly. The matching and sampling methods instead
aim to reproduce the treatment's load level, for which the modeled load is the
natural basis.

## The feature pipeline

Clustering runs on the comparison pool's loadshapes alone. The treatment meters
are matched to the resulting clusters afterward, so no treatment data enters the
clustering itself. The steps below are applied in order to the pool loadshape
matrix, one row per pool meter, one column per loadshape timestep.

### Normalization

Each loadshape is first normalized. CG clustering uses min-max quantile
normalization with a quantile of 0.1. The 10th and 90th percentiles of a
loadshape are mapped to $-1$ and $+1$, and the rest of the shape is scaled
linearly between them:

$$
\tilde{x}_j = 2 \, \frac{x_j - q_{0.1}}{q_{0.9} - q_{0.1}} - 1
$$

where $x_j$ is the loadshape value at timestep $j$, $q_{0.1}$ and $q_{0.9}$ are
the meter's own 10th and 90th percentiles, and $\tilde{x}_j$ is the normalized
value. Using quantiles rather than the raw minimum and maximum keeps a single
spike from compressing the rest of the shape. Normalization is per meter, so a
meter that uses ten times the energy of another but follows the same daily
behavior lands in the same region of feature space. A loadshape whose 10th and
90th percentiles are within a small tolerance of each other is treated as flat
and mapped to the midpoint rather than divided by a near-zero range.

A variance cap follows normalization. Per-meter normalization can amplify noise
on a nearly flat loadshape, because dividing by a small quantile range inflates
whatever small fluctuations remain. Any meter whose post-normalization variance
exceeds the 5th percentile of the pool's variances is rescaled down to that cap.
The relative shape is preserved and only the amplitude is attenuated.

### Functional PCA

The normalized loadshapes are then reduced by functional principal component
analysis. Functional PCA fits a smooth Fourier basis to the loadshapes and keeps
the leading components. The number of components is chosen automatically as the
smallest number whose cumulative explained-variance ratio reaches the
`min_var_ratio` threshold, which CG clustering sets to 0.97. Retaining 97
percent of the variance drops the high-frequency residual that is mostly noise
while keeping the shape structure that distinguishes meters. As a worked example,
a daily loadshape of 24 hourly points might reduce to five or six functional
components that together explain 97 percent of the pool's shape variance, so the
clustering runs in six dimensions rather than twenty-four.

The wavelet transform is available in the shared clustering library but is
disabled for CG clustering.

### Bisecting k-means with scored cluster-count selection

The reduced features are clustered with bisecting k-means. Bisecting k-means
starts with all meters in one cluster and repeatedly splits a chosen cluster in
two with a k-means step, building a hierarchy of partitions. The default
bisecting strategy splits the largest cluster at each step. Each split is
reclustered several times internally and the best split kept, which reduces the
sensitivity to any single k-means initialization.

The number of clusters is not fixed. The algorithm produces a partition at each
candidate cluster count within a range, and the counts are scored so the best
one is selected. CG clustering scores with the Calinski-Harabasz index, a
variance-ratio criterion that rewards partitions whose clusters are tight
internally and well separated from each other. The count with the best score is
chosen, subject to a ceiling on the number of non-outlier clusters, currently
200.

The candidate range is derived from the pool size rather than fixed, so a small
pool is not asked to support hundreds of clusters and a large pool is not capped
too low. The lower and upper bounds come from the data size $N$, the minimum
cluster size $s_{\min}$, and configured hard bounds. The lower bound is

$$
k_{\text{low}} = \min\!\left( L, \; \max\!\left( \left\lfloor \frac{30 + 4.58 \, e^{N/335}}{s_{\min}} \right\rfloor, \; 2 \right) \right)
$$

where $L$ is the configured lower hard bound, currently 8. The upper bound
follows a saturating curve calibrated so that a pool of 1000 meters targets
about 250 clusters, capped by the configured upper hard bound $U$, currently
1500, and never exceeding the pool size. As a worked example, a pool of 600
meters with a minimum cluster size of 15 yields a lower bound near 8 and an
upper bound well below 1500, and the scored search runs over that range.

### Minimum cluster size and the outlier cluster

CG clustering sets a minimum cluster size of 15. Clusters below that size are
relabeled to $-1$, an outlier cluster, and excluded from scoring. The outlier
label is negative by design. The correction ignores every meter whose cluster
label is negative, so pool meters that fall into thin clusters are held out of
the correction rather than forming a comparison group too small to be
trustworthy. A post-clustering pass also flags points that sit far from their
cluster median along any principal component, beyond a Gaussian sigma threshold
of 3.0, and routes them to the outlier cluster as well.

## Fitting treatment meters to clusters

Clustering fixes the clusters. Each treatment meter is then assigned a weight on
each cluster, so a treatment meter can draw on more than one cluster when its
loadshape sits between them. The fit is per treatment meter, independent of
every other treatment meter.

Each cluster is first summarized by an aggregate loadshape, the median of its
member loadshapes by default. The cluster aggregates and the treatment
loadshapes are normalized the same way the pool loadshapes were. For one
treatment meter the weights $x_c$ over clusters $c$ are found by minimizing the
misfit between the treatment loadshape and the weighted sum of cluster
loadshapes:

$$
\min_{x} \; \sum_j \, W_j \left( t_j - \sum_c x_c \, \ell_{c,j} \right)^2
\qquad \text{subject to} \quad \sum_c x_c = 1, \; x_c \ge 0
$$

where $t_j$ is the treatment loadshape at timestep $j$, $\ell_{c,j}$ is cluster
$c$'s aggregate loadshape at timestep $j$, $x_c$ is the weight on cluster $c$,
and $W_j$ is a per-residual robustness weight. The weights lie on the simplex,
so they are non-negative and sum to one, and the problem is solved with
sequential least-squares programming (SLSQP).

The robustness weights $W_j$ come from an adaptive loss, which by default
behaves like a mean-absolute-error criterion rather than a plain sum of squares.
An adaptive loss downweights timesteps where the treatment loadshape is far from
any weighted cluster combination, so a few anomalous hours do not dominate the
fit. When the loss is set to a plain squared error the weights are uniform and
the objective reduces to ordinary least squares.

The optimizer is seeded by inverse-distance weights rather than started
uniformly. For a treatment meter the Euclidean distance to each cluster
aggregate is computed, the nearest cluster's distance divides every distance to
give a value in $(0, 1]$, that ratio is raised to a high power so the nearest
cluster dominates, and the result is normalized to sum to one. A treatment meter
that sits exactly on a cluster aggregate receives full initial weight on that
cluster rather than a divide-by-zero. Seeding near the nearest cluster gives the
optimizer a head start close to the likely solution.

After the fit, small weights are pruned. Any weight below
`percent_cluster_minimum`, currently $10^{-6}$, is set to zero and the remaining
weights are renormalized to sum to one. Pruning keeps a treatment meter from
carrying a long tail of negligible cluster memberships that would each pull a few
pool meters into its comparison group for no real benefit. The pruned,
renormalized weights are what the correction reads as the per-cluster treatment
weights.

## Settings

The table lists the CG clustering settings and their defaults.

| Setting | Default | Meaning |
| --- | --- | --- |
| `feature_transform.normalize.method` | `min_max_quantile` | per-meter loadshape normalization |
| `feature_transform.normalize.quantile` | 0.1 | quantile mapped to the normalized bounds |
| `feature_transform.fpca.enabled` | `True` | reduce loadshapes by functional PCA |
| `feature_transform.fpca.min_var_ratio` | 0.97 | cumulative variance the retained components must reach |
| `feature_transform.wavelet.enabled` | `False` | wavelet transform (off for CG clustering) |
| `min_cluster_size` | 15 | smallest cluster kept before relabeling to the outlier cluster |
| `small_cluster_mode` | `outlier` | relabel sub-threshold clusters to $-1$ |
| `algorithm_selection` | `bisecting_kmeans` | clustering algorithm |
| `bisecting_kmeans.bisecting_strategy` | `largest_cluster` | which cluster to split next |
| `bisecting_kmeans.n_cluster.lower` | 8 | lower hard bound on cluster count before data-derived narrowing |
| `bisecting_kmeans.n_cluster.upper` | 1500 | upper hard bound on cluster count |
| `bisecting_kmeans.scoring.weights` | `{calinski_harabasz_index: 1.0}` | cluster-count scoring criterion |
| `bisecting_kmeans.scoring.max_non_outlier_cluster_count` | 200 | ceiling on scored cluster counts |
| `treatment_match.agg_type` | `median` | how cluster loadshapes are aggregated for the fit |
| `treatment_match.adaptive_loss_alpha` | `mae` | robustness of the treatment-to-cluster fit |
| `treatment_match.percent_cluster_minimum` | $10^{-6}$ | weight below which a cluster membership is pruned |
| `seed` | 42 | random seed for the clustering |

## When to prefer CG clustering and its costs

CG clustering is the method to reach for when a per-meter comparison group
matters. It is the only method that varies the comparison group across treatment
meters, so a heterogeneous treatment group with distinct usage behaviors is
served better here than by a single shared sample. Because it matches on the
error basis, it also aligns the comparison group with the exact residual
structure the correction exploits, which is the closest available proxy for what
the correction needs.

The cost is that clustering needs a pool large enough to fill clusters. With a
minimum cluster size of 15, a pool that produces only a handful of meters per
cluster will push most of them into the outlier cluster and leave little to
correct with. A treatment meter that ends up with fewer than five usable
comparison-group meters is dropped by the correction. Clustering is also the
most involved method to reason about, since the comparison group for a treatment
meter is an implicit function of the whole pool's cluster structure and that
meter's fitted weights, rather than an explicit list of matched meters. When the
pool is small or the treatment group is uniform, a simpler method may serve as
well at lower cost.
