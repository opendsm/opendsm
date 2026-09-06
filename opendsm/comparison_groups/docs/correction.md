# The comparison-group correction

The correction is where a comparison group does its work. Selection chooses which
pool meters stand in for a treatment meter. The correction takes that comparison
group's own model error over the reporting period, scales it, and subtracts it
from the treatment counterfactual. This page derives the correction from first
principles, states the assumption each scale form rests on, and works through the
aggregation, degradation, and uncertainty exactly as the code computes them. The
math here matches the correction kernel, and the scalar per-timestep path and the
vectorized whole-window path share it.

## The generalized correction

A savings measurement subtracts observed reporting usage from a counterfactual,
the energy a building would have used without an intervention. The counterfactual
is a baseline model projected into the reporting period. That projection cannot
anticipate exogenous conditions the baseline never saw, so it reads shared
exogenous movement as savings. The correction removes that movement by borrowing
it from meters that experienced the same conditions but received no intervention.

The correction is applied per treatment meter, per comparison-group meter, at
each timestep:

$$
m_{cT} = m_T - s_{CG} \left( m_{CG} - o_{CG} \right)
$$

Here $m_T$ is the treatment meter's model value at the timestep, the uncorrected
counterfactual. $m_{CG}$ is a comparison-group meter's model value and $o_{CG}$
is its observed value at the same timestep. $s_{CG}$ is a scale factor described
below, and $m_{cT}$ is the corrected counterfactual.

The bracketed term is the comparison-group meter's own model error over the
reporting period:

$$
\Delta_{CG} = m_{CG} - o_{CG}
$$

This is the portion of the comparison meter's usage its model failed to
anticipate. If conditions suppressed usage, the model overpredicts and
$\Delta_{CG}$ is positive. Scaling that error and subtracting it from the
treatment model removes the shared exogenous movement from the treatment
counterfactual. As a worked example, suppose a comparison meter's reporting model
predicted 100 units but the meter used only 72. Then $\Delta_{CG} = 100 - 72 =
28$ units, an overprediction because conditions suppressed usage. That 28-unit
error, scaled to the treatment meter, is subtracted from the treatment
counterfactual, lowering the estimate of what the treatment building would have
used.

## Scale algorithms

The scale $s_{CG}$ rescales the comparison error to the treatment meter. Three
forms are available, set by the `algorithm` setting, each carrying its own
assumption about how the exogenous effect relates the two meters.

### Ordinary

$$
s_{CG} = 1
$$

The ordinary form subtracts the comparison error directly. Its assumption is that
the exogenous effect is additive and magnitude-comparable between the treatment
meter and the comparison meter, so a 28-unit unmodeled swing at the comparison
meter corresponds to a 28-unit swing at the treatment meter. As a worked example,
a treatment model of $m_T = 105$ corrected against $\Delta_{CG} = 28$ gives $m_{cT}
= 105 - 1 \times 28 = 77$. This holds only when the meters sit on the same usage
scale.

### Percent

$$
s_{CG} = \frac{m_T}{m_{CG}}
$$

The percent form rescales the comparison error by the ratio of the two model
magnitudes. Its assumption is that the exogenous effect scales multiplicatively
with modeled usage, so a comparison meter twice the treatment meter's size
absorbs twice the absolute swing for the same conditions. Dividing by $m_{CG}$
puts the correction on the treatment meter's scale. As a worked example, a
treatment model of $m_T = 210$ against a comparison model of $m_{CG} = 100$ gives
$s_{CG} = 2.1$, so the 28-unit comparison error becomes a $2.1 \times 28 = 58.8$
unit correction and $m_{cT} = 210 - 58.8 = 151.2$. A comparison meter with a zero
model magnitude makes the ratio undefined and is guarded to contribute no
correction rather than dividing to infinity.

### Absolute percent

$$
s_{CG} = \left| \frac{m_T}{m_{CG}} \right|
$$

The absolute-percent form is the default. It behaves like the percent form but
takes the absolute value of the ratio. Its assumption is the same multiplicative
one, with added robustness to sign for meters whose model magnitude can go
negative, such as solar meters whose net-metered load exports to the grid. When
$m_{CG}$ is negative the signed ratio flips the correction's sign, which would add
the exogenous movement to the counterfactual instead of removing it. The absolute
value keeps the scale a positive magnitude rescaling. As a worked example,
consider a solar comparison meter with $m_{CG} = -40$ and $o_{CG} = -50$, so
$\Delta_{CG} = -40 - (-50) = 10$, against a treatment model of $m_T = 210$. The
signed percent scale is $210 / -40 = -5.25$, giving a correction of $-52.5$ that
would raise the counterfactual. The absolute scale is $5.25$, giving $+52.5$ and
$m_{cT} = 210 - 52.5 = 157.5$, the intended direction.

## Correction caps

A percent scale grows without bound as the comparison model magnitude approaches
zero, so a comparison meter that barely uses any modeled energy can produce an
enormous correction. A cap bounds each per-meter correction to a multiple of the
treatment model magnitude:

$$
\left| s_{CG} \, \Delta_{CG} \right| \le \kappa \left| m_T \right|
$$

where $\kappa$ is the `correction_cap.value`, currently 3. A correction beyond
the cap is clipped to it.

The cap has two variants. The global variant applies the cap to every
comparison-group meter. The solar variant, the default, applies the cap only
where the comparison model magnitude falls below a small threshold,
`correction_cap.solar_threshold`, currently $1/3$. The solar variant targets
exactly the meters where the percent scale is unstable, the net-metered solar
meters whose model magnitude crosses zero, while leaving large-magnitude meters
uncapped. As a worked example, a treatment model of $m_T = 200$ with $\kappa = 3$
caps any single correction at $\pm 600$, so a runaway $5000$-unit correction from
a near-zero comparison model is clipped to $600$.

## Outlier rejection

Outlier rejection is available but disabled by default. When enabled, it runs per
cluster, per timestep, over the per-meter corrections before they are averaged.
The corrections may first be passed through a transform, then an outlier rule
removes corrections that sit too far from the cluster's center. The rule combines
an interquartile fence at the `quantile` setting, currently 0.25, with a sigma
threshold, `std_threshold`, currently 3.0. The transform choices are
standardize, bisymlog, Yeo-Johnson, robust Yeo-Johnson, Box-Cox, and robust
Box-Cox. After rejection the cluster weights are renormalized over the surviving
meters. When rejection drives a cluster below three finite meters, the cluster
degrades at that timestep, as described below.

## Cluster aggregation

Within a cluster the per-meter corrections are combined into one cluster
correction by a weighted average. The default weighting is by model magnitude, so
a comparison meter with a larger modeled load carries proportionally more of the
cluster correction. The raw weight for meter $i$ is

$$
w_i^{\text{raw}} = \frac{\left| m_{CG,i} \right|}{\sum_j \left| m_{CG,j} \right|}
$$

When every model magnitude in a cluster is zero the weighting falls back to
uniform.

### The weight cap and water-filling

Model-magnitude weighting can concentrate nearly all the weight on a single large
meter in a small or magnitude-skewed cluster, which would make the cluster
correction depend on one meter. A weight cap bounds that concentration. No single
meter's weight may exceed the cap $w_{\max}$, currently 0.5. Weight above the cap
is clipped to it, and the excess is redistributed proportionally over the
uncapped meters, iterating until no weight exceeds the cap. If every uncapped
meter carries zero weight the excess is spread equally. This is a water-filling
procedure. When the cap is infeasible, meaning $w_{\max}$ times the number of
valid meters is below one so no capped distribution can sum to one, the cluster
falls back to uniform weights.

The cap matters because a magnitude-skewed cluster otherwise degenerates to
single-meter matching. As a worked example, take a cluster of four meters with
model magnitudes $100$, $10$, $5$, and $5$. The raw weights are $0.833$, $0.083$,
$0.042$, and $0.042$, so the largest meter carries five-sixths of the cluster.
Capping at $0.5$ clips that meter and redistributes its $0.333$ excess over the
other three in proportion to their weights, giving roughly $0.500$, $0.249$,
$0.126$, and $0.126$. The cluster correction now draws on all four meters rather
than tracking the single largest one.

### Effective sample size

The concentration of a weight vector is measured by Kish's effective sample size:

$$
n_{\text{eff}} = \frac{1}{\sum_i w_i^2}
$$

Uniform weights over $n$ meters give $n_{\text{eff}} = n$, and weight
concentrated on one meter gives $n_{\text{eff}} = 1$. For the capped cluster
above, $n_{\text{eff}} = 1 / (0.5^2 + 0.249^2 + 0.126^2 + 0.126^2) \approx 2.9$,
while the uncapped weights give $n_{\text{eff}} \approx 1.4$. A cap of 0.5 or
below is chosen because it guarantees $n_{\text{eff}} \ge 2$ for any cluster of
two or more meters, which is the minimum a weighted spread estimate can support.

### Combining clusters

Each cluster produces a mean correction $\bar{c}_k = \sum_i w_i c_i$. The clusters
are then combined into the treatment meter's correction by their selection
weights, and the correction is subtracted from the treatment model:

$$
m_{cT} = m_T - \sum_k W_k \, \bar{c}_k
$$

where $W_k$ is the treatment meter's normalized weight on cluster $k$, from the
selection stage, renormalized over the clusters present. For a matching or
sampling method there is a single cluster and $W_k = 1$.

## Per-cluster degradation

The correction degrades per cluster, not per timestep. At each timestep a cluster
with fewer than three finite comparison-group meters is dropped, and the surviving
clusters are averaged with their weights renormalized over the survivors. A sparse
cluster therefore does not abort the timestep. A corrected value is missing only
when no cluster survives at that timestep or the treatment model itself is
non-finite.

Renormalizing over survivors is deliberate. Losing a comparison meter to missing
data should never block a treatment meter's measurement as long as some comparison
signal remains, because the measurement is what the analysis exists to produce.
The whole-window kernel diverges from the scalar path here on purpose. The scalar
path raises when a cluster has too few finite meters, while the whole-window path
drops that cluster at the affected timesteps and keeps going.

There is a floor on the whole comparison group. A treatment meter with fewer than
five comparison-group meters available is excluded and recorded on the
disqualification ledger, because a correction drawn from too few meters is too
sensitive to any one of them to be trustworthy.

## Uncertainty propagation

The uncertainty output is an honest heuristic band, not a calibrated interval,
and it should be read as such. Three constructions combine to produce it, each
carrying an approximation. The derivation below follows the code term by term.

### Per-meter correction variance

Each per-meter correction $c = s_{CG} \Delta_{CG}$ carries a variance propagated
from the treatment and comparison uncertainties. The comparison error variance
accounts for the correlation between a comparison meter's observed and model
series over the reporting period:

$$
\operatorname{Var}(\Delta_{CG}) = u_{m_{CG}}^2 + u_{o_{CG}}^2 - 2 \, \rho_{CG} \, u_{m_{CG}} \, u_{o_{CG}}
$$

where $u_{m_{CG}}$ and $u_{o_{CG}}$ are the comparison model and observed
uncertainties and $\rho_{CG}$ is their per-meter correlation over the reporting
period, passed in as `CGr_corr`. When the observed series carries no uncertainty
the expression reduces to $\operatorname{Var}(\Delta_{CG}) = u_{m_{CG}}^2$.

The scale carries variance only for the percent forms. The ordinary scale is
constant, so its variance is zero. The percent scale variance is propagated in an
absolute form that avoids dividing by the treatment model, which would be singular
when $m_T = 0$:

$$
\operatorname{Var}(s_{CG}) = \frac{u_{m_T}^2}{m_{CG}^2} + \frac{m_T^2 \, u_{m_{CG}}^2}{m_{CG}^4}
$$

where $u_{m_T}$ is the treatment model uncertainty. This is the standard ratio
propagation for $m_T / m_{CG}$ treating numerator and denominator as independent.
The absolute value in the absolute-percent scale has unit-magnitude derivative, so
it carries the same variance as the signed percent scale. A guarded zero
$m_{CG}$ contributes no scale variance.

The per-meter correction variance then combines the two, again in an absolute form
that neglects the covariance between the scale and the comparison error so that a
zero $\Delta_{CG}$ or zero scale does not divide by zero:

$$
\operatorname{Var}(c) = s_{CG}^2 \, \operatorname{Var}(\Delta_{CG}) + \Delta_{CG}^2 \, \operatorname{Var}(s_{CG})
$$

### Within-cluster uncertainty

A cluster's uncertainty combines the spread of its per-meter corrections with the
per-meter model uncertainties. The spread of the corrections around the weighted
mean is

$$
\hat{\sigma}_c^2 = \frac{\sum_i w_i \left( c_i - \bar{c} \right)^2}{1 - 1/n}
$$

for the weighted path, where $n$ is the count of finite meters in the cluster. The
unweighted path uses the population form $\hat{\sigma}_c^2 = \frac{1}{n} \sum_i
(c_i - \bar{c})^2$. That spread is scaled to a band by a t-based factor:

$$
f(\nu) = \frac{t_{1 - \alpha/2, \, \nu - 1}}{\sqrt{\nu}}
$$

where $t$ is the Student-t quantile, $\alpha$ is the significance level, currently
0.10, and $\nu$ is the effective sample size. On the weighted path $\nu$ is the
Kish effective sample size $n_{\text{eff}}$, and on the uniform path it is the
finite meter count. The aggregation uncertainty is $u_{\text{agg}} =
\hat{\sigma}_c \, f(\nu)$. The model-uncertainty term is the weighted mean of the
per-meter correction variances:

$$
\overline{u^2} = \sum_i w_i \, \operatorname{Var}(c_i)
$$

and the cluster uncertainty is the quadrature sum of the two:

$$
u_{\text{cl}} = \sqrt{u_{\text{agg}}^2 + \overline{u^2}}
$$

When the Kish effective sample size falls below two the point correction stays
weighted, but the spread and the t-factor revert to uniform weights over the
finite meters, because an effective sample size below two cannot support a
weighted interval. When fewer than two finite meters remain the cluster carries no
uncertainty and its band is dropped.

### Combining across clusters and meters

The clusters are combined into the treatment meter's corrected uncertainty by a
quadrature over the surviving clusters, with the treatment model uncertainty
carried alongside:

$$
u_{cT} = \sqrt{u_{m_T}^2 + \sum_k W_k^2 \, u_{\text{cl},k}^2}
$$

using the same survivor-renormalized weights $W_k$ as the point correction. A
surviving cluster whose uncertainty is non-finite stays in the point correction
but is omitted from the band, which understates it. The quadrature treats the
comparison-group meters as independent, so the result is a heuristic band rather
than a calibrated interval and should be read as a relative indicator.

## Assumptions

The correction removes the exogenous effect only when its assumptions hold.
Collected in one place, they are the following.

The comparison group experiences the same exogenous conditions as the treatment
group. If the comparison meters see different weather, a different local economy,
or a different shock, their model error does not stand in for the treatment's.

The exogenous response is captured by the chosen scale form. The ordinary form
assumes an additive, magnitude-comparable effect. The percent forms assume a
multiplicative effect that scales with modeled usage. A mismatch between the true
response and the chosen form leaves residual exogenous movement in the
counterfactual.

The comparison meters received no intervention. A comparison meter that also
underwent an intervention carries a treatment effect in its model error, which the
correction would subtract from the treatment counterfactual as if it were
exogenous.

The baseline models are of comparable quality across the two groups. The
correction differences the comparison group's model error against the treatment's,
so a systematic difference in model quality between the groups enters the
correction as if it were an exogenous signal.

## Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `algorithm` | `absolute_percent_difference_in_differences` | scale form (`None` disables the correction) |
| `weight_cluster_aggregation` | `model_magnitude` | within-cluster weighting (`None` for uniform) |
| `weight_cap` | 0.5 | upper bound on any single meter's within-cluster weight |
| `outlier_rejection.enabled` | `False` | reject per-cluster correction outliers |
| `outlier_rejection.transform` | `None` | transform applied before outlier rejection |
| `outlier_rejection.std_threshold` | 3.0 | sigma threshold for outlier rejection |
| `outlier_rejection.quantile` | 0.25 | interquartile fence for outlier rejection |
| `correction_cap.enabled` | `True` | cap each correction to a multiple of the treatment model |
| `correction_cap.type` | `solar` | cap everywhere (`global`) or only near-zero comparison models (`solar`) |
| `correction_cap.value` | 3.0 | cap multiple $\kappa$ of the treatment model magnitude |
| `correction_cap.solar_threshold` | $1/3$ | comparison model magnitude below which the solar cap applies |
| `alpha` | 0.10 | significance level for the uncertainty bands |
| `min_window_coverage` | 0.9 | minimum reporting-window coverage a meter needs to be corrected |

## Calibrated limits

The uncertainty bands are not yet calibrated to a stated coverage level.
Calibration is an open area, and the bands should be read as relative indicators
rather than exact confidence statements. Two mechanisms push the band away from a
true interval. The quadrature treats the comparison-group meters as independent,
so exogenous residuals they share make the band understate. In the other
direction, a small cluster whose weights
concentrate falls back to a uniform-weight spread over few meters, which can
inflate a cluster's band when its effective sample size is near the floor. The net
band is a useful quantification of what would otherwise go unquantified, but it is
a heuristic, and it may benefit from additional research toward calibration.
