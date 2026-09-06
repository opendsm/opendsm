# Individual meter matching

Individual meter matching pairs each treatment meter with its closest comparison
pool meters by a distance on their loadshapes. It is the most direct selection
method. Where CG clustering assigns treatment meters to clusters, matching hands
each treatment meter an explicit short list of pool meters. The union of those
lists becomes the comparison group. This page describes the distance, the two
ways matches are chosen, how duplicates and candidate pre-filtering are handled,
and how the correction treats the result.

Matching defaults to the modeled-load basis, because its goal is to reproduce the
treatment's load level rather than its model-error structure.

## Distance

Each meter is represented by its loadshape, a vector with one entry per loadshape
timestep. The distance between a treatment meter and a pool meter is a metric on
those vectors, Euclidean by default. For a treatment loadshape $t$ and a pool
loadshape $p$ the Euclidean distance is

$$
d(t, p) = \sqrt{\sum_j \left( t_j - p_j \right)^2}
$$

where $t_j$ and $p_j$ are the loadshape values at timestep $j$. The Manhattan
and cosine metrics are also available. Optional per-feature weights scale the
loadshape entries before the distance is computed, so certain hours or seasons
can be given more influence over the match. As a worked example, weighting the
summer-afternoon hours more heavily makes the match prioritize agreement during
the peak-demand window over agreement overnight.

Distances are computed in single precision and the pool is processed in chunks,
because a large pool against a large treatment group produces a distance matrix
too big to hold at full precision at once. The chunk size is
`n_pool_meters_per_chunk`, currently 10,000 pool meters per chunk.

## The two selection methods

The `selection_method` setting chooses how matches are drawn from the distances.

Minimizing meter distance is the default. Each treatment meter is assigned its
`n_matches_per_treatment` nearest pool meters, currently four. This is a direct
nearest-neighbor match, treatment meter by treatment meter. When duplicates are
disallowed the assignment is made greedily, treatment meters processed
closest-match-first so the meters with the tightest available matches claim
their neighbors before the harder-to-match meters do.

Minimizing loadshape distance instead fits the comparison group to the mean
treatment loadshape as a whole. Rather than matching each treatment meter
independently, it solves a constrained least-squares problem for a combination
of pool meters whose aggregate loadshape best approximates the scaled mean
treatment loadshape, then takes the highest-weighted pool meters as the matches.
This targets a comparison group that reproduces the treatment group's average
shape, which can differ from the set of individually nearest meters when the
treatment group is spread across loadshape space.

## Matches per treatment, duplicates, and weights

`n_matches_per_treatment` sets how many pool meters each treatment meter is
matched to. `allow_duplicate_matches` controls whether one pool meter may be
matched to more than one treatment meter. Duplicates are disallowed by default,
and duplicates are only permitted at all under the minimize-meter-distance
method, because the loadshape-distance solver assigns from a shared pool and has
no per-treatment notion of a duplicate.

When duplicates are disallowed, the requested number of matches can exceed what
the pool can supply without reuse. If a pool of $P$ meters must serve $T$
treatment meters with unique matches, the matches per treatment cannot exceed
$\lfloor P / T \rfloor$. When the request is larger it is reduced to that floor
and a warning is logged. As a worked example, a pool of 30 meters serving 8
treatment meters can supply at most $\lfloor 30 / 8 \rfloor = 3$ unique matches
per treatment, so a request for four is reduced to three.

When duplicates are allowed, a pool meter matched to several treatment meters is
carried once in the comparison group with a weight equal to the number of
treatment meters that matched it. The first occurrence of a duplicated pool
meter carries the full count and later occurrences are set to zero weight, so
the meter is counted once rather than repeated. A `duplicated` flag marks these
meters. An optional `max_distance_threshold` drops any match whose distance
exceeds the threshold after matching, which filters out matches that are nominally
nearest but still too far to be credible.

## Candidate pre-filtering

Before distances are computed against the whole pool, the pool is pre-filtered to
a candidate set when it is large. Each treatment meter contributes its own
nearest candidates, `n_matches_per_treatment` times the `candidate_multiplier`,
currently 10, and the union of those per-treatment candidates forms the pool the
match actually runs against. Taking each treatment meter's own nearest candidates
rather than a single global neighborhood preserves the true neighbors of
treatment meters that sit in different regions of loadshape space. The
pre-filter is skipped when the pool is small enough that the candidate set would
cover most of it. Setting `candidate_multiplier` to `None` disables pre-filtering
entirely.

## How the correction treats the matches

Every method returns the normalized selection the correction consumes, and
individual meter matching places all matched pool meters in a single cluster. The
correction therefore treats the union of all treatment meters' matches as one
shared comparison group, not as per-treatment groups. Each treatment
meter is corrected against that shared group. The per-treatment match lists
remain available as a diagnostic, but they do not partition the correction. This
is the practical difference from CG clustering, which is the only method that
gives each treatment meter its own comparison group.

## Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `distance_metric` | `euclidean` | metric on loadshapes |
| `selection_method` | `minimize_meter_distance` | per-meter nearest match or group loadshape fit |
| `n_matches_per_treatment` | 4 | pool meters matched to each treatment meter |
| `allow_duplicate_matches` | `False` | whether a pool meter may serve multiple treatment meters |
| `max_distance_threshold` | `None` | drop matches beyond this distance |
| `candidate_multiplier` | 10 | per-treatment candidate multiple for pre-filtering (`None` disables) |
| `n_pool_meters_per_chunk` | 10000 | pool meters per distance chunk |
