# Clustering improvements — planned work

This branch is the staging area for clustering algorithm improvements that
go beyond the extraction work in PRs against the local opendsm-integration
plan. Land items here as they're designed and implemented, then open
upstream PRs once each piece is independently shippable.

## spectral_divisive stabilization

Two related cross-platform issues surfaced during the clustering reorg
(#586) CI. Both were worked around in #586 rather than fixed:

### 1. Degenerate eigenspace on pathologically separated data

With `sep/std > 50`, inter-cluster affinity underflows to 0 in float32,
the top-k Laplacian eigenvalues collapse within ULP, and different BLAS
implementations pick different orthonormal bases. The score function
then reads them as different *confident* winners (k=2/3/4 across
Windows/Linux/macOS for the same input).

Workaround in #586: changed the test fixture to a non-degenerate regime
(`sep/std ≈ 5`, `n_per_cluster ≥ 50`). The original pathological
fixtures (`sep/std=67`, `n=80, center_box=(-8, 8)`) couldn't be
restored as test inputs without algorithmic stabilization.

Proposed fix: detect degenerate eigenspace via eigenvalue spacing and
bypass score-based selection in favour of eigengap when the top
eigenvalues are within tolerance of each other.

### 2. `_power_iteration_fiedler` non-convergence on macOS py3.11

Returns `lambda2 = 0.0` on macOS py3.11 for a connected graph where
`scipy.sparse.linalg.eigsh` correctly returns `~0.19`. Indicates the
power iteration fails to converge or fails to deflate the trivial
eigenvector on Apple Accelerate.

Proposed fix: add convergence guards and better initialization to
`_power_iteration_fiedler`.

### Done when

Both pathological fixtures (`sep/std=67`, `n=80, center_box=(-8, 8)`)
can be restored as test inputs and the score-based k-selection
produces the same winner across Linux/macOS/Windows runners.
