#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2014-2025 OpenDSM contributors
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


def _rolling_median5(y):
    """Fast sliding-window median with window=5 for small numpy arrays.

    Replaces scipy.ndimage.median_filter for 1-D arrays, avoiding the overhead
    of the ndimage dispatch path.  Edge elements are left unchanged (same
    behaviour as median_filter with default reflect padding gives for size=5
    when the signal is long enough).
    """
    n = len(y)
    if n < 5:
        return y.copy()
    out = y.copy()
    # Use np.partition for O(n) per window rather than full sort
    for i in range(2, n - 2):
        window = y[i - 2 : i + 3]
        # partition so that index 2 holds the median
        out[i] = np.partition(window, 2)[2]
    return out


def ellipsoid_intersection_test(mu_A, mu_B, cov_A, cov_B, a_A=None, b_A=None, a_B=None, b_B=None):
    """
    Tests whether two ellipsoids intersect or not. The ellipsoids are defined by their mean vectors and covariance matrices.
    The function uses the K-function to calculate the intersection of the ellipsoids. If the K-function is greater than or equal to 0,
    then the ellipsoids intersect, otherwise they do not.

    Parameters:
    mu_A (numpy.ndarray): Mean vector of the first ellipsoid.
    mu_B (numpy.ndarray): Mean vector of the second ellipsoid.
    cov_A (numpy.ndarray): Covariance matrix of the first ellipsoid.
    cov_B (numpy.ndarray): Covariance matrix of the second ellipsoid.
    a_A, b_A (float, optional): Semi-axes of ellipsoid A for fast bbox pre-filter.
    a_B, b_B (float, optional): Semi-axes of ellipsoid B for fast bbox pre-filter.

    Returns:
    bool: True if the ellipsoids intersect, False otherwise.
    """

    # --- Cheap bounding-box pre-filter (avoids eigh + minimize_scalar) ---
    # If the Euclidean distance between centres exceeds the sum of the
    # major semi-axes in each dimension the ellipsoids cannot overlap.
    if (a_A is not None) and (a_B is not None) and (b_A is not None) and (b_B is not None):
        d2 = np.sum((mu_A - mu_B) ** 2)
        # Conservative bound: if d > sum of the *largest* semi-axis of each ellipse
        max_r_A = max(abs(a_A), abs(b_A))
        max_r_B = max(abs(a_B), abs(b_B))
        if d2 > (max_r_A + max_r_B) ** 2:
            return False

    # Fix if all values are the same in 1 direction, "brent" doesn't work well with this
    if cov_A[1, 1] == 0:
        cov_A[1, 1] = 1e-14

    if cov_B[1, 1] == 0:
        cov_B[1, 1] = 1e-14

    lambdas, phi = eigh(cov_A, b=cov_B)
    v_squared = np.dot(phi.T, mu_A - mu_B) ** 2

    res = minimize_scalar(
        ellipsoid_K_function,
        #   bracket = [0.0, 0.5, 1.0],
        bounds=[0.0, 1.0],
        args=(lambdas, v_squared),
        method="bounded",
    )

    if res.fun[0] >= 0:
        return True
    return False


def ellipsoid_K_function(ss, lambdas, v_squared):
    """
    The K-function is a measure of spatial point pattern, often used in spatial statistics
    to analyze the clustering or dispersion of points in a dataset. The formula used in this
    code is a specific calculation for an ellipsoid.

    Parameters:
    ss (float): A scalar value between 0 and 1.
    lambdas (numpy.ndarray): A 1D numpy array of eigenvalues of the covariance matrix.
    v_squared (numpy.ndarray): A 1D numpy array of squared differences between the means of two ellipsoids.

    Returns:
    float: The value of the K-function for the given input values.
    """
    ss = np.array(ss).reshape((-1, 1))
    lambdas = np.array(lambdas).reshape((1, -1))
    v_squared = np.array(v_squared).reshape((1, -1))

    return 1 - np.sum(v_squared * ((ss * (1 - ss)) / (1 + ss * (lambdas - 1))), axis=1)


def confidence_ellipse(x, y, var=np.ones([2, 2]) * 1.96):
    """
    Compute the confidence ellipse for a 2D dataset.

    Parameters:
    x (numpy.ndarray): The x-coordinates of the data points.
    y (numpy.ndarray): The y-coordinates of the data points.
    var (numpy.ndarray): The variance of the data points. Default is 1.96.

    Returns:
    list: A list containing the mean, covariance, major and minor axis lengths, and rotation angle of the ellipse.

    """

    # Applying a median filter to help with outliers
    idx_sorted = np.argsort(x).flatten()
    idx_original = np.argsort(idx_sorted).flatten()

    # size could be changed with justification — use inline numpy rolling median
    # instead of scipy.ndimage.median_filter to avoid dispatch overhead
    y = _rolling_median5(y[idx_sorted])[idx_original]

    # Computing the covariance and ellipse parameter values
    cov = np.cov(x, y) * var  # scale covariances by std choice

    # Analytical 2×2 symmetric eigendecomposition — avoids np.linalg.eig
    # (general solver) for the common real symmetric 2×2 case.
    # Eigenvalue ordering matches the original np.linalg.eig convention used
    # by this function: a = smaller semi-axis, b = larger semi-axis.
    ca, cb, cd = cov[0, 0], cov[0, 1], cov[1, 1]
    tr_half = (ca + cd) * 0.5
    disc = np.sqrt(max(((ca - cd) * 0.5) ** 2 + cb ** 2, 0.0))
    lam1 = tr_half + disc   # larger eigenvalue
    lam2 = tr_half - disc   # smaller eigenvalue
    # Clamp negative eigenvalues to zero before sqrt (rounding artefacts)
    a = np.sqrt(max(lam2, 0.0))   # smaller semi-axis
    b = np.sqrt(max(lam1, 0.0))   # larger semi-axis
    # Eigenvector angle for the eigenvalue corresponding to 'a' (lam2)
    if cb != 0.0:
        phi = np.arctan2(lam2 - ca, cb)
    else:
        phi = 0.0 if ca <= cd else np.pi / 2.0

    mu = np.array([np.mean(x), np.mean(y)])

    return mu, cov, a, b, phi


def robust_confidence_ellipse(x, y, var=np.ones([2, 2]) * 1.96, outlier_std=3, N=3):
    """
    Computes a robust confidence ellipse for a set of points.

    Parameters:
    x (numpy.ndarray): Array of x-coordinates of the points.
    y (numpy.ndarray): Array of y-coordinates of the points.
    var (numpy.ndarray): Variance-covariance matrix. Default is a 2x2 matrix with 1.96 in the diagonal.
    outlier_std (float): Standard deviation for outlier detection. Default is 3.
    N (int): Number of iterations for outlier removal. Default is 3.

    Returns:
    list: A list containing the mean, covariance matrix, major and minor axis lengths, and rotation angle of the ellipse.
    """

    var_outlier = np.ones([2, 2]) * outlier_std**2

    # remove outliers in N iterations
    for n in range(N):
        if len(x) <= 1 or np.all(x == x[0]) or np.all(y == y[0]):
            break

        mu, cov, a, b, phi = confidence_ellipse(x, y, var_outlier)

        if a == 0 or b == 0:
            break

        # Center points
        xc = x - mu[0]
        yc = y - mu[1]

        # Rotate points so ellipse is aligned with axes
        phi *= -1
        xct = xc * np.cos(phi) - yc * np.sin(phi)
        yct = xc * np.sin(phi) + yc * np.cos(phi)

        # normalize to a circle of radius 1
        r = (xct / a) ** 2 + (yct / b) ** 2

        idx = np.argwhere(r <= 1).flatten()  # non-outlier points

        # if all outliers, break
        if len(idx) < 3:
            break

        # if no outliers, break
        if len(idx) == len(x):
            break

        x = x[idx]
        y = y[idx]

    if (len(x) < 3) or np.all(x == x[0]) or np.all(y == y[0]):
        mu = cov = a = b = phi = None
        return [mu, cov, a, b, phi]

    return confidence_ellipse(x, y, var)


def ellipsoid_split_filter(meter, n_std=[1.4, 1.4]):
    """
    Filters a set of points based on a robust confidence ellipse. The points are split into groups using robust ellipses computed
    and then tested for intersection. This determines whether separate keys are needed for different seasons and day types.

    Parameters:
    meter (pandas.DataFrame): Dataframe containing the points to be filtered.
    n_std (float or list): Standard deviation for outlier detection. Default is [1.4, 1.4].

    Returns:
    dict: A dictionary containing the filtered points for each season and day type.
    """

    if isinstance(n_std, float):
        var = np.ones([2, 2]) * n_std**2
    else:
        std = np.array(n_std)[:, None]
        var = std.T * std

    # Pre-extract numpy arrays once — avoids 6× repeated pandas boolean
    # indexing and sort_values on the full DataFrame inside the inner loop.
    arr_season = meter["season"].to_numpy()
    arr_day = meter["weekday_weekend"].to_numpy()
    arr_obs = meter["observed"].to_numpy()
    arr_temp = meter["temperature"].to_numpy()
    arr_valid = ~np.isnan(arr_obs)

    # Precompute day-type masks (constant across the season loop)
    wd_mask = arr_day == "weekday"
    we_mask = arr_day == "weekend"

    cluster_ellipse = {}
    for season in ["summer", "shoulder", "winter"]:
        mask_season = arr_season == season
        for day_mask, key in [(wd_mask, f"wd-{season[:2]}"), (we_mask, f"we-{season[:2]}")]:
            mask = mask_season & day_mask & arr_valid
            T = arr_temp[mask]
            obs = arr_obs[mask]

            if len(T) < 3 or len(obs) < 3:
                mu = cov = a = b = phi = None
            else:
                # Sort by temperature (replaces sort_values)
                idx_sorted = np.argsort(T)
                T = T[idx_sorted]
                obs = obs[idx_sorted]
                mu, cov, a, b, phi = robust_confidence_ellipse(
                    T, obs, var, outlier_std=3.6, N=3
                )

            cluster_ellipse[key] = {"mu": mu, "cov": cov, "a": a, "b": b, "phi": phi}

    combos = {
        "summer": [
            [["wd-su", "wd-sh"], ["we-su", "we-sh"]],
            [["wd-su", "wd-wi"], ["we-su", "we-wi"]],
        ],
        "shoulder": [
            [["wd-su", "wd-sh"], ["we-su", "we-sh"]],
            [["wd-sh", "wd-wi"], ["we-sh", "we-wi"]],
        ],
        "winter": [
            [["wd-sh", "wd-wi"], ["we-sh", "we-wi"]],
            [["wd-su", "wd-wi"], ["we-su", "we-wi"]],
        ],
        "weekday_weekend": [
            [["wd-su", "we-su"], ["wd-sh", "we-sh"], ["wd-wi", "we-wi"]]
        ],
    }

    ellipse_overlap = {}
    allow_separate = {
        "summer": [False, False],
        "shoulder": [False, False],
        "winter": [False, False],
        "weekday_weekend": [False],
    }
    for key in allow_separate.keys():
        for i, season_wd_we in enumerate(combos[key]):
            for combo in season_wd_we:
                combo_str = "__".join(combo)

                if combo_str not in ellipse_overlap:
                    ea = cluster_ellipse[combo[0]]
                    eb = cluster_ellipse[combo[1]]
                    mu_A, cov_A = ea["mu"], ea["cov"]
                    mu_B, cov_B = eb["mu"], eb["cov"]

                    if all([coef is not None for coef in [mu_A, mu_B, cov_A, cov_B]]):
                        ellipse_overlap[combo_str] = ellipsoid_intersection_test(
                            mu_A, mu_B, cov_A, cov_B,
                            a_A=ea["a"], b_A=ea["b"],
                            a_B=eb["a"], b_B=eb["b"],
                        )
                    else:
                        ellipse_overlap[combo_str] = False

                if not ellipse_overlap[combo_str]:
                    allow_separate[key][i] = True
                    break

        allow_separate[key] = all(allow_separate[key])

    return allow_separate
