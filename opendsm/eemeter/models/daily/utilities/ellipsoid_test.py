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
from scipy.ndimage import median_filter
from scipy.optimize import minimize_scalar


def ellipsoid_intersection_test(mu_A, mu_B, cov_A, cov_B):
    """
    Tests whether two ellipsoids intersect or not. The ellipsoids are defined by their mean vectors and covariance matrices.
    The function uses the K-function to calculate the intersection of the ellipsoids. If the K-function is greater than or equal to 0,
    then the ellipsoids intersect, otherwise they do not.

    Parameters:
    mu_A (numpy.ndarray): Mean vector of the first ellipsoid.
    mu_B (numpy.ndarray): Mean vector of the second ellipsoid.
    cov_A (numpy.ndarray): Covariance matrix of the first ellipsoid.
    cov_B (numpy.ndarray): Covariance matrix of the second ellipsoid.

    Returns:
    bool: True if the ellipsoids intersect, False otherwise.
    """

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

    # size could be changed with justification
    y = median_filter(y[idx_sorted], size=5)[idx_original]

    # Computing the covariance and ellipse parameter values
    cov = np.cov(x, y) * var  # scale covariances by std choice
    ab_sqr, v = np.linalg.eig(cov)
    [a, b] = np.sqrt(ab_sqr)
    phi = np.arctan2(*v[:, 0][::-1])

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

    cluster_ellipse = {}
    for season in ["summer", "shoulder", "winter"]:
        for day_type, day_num in enumerate([[1, 2, 3, 4, 5], [6, 7]]):
            if day_type == 0:
                key = f"wd-{season[:2]}"
            else:
                key = f"we-{season[:2]}"

            meter_season = meter[
                (meter["season"] == season) & (meter["observed"].notna())
            ]
            meter_season = meter_season[meter_season["day_of_week"].isin(day_num)]
            meter_season = meter_season.sort_values(by=["temperature"])

            T = meter_season["temperature"].values
            obs = meter_season["observed"].values

            if (len(T) < 3) or (len(obs) < 3):
                mu = cov = a = b = phi = None
            else:
                mu, cov, a, b, phi = robust_confidence_ellipse(
                    T, obs, var, outlier_std=3.6, N=3
                )
                # mu, cov, a, b, phi = confidence_ellipse(T, obs, std_sqr)

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
                    mu_A = cluster_ellipse[combo[0]]["mu"]
                    cov_A = cluster_ellipse[combo[0]]["cov"]
                    mu_B = cluster_ellipse[combo[1]]["mu"]
                    cov_B = cluster_ellipse[combo[1]]["cov"]

                    if all([coef is not None for coef in [mu_A, mu_B, cov_A, cov_B]]):
                        ellipse_overlap[combo_str] = ellipsoid_intersection_test(
                            mu_A, mu_B, cov_A, cov_B
                        )
                    else:
                        ellipse_overlap[combo_str] = False

                if not ellipse_overlap[combo_str]:
                    allow_separate[key][i] = True
                    break

        allow_separate[key] = all(allow_separate[key])

    return allow_separate
