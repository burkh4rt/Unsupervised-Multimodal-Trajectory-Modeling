#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implements a state space model with linear Gaussian state transitions and
a kNN-based measurement model
"""

import pickle

import numpy as np
import scipy.stats as sp_stats
import sklearn.model_selection as skl_ms
import sklearn.neighbors as skl_neighbors

from ..util import util_state_space as util
from . import state_space_model as ssm

np_eps = np.finfo(float).eps


class StateSpaceHybrid(ssm.StateSpaceModel):
    """State space model where the state transition model is linear, Gaussian
    and the measurement model is estimated with as Gaussian with a
    cross-validated k-NN mean and homoskedastic covariance;
    allows for a non-linear relationship between the states and measurements
    """

    def __init__(
        self,
        *,
        n_neighbors: int | list = 10,
        n_folds: int = 3,
        alpha: float = 0.0,
    ):
        """instantiates model with hyperparameters as specified

        Parameters
        ----------
        n_neighbors
            number of neighbors to use for the k-NN models, or list thereof
        n_folds
            number of cross-validation folds to use
        alpha
            regularisation for linear model

        """
        super().__init__()
        self.n_neighbors = (
            n_neighbors if isinstance(n_neighbors, list) else [n_neighbors]
        )
        self.n_folds = n_folds
        self.alpha = alpha if alpha > 2 * np_eps else 0

    def __str__(self):
        return (
            "State space model with linear state model and "
            f"k={self.n_neighbors}-NN-based measurement model "
            f"({self.n_folds=}, {self.alpha=})"
        )

    def fit(self, data: tuple[np.ndarray, np.ndarray]):
        self.data = tuple(map(np.atleast_3d, data))
        states, measurements = data

        self.state_init = {
            "mean": np.nanmean(states[0], axis=0),
            "cov": np.cov(
                util.take_finite_along_axis(states[0]), rowvar=False
            ),
        }

        self.state_model = dict(
            zip(
                ["coeff", "covar"],
                util.regress_alpha(
                    np.vstack(states[:-1]),
                    np.vstack(states[1:]),
                    self.alpha,
                )
                if self.alpha > 2 * np_eps
                else util.regress(
                    np.vstack(states[:-1]), np.vstack(states[1:])
                ),
            )
        )

        outp = np.vstack(measurements[:])
        inp = np.vstack(states[:])
        meas_idx = np.isfinite(np.column_stack([inp, outp])).all(axis=1)
        out_mdl = skl_ms.GridSearchCV(
            skl_neighbors.KNeighborsRegressor(),
            param_grid={"n_neighbors": self.n_neighbors},
            cv=self.n_folds,
            scoring="neg_mean_squared_error",
        )
        out_mdl.fit(inp[meas_idx], outp[meas_idx])
        out_inf = out_mdl.predict(inp[meas_idx])

        self.measurement_model = {
            "mean": skl_neighbors.KNeighborsRegressor(
                n_neighbors=out_mdl.best_params_["n_neighbors"]
            ).fit(inp[meas_idx], out_inf),
            "cov": np.cov(outp[meas_idx] - out_inf, rowvar=False),
        }

        return self

    def to_pickle(self) -> bytes:
        return pickle.dumps(
            {
                "n_folds": self.n_folds,
                "n_neighbors": self.n_neighbors,
                "data_hash": self.data_hash,
                "state_init": self.state_init,
                "state_model": self.state_model,
                "measurement_model": self.measurement_model,
                "alpha": self.alpha,
            }
        )

    def from_pickle(self, p: bytes):
        pickle_dict = pickle.loads(p)
        self.n_folds = pickle_dict["n_folds"]
        self.n_neighbors = pickle_dict["n_neighbors"]
        self.data_hash = pickle_dict["data_hash"]
        self.state_init = pickle_dict["state_init"]
        self.state_model = pickle_dict["state_model"]
        self.measurement_model = pickle_dict["measurement_model"]
        self.alpha = pickle_dict["alpha"] if "alpha" in pickle_dict else 0
        return self

    def score(self, data: tuple[np.ndarray, np.ndarray]):
        if data is None:
            data = self.data
        states, measurements = data
        T = states.shape[0]
        log_likelihoods = sp_stats.multivariate_normal(
            mean=self.state_init["mean"],
            cov=self.state_init["cov"],
            allow_singular=True,
        ).logpdf(states[0])
        for t in range(T - 1):
            states0, states1 = states[t], states[t + 1]
            idx_fin = np.isfinite(np.column_stack([states0, states1])).all(
                axis=1
            )
            log_likelihoods[idx_fin] += sp_stats.multivariate_normal(
                cov=self.state_model["covar"], allow_singular=True
            ).logpdf(
                states1[idx_fin] - states0[idx_fin] @ self.state_model["coeff"]
            )
        for t in range(T):
            states0, meas0 = states[t], measurements[t]
            idx_fin = np.isfinite(np.column_stack([states0, meas0])).all(
                axis=1
            )
            log_likelihoods[idx_fin] += sp_stats.multivariate_normal(
                cov=self.measurement_model["cov"], allow_singular=True
            ).logpdf(
                meas0[idx_fin]
                - self.measurement_model["mean"].predict(states0[idx_fin])
            )
        return log_likelihoods
