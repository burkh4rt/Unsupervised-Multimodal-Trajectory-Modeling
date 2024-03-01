#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implements a linear Gaussian state space model
"""

import pickle

import numpy as np

from ..linear_gaussian import marginalizable_state_space_model as mssm
from ..util import util_state_space as util
from . import state_space_model as ssm

np_eps = np.finfo(float).eps


class StateSpaceLinearGaussian(ssm.StateSpaceModel):
    """Linear Gaussian state-space model;
    also known as a Linear Dynamical System / Kalman-type model
    """

    def __init__(self, alpha: float = 0.0):
        """instantiates model with hyperparameters as specified

        Parameters
        ----------
        alpha
            regularisation for linear model

        """
        super().__init__()
        self.alpha = alpha if alpha > 2 * np_eps else 0

    def __str__(self):
        return "State space model with linear Gaussian components"

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
                    np.row_stack(states[:-1]),
                    np.row_stack(states[1:]),
                    self.alpha,
                )
                if self.alpha > 2 * np_eps
                else util.regress(
                    np.row_stack(states[:-1]), np.row_stack(states[1:])
                ),
            )
        )
        self.measurement_model = dict(
            zip(
                ["coeff", "covar"],
                util.regress_alpha(
                    np.row_stack(states[:]),
                    np.row_stack(measurements[:]),
                    self.alpha,
                )
                if self.alpha > 2 * np_eps
                else util.regress(
                    np.row_stack(states[:]), np.row_stack(measurements[:])
                ),
            )
        )

        return self

    def to_pickle(self) -> bytes:
        return pickle.dumps(
            {
                "state_init": self.state_init,
                "state_model": self.state_model,
                "measurement_model": self.measurement_model,
                "alpha": self.alpha,
            }
        )

    def from_pickle(self, p: bytes):
        pickle_dict = pickle.loads(p)
        self.state_init = pickle_dict["state_init"]
        self.state_model = pickle_dict["state_model"]
        self.measurement_model = pickle_dict["measurement_model"]
        self.alpha = pickle_dict["alpha"] if "alpha" in pickle_dict else 0
        return self

    def score(self, data: tuple[np.ndarray, np.ndarray] = None):
        if data is None:
            data = self.data
        states, measurements = data
        T = states.shape[0]

        full_mean_T0 = mssm.mm(
            T,
            self.state_init["mean"],
            self.state_model["coeff"],
            self.measurement_model["coeff"],
        )

        full_cov_T0 = mssm.CC(
            T,
            self.state_init["cov"],
            self.state_model["coeff"],
            self.state_model["covar"],
            self.measurement_model["coeff"],
            self.measurement_model["covar"],
        )

        return mssm.multivariate_normal_log_likelihood(
            np.hstack((*states, *measurements)),
            full_mean_T0,
            full_cov_T0,
            np.zeros(states.shape[1]),
        )

    def score_alt(self, data: tuple[np.ndarray, np.ndarray] = None):
        if data is None:
            data = self.data
        states, measurements = data
        T = states.shape[0]

        return mssm.full_marginalizable_log_prob(
            z=states,
            x=measurements,
            T=T,
            m=self.state_init["mean"],
            S=self.state_init["cov"],
            A=self.state_model["coeff"],
            Γ=self.state_model["covar"],
            H=self.measurement_model["coeff"],
            Λ=self.measurement_model["covar"],
        )
