#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implements a Bayesian classifier for state space models
"""

import numpy as np
from sklearn import base as skl_base

from . import state_space_model as ssm


class StateSpaceModelClassifier(
    skl_base.BaseEstimator, skl_base.DensityMixin, skl_base.ClassifierMixin
):
    """A generative classifier where p(data|class) is learned as a state
    space model
    """

    def __init__(
        self,
        component_model: type(ssm.StateSpaceModel),
    ):
        super().__init__()

        self.component_model = component_model
        self.classes, self.n_classes = None, None
        self.propensities = None
        self.class_models = None
        self.data = None

    def fit(self, data: tuple[np.ndarray, np.ndarray], labels: np.ndarray):
        self.data = tuple(map(np.atleast_3d, data))
        states, measurements = data
        self.classes, cts = np.unique(labels, return_counts=True)
        self.n_classes = len(self.classes)
        self.propensities = cts / np.sum(cts)
        self.class_models = [self.component_model() for _ in self.classes]
        for i, c in enumerate(self.classes):
            self.class_models[i].fit(
                data=(states[:, labels == c], measurements[:, labels == c])
            )
        return self

    def score(self, data: tuple[np.ndarray, np.ndarray] = None) -> float:
        if data is None:
            data = self.data
        else:
            data = tuple(map(np.atleast_3d, data))

        jt_p = np.sum(
            np.column_stack(
                [
                    self.propensities[i]
                    * np.exp(self.class_models[i].score(data=data))
                    for i in range(self.n_classes)
                ]
            ),
            axis=1,
        )
        assert jt_p.shape[0] == data[0].shape[1]
        return float(np.sum(np.log(jt_p)))

    def predict_proba(
        self, data: tuple[np.ndarray, np.ndarray] = None
    ) -> np.ndarray:
        if data is None:
            data = self.data
        else:
            data = tuple(map(np.atleast_3d, data))

        pc = np.column_stack(
            [
                self.propensities[i]
                * np.exp(self.class_models[i].score(data=data))
                for i in range(self.n_classes)
            ]
        )
        pc /= np.sum(pc, axis=1, keepdims=True)

        assert pc.shape[0] == data[0].shape[1]
        assert np.all(pc >= 0.0) and np.allclose(np.sum(pc, axis=-1), 1.0)
        return pc

    def predict(
        self, data: tuple[np.ndarray, np.ndarray] = None
    ) -> np.ndarray:
        if data is None:
            data = self.data
        else:
            data = tuple(map(np.atleast_3d, data))

        preds = self.classes[np.argmax(self.predict_proba(data), axis=1)]
        assert preds.size == data[0].shape[1]
        return preds
