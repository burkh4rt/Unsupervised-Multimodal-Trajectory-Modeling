#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
We explicitly derive the joint distribution of a linear Gaussian
latent state space model.

The latent process
Z[1], Z[2], ..., Z[T]
in ℝˡ is governed by the state model
Z[i] | Z[i-1] ~ N(Z[i-1]*A, Γ) for i = 2, ..., T
with initialisation Z[1] ~ N(m, S)
and the observed latent states
X[1], X[2], ..., X[T]
in ℝᵈ are generated using the measurement model
X[i] | Z[i] ~ N(Z[i]*H, Λ) for i = 1, ..., T

As the resulting joint distribution (Z[1], ..., Z[T]; X[1], ..., X[T])
is multivariate Gaussian, this enables us to calculate marginal
distributions when we encounter hidden variables or missing data.
"""

import warnings

import numba as nb
import numpy as np
import scipy.stats as sp_stats

warnings.simplefilter("ignore")


@nb.jit(
    nb.float64[:, :](
        nb.int64,
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
    ),
    nopython=True,
    parallel=True,
)
def _CZZii(i: int, S: np.array, A: np.array, Γ: np.array) -> np.array:
    """Covariance matrix for the ith hidden random variable (1-indexed)

    Parameters
    ----------
    i
        index 1<=i<=T
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance

    Returns
    -------
    Var(Z[i])
        d×d symmetric positive semi-definite matrix

    """
    if i == 1:
        return S
    return Γ + A.T @ _CZZii(i - 1, S, A, Γ) @ A


@nb.jit(
    nb.float64[:, :](
        nb.int64,
        nb.int64,
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
    ),
    nopython=True,
    parallel=True,
)
def _CZZij(i: int, j: int, S: np.array, A: np.array, Γ: np.array) -> np.array:
    """Covariance between the ith and jth hidden random variables (1-indexed)

    Parameters
    ----------
    i
        index 1<=i<=T
    j
        index 1<=j<=T
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance

    Returns
    -------
    Cov(Z[i], Z[j])
        d×d matrix

    """
    if i == j:
        return _CZZii(i, S, A, Γ)
    elif j > i:
        return _CZZii(i, S, A, Γ) @ np.linalg.matrix_power(A, j - i)
    else:  # j < i
        return _CZZij(j, i, S, A, Γ).T


def CZZ(T: int, S: np.array, A: np.array, Γ: np.array) -> np.array:
    """Covariance for the full hidden autoregressive process

    Parameters
    ----------
    T
        integer number of time steps
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance

    Returns
    -------
    Var(Z)
        dT×dT matrix

    """
    return np.block(
        [
            [_CZZij(i, j, S, A, Γ) for j in range(1, T + 1)]
            for i in range(1, T + 1)
        ]
    )


def _CZX(
    T: int, S: np.array, A: np.array, Γ: np.array, H: np.array
) -> np.array:
    """Covariance between the full hidden and observed processes
    Z & X, respectively

    Parameters
    ----------
    T
        integer number of time steps
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance
    H
        measurement model coefficients

    Returns
    -------
    Cov(X, Z)
        dT×ℓT matrix

    """
    return np.block(
        [
            [_CZZij(i, j, S, A, Γ) @ H for j in range(1, T + 1)]
            for i in range(1, T + 1)
        ]
    )


@nb.jit(
    nb.float64[:, :](
        nb.int64,
        nb.int64,
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
    ),
    nopython=True,
    parallel=True,
)
def _CXXij(
    i: int,
    j: int,
    S: np.array,
    A: np.array,
    Γ: np.array,
    H: np.array,
    Λ: np.array,
) -> np.array:
    """Covariance between the ith and jth observed random variables
    (1-indexed)

    Parameters
    ----------
    i
        index 1<=i<=T
    j
        index 1<=j<=T
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance
    H
        measurement model coefficients
    Λ
        measurement model covariance

    Returns
    -------
    Cov(X[i], X[j])
        ℓ×ℓ matrix

    """
    if i == j:
        return Λ + H.T @ _CZZii(i, S, A, Γ) @ H
    elif j > i:  # i!=j
        return H.T @ _CZZij(i, j, S, A, Γ) @ H
    else:
        return _CXXij(j, i, S, A, Γ, H, Λ).T


def CXX(
    T: int, S: np.array, A: np.array, Γ: np.array, H: np.array, Λ: np.array
) -> np.array:
    """Covariance over all observed random variables

    Parameters
    ----------
    T
        integer number of time steps
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance
    H
        measurement model coefficients
    Λ
        measurement model covariance

    Returns
    -------
    Var(X)
        ℓT×ℓT matrix

    """
    return np.block(
        [
            [_CXXij(i, j, S, A, Γ, H, Λ) for j in range(1, T + 1)]
            for i in range(1, T + 1)
        ]
    )


def CC(
    T: int, S: np.array, A: np.array, Γ: np.array, H: np.array, Λ: np.array
) -> np.array:
    """Full covariance matrix for the joint distribution (Z,X)

    Parameters
    ----------
    T
        integer number of time steps
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance
    H
        measurement model coefficients
    Λ
        measurement model covariance

    Returns
    -------
    Var([Z,X])
        (d+ℓ)T×(d+ℓ)T matrix

    """
    S, A, Γ, H, Λ = map(np.atleast_2d, (S, A, Γ, H, Λ))
    return np.block(
        [
            [CZZ(T, S, A, Γ), _CZX(T, S, A, Γ, H)],
            [_CZX(T, S, A, Γ, H).T, CXX(T, S, A, Γ, H, Λ)],
        ]
    )


def mmZ(T: int, m: np.array, A: np.array) -> np.array:
    """Full mean vector for the latent process Z

    Parameters
    ----------
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    A
        state space model coefficients

    Returns
    -------
    𝔼(Z)
        dT-length vector

    """
    return np.hstack(
        [m @ np.linalg.matrix_power(A, i) for i in range(T)]
    ).ravel()


def mmX(T: int, m: np.array, A: np.array, H: np.array) -> np.array:
    """Full mean vector for the observed process X

    Parameters
    ----------
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    A
        state space model coefficients
    H
        measurement model coefficients

    Returns
    -------
    𝔼(X)
        ℓT-length vector

    """
    return np.hstack(
        [[m @ np.linalg.matrix_power(A, i) @ H] for i in range(T)]
    ).ravel()


def mm(T: int, m: np.array, A: np.array, H: np.array) -> np.array:
    """Full mean vector for the joint distribution (Z, X)

    Parameters
    ----------
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    A
        state space model coefficients
    H
        measurement model coefficients

    Returns
    -------
    𝔼([Z,X])
        (d+ℓ)T-length vector

    """
    A, H = map(np.atleast_2d, (A, H))
    m = np.atleast_1d(m)
    return np.hstack([mmZ(T, m, A), mmX(T, m, A, H)]).ravel()


def full_log_prob(
    z: np.array,
    x: np.array,
    T: np.array,
    m: np.array,
    S: np.array,
    A: np.array,
    Γ: np.array,
    H: np.array,
    Λ: np.array,
) -> np.array:
    """log of joint distribution function (p.d.f.) for (Z,X) calculated using
    our analytically calculated mean and variance functions

    Parameters
    ----------
    z
        T×n×d array of hidden states where
            T length of trajectories
            n number of trajectories
            d dimensionality of the latent space
    x
        T×n×ℓ array of observed variables  where
            T length of trajectories
            n number of trajectories
            ℓ dimensionality of the latent space
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance
    H
        measurement model coefficients
    Λ
        measurement model covariance

    Returns
    -------
    n-dimensional vector
        log of joint (Gaussian) distribution of the (Z,X) evaluated at (z,x)

    See Also
    --------
    mm
        the mean function we use here
    CC
        the covariance function we use here

    """
    z, x = map(np.atleast_3d, (z, x))
    return sp_stats.multivariate_normal(
        mean=mm(T, m, A, H),
        cov=CC(T, S, A, Γ, H, Λ),
        allow_singular=True,
    ).logpdf(np.hstack((*z[:], *x[:])))


def composite_log_prob(
    z: np.array,
    x: np.array,
    T: int,
    m: np.array,
    S: np.array,
    A: np.array,
    Γ: np.array,
    H: np.array,
    Λ: np.array,
) -> np.array:
    """log of joint distribution function (p.d.f.) for (Z,X) calculated using
    the generation process

    Parameters
    ----------
    z
        T×n×d array of hidden states where
            T length of trajectories
            n number of trajectories
            d dimensionality of the latent space
    x
        T×n×ℓ array of observed variables where
            T length of trajectories
            n number of trajectories
            ℓ dimensionality of the latent space
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance
    H
        measurement model coefficients
    Λ
        measurement model covariance

    Returns
    -------
    n-dimensional vector
        log of joint (Gaussian) distribution of (Z,X) evaluated at (z,x)

    See Also
    --------
    full_log_prob
        a different way to calculate this quantity

    """
    z, x = map(np.atleast_3d, (z, x))
    S, A, Γ, H, Λ = map(np.atleast_2d, (S, A, Γ, H, Λ))
    log_likelihoods = sp_stats.multivariate_normal(
        mean=m,
        cov=S,
        allow_singular=True,
    ).logpdf(z[0, :, :])
    for t in range(T - 1):
        log_likelihoods += sp_stats.multivariate_normal(
            cov=Γ, allow_singular=True
        ).logpdf(z[t + 1, :, :] - z[t, :, :] @ A)
    for t in range(T):
        log_likelihoods += sp_stats.multivariate_normal(
            cov=Λ, allow_singular=True
        ).logpdf(x[t, :, :] - z[t, :, :] @ H)
    return log_likelihoods


def hidden_log_prob(
    z: np.array,
    T: np.array,
    m: np.array,
    S: np.array,
    A: np.array,
    Γ: np.array,
) -> np.array:
    """log of distribution of Z evaluated at z evaluated using the calculations
    we've made

    Parameters
    ----------
    z
        T×n×d array of hidden states
            T length of trajectories
            n number of trajectories
            d dimensionality of the latent space
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance

    Returns
    -------
    n-dimensional vector
        log of (Gaussian) distribution of Z evaluated at z

    """
    z = np.atleast_3d(z)
    S, A, Γ = map(np.atleast_2d, (S, A, Γ))
    return sp_stats.multivariate_normal(
        mean=mmZ(T, m, A),
        cov=CZZ(T, S, A, Γ),
        allow_singular=True,
    ).logpdf(np.hstack(z[:]))


def composite_hidden_log_prob(
    z: np.array,
    T: np.array,
    m: np.array,
    S: np.array,
    A: np.array,
    Γ: np.array,
) -> np.array:
    """log of distribution of Z evaluated at z calculated using the generation
    process

    Parameters
    ----------
    z
        T×n×d array of hidden states where
            T length of trajectories
            n number of trajectories
            d dimensionality of the latent space
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance

    Returns
    -------
    n-dimensional vector
        of the log of (Gaussian) distribution of Z evaluated at z

    See Also
    --------
    hidden_log_prob
        different way to calculate this quantity

    """
    z = np.atleast_3d(z)
    S, A, Γ = map(np.atleast_2d, (S, A, Γ))
    log_likelihoods = sp_stats.multivariate_normal(
        mean=m,
        cov=S,
        allow_singular=True,
    ).logpdf(z[0, :, :])
    for t in range(T - 1):
        log_likelihoods += sp_stats.multivariate_normal(
            cov=Γ, allow_singular=True
        ).logpdf(z[t + 1, :, :] - z[t, :, :] @ A)
    return log_likelihoods


def observed_log_prob(
    x: np.array,
    T: np.array,
    m: np.array,
    S: np.array,
    A: np.array,
    Γ: np.array,
    H: np.array,
    Λ: np.array,
) -> np.array:
    """log of joint distribution function (p.d.f.) for X calculated using
    our analytically calculated mean and variance functions

    Parameters
    ----------
    x
        T×n×ℓ array of observed variables where
            T length of trajectories
            n number of trajectories
            ℓ dimensionality of the latent space
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance
    H
        measurement model coefficients
    Λ
        measurement model covariance

    Returns
    -------
        n-dimensional vector of the log of joint (Gaussian) distribution of
        X evaluated at x

    See Also
    --------
    mmX
        the mean function we use here
    CXX
        the covariance function we use here

    """
    x = np.atleast_3d(x)
    S, A, Γ, H, Λ = map(np.atleast_2d, (S, A, Γ, H, Λ))
    return sp_stats.multivariate_normal(
        mean=mmX(T, m, A, H),
        cov=CXX(T, S, A, Γ, H, Λ),
        allow_singular=True,
    ).logpdf(np.hstack(x[:]))


def full_marginalizable_log_prob(
    z: np.array,
    x: np.array,
    T: np.array,
    m: np.array,
    S: np.array,
    A: np.array,
    Γ: np.array,
    H: np.array,
    Λ: np.array,
) -> np.array:
    """log of joint distribution function (p.d.f.) for (Z,X),
    marginalising over missing dimensions of the data

    Parameters
    ----------
    z
        T×n×d array of hidden states, potentially including np.nan's, where
            T length of trajectories
            n number of trajectories
            d dimensionality of the latent space
    x
        T×n×ℓ array of observed variables, potentially including np.nan's,
        where
            T length of trajectories
            n number of trajectories
            ℓ dimensionality of the latent space
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance
    H
        measurement model coefficients
    Λ
        measurement model covariance

    Returns
    -------
    log of joint (Gaussian) distribution of (Z,X) evaluated at (z,x)
        non-finite dimensions have been marginalized out

    See Also
    --------
    mm
        the mean function we use here
    CC
        the covariance function we use here

    """
    z, x = map(np.atleast_3d, (z, x))
    zx = np.ma.masked_invalid(np.hstack((*z[:], *x[:])))
    p = np.zeros(zx.shape[0])

    # compute mean and covariance once
    mean = mm(T, m, A, H)
    cov = CC(T, S, A, Γ, H, Λ)

    # loop over data and calculate for each instance
    for i in range(zx.shape[0]):
        zx_i = zx[i, :]
        p[i] = sp_stats.multivariate_normal(
            mean=mean[~zx_i.mask],
            cov=cov[~zx_i.mask, :][:, ~zx_i.mask],
            allow_singular=True,
        ).logpdf(zx_i.compressed())
    return p


@nb.guvectorize(
    [
        (
            nb.float64[:, :],
            nb.float64[:],
            nb.float64[:, :],
            nb.float64[:],
        )
    ],
    "(n,d),(d),(d,d)->(n)",
    nopython=True,
    # fastmath=True,
)
def multivariate_normal_log_likelihood(
    x: np.array, μ: np.array, Σ: np.array, p: np.array
):
    """computes the log likelihood of a multivariate N(μ,Σ) distribution
    evaluated at the rows of x and assigns it to p;
    if x[i], 1<=i<=n, contains np.nan's or np.inf's for some elements
    (anything that is not finite), it marginalizes these out of the computation

    Parameters
    ----------
    x
        n×d-dimensional matrix containing rows of observations
    μ
        d-dimensional vector
    Σ
        d×d-dimensional covariance matrix
    p
        n-dimensional vector containing the log-likelihoods of interest

    Returns
    -------
        assigns p[i] to be the log likelihood of X~N(μ,Σ) at X=x[i,:]

    """
    x = np.atleast_2d(x)
    Σ = np.atleast_2d(Σ)

    for i in range(p.size):
        m = x[i, :].ravel() - μ.ravel()
        idx = np.argwhere(np.isfinite(m)).ravel()
        p[i] = -0.5 * np.log(
            (2 * np.pi) ** idx.size * np.linalg.det(Σ[idx, :][:, idx])
        ) - 0.5 * m[idx] @ np.linalg.solve(Σ[idx, :][:, idx], m[idx])


def sample_trajectory(
    n: int,
    T: int,
    m: np.array,
    S: np.array,
    A: np.array,
    Γ: np.array,
    H: np.array,
    Λ: np.array,
    rng: np.random.Generator = np.random.default_rng(42),
) -> tuple[np.array, np.array]:
    """Given model parameters, this function creates n samples of (Z,X)

    Parameters
    ----------
    n
        integer number of samples to create
    T
        integer number of time steps
    m
        mean of initial hidden random variable
    S
        covariance of initial hidden random variable
    A
        state space model coefficients
    Γ
        state space model covariance
    H
        measurement model coefficients
    Λ
        measurement model covariance
    rng
        random number generator

    Returns
    -------
    a tuple (z,x)
        n samples from from the joint distribution of (Z,X)
        with provided parameters
        z has shape T×n×d
        x has shape T×n×ℓ
    """
    S, A, Γ, H, Λ = map(np.atleast_2d, (S, A, Γ, H, Λ))

    z = np.zeros(shape=(T, n, m.shape[0]))
    x = np.zeros(shape=(T, n, H.shape[1]))

    z[0, :, :] = sp_stats.multivariate_normal(mean=m, cov=S).rvs(
        size=n, random_state=rng
    )
    x[0, :, :] = z[0, :, :] @ H + sp_stats.multivariate_normal(cov=Λ).rvs(
        size=n, random_state=rng
    )
    for t in range(T - 1):
        z[t + 1, :, :] = z[t, :, :] @ A + sp_stats.multivariate_normal(
            cov=Γ,
        ).rvs(size=n, random_state=rng)
        x[t + 1, :, :] = z[t + 1, :, :] @ H + sp_stats.multivariate_normal(
            cov=Λ
        ).rvs(size=n, random_state=rng)
    return z, x


def sample_nonlinear_nongaussian_trajectory(
    n: int,
    dz: int,
    dx: int,
    T: int,
    m: callable,
    f: callable,
    Γ: callable,
    h: callable,
    Λ: callable,
    rng: np.random.Generator = np.random.default_rng(42),
) -> tuple[np.array, np.array]:
    """Given model parameters, this function creates n samples of (Z,X)

    Parameters
    ----------
    n
        integer number of samples to create
    dz
        dimensionality of hidden / latent states
    dx
        dimensionality of measurements / observations
    T
        integer number of time steps
    m
        sampler for initial hidden random variable (taking size, rng as input)
        e.g. lambda size, rng: sp_stats.multivariate_normal(cov=Λ).rvs(
            size=size, random_state=rng
        )
    f
        function for state space model
        e.g. lambda z: z ** 2
    Γ
        noise part of state space model (taking size, rng as input)
    h
        function for measurement model
        e.g. lambda z: np.sin(z)
    Λ
        noise part of measurement model (taking size, rng as input)
    rng
        random number generator

    Returns
    -------
    a tuple (z,x)
        n samples from from the joint distribution of (Z,X)
        with provided parameters
        z has shape T×n×d
        x has shape T×n×ℓ
    """

    z = np.zeros(shape=(T, n, dz))
    x = np.zeros(shape=(T, n, dx))

    z[0, :, :] = m(n, rng)
    x[0, :, :] = np.apply_along_axis(func1d=h, axis=-1, arr=z[0, :, :]) + Λ(
        n, rng
    )

    for t in range(T - 1):
        z[t + 1, :, :] = np.apply_along_axis(
            func1d=f, axis=-1, arr=z[t, :, :]
        ) + Γ(n, rng)
        x[t + 1, :, :] = np.apply_along_axis(
            func1d=h, axis=-1, arr=z[t + 1, :, :]
        ) + Λ(n, rng)
    return z, x


def marginalizable_gaussian_log_prob(
    x: np.array, μ: np.array = None, Σ: np.array = None
):
    """gaussian log probability that marginalizes over np.nan values

    Parameters
    ----------
    x
        n×d-dimensional matrix where rows are observations
    μ
        d-dimensional vector; defaults to the zero vector
    Σ
        d×d-dimensional covariance matrix; defaults to the identity

    Returns
    -------
    log_probs
        n-dimensional vector of log(η(x[i];μ,Σ)) indexed over the rows i
    """
    x = np.atleast_2d(x)
    n, d = x.shape
    if μ is None:
        μ = np.zeros(shape=d)
    if Σ is None:
        Σ = np.eye(d)
    else:
        Σ = np.atleast_2d(Σ)
    xm = np.ma.masked_invalid(x)
    p = np.zeros(n)
    for i in range(n):
        p[i] = sp_stats.multivariate_normal(
            mean=μ[~xm[i].mask],
            cov=Σ[~xm[i].mask, :][:, ~xm[i].mask],
            allow_singular=True,
        ).logpdf(xm[i].compressed())
    return p
