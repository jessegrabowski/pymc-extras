"""Closed-form regularization losses for prior calibration.

KL helpers between two distributions of the same family — useful as the
"smallest adjustment" pull in a calibration loss. KL is finite for these
families whenever the two parameterizations have the same support, which
is always for Normal/HalfNormal/LogNormal/Gamma/Beta/Exponential pairs.

For families where lifting a parameter shifts the support boundary
(Uniform's bounds, Pareto's `m`, truncated distributions), KL would be
ill-defined; use `moment_distance` instead — it works for any family
that exposes closed-form `mean` and `var` via `pytensor-distributions`.

Parameter naming follows PyMC's canonical conventions (rate `beta` for
Gamma, rate `lam` for Exponential, etc.).
"""

import pytensor.tensor as pt

__all__ = [
    "kl_normal",
    "kl_halfnormal",
    "kl_lognormal",
    "kl_gamma",
    "kl_beta",
    "kl_exponential",
    "moment_distance",
]


def kl_normal(mu_p, sigma_p, mu_q, sigma_q):
    """KL(N(mu_p, sigma_p) || N(mu_q, sigma_q))."""
    return pt.log(sigma_q / sigma_p) + (sigma_p**2 + (mu_p - mu_q) ** 2) / (2 * sigma_q**2) - 0.5


def kl_halfnormal(sigma_p, sigma_q):
    """KL(HN(sigma_p) || HN(sigma_q)).

    Folded zero-mean Gaussian: same form as Normal-Normal with mu_p = mu_q = 0.
    """
    return pt.log(sigma_q / sigma_p) + sigma_p**2 / (2 * sigma_q**2) - 0.5


def kl_lognormal(mu_p, sigma_p, mu_q, sigma_q):
    """KL(LogNormal(mu_p, sigma_p) || LogNormal(mu_q, sigma_q)).

    Log-transform is bijective, so KL is identical to the underlying Gaussian KL.
    """
    return kl_normal(mu_p, sigma_p, mu_q, sigma_q)


def kl_gamma(alpha_p, beta_p, alpha_q, beta_q):
    """KL(Gamma(alpha_p, beta_p) || Gamma(alpha_q, beta_q)). Rate parameterization.

    Also valid for KL(InvGamma(α_p, β_p) || InvGamma(α_q, β_q))`: since the
    map `X → 1/X` between Gamma and InvGamma is bijective, the KL is identical.
    """
    return (
        (alpha_p - alpha_q) * pt.digamma(alpha_p)
        - pt.gammaln(alpha_p)
        + pt.gammaln(alpha_q)
        + alpha_q * (pt.log(beta_p) - pt.log(beta_q))
        + alpha_p * (beta_q - beta_p) / beta_p
    )


def kl_beta(alpha_p, beta_p, alpha_q, beta_q):
    """KL(Beta(alpha_p, beta_p) || Beta(alpha_q, beta_q))."""
    log_B_p = pt.gammaln(alpha_p) + pt.gammaln(beta_p) - pt.gammaln(alpha_p + beta_p)
    log_B_q = pt.gammaln(alpha_q) + pt.gammaln(beta_q) - pt.gammaln(alpha_q + beta_q)
    psi_sum = pt.digamma(alpha_p + beta_p)
    return (
        log_B_q
        - log_B_p
        + (alpha_p - alpha_q) * (pt.digamma(alpha_p) - psi_sum)
        + (beta_p - beta_q) * (pt.digamma(beta_p) - psi_sum)
    )


def kl_exponential(lam_p, lam_q):
    """KL(Exp(lam_p) || Exp(lam_q)). Rate parameterization."""
    return pt.log(lam_p) - pt.log(lam_q) + lam_q / lam_p - 1.0


def moment_distance(family, params_p, params_q):
    """Squared distance between mean+variance of two distributions in a family.

    A family-agnostic, support-shift-tolerant alternative to KL. Useful when
    a lifted parameter changes the distribution's support (Uniform bounds,
    Pareto's `m`, Truncated bounds).

    Parameters
    ----------
    family : module
        A submodule of `pytensor_distributions` (e.g. `pytensor_distributions.beta`).
    params_p, params_q : tuple
        Positional parameters for the two distributions, in the order the
        family's `mean(*params)` / `var(*params)` accept.
    """
    mean_p = family.mean(*params_p)
    mean_q = family.mean(*params_q)
    var_p = family.var(*params_p)
    var_q = family.var(*params_q)
    return (mean_p - mean_q) ** 2 + (var_p - var_q) ** 2
