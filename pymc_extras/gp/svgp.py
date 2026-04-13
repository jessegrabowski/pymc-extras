from functools import partial

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pymc.gp.util import stabilize


class SVGP:
    """Sparse variational Gaussian process.

    This class only builds PyTensor graphs — ELBO, KL, predictive mean/covariance.
    Minibatching, optimizer choice, backend selection, and the training loop are
    the user's responsibility.
    """

    def __init__(
        self,
        input_dim,
        n_data,
        mean_func,
        cov_func,
        sigma,
        z_init,
        jitter=1e-6,
    ):
        self.mean_func = mean_func
        self.cov_func = cov_func
        self.sigma = sigma
        self.jitter = jitter

        self.input_dim = input_dim
        self.n_inducing = z_init.shape[0]
        self.n_data = n_data

        self.z_init = z_init

        self.initialize()

    def initialize(self, model=None):
        n = self.n_inducing
        floatX = pytensor.config.floatX

        # Packed lower-triangular index of the diagonal entries (row-major order).
        diag_idxs = np.arange(1, n + 1).cumsum() - 1
        vrc_init = np.zeros(n * (n + 1) // 2, dtype=floatX)
        vrc_init[diag_idxs] = 1.0  # initialize L = I  =>  Cov(q(u)) = I
        mean_init = np.zeros(n, dtype=floatX)
        z_init = np.asarray(self.z_init, dtype=floatX)

        with pm.modelcontext(model):
            self.z = pm.Flat("z", shape=(n, self.input_dim), initval=z_init)
            self.variational_mean = pm.Flat("variational_mean", shape=n, initval=mean_init)
            vrc_packed = pm.Flat("vrc", shape=n * (n + 1) // 2, initval=vrc_init)
            # Lower-triangular Cholesky root L such that Cov(q(u)) = L @ L.T.
            self.variational_root_covariance = pm.expand_packed_triangular(n, vrc_packed)

    def _kzz_cholesky(self):
        muz = self.mean_func(self.z)
        Kzz = stabilize(self.cov_func(self.z), self.jitter)
        Lz = pt.linalg.cholesky(Kzz, lower=True)
        return muz, Lz

    def kl_divergence(self):
        muz, Lz = self._kzz_cholesky()
        return self.kl_mvn_chol(self.variational_mean, self.variational_root_covariance, muz, Lz)

    @staticmethod
    def kl_mvn_chol(mu_q, L_q, mu_p, L_p):
        """KL[ N(mu_q, L_q L_qᵀ) || N(mu_p, L_p L_pᵀ) ]."""
        d = mu_p - mu_q
        # tr(K_p⁻¹ K_q) = ‖L_p⁻¹ L_q‖_F²
        M = pt.linalg.solve_triangular(L_p, L_q, lower=True)
        # (μ_p − μ_q)ᵀ K_p⁻¹ (μ_p − μ_q) = ‖L_p⁻¹ d‖²
        v = pt.linalg.solve_triangular(L_p, d, lower=True)
        trace = pt.sum(M**2)
        mahalanobis = pt.sum(v**2)
        logdet_q = 2.0 * pt.sum(pt.log(pt.diag(L_q)))
        logdet_p = 2.0 * pt.sum(pt.log(pt.diag(L_p)))
        n = mu_q.shape[0]
        return 0.5 * (trace + mahalanobis + logdet_p - logdet_q - n)

    def predict(self, t):
        """Variational posterior over f(t). Returns (mean, covariance) with no observation noise."""
        mu_q = self.variational_mean
        L_q = self.variational_root_covariance

        muz, Lz = self._kzz_cholesky()
        mut = self.mean_func(t)
        Kzt = self.cov_func(self.z, t)
        Ktt = self.cov_func(t)

        Lz_inv_Kzt = pt.linalg.solve_triangular(Lz, Kzt, lower=True)
        Kzz_inv_Kzt = pt.linalg.solve_triangular(Lz.T, Lz_inv_Kzt, lower=False)
        Lq_T_Kzz_inv_Kzt = L_q.T @ Kzz_inv_Kzt

        mean = mut + Kzz_inv_Kzt.T @ (mu_q - muz)
        covariance = Ktt - Lz_inv_Kzt.T @ Lz_inv_Kzt + Lq_T_Kzz_inv_Kzt.T @ Lq_T_Kzz_inv_Kzt
        return mean, covariance

    def predict_diag(self, X):
        """Variational posterior over f(X). Returns (mean, variance) — only the diagonal.

        Shares one Kzz Cholesky across the whole batch rather than vectorizing per-point.
        """
        mu_q = self.variational_mean
        L_q = self.variational_root_covariance

        muz, Lz = self._kzz_cholesky()
        mux = self.mean_func(X)
        Kzx = self.cov_func(self.z, X)
        Kxx_diag = self.cov_func(X, diag=True)

        Lz_inv_Kzx = pt.linalg.solve_triangular(Lz, Kzx, lower=True)
        Kzz_inv_Kzx = pt.linalg.solve_triangular(Lz.T, Lz_inv_Kzx, lower=False)
        Lq_T_Kzz_inv_Kzx = L_q.T @ Kzz_inv_Kzx

        mean = mux + Kzz_inv_Kzx.T @ (mu_q - muz)
        variance = Kxx_diag - pt.sum(Lz_inv_Kzx**2, axis=0) + pt.sum(Lq_T_Kzz_inv_Kzx**2, axis=0)
        return mean, variance

    def variational_expectation(self, X_batch, y_batch):
        mean, variance = self.predict_diag(X_batch)
        y_flat = pt.squeeze(y_batch, axis=-1)
        sq_error = pt.square(y_flat - mean)
        log_two_pi = np.log(2.0 * np.pi)
        log_sigma2 = 2.0 * pt.log(self.sigma)
        return -0.5 * (log_two_pi + log_sigma2 + (sq_error + variance) / self.sigma**2)

    def elbo(self, X_batch, y_batch):
        var_exp = self.variational_expectation(X_batch, y_batch)
        batch_size = y_batch.shape[0]
        return (self.n_data / batch_size) * pt.sum(var_exp) - self.kl_divergence()

    def compile_pred_func(self, sigma=None, diag=False, mode="FAST_RUN", model=None):
        """Compile a prediction function.

        Parameters
        ----------
        sigma : pytensor variable, float, or None
            If provided, add sigma² to the predictive variance / covariance diagonal,
            producing the posterior over y. If None, return the posterior over f.
        diag : bool
            If True, compute only the diagonal of the predictive covariance (much cheaper).
        """
        t = pt.tensor("t", shape=(None, self.input_dim))

        if diag:
            mu, cov = self.predict_diag(t)
            if sigma is not None:
                cov = cov + sigma**2
        else:
            mu, cov = self.predict(t)
            if sigma is not None:
                cov = cov + sigma**2 * pt.identity_like(cov)

        with pm.modelcontext(model) as model:
            mu_value, cov_value = model.replace_rvs_by_values([mu, cov])

        inputs = pm.inputvars([mu_value, cov_value])
        f_predict = pytensor.function(
            inputs=inputs,
            outputs=[mu_value.squeeze(), cov_value.squeeze()],
            on_unused_input="ignore",
            mode=mode,
        )
        return partial(
            self._predict_f,
            inputs=inputs,
            f_predict=f_predict,
        )

    def _predict_f(self, X_pred, result_dict, inputs, f_predict):
        input_names = [x.name for x in inputs]
        mu_pred, cov_pred = f_predict(
            **{k: v for k, v in result_dict.items() if k in input_names}, t=X_pred
        )
        return mu_pred, cov_pred


class WhitenedSVGP(SVGP):
    r"""SVGP with the whitened parameterization.

    Instead of parameterizing ``q(u) = N(μ_q, L_q L_qᵀ)`` directly over the
    inducing values ``u = f(z)``, parameterize the whitened variables
    ``v = L_z⁻¹ (u − μ_z)`` as ``q(v) = N(μ̃, L̃ L̃ᵀ)``. Under the prior,
    ``v ~ N(0, I)``.

    Advantages over :class:`SVGP`:

    - ``KL[q(v) ‖ p(v)] = KL[N(μ̃, L̃ L̃ᵀ) ‖ N(0, I)]`` — no ``K_zz`` inversion
      and no Cholesky of ``K_zz`` in the KL computation.
    - The optimal variational distribution over ``v`` does not depend on
      ``K_zz``, so optimizer conditioning is dramatically better when kernel
      hyperparameters are learned jointly.
    - One fewer triangular solve per predictive call (only ``L_z⁻¹ K_zt``; no
      separate ``K_zz⁻¹ K_zt``).

    ``self.variational_mean`` and ``self.variational_root_covariance`` now
    represent ``μ̃`` and ``L̃`` rather than ``μ_q`` and ``L_q``.
    """

    def kl_divergence(self):
        mu_tilde = self.variational_mean
        L_tilde = self.variational_root_covariance
        m = mu_tilde.shape[0]
        # tr(L̃ L̃ᵀ) = ‖L̃‖_F²
        trace = pt.sum(L_tilde**2)
        mahalanobis = pt.sum(mu_tilde**2)
        logdet_q = 2.0 * pt.sum(pt.log(pt.diag(L_tilde)))
        return 0.5 * (trace + mahalanobis - logdet_q - m)

    def predict(self, t):
        """Variational posterior over f(t). Returns (mean, covariance) with no observation noise."""
        mu_tilde = self.variational_mean
        L_tilde = self.variational_root_covariance

        _, Lz = self._kzz_cholesky()
        mut = self.mean_func(t)
        Kzt = self.cov_func(self.z, t)
        Ktt = self.cov_func(t)

        # A = L_z⁻¹ K_zt,  shape (M, T)
        A = pt.linalg.solve_triangular(Lz, Kzt, lower=True)
        # L̃ᵀ A,  shape (M, T)
        B = L_tilde.T @ A

        mean = mut + A.T @ mu_tilde
        covariance = Ktt - A.T @ A + B.T @ B
        return mean, covariance

    def predict_diag(self, X):
        """Variational posterior over f(X). Returns (mean, variance) — only the diagonal."""
        mu_tilde = self.variational_mean
        L_tilde = self.variational_root_covariance

        _, Lz = self._kzz_cholesky()
        mux = self.mean_func(X)
        Kzx = self.cov_func(self.z, X)
        Kxx_diag = self.cov_func(X, diag=True)

        A = pt.linalg.solve_triangular(Lz, Kzx, lower=True)
        B = L_tilde.T @ A

        mean = mux + A.T @ mu_tilde
        variance = Kxx_diag - pt.sum(A**2, axis=0) + pt.sum(B**2, axis=0)
        return mean, variance
