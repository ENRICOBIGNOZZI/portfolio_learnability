import numpy as np
from scipy.linalg import cho_factor, cho_solve


class PortfolioKernel:

    def __init__(
        self,
        N=None,
        kernel="gaussian",
        ell=1.0,
        n_random_features=None,
        random_state=0,
        batch_size=8192,
    ):
        self.N = N
        self.I_N = np.eye(N) if N is not None else 1.0
        self.kernel = kernel
        self.ell = ell
        self.n_random_features = n_random_features
        self.random_state = random_state
        self.batch_size = batch_size

    # =========================================================
    # SCALAR KERNELS
    # =========================================================

    @staticmethod
    def static_gram(X, Y):
        return np.ones((len(X), len(Y)))

    @staticmethod
    def linear_gram(X, Y):
        return 1.0 + X @ Y.T

    def gaussian_gram(self, X, Y):
        d2 = (
            np.sum(X**2, axis=1)[:, None]
            + np.sum(Y**2, axis=1)[None, :]
            - 2.0 * X @ Y.T
        )
        return np.exp(
            -np.maximum(d2, 0.0) / (2.0 * self.ell**2)
        )

    def matern32_gram(self, X, Y):
        d2 = (
            np.sum(X**2, axis=1)[:, None]
            + np.sum(Y**2, axis=1)[None, :]
            - 2.0 * X @ Y.T
        )
        d = np.sqrt(np.maximum(d2, 0.0))
        a = np.sqrt(3.0) * d / self.ell
        return (1.0 + a) * np.exp(-a)

    @staticmethod
    def ntk_gram(X, Y):
        dot = X @ Y.T

        norm_x = np.linalg.norm(X, axis=1)
        norm_y = np.linalg.norm(Y, axis=1)

        denominator = (
            norm_x[:, None]
            * norm_y[None, :]
        )

        cosine = np.divide(
            dot,
            denominator,
            out=np.zeros_like(dot),
            where=denominator != 0,
        )

        theta = np.arccos(
            np.clip(cosine, -1.0, 1.0)
        )

        nngp = (
            denominator
            * (
                np.sin(theta)
                + (np.pi - theta) * np.cos(theta)
            )
            / (2.0 * np.pi)
        )

        derivative = (
            np.pi - theta
        ) / (2.0 * np.pi)

        return nngp + dot * derivative

    # =========================================================
    # GENERIC KERNEL CALL
    # =========================================================

    def gram(self, X, Y=None):

        X = np.atleast_2d(
            np.asarray(X, dtype=float)
        )

        Y = (
            X
            if Y is None
            else np.atleast_2d(
                np.asarray(Y, dtype=float)
            )
        )

        return getattr(
            self,
            f"{self.kernel}_gram"
        )(X, Y)

    def __call__(self, x, y):

        k = self.gram(
            np.asarray(x)[None, :],
            np.asarray(y)[None, :]
        )[0, 0]

        return k * self.I_N

    # =========================================================
    # FINITE FEATURE MAPS
    # =========================================================

    @staticmethod
    def static_features(X):
        return np.ones((len(X), 1))

    @staticmethod
    def linear_features(X):
        return np.column_stack(
            (
                np.ones(len(X)),
                X,
            )
        )

    @staticmethod
    def ntk_features(X):
        x = X[:, 0]

        return np.column_stack(
            (
                np.maximum(x, 0.0),
                np.maximum(-x, 0.0),
            )
        )

    def _initialize_rff(self, d):

        rng = np.random.default_rng(
            self.random_state
        )

        P = self.n_random_features

        W = rng.normal(
            size=(d, P)
        )

        if self.kernel == "gaussian":
            W = W / self.ell

        if self.kernel == "matern32":
            chi2 = rng.chisquare(
                df=3.0,
                size=P,
            )

            W = (
                W
                / np.sqrt(
                    chi2[None, :] / 3.0
                )
                / self.ell
            )

        self.rff_weights_ = W

        self.rff_phases_ = rng.uniform(
            0.0,
            2.0 * np.pi,
            size=P,
        )

    def rff_features(self, X):

        if not hasattr(
            self,
            "rff_weights_"
        ):
            self._initialize_rff(
                X.shape[1]
            )

        return (
            np.sqrt(
                2.0
                / self.n_random_features
            )
            * np.cos(
                X @ self.rff_weights_
                + self.rff_phases_
            )
        )

    def features(self, X):

        X = np.atleast_2d(
            np.asarray(X, dtype=float)
        )

        if self.kernel in {
            "gaussian",
            "matern32",
        }:
            return self.rff_features(X)

        return getattr(
            self,
            f"{self.kernel}_features"
        )(X)

    # =========================================================
    # MANAGED PAYOFFS
    # =========================================================

    def managed_feature_statistics(
        self,
        X,
        returns,
    ):

        Phi = self.features(X)

        returns = np.asarray(
            returns,
            dtype=float,
        )

        return {
            "n_observations": len(X),

            "return_loading":
                Phi.T @ returns,

            "net_loading":
                Phi.sum(axis=0),
        }

    # =========================================================
    # KRR
    # =========================================================

    def fit(
        self,
        X,
        y,
        lambda_=1e-6,
    ):

        X = np.atleast_2d(
            np.asarray(X, dtype=float)
        )

        y = np.asarray(
            y,
            dtype=float,
        )

        self.lambda_ = lambda_

        explicit_features = (
            self.kernel
            in {"static", "linear"}
            or (
                self.kernel == "ntk"
                and X.shape[1] == 1
            )
            or self.n_random_features
            is not None
        )

        if explicit_features:

            Phi = self.features(X)

            A = Phi.T @ Phi

            A.flat[
                :: len(A) + 1
            ] += (
                len(X)
                * lambda_
            )

            self.coef_ = cho_solve(
                cho_factor(
                    A,
                    lower=True,
                ),
                Phi.T @ y,
            )

            self.backend_ = "primal"

        else:

            K = self.gram(X)

            K.flat[
                :: len(K) + 1
            ] += (
                len(X)
                * lambda_
            )

            self.alpha_ = cho_solve(
                cho_factor(
                    K,
                    lower=True,
                ),
                y,
            )

            self.X_train_ = X

            self.backend_ = "dual"

        return self

    def predict(self, X):

        X = np.atleast_2d(
            np.asarray(X, dtype=float)
        )

        if self.backend_ == "primal":
            return (
                self.features(X)
                @ self.coef_
            )

        return (
            self.gram(
                X,
                self.X_train_,
            )
            @ self.alpha_
        )