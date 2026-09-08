import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gamma, kv


class PortfolioKernel:
    """
    Kernel functions and kernel-ridge estimation.

    With ``fit(X, y)`` the estimator has the representer form

        f(x) = sum_i alpha_i k(X_i, x)

    and stores the fitted coefficients in ``alpha_``.  Linear KRR and the
    one-dimensional NTK are solved through exact finite feature maps.  This
    is algebraically identical to the dual solution, without allocating the
    n x n Gram matrix.

    If ``N`` is supplied, point-wise methods also return portfolio-valued
    kernels of the form

        K(x, y) = k(x, y) I_N
    """

    def __init__(
        self,
        N=None,
        kernel="linear",
        ell=1.0,
        n_random_features=None,
        random_state=0,
        batch_size=8192,
    ):
        if isinstance(N, str):
            # Convenience form: PortfolioKernel("gaussian", ell=0.5)
            kernel, N = N, None

        if kernel not in {"linear", "gaussian", "matern32", "ntk"}:
            raise ValueError(
                "kernel must be 'linear', 'gaussian', 'matern32', or 'ntk'."
            )
        self._check_lengthscale(ell)

        self.N = N
        self.I_N = np.eye(N) if N is not None else None
        self.kernel = kernel
        self.ell = float(ell)
        if n_random_features is not None and (
            not isinstance(n_random_features, (int, np.integer))
            or n_random_features <= 0
        ):
            raise ValueError("n_random_features must be a positive integer or None.")
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.n_random_features = n_random_features
        self.random_state = random_state
        self.batch_size = int(batch_size)

    def lift(self, k):
        """
        Lift a scalar kernel k(x,y) to the portfolio-valued kernel

            K(x,y) = k(x,y) I_N.
        """
        if self.I_N is None:
            return k
        return k * self.I_N

    # ========================================================
    # 1. STATIC
    #
    # K(x,y) = I_N
    #
    # Portfolio policy:
    #     f(x) = w
    # ========================================================

    def static(self, x=None, y=None):
        return self.I_N.copy()

    # ========================================================
    # 2. LINEAR
    #
    # k(x,y) = 1 + x'y
    #
    # K(x,y) = (1 + x'y) I_N
    #
    # Portfolio policy:
    #     f(x) = a + Bx
    # ========================================================

    def linear(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        self._check_inputs(x, y)

        k = 1.0 + x @ y

        return self.lift(k)

    # ========================================================
    # 3. FINITE FEATURES
    #
    # k(x,y) = Phi(x)' Phi(y)
    #
    # K(x,y) = Phi(x)' Phi(y) I_N
    #
    # Portfolio policy:
    #     f(x) = B Phi(x)
    # ========================================================

    def finite(self, x, y, Phi):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        self._check_inputs(x, y)

        phi_x = np.asarray(Phi(x), dtype=float).reshape(-1)
        phi_y = np.asarray(Phi(y), dtype=float).reshape(-1)

        if phi_x.shape != phi_y.shape:
            raise ValueError(
                "Phi(x) and Phi(y) must have the same dimension."
            )

        k = phi_x @ phi_y

        return self.lift(k)

    # ========================================================
    # 4. GAUSSIAN / RBF
    #
    # k(x,y)
    # =
    # exp(-||x-y||^2 / (2 ell^2))
    #
    # Infinitely smooth kernel.
    # ========================================================

    def gaussian(self, x, y, ell=1.0):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        self._check_inputs(x, y)
        self._check_lengthscale(ell)

        dist2 = np.sum((x - y) ** 2)

        k = np.exp(
            -dist2 / (2.0 * ell**2)
        )

        return self.lift(k)

    # ========================================================
    # 5. NEURAL TANGENT KERNEL (two-layer ReLU)
    #
    # Infinite-width NTK for a fully-connected network with a
    # ReLU activation and no bias.  The scalar kernel is lifted
    # to a portfolio-valued kernel in the same way as above.
    # ========================================================

    def ntk(self, x, y):
        k = self.ntk_value(x, y)
        return self.lift(k)

    @staticmethod
    def ntk_value(x, y):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        if x.shape != y.shape:
            raise ValueError("x and y must have the same dimension.")

        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        if np.isclose(norm_x, 0.0) or np.isclose(norm_y, 0.0):
            return 0.0

        cosine = np.clip((x @ y) / (norm_x * norm_y), -1.0, 1.0)
        theta = np.arccos(cosine)
        nngp = norm_x * norm_y * (
            np.sin(theta) + (np.pi - theta) * np.cos(theta)
        ) / (2.0 * np.pi)
        derivative = (np.pi - theta) / (2.0 * np.pi)
        return nngp + (x @ y) * derivative

    # ========================================================
    # BATCH GRAM MATRICES
    #
    # These scalar kernels are used by scalable estimation code.
    # ========================================================

    @staticmethod
    def linear_gram(X, Y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        return 1.0 + X @ Y.T

    @staticmethod
    def gaussian_gram(X, Y, ell=1.0):
        if ell <= 0:
            raise ValueError("The length scale ell must be strictly positive.")

        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        dist2 = (
            np.sum(X**2, axis=1)[:, None]
            + np.sum(Y**2, axis=1)[None, :]
            - 2.0 * X @ Y.T
        )
        return np.exp(-np.maximum(dist2, 0.0) / (2.0 * ell**2))

    @staticmethod
    def matern32_gram(X, Y, ell=1.0):
        """Matérn-3/2 Gram matrix for row-wise observations."""
        if ell <= 0:
            raise ValueError("The length scale ell must be strictly positive.")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        dist2 = (
            np.sum(X**2, axis=1)[:, None]
            + np.sum(Y**2, axis=1)[None, :]
            - 2.0 * X @ Y.T
        )
        distance = np.sqrt(np.maximum(dist2, 0.0))
        scaled = np.sqrt(3.0) * distance / ell
        return (1.0 + scaled) * np.exp(-scaled)

    @staticmethod
    def ntk_gram(X, Y):
        """Two-layer ReLU NTK Gram matrix for row-wise observations."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.atleast_2d(np.asarray(Y, dtype=float))

        norm_x = np.linalg.norm(X, axis=1)
        norm_y = np.linalg.norm(Y, axis=1)
        denominator = norm_x[:, None] * norm_y[None, :]
        dot = X @ Y.T

        cosine = np.divide(
            dot,
            denominator,
            out=np.zeros_like(dot),
            where=denominator > 0,
        )
        theta = np.arccos(np.clip(cosine, -1.0, 1.0))
        nngp = denominator * (
            np.sin(theta) + (np.pi - theta) * np.cos(theta)
        ) / (2.0 * np.pi)
        derivative = (np.pi - theta) / (2.0 * np.pi)
        return nngp + dot * derivative

    # ========================================================
    # KERNEL-RIDGE FIT
    #
    # The fitted alpha_ has one coefficient per training observation:
    #
    # alpha_ = (K_train + n * lambda_ * I)^(-1) y_train
    # ========================================================

    def gram(self, X, Y=None):
        """Return the scalar Gram matrix for the configured kernel."""
        X = np.asarray(X, dtype=float)
        Y = X if Y is None else np.asarray(Y, dtype=float)

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be two-dimensional (n_observations, n_features).")
        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have the same number of features.")

        if self.kernel == "linear":
            return self.linear_gram(X, Y)
        if self.kernel == "gaussian":
            return self.gaussian_gram(X, Y, ell=self.ell)
        if self.kernel == "matern32":
            return self.matern32_gram(X, Y, ell=self.ell)
        return self.ntk_gram(X, Y)

    def fit(self, X, y, lambda_=1e-6):
        """Estimate and store the kernel coefficients ``alpha_``.

        Parameters
        ----------
        X : array-like, shape (n_observations, n_features)
            Training characteristics.  For the current project use the
            monthly rank-normalized values in ``panel['x']``.
        y : array-like, shape (n_observations,) or (n_observations, n_outputs)
            Training target(s), e.g. next-month excess returns.
        lambda_ : float
            Ridge regularization.  It is distinct from ``alpha_`` and is
            scaled by the number of training observations.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be two-dimensional (n_observations, n_features).")
        if y.ndim not in {1, 2}:
            raise ValueError("y must be one- or two-dimensional.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of observations.")
        if len(X) == 0:
            raise ValueError("X and y cannot be empty.")
        if lambda_ <= 0:
            raise ValueError("lambda_ must be strictly positive.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite values.")

        self.lambda_ = float(lambda_)
        self._target_was_vector = y.ndim == 1

        if self.kernel == "linear":
            # k(x, y) = 1 + x'y = phi(x)'phi(y), with phi(x) = [1, x].
            # Solving the (d + 1) dimensional primal system is exactly KRR.
            Phi = self._linear_features(X)
            return self._fit_exact_features(Phi, y, feature_map="linear")

        if self.kernel == "ntk" and X.shape[1] == 1:
            # For scalar inputs, the current bias-free ReLU NTK has an exact
            # two-dimensional feature map. No kernel approximation is used.
            Phi = self._ntk_1d_features(X)
            return self._fit_exact_features(Phi, y, feature_map="ntk_1d")

        if self.kernel in {"gaussian", "matern32"} and self.n_random_features is not None:
            return self._fit_gaussian_rff(X, y)

        self.fit_backend_ = "exact_dual"
        self.X_train_ = X.copy()
        K_train = self.gram(self.X_train_)
        K_train.flat[:: len(X) + 1] += len(X) * self.lambda_
        factor = cho_factor(
            K_train,
            lower=True,
            overwrite_a=True,
            check_finite=False,
        )
        self.alpha_ = cho_solve(factor, y, check_finite=False)
        return self

    def _fit_gaussian_rff(self, X, y):
        """Fit the RFF approximation to Gaussian KRR in bounded memory."""
        statistics = self.gaussian_rff_statistics(X, y)
        self.fit_from_gaussian_rff_statistics([statistics], self.lambda_)

        # Exact dual coefficients for the approximate RFF kernel.
        alpha_parts = []
        scale = len(X) * self.lambda_
        for start in range(0, len(X), self.batch_size):
            stop = min(start + self.batch_size, len(X))
            prediction = self._gaussian_rff_features(X[start:stop]) @ self.coef_
            alpha_parts.append((y[start:stop] - prediction) / scale)
        self.alpha_ = np.concatenate(alpha_parts, axis=0)
        return self

    def _initialize_gaussian_rff(self, n_inputs):
        """Initialize a reproducible stationary RFF map."""
        if hasattr(self, "rff_weights_"):
            if self.rff_weights_.shape[0] != n_inputs:
                raise ValueError("The RFF input dimension changed after initialization.")
            return

        n_features = int(self.n_random_features)
        rng = np.random.default_rng(self.random_state)
        gaussian_draws = rng.normal(size=(n_inputs, n_features))
        if self.kernel == "gaussian":
            self.rff_weights_ = gaussian_draws / self.ell
        elif self.kernel == "matern32":
            # The Matérn-ν spectral distribution is multivariate Student-t
            # with 2ν degrees of freedom and scale I/ell². For ν=3/2 this
            # gives df=3 and an exact Bochner random-feature sampler.
            chi_square = rng.chisquare(df=3.0, size=n_features)
            self.rff_weights_ = (
                gaussian_draws
                / np.sqrt(chi_square[None, :] / 3.0)
                / self.ell
            )
        else:
            raise ValueError("Stationary RFF initialization requires Gaussian or Matérn-3/2.")
        self.rff_phases_ = rng.uniform(0.0, 2.0 * np.pi, size=n_features)

    def gaussian_rff_statistics(self, X, y):
        """Compute additive Z'Z and Z'y stationary-RFF statistics."""
        if self.kernel not in {"gaussian", "matern32"} or self.n_random_features is None:
            raise ValueError("RFF statistics require Gaussian or Matérn-3/2 RFF.")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim not in {1, 2} or len(X) != len(y):
            raise ValueError("X and y have incompatible shapes.")
        if len(X) == 0:
            raise ValueError("X and y cannot be empty.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite values.")

        n_observations, n_inputs = X.shape
        n_features = int(self.n_random_features)
        self._initialize_gaussian_rff(n_inputs)
        feature_gram = np.zeros((n_features, n_features), dtype=float)
        rhs_shape = (n_features,) if y.ndim == 1 else (n_features, y.shape[1])
        rhs = np.zeros(rhs_shape, dtype=float)
        feature_sum = np.zeros(n_features, dtype=float)

        for start in range(0, n_observations, self.batch_size):
            stop = min(start + self.batch_size, n_observations)
            Phi = self._gaussian_rff_features(X[start:stop])
            feature_gram += Phi.T @ Phi
            rhs += Phi.T @ y[start:stop]
            feature_sum += Phi.sum(axis=0)

        return {
            "n_observations": n_observations,
            "n_inputs": n_inputs,
            "feature_gram": feature_gram,
            "rhs": rhs,
            "feature_sum": feature_sum,
            "ell": self.ell,
            "n_random_features": n_features,
            "random_state": self.random_state,
        }

    def gaussian_rff_features(self, X):
        """Return the configured Gaussian random Fourier feature map."""
        if self.kernel != "gaussian" or self.n_random_features is None:
            raise ValueError("Gaussian RFF features require a Gaussian RFF model.")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.ndim != 2 or len(X) == 0 or not np.isfinite(X).all():
            raise ValueError("X must be a finite, non-empty matrix.")
        self._initialize_gaussian_rff(X.shape[1])
        return self._gaussian_rff_features(X)

    def feature_map(self, X):
        """Return the finite feature representation used by scalable fits."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.ndim != 2 or len(X) == 0 or not np.isfinite(X).all():
            raise ValueError("X must be a finite, non-empty matrix.")
        if self.kernel == "linear":
            return self._linear_features(X)
        if self.kernel in {"gaussian", "matern32"}:
            if self.n_random_features is None:
                raise ValueError("Stationary scalable features require n_random_features.")
            self._initialize_gaussian_rff(X.shape[1])
            return self._gaussian_rff_features(X)
        if self.kernel == "ntk" and X.shape[1] == 1:
            return self._ntk_1d_features(X)
        raise ValueError("No scalable finite feature map is configured for this kernel.")

    def managed_feature_statistics(self, X, returns):
        """Compress one month into return and net-exposure feature loadings."""
        X = np.asarray(X, dtype=float)
        returns = np.asarray(returns, dtype=float)
        if X.ndim != 2 or returns.ndim != 1 or len(X) != len(returns):
            raise ValueError("X and returns have incompatible shapes.")
        if len(X) == 0 or not np.isfinite(X).all() or not np.isfinite(returns).all():
            raise ValueError("X and returns must be finite and non-empty.")

        if self.kernel == "linear":
            n_features = X.shape[1] + 1
        elif self.n_random_features is not None:
            n_features = int(self.n_random_features)
        else:
            raise ValueError("The configured kernel has no scalable feature map.")
        return_loading = np.zeros(n_features, dtype=float)
        net_loading = np.zeros(n_features, dtype=float)
        for start in range(0, len(X), self.batch_size):
            stop = min(start + self.batch_size, len(X))
            Phi = self.feature_map(X[start:stop])
            return_loading += Phi.T @ returns[start:stop]
            net_loading += Phi.sum(axis=0)
        return {
            "n_observations": len(X),
            "return_loading": return_loading,
            "net_loading": net_loading,
        }

    def gaussian_rff_managed_statistics(self, X, returns):
        """Compress one month into paper-consistent managed-payoff loadings.

        For free weights ``w_i = phi(x_i)' beta``, these loadings satisfy
        ``portfolio_return = return_loading' beta`` and
        ``sum_i w_i = net_loading' beta``. No division by the cross-sectional
        asset count is applied.
        """
        if self.kernel != "gaussian" or self.n_random_features is None:
            raise ValueError("Managed RFF statistics require a Gaussian RFF model.")
        X = np.asarray(X, dtype=float)
        returns = np.asarray(returns, dtype=float)
        if X.ndim != 2 or returns.ndim != 1 or len(X) != len(returns):
            raise ValueError("X and returns have incompatible shapes.")
        if len(X) == 0 or not np.isfinite(X).all() or not np.isfinite(returns).all():
            raise ValueError("X and returns must be finite and non-empty.")

        self._initialize_gaussian_rff(X.shape[1])
        n_features = int(self.n_random_features)
        return_loading = np.zeros(n_features, dtype=float)
        net_loading = np.zeros(n_features, dtype=float)
        for start in range(0, len(X), self.batch_size):
            stop = min(start + self.batch_size, len(X))
            Phi = self._gaussian_rff_features(X[start:stop])
            return_loading += Phi.T @ returns[start:stop]
            net_loading += Phi.sum(axis=0)
        return {
            "n_observations": len(X),
            "return_loading": return_loading,
            "net_loading": net_loading,
        }

    def fit_from_gaussian_rff_statistics(self, statistics, lambda_=1e-6):
        """Fit Gaussian RFF KRR by summing reusable block statistics."""
        if self.kernel not in {"gaussian", "matern32"} or self.n_random_features is None:
            raise ValueError("This method requires Gaussian or Matérn-3/2 RFF.")
        if lambda_ <= 0:
            raise ValueError("lambda_ must be strictly positive.")

        blocks = list(statistics)
        if not blocks:
            raise ValueError("At least one statistics block is required.")

        first = blocks[0]
        expected = (
            first["ell"],
            first["n_random_features"],
            first["random_state"],
            first["n_inputs"],
        )
        configured = (
            self.ell,
            int(self.n_random_features),
            self.random_state,
            first["n_inputs"],
        )
        if expected != configured:
            raise ValueError("RFF statistics do not match the model configuration.")
        for block in blocks[1:]:
            observed = (
                block["ell"],
                block["n_random_features"],
                block["random_state"],
                block["n_inputs"],
            )
            if observed != expected:
                raise ValueError("Cannot combine statistics from different RFF maps.")

        self._initialize_gaussian_rff(first["n_inputs"])
        n_observations = sum(block["n_observations"] for block in blocks)
        feature_gram = blocks[0]["feature_gram"].copy()
        rhs = blocks[0]["rhs"].copy()
        for block in blocks[1:]:
            feature_gram += block["feature_gram"]
            rhs += block["rhs"]

        n_features = int(self.n_random_features)
        self.lambda_ = float(lambda_)
        feature_gram.flat[:: n_features + 1] += n_observations * self.lambda_
        factor = cho_factor(
            feature_gram,
            lower=True,
            overwrite_a=True,
            check_finite=False,
        )
        self.coef_ = cho_solve(factor, rhs, check_finite=False)
        self.fit_backend_ = "rff_primal"
        self.feature_map_ = "gaussian_rff"
        self.n_train_ = n_observations
        return self

    @staticmethod
    def _linear_features(X):
        X = np.asarray(X, dtype=float)
        return np.column_stack((np.ones(len(X), dtype=X.dtype), X))

    @staticmethod
    def _ntk_1d_features(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError("The exact NTK feature map requires one input feature.")
        x = X[:, 0]
        return np.column_stack((np.maximum(x, 0.0), np.maximum(-x, 0.0)))

    def _gaussian_rff_features(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        projection = X @ self.rff_weights_ + self.rff_phases_
        return np.sqrt(2.0 / self.n_random_features) * np.cos(projection)

    def _fit_exact_features(self, Phi, y, feature_map):
        """Solve KRR exactly using a finite-dimensional kernel feature map."""
        n_observations, n_features = Phi.shape
        feature_gram = Phi.T @ Phi
        feature_gram.flat[:: n_features + 1] += n_observations * self.lambda_
        rhs = Phi.T @ y
        factor = cho_factor(
            feature_gram,
            lower=True,
            overwrite_a=True,
            check_finite=False,
        )
        self.coef_ = cho_solve(factor, rhs, check_finite=False)
        self.fit_backend_ = "exact_primal"
        self.feature_map_ = feature_map
        self.n_train_ = n_observations

        # Preserve the dual representer coefficients without an n x n solve:
        # alpha = (y - Phi beta) / (n lambda).
        self.alpha_ = (y - Phi @ self.coef_) / (
            n_observations * self.lambda_
        )
        return self

    def predict(self, X):
        """Predict using the coefficients estimated by ``fit``."""
        if not hasattr(self, "fit_backend_"):
            raise RuntimeError("Call fit(X, y) before predict(X).")

        if getattr(self, "fit_backend_", None) == "exact_primal":
            if self.feature_map_ == "linear":
                Phi = self._linear_features(X)
            else:
                Phi = self._ntk_1d_features(X)
            return Phi @ self.coef_

        if getattr(self, "fit_backend_", None) == "rff_primal":
            X = np.asarray(X, dtype=float)
            predictions = []
            for start in range(0, len(X), self.batch_size):
                stop = min(start + self.batch_size, len(X))
                predictions.append(
                    self._gaussian_rff_features(X[start:stop]) @ self.coef_
                )
            return np.concatenate(predictions, axis=0)

        K_new_train = self.gram(X, self.X_train_)
        return K_new_train @ self.alpha_

    # ========================================================
    # 6. MATERN
    #
    # Parametrized by Sobolev regularity s.
    #
    # Relation:
    #
    #     nu = s - d/2
    #
    # hence
    #
    #     s = nu + d/2.
    #
    # The Matérn RKHS is equivalent to H^s.
    #
    # Requires:
    #
    #     s > d/2.
    # ========================================================

    def matern(self, x, y, s, ell=1.0):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        self._check_inputs(x, y)
        self._check_lengthscale(ell)

        d = x.size

        nu = s - d / 2.0

        if nu <= 0:
            raise ValueError(
                f"s must satisfy s > d/2 = {d / 2:.4f}."
            )

        r = np.linalg.norm(x - y)

        # Matérn kernel satisfies k(x,x) = 1
        if np.isclose(r, 0.0):
            return self.lift(1.0)

        z = np.sqrt(2.0 * nu) * r / ell

        k = (
            2.0 ** (1.0 - nu)
            / gamma(nu)
            * z**nu
            * kv(nu, z)
        )

        return self.lift(k)

    # ========================================================
    # AUXILIARY METHODS
    # ========================================================

    @staticmethod
    def _check_inputs(x, y):
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError(
                "x and y must be one-dimensional arrays."
            )

        if x.shape != y.shape:
            raise ValueError(
                "x and y must have the same dimension."
            )

    @staticmethod
    def _check_lengthscale(ell):
        if ell <= 0:
            raise ValueError(
                "The length scale ell must be strictly positive."
            )
