import numpy as np


class PortfolioKernel:
    """Scalar kernels and their finite feature maps."""

    def __init__(
        self,
        kernel="linear",
        ell=1.0,
        n_random_features=None,
        random_state=0,
    ):
        self.kernel = kernel
        self.ell = ell
        self.n_random_features = n_random_features
        self.random_state = random_state

    # ---------------------------------------------------------
    # Scalar kernels
    # ---------------------------------------------------------

    @staticmethod
    def static_gram(X, Y):
        return np.ones((len(X), len(Y)))

    @staticmethod
    def linear_gram(X, Y):
        return 1.0 + X @ Y.T

    def gaussian_gram(self, X, Y):
        distance_squared = (
            np.sum(X**2, axis=1)[:, None]
            + np.sum(Y**2, axis=1)[None, :]
            - 2.0 * X @ Y.T
        )

        distance_squared = np.maximum(distance_squared, 0.0)

        return np.exp(
            -distance_squared / (2.0 * self.ell**2)
        )

    def matern32_gram(self, X, Y):
        distance_squared = (
            np.sum(X**2, axis=1)[:, None]
            + np.sum(Y**2, axis=1)[None, :]
            - 2.0 * X @ Y.T
        )

        distance = np.sqrt(
            np.maximum(distance_squared, 0.0)
        )

        scaled_distance = (
            np.sqrt(3.0) * distance / self.ell
        )

        return (
            (1.0 + scaled_distance)
            * np.exp(-scaled_distance)
        )

    def matern12_gram(self, X, Y):
        distance_squared = (
            np.sum(X**2, axis=1)[:, None]
            + np.sum(Y**2, axis=1)[None, :]
            - 2.0 * X @ Y.T
        )

        distance = np.sqrt(
            np.maximum(distance_squared, 0.0)
        )

        return np.exp(-distance / self.ell)

    def matern52_gram(self, X, Y):
        distance_squared = (
            np.sum(X**2, axis=1)[:, None]
            + np.sum(Y**2, axis=1)[None, :]
            - 2.0 * X @ Y.T
        )

        distance = np.sqrt(
            np.maximum(distance_squared, 0.0)
        )
        scaled_distance = (
            np.sqrt(5.0) * distance / self.ell
        )

        return (
            1.0
            + scaled_distance
            + scaled_distance**2 / 3.0
        ) * np.exp(-scaled_distance)

    @staticmethod
    def ntk_gram(X, Y):
        dot_product = X @ Y.T

        norm_x = np.linalg.norm(X, axis=1)
        norm_y = np.linalg.norm(Y, axis=1)
        denominator = norm_x[:, None] * norm_y[None, :]

        cosine = np.divide(
            dot_product,
            denominator,
            out=np.zeros_like(dot_product),
            where=denominator != 0,
        )

        angle = np.arccos(
            np.clip(cosine, -1.0, 1.0)
        )

        nngp = (
            denominator
            * (
                np.sin(angle)
                + (np.pi - angle) * np.cos(angle)
            )
            / (2.0 * np.pi)
        )

        derivative = (
            np.pi - angle
        ) / (2.0 * np.pi)

        return nngp + dot_product * derivative

    def gram(self, X, Y=None):
        X = np.atleast_2d(
            np.asarray(X, dtype=float)
        )

        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(
                np.asarray(Y, dtype=float)
            )

        kernel_function = getattr(
            self,
            f"{self.kernel}_gram",
        )

        return kernel_function(X, Y)

    def __call__(self, x, y):
        value = self.gram(
            np.asarray(x)[None, :],
            np.asarray(y)[None, :],
        )[0, 0]

        return float(value)

    # ---------------------------------------------------------
    # Finite feature maps
    # ---------------------------------------------------------

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

    def initialize_random_features(self, input_dimension):
        if self.n_random_features is None:
            raise ValueError(
                "n_random_features is required for Gaussian and Matérn features."
            )

        random_generator = np.random.default_rng(
            self.random_state
        )

        weights = random_generator.normal(
            size=(
                input_dimension,
                self.n_random_features,
            )
        )

        if self.kernel == "gaussian":
            weights = weights / self.ell

        matern_degrees = {
            "matern12": 1.0,
            "matern32": 3.0,
            "matern52": 5.0,
        }

        if self.kernel in matern_degrees:
            degrees = matern_degrees[self.kernel]
            chi_squared = random_generator.chisquare(
                df=degrees,
                size=self.n_random_features,
            )

            weights = (
                weights
                / np.sqrt(chi_squared[None, :] / degrees)
                / self.ell
            )

        self.random_weights = weights
        self.random_phases = random_generator.uniform(
            0.0,
            2.0 * np.pi,
            size=self.n_random_features,
        )

    def initialize_ntk_features(self, input_dimension):
        random_generator = np.random.default_rng(
            self.random_state
        )

        self.ntk_weights = random_generator.normal(
            size=(
                input_dimension,
                self.n_random_features,
            )
        )
        self.ntk_directions = random_generator.normal(
            size=(
                input_dimension,
                self.n_random_features,
            )
        )

    def random_features(self, X):
        if not hasattr(self, "random_weights"):
            self.initialize_random_features(
                X.shape[1]
            )

        return (
            np.sqrt(2.0 / self.n_random_features)
            * np.cos(
                X @ self.random_weights
                + self.random_phases
            )
        )

    def ntk_features(self, X):
        if self.n_random_features is None:
            raise ValueError(
                "n_random_features is required for NTK features."
            )

        if not hasattr(self, "ntk_weights"):
            self.initialize_ntk_features(
                X.shape[1]
            )

        projection = X @ self.ntk_weights
        scale = np.sqrt(self.n_random_features)

        first_part = np.maximum(
            projection,
            0.0,
        ) / scale

        second_part = (
            (projection > 0)
            * (X @ self.ntk_directions)
            / scale
        )

        return np.column_stack(
            (
                first_part,
                second_part,
            )
        )

    def features(self, X):
        X = np.atleast_2d(
            np.asarray(X, dtype=float)
        )

        if self.kernel == "static":
            return self.static_features(X)

        if self.kernel == "linear":
            return self.linear_features(X)

        if self.kernel in {
            "gaussian",
            "matern12",
            "matern32",
            "matern52",
        }:
            return self.random_features(X)

        if self.kernel == "ntk":
            return self.ntk_features(X)

        raise ValueError(
            f"No finite feature map is available for kernel '{self.kernel}'."
        )



def make_return_row(month, kernel):
    features = kernel.features(
        month["x"]
    )

    return (
        features.T @ month["r"]
    ) / month["n_assets"]


def make_return_dictionary(months, kernel):
    """Compute every monthly return row only once."""
    return {
        month["formation_date"]: make_return_row(
            month,
            kernel,
        )
        for month in months
    }


def make_return_matrix(months, kernel, return_dictionary=None):
    if return_dictionary is None:
        rows = [
            make_return_row(month, kernel)
            for month in months
        ]
    else:
        rows = [
            return_dictionary[month["formation_date"]]
            for month in months
        ]

    return np.vstack(rows)


def kernel_eigenvalues(return_matrix):
    """Eigenvalues of the empirical kernel second-moment matrix."""
    number_of_months = len(return_matrix)

    if return_matrix.shape[1] <= number_of_months:
        covariance_matrix = (
            return_matrix.T @ return_matrix
        ) / number_of_months
    else:
        covariance_matrix = (
            return_matrix @ return_matrix.T
        ) / number_of_months

    eigenvalues = np.linalg.eigvalsh(
        covariance_matrix
    )
    eigenvalues = np.maximum(
        eigenvalues,
        0.0,
    )

    return eigenvalues[::-1]


def effective_dimension(eigenvalues, lambda_value):
    """Compute sum(eigenvalue / (eigenvalue + lambda))."""
    return np.sum(
        eigenvalues
        / (eigenvalues + lambda_value)
    )


def make_complexity_lambda_grid(
    eigenvalues,
    number_of_lambdas=60,
    maximum_fraction=0.995,
):
    """Make lambdas that are evenly spaced in effective dimension."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvalues = eigenvalues[np.isfinite(eigenvalues)]

    if number_of_lambdas < 2:
        raise ValueError(
            "number_of_lambdas must be at least 2."
        )
    if not 0.0 < maximum_fraction < 1.0:
        raise ValueError(
            "maximum_fraction must be between 0 and 1."
        )
    if len(eigenvalues) == 0 or eigenvalues.max() <= 0:
        raise ValueError(
            "eigenvalues must contain a positive value."
        )

    largest = eigenvalues.max()
    active = eigenvalues[
        eigenvalues > largest * 1e-12
    ]

    target_complexities = np.linspace(
        0.25,
        maximum_fraction * len(active),
        number_of_lambdas,
    )

    smallest_lambda = active.min() * 1e-9
    largest_lambda = largest * len(active) * 100.0
    lambda_values = []

    for target in target_complexities:
        lower = smallest_lambda
        upper = largest_lambda

        for _ in range(100):
            middle = np.sqrt(lower * upper)
            complexity = effective_dimension(
                active,
                middle,
            )

            if complexity > target:
                lower = middle
            else:
                upper = middle

        lambda_values.append(
            np.sqrt(lower * upper)
        )

    return np.sort(np.asarray(lambda_values))


def fit_lambda_grid(return_matrix, lambda_grid):
    """Estimate one beta for every lambda with one SVD."""
    number_of_months = len(return_matrix)

    left_vectors, singular_values, right_vectors = np.linalg.svd(
        return_matrix,
        full_matrices=False,
    )

    target = (
        left_vectors.T
        @ np.ones(number_of_months)
    )

    betas = {}

    for lambda_value in lambda_grid:
        multiplier = (
            singular_values
            / (
                singular_values**2
                + number_of_months * lambda_value
            )
        )

        betas[lambda_value] = (
            right_vectors.T
            @ (multiplier * target)
        )

    eigenvalues = (
        singular_values**2
        / number_of_months
    )

    return betas, eigenvalues


def estimate_beta(return_matrix, lambda_value):
    betas, _ = fit_lambda_grid(
        return_matrix,
        [lambda_value],
    )

    return betas[lambda_value]
