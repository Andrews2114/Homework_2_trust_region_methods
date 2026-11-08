"""Real phase retrieval: data generation, cost, gradient, Hessian-vector product.

f(x) = 1/(4n) || (Ax)^{.2} - y^{.2} ||^2
grad f(x)      = (1/n) A^T ( (Ax)^3 - y^2 . (Ax) )          (entrywise ops)
Hess f(x)[v]   = (1/n) A^T ( (3 (Ax)^2 - y^2) . (A v) )
"""
import numpy as np


def generate_data(n=200, m_factor=6, sigma=1e-3, seed=0):
    """Exactly the instance of the assignment (same draw order as the handout)."""
    rng = np.random.default_rng(seed)
    m = m_factor * n
    A = rng.uniform(size=(m, n)) / np.sqrt(n)
    x_true = rng.standard_normal(n)
    y = np.abs(A @ x_true) + sigma * rng.standard_normal(m)
    x0 = rng.standard_normal(n)
    return A, y, x_true, x0


def relative_error(x, x_true):
    """e(x) = min(||x - x_true||, ||x + x_true||) / ||x_true||  (global sign ambiguity)."""
    nt = np.linalg.norm(x_true)
    return min(np.linalg.norm(x - x_true), np.linalg.norm(x + x_true)) / nt


class PhaseRetrievalProblem:
    """Cost, gradient and Hessian-vector product, with a counter of products with A / A^T.

    The counter `matvecs` is used to compare algorithms fairly (see report, Question 4).
    A one-entry cache avoids recomputing A x when cost / grad / hess_op are called at the same x.
    """

    def __init__(self, A, y):
        self.A, self.y = A, y
        self.y2 = y ** 2
        self.m, self.n = A.shape
        self.matvecs = 0
        self._cache_key, self._cache_z = None, None

    # -- low-level counted products -------------------------------------------------
    def _A(self, v):
        self.matvecs += 1
        return self.A @ v

    def _AT(self, u):
        self.matvecs += 1
        return self.A.T @ u

    def _forward(self, x):
        key = x.tobytes()
        if key != self._cache_key:
            self._cache_z, self._cache_key = self._A(x), key
        return self._cache_z

    # -- the three objects asked for in Question 3 ------------------------------------
    def cost(self, x):
        r = self._forward(x) ** 2 - self.y2
        return (r @ r) / (4 * self.n)

    def grad(self, x):
        z = self._forward(x)
        return self._AT((z ** 2 - self.y2) * z) / self.n

    def hess_vec(self, x, v):
        z = self._forward(x)
        return self._AT((3 * z ** 2 - self.y2) * self._A(v)) / self.n

    # -- interface used by trust_region_tcg ------------------------------------------
    def hess_op(self, x):
        """Return v -> Hess f(x)[v]; the weights 3 z^2 - y^2 are computed once per x."""
        w = 3 * self._forward(x) ** 2 - self.y2
        return lambda v: self._AT(w * self._A(v)) / self.n

    # -- slow reference implementation (dense Hessian), only for checks / timing -------
    def hess_dense(self, x):
        w = 3 * (self.A @ x) ** 2 - self.y2
        return (self.A.T * w) @ self.A / self.n


class FunctionProblem:
    """Wrap plain callables cost(x), grad(x), hess_vec(x, v) (used for the test functions)."""

    def __init__(self, cost, grad, hess_vec):
        self.cost, self.grad, self.hess_vec = cost, grad, hess_vec
        self.matvecs = 0

    def hess_op(self, x):
        return lambda v: self.hess_vec(x, v)
