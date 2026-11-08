"""Tests for tCG (Question 1), TR-tCG (Question 2) and the derivatives (Question 3).

Each test_* function raises AssertionError on failure.  run_all() returns a list of
(name, description, passed) used by main.py to fill the table in the report.
"""
import numpy as np
from tcg import tcg
from trust_region import trust_region_tcg
from phase_retrieval import PhaseRetrievalProblem, FunctionProblem, generate_data

TOL = 1e-10


def _model(H, g, v):
    return 0.5 * v @ H @ v + g @ v


def _run(H, g, Delta, **kw):
    return tcg(lambda u: H @ u, np.asarray(g, float), Delta, **kw)


# ----------------------------------------------------------------------------- tCG
def test_tcg_1d_interior():
    v, _, info = _run(np.array([[2.0]]), [-1.0], 1.0)
    assert abs(v[0] - 0.5) < TOL and info['reason'] == 'tolerance'


def test_tcg_1d_boundary():
    v, _, info = _run(np.array([[2.0]]), [-4.0], 1.0)           # unconstrained minimiser is 2
    assert abs(v[0] - 1.0) < TOL and info['reason'] == 'boundary'


def test_tcg_1d_negative_curvature():
    v, _, info = _run(np.array([[-1.0]]), [0.5], 1.0)           # global minimiser on [-1,1] is -1
    assert abs(v[0] + 1.0) < TOL and info['reason'] == 'negative_curvature'


def test_tcg_zero_gradient():
    v, _, info = _run(np.diag([1.0, -1.0]), [0.0, 0.0], 1.0)    # corner case: g = 0 -> v = 0
    assert np.allclose(v, 0) and info['reason'] == 'zero_gradient'


def test_tcg_zero_hessian():
    v, _, info = _run(np.zeros((2, 2)), [3.0, 4.0], 2.0)        # linear model: boundary along -g
    assert np.allclose(v, [-1.2, -1.6], atol=TOL)


def test_tcg_tiny_radius():
    g = np.array([1.0, 2.0, -2.0])
    v, _, _ = _run(np.diag([1.0, 2.0, 3.0]), g, 1e-6)           # Cauchy-like step -Delta g/||g||
    assert np.allclose(v, -1e-6 * g / 3, atol=1e-16)


def test_tcg_2d_negative_curvature():
    v, _, info = _run(np.diag([-1.0, 1.0]), [1.0, 0.0], 1.0)    # exact solution (-1, 0)
    assert np.allclose(v, [-1, 0], atol=TOL) and info['reason'] == 'negative_curvature'


def test_tcg_3d_interior_equals_newton():
    H = np.diag([1.0, 2.0, 3.0]); g = np.array([1.0, -1.0, 0.5])
    v, _, info = _run(H, g, 10.0, kappa=1e-12, theta=0.0)       # tiny tolerance -> exact CG
    assert np.allclose(v, -np.linalg.solve(H, g), atol=1e-9)


def test_tcg_singular_psd_in_range():
    H = np.diag([1.0, 0.0, 0.0]); g = np.array([-1.0, 0.0, 0.0])
    v, _, _ = _run(H, g, 5.0)                                   # g in range(H): v = (1, 0, 0)
    assert np.allclose(v, [1, 0, 0], atol=TOL)


def test_tcg_singular_psd_outside_range():
    H = np.diag([1.0, 0.0]); g = np.array([0.0, -1.0])          # zero curvature along -g
    v, _, _ = _run(H, g, 3.0)
    assert np.allclose(v, [0, 3.0], atol=TOL)


def test_tcg_random_properties():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = rng.integers(1, 8)
        B = rng.standard_normal((n, n)); H = (B + B.T) / 2
        g = rng.standard_normal(n); Delta = 10 ** rng.uniform(-2, 1)
        v, Hv_v, info = _run(H, g, Delta)
        assert np.linalg.norm(v) <= Delta * (1 + 1e-12)                          # feasibility
        assert np.allclose(Hv_v, H @ v, atol=1e-9 * (1 + np.linalg.norm(H @ v)))  # tracked H v
        assert _model(H, g, v) <= -0.5 * np.linalg.norm(g) * min(
            Delta, np.linalg.norm(g) / max(np.linalg.norm(H, 2), 1e-300)) + 1e-12  # Cauchy decrease
        assert abs(info['model_decrease'] + _model(H, g, v)) < 1e-9 * (1 + abs(_model(H, g, v)))


def test_tcg_matrix_free_callable():
    rng = np.random.default_rng(2)
    B = rng.standard_normal((6, 6)); H = B @ B.T; g = rng.standard_normal(6)
    v1, _, _ = tcg(lambda u: H @ u, g, 0.3)
    v2, _, _ = tcg(lambda u: B @ (B.T @ u), g, 0.3)             # H only available through a function
    assert np.allclose(v1, v2, atol=1e-10)


# ------------------------------------------------------------------------- TR-tCG
def test_tr_quadratic():
    rng = np.random.default_rng(3)
    B = rng.standard_normal((8, 8)); Q = B @ B.T + np.eye(8); b = rng.standard_normal(8)
    P = FunctionProblem(lambda x: 0.5 * x @ Q @ x - b @ x, lambda x: Q @ x - b, lambda x, v: Q @ v)
    x, H = trust_region_tcg(P, np.zeros(8), Delta0=0.1, gradtol=1e-10)
    assert np.allclose(x, np.linalg.solve(Q, b), atol=1e-8)


def test_tr_rosenbrock():
    f = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    g = lambda x: np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])
    hv = lambda x, v: np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]]) @ v
    x, H = trust_region_tcg(FunctionProblem(f, g, hv), np.array([-1.2, 1.0]), gradtol=1e-10, maxiter=200)
    assert np.allclose(x, [1, 1], atol=1e-8)
    assert all(np.diff(np.array(H['f'])) <= 1e-14)                 # monotone decrease of f


def test_tr_escapes_saddle():
    f = lambda x: x[0] ** 2 - x[1] ** 2 + 0.25 * x[1] ** 4          # saddle at 0, minima at (0, +-sqrt 2)
    g = lambda x: np.array([2 * x[0], -2 * x[1] + x[1] ** 3])
    hv = lambda x, v: np.array([2 * v[0], (-2 + 3 * x[1] ** 2) * v[1]])
    x, H = trust_region_tcg(FunctionProblem(f, g, hv), np.array([0.1, 1e-3]), gradtol=1e-9)
    assert abs(abs(x[1]) - np.sqrt(2)) < 1e-7 and abs(x[0]) < 1e-7


def test_tr_radius_rules():
    P = FunctionProblem(lambda x: x @ x, lambda x: 2 * x, lambda x, v: 2 * v)
    _, H = trust_region_tcg(P, np.array([10.0, 0.0]), Delta0=1.0, Delta_bar=4.0, gradtol=1e-12)
    assert max(H['Delta']) <= 4.0 and H['Delta'][1] == 2.0 and H['gradnorm'][-1] <= 1e-12


# --------------------------------------------------------------------- derivatives
def _small_problem():
    A, y, xt, x0 = generate_data(n=20, seed=5)
    return PhaseRetrievalProblem(A, y), x0


def test_grad_vs_dense_formula():
    P, x = _small_problem()
    z = P.A @ x
    g_ref = P.A.T @ ((z ** 2 - P.y2) * z) / P.n
    assert np.allclose(P.grad(x), g_ref, atol=1e-12)


def test_hessvec_vs_dense_and_symmetry():
    P, x = _small_problem()
    rng = np.random.default_rng(6)
    u, v = rng.standard_normal(P.n), rng.standard_normal(P.n)
    H = P.hess_dense(x)
    assert np.allclose(P.hess_vec(x, v), H @ v, atol=1e-10)
    assert np.allclose(P.hess_op(x)(v), H @ v, atol=1e-10)
    assert abs(u @ P.hess_vec(x, v) - v @ P.hess_vec(x, u)) < 1e-10 * (1 + abs(u @ H @ v))


def test_hessvec_vs_finite_difference_of_gradient():
    P, x = _small_problem()
    v = np.random.default_rng(7).standard_normal(P.n); t = 1e-5
    fd = (P.grad(x + t * v) - P.grad(x - t * v)) / (2 * t)
    assert np.linalg.norm(fd - P.hess_vec(x, v)) < 1e-6 * np.linalg.norm(fd)


ALL_TESTS = [
    ('test_tcg_1d_interior', 'tCG 1-D, minimiser strictly inside the ball'),
    ('test_tcg_1d_boundary', 'tCG 1-D, minimiser outside: stops on the boundary'),
    ('test_tcg_1d_negative_curvature', 'tCG 1-D, concave model: goes to the correct boundary point'),
    ('test_tcg_zero_gradient', 'tCG corner case $g=0$ returns $v=0$'),
    ('test_tcg_zero_hessian', 'tCG with $H=0$: boundary step along $-g$'),
    ('test_tcg_tiny_radius', 'tCG with tiny $\\Delta$: Cauchy-like step $-\\Delta g/\\|g\\|$'),
    ('test_tcg_2d_negative_curvature', 'tCG 2-D indefinite $H$, exact solution known'),
    ('test_tcg_3d_interior_equals_newton', 'tCG 3-D, large ball, tiny tolerance: equals $-H^{-1}g$'),
    ('test_tcg_singular_psd_in_range', 'tCG 3-D singular PSD $H$, $g\\in\\mathrm{range}(H)$'),
    ('test_tcg_singular_psd_outside_range', 'tCG 2-D singular PSD $H$, zero curvature along $-g$'),
    ('test_tcg_random_properties', '200 random indefinite models: feasibility, Cauchy decrease, tracked $Hv$'),
    ('test_tcg_matrix_free_callable', 'tCG only needs a function $v\\mapsto Hv$'),
    ('test_tr_quadratic', 'TR-tCG on a strongly convex quadratic finds $Q^{-1}b$'),
    ('test_tr_rosenbrock', 'TR-tCG on Rosenbrock reaches $(1,1)$, $f$ monotone'),
    ('test_tr_escapes_saddle', 'TR-tCG leaves a saddle point (negative curvature)'),
    ('test_tr_radius_rules', 'TR-tCG radius updates respect $\\bar\\Delta$ and doubling rule'),
    ('test_grad_vs_dense_formula', 'gradient equals explicit formula'),
    ('test_hessvec_vs_dense_and_symmetry', 'Hessian-vector product equals dense Hessian; symmetry'),
    ('test_hessvec_vs_finite_difference_of_gradient', 'Hessian-vector product vs central differences of $\\nabla f$'),
]


def run_all():
    results = []
    for name, desc in ALL_TESTS:
        try:
            globals()[name]()
            ok = True
        except AssertionError:
            ok = False
        results.append((name, desc, ok))
    return results


if __name__ == '__main__':
    for name, desc, ok in run_all():
        print(('PASS ' if ok else 'FAIL ') + name)
