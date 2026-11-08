"""Gradient descent with Armijo backtracking line search (comparison baseline)."""
import time
import numpy as np


def gradient_descent(problem, x0, gradtol=1e-8, maxiter=200000, max_matvecs=None,
                     c1=1e-4, shrink=0.5, grow=2.0, alpha0=1.0, store_x=True):
    """Backtracking steps start from `grow` times the last accepted step (cheap and effective)."""
    x = np.array(x0, dtype=float)
    t0, mv0 = time.perf_counter(), problem.matvecs
    f, g = problem.cost(x), problem.grad(x)
    H = dict(f=[f], gradnorm=[np.linalg.norm(g)], matvecs=[problem.matvecs - mv0],
             time=[time.perf_counter() - t0], x=[x.copy()] if store_x else [])
    alpha, k = alpha0, 0
    while H['gradnorm'][-1] > gradtol and k < maxiter:
        if max_matvecs is not None and problem.matvecs - mv0 > max_matvecs:
            break
        k += 1
        gg = g @ g
        alpha *= grow
        while True:
            x_new = x - alpha * g
            f_new = problem.cost(x_new)
            if f_new <= f - c1 * alpha * gg or alpha < 1e-16:
                break
            alpha *= shrink
        x, f = x_new, f_new
        g = problem.grad(x)
        H['f'].append(f); H['gradnorm'].append(np.linalg.norm(g))
        H['matvecs'].append(problem.matvecs - mv0); H['time'].append(time.perf_counter() - t0)
        if store_x:
            H['x'].append(x.copy())
    return x, H
