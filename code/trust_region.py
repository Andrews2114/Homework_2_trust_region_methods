"""Trust-region method with truncated CG subproblem solver (TR-tCG)."""
import time
import numpy as np
from tcg import tcg


def trust_region_tcg(problem, x0, Delta0=1.0, Delta_bar=100.0, rho_prime=0.1,
                     gradtol=1e-8, maxiter=500, kappa=0.1, theta=1.0, store_x=True):
    """Minimise problem.cost using problem.grad and problem.hess_op (Hessian-vector products).

    The algorithm never sees x_true.  History (one entry per iteration, entry 0 = x0):
    f, gradnorm, Delta, rho, accepted, tcg_iters, tcg_reason, matvecs, time, x.
    """
    x = np.array(x0, dtype=float)
    Delta = Delta0
    t0 = time.perf_counter()
    mv0 = problem.matvecs
    f, g = problem.cost(x), problem.grad(x)
    Hop = problem.hess_op(x)
    H = dict(f=[f], gradnorm=[np.linalg.norm(g)], Delta=[Delta], rho=[np.nan], accepted=[True],
             tcg_iters=[0], tcg_reason=[''], matvecs=[problem.matvecs - mv0],
             time=[time.perf_counter() - t0], x=[x.copy()] if store_x else [])
    k = 0
    while H['gradnorm'][-1] > gradtol and k < maxiter and Delta > 1e-14:
        k += 1
        s, Hs, info = tcg(Hop, g, Delta, kappa=kappa, theta=theta)
        model_dec = info['model_decrease']
        x_new = x + s
        f_new = problem.cost(x_new)
        reg = max(1.0, abs(f)) * np.finfo(float).eps * 1e3      # guards ratio against round-off
        rho = (f - f_new + reg) / (model_dec + reg)
        hit = np.linalg.norm(s) >= (1 - 1e-8) * Delta
        if rho < 0.25:
            Delta_next = Delta / 4
        elif rho > 0.75 and hit:
            Delta_next = min(2 * Delta, Delta_bar)
        else:
            Delta_next = Delta
        accepted = bool(rho > rho_prime)
        if accepted:
            x, f = x_new, f_new
            g = problem.grad(x)                                  # reuses cached A x
            Hop = problem.hess_op(x)
        Delta = Delta_next
        H['f'].append(f); H['gradnorm'].append(np.linalg.norm(g)); H['Delta'].append(Delta)
        H['rho'].append(rho); H['accepted'].append(accepted)
        H['tcg_iters'].append(info['iters']); H['tcg_reason'].append(info['reason'])
        H['matvecs'].append(problem.matvecs - mv0); H['time'].append(time.perf_counter() - t0)
        if store_x:
            H['x'].append(x.copy())
    return x, H
