"""Truncated conjugate gradients (Steihaug-Toint) for the trust-region subproblem

    min_{||v|| <= Delta}  m(v) = 1/2 <v, H v> + <g, v>

H is given as a function Hv: v -> H v (never as a matrix).
"""
import numpy as np


def _boundary_tau(v, p, Delta):
    """Positive tau with ||v + tau p|| = Delta (requires ||v|| <= Delta), numerically stable."""
    a, b, c = p @ p, 2 * (v @ p), v @ v - Delta ** 2          # a tau^2 + b tau + c = 0, c <= 0
    disc = np.sqrt(max(b * b - 4 * a * c, 0.0))
    if b < 0:
        return (-b + disc) / (2 * a)
    return -2 * c / (b + disc) if (b + disc) > 0 else Delta / np.sqrt(a)


def tcg(Hv, g, Delta, kappa=0.1, theta=1.0, maxiter=None):
    """Return (v, Hv_of_v, info).

    Stopping: residual ||r|| <= ||g|| min(kappa, ||g||^theta), boundary hit, negative / zero
    curvature direction (then we go to the boundary along it), or maxiter (default n).
    info: dict(reason, iters, model_decrease = m(0) - m(v)).
    """
    g = np.asarray(g, dtype=float)
    n = g.size
    maxiter = n if maxiter is None else maxiter
    v, Hv_v = np.zeros(n), np.zeros(n)
    r, rr = g.copy(), g @ g
    gnorm = np.sqrt(rr)
    if gnorm == 0.0:
        return v, Hv_v, dict(reason='zero_gradient', iters=0, model_decrease=0.0)
    tol = gnorm * min(kappa, gnorm ** theta)
    p = -r
    reason, j = 'maxiter', 0
    for j in range(1, maxiter + 1):
        Hp = Hv(p)
        pHp = p @ Hp
        if pHp <= 0:                                           # nonpositive curvature
            tau = _boundary_tau(v, p, Delta)
            v, Hv_v, reason = v + tau * p, Hv_v + tau * Hp, 'negative_curvature'
            break
        alpha = rr / pHp
        v_new = v + alpha * p
        if v_new @ v_new >= Delta ** 2:                        # leaves the ball
            tau = _boundary_tau(v, p, Delta)
            v, Hv_v, reason = v + tau * p, Hv_v + tau * Hp, 'boundary'
            break
        v, Hv_v = v_new, Hv_v + alpha * Hp
        r = r + alpha * Hp
        rr_new = r @ r
        if np.sqrt(rr_new) <= tol:
            reason = 'tolerance'
            break
        p = -r + (rr_new / rr) * p
        rr = rr_new
    dec = -(g @ v) - 0.5 * (v @ Hv_v)
    return v, Hv_v, dict(reason=reason, iters=j, model_decrease=dec)
