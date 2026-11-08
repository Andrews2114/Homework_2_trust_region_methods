"""MATH-329 HW2 -- run everything: tests, derivative checks, TR-tCG vs GD, figures, LaTeX macros.

    python main.py            (from the folder code/;  writes into ../results/)

Every number quoted in the report is written to results/numbers.tex by this script and read
by ../S.tex via \\input, so code, results and report always agree after one run.
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_retrieval import PhaseRetrievalProblem, generate_data, relative_error
from trust_region import trust_region_tcg
from gradient_descent import gradient_descent
import tests

# ------------------------------------------------------------------------ configuration
GROUP_ID = 'S'
N, M_FACTOR, SIGMA, SEED = 200, 6, 1e-3, 0           # assignment: n=200, m=6n, sigma=1e-3
GRADTOL = 1e-8
DELTA0, DELTA_BAR, RHO_PRIME, KAPPA, THETA = 1.0, 100.0, 0.1, 0.1, 1.0
N_STARTS = 10                                        # extra robustness experiment
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS, exist_ok=True)

macros = {}

# Formatting numbers for scientific notation
def sci(x, d=2):
    """LaTeX scientific notation."""
    if x == 0:
        return r'\ensuremath{0}'
    e = int(np.floor(np.log10(abs(x))))
    mant = x / 10 ** e
    if round(mant, d) >= 10:
        mant, e = mant / 10, e + 1
    return rf'\ensuremath{{{mant:.{d}f}\times10^{{{e}}}}}'

# Quick formatting for precision in floats
def fix(x, d=1):
    return f'{x:.{d}f}'

# Get first value index from threshold
def first_below(values, tol):
    idx = np.nonzero(np.asarray(values) <= tol)[0]
    return int(idx[0]) if idx.size else None

# Get logarithmic slope fit
def fit_slope(ts, errs, lo, hi):
    mask = (ts >= lo) & (ts <= hi)
    return np.polyfit(np.log10(ts[mask]), np.log10(errs[mask]), 1)[0]


# ------------------------------------------------------------------------- parameters
macros.update(ParamGroup=GROUP_ID, ParamN=str(N), ParamM=str(M_FACTOR * N), ParamSigma=sci(SIGMA, 0),
              ParamSeed=str(SEED), ParamGradTol=sci(GRADTOL, 0), ParamDeltaZero=fix(DELTA0),
              ParamDeltaBar=fix(DELTA_BAR, 0), ParamRhoPrime=fix(RHO_PRIME, 1),
              ParamKappa=fix(KAPPA, 1), ParamTheta=fix(THETA, 0), ParamNStarts=str(N_STARTS))

# Generate synthetic data
A, y, x_true, x0 = generate_data(N, M_FACTOR, SIGMA, SEED)

# ------------------------------------------------------------------------------ tests
res = tests.run_all()
n_pass = sum(ok for _, _, ok in res)
macros.update(NumTestsTotal=str(len(res)), NumTestsPassed=str(n_pass))
with open(os.path.join(RESULTS, 'tests_table.tex'), 'w') as fh:
    for name, desc, ok in res:
        fh.write(rf"\texttt{{{name.replace('_', chr(92) + '_')}}} & {desc} & {'pass' if ok else 'FAIL'} \\" + '\n')
with open(os.path.join(RESULTS, 'tests_table.tex'), 'a') as fh:
    fh.write(r'\bottomrule' + '\n')
with open(os.path.join(RESULTS, 'test_log.txt'), 'w') as fh:
    for name, desc, ok in res:
        fh.write(('PASS ' if ok else 'FAIL ') + name + '\n')
print(f'tests: {n_pass}/{len(res)} passed')

# ------------------------------------------------- Question 3: derivative checks + timing
P = PhaseRetrievalProblem(A, y)
rng = np.random.default_rng(123)
xc = x0.copy()                                       # a generic point, far from critical points
v = rng.standard_normal(N); v /= np.linalg.norm(v)
f0, g0, Hv0 = P.cost(xc), P.grad(xc), P.hess_vec(xc, v)
ts = np.logspace(-9, 1, 60)
err1 = np.array([abs(P.cost(xc + t * v) - f0 - t * (g0 @ v)) for t in ts])
err2 = np.array([abs(P.cost(xc + t * v) - f0 - t * (g0 @ v) - 0.5 * t ** 2 * (v @ Hv0)) for t in ts])
slope1, slope2 = fit_slope(ts, err1, 1e-4, 1e-2), fit_slope(ts, err2, 1e-3, 3e-2)
macros.update(CheckSlopeGrad=fix(slope1, 2), CheckSlopeHess=fix(slope2, 2))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for a, e, s, ref, lab in [(ax[0], err1, slope1, 2, r'$|f(x+tv)-f(x)-t\langle\nabla f(x),v\rangle|$'),
                          (ax[1], err2, slope2, 3, r'$|f(x+tv)-f(x)-t\langle\nabla f(x),v\rangle-\frac{t^2}{2}\langle v,\nabla^2 f(x)[v]\rangle|$')]:
    a.loglog(ts, e, 'o-', ms=3, label=f'error (fitted slope {s:.2f})')
    a.loglog(ts, e[np.argmin(abs(ts - 1e-3))] * (ts / 1e-3) ** ref, 'k--', lw=1, label=f'slope {ref}')
    a.set_xlabel('$t$'); a.set_title(lab, fontsize=8); a.legend(fontsize=8); a.grid(True, which='both', alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_derivative_checks.pdf')); plt.close(fig)

reps = 300
t_ = time.perf_counter()
for _ in range(reps):
    P.hess_vec(xc, v)
t_mf = (time.perf_counter() - t_) / reps
reps_d = 20
t_ = time.perf_counter()
for _ in range(reps_d):
    Hd = P.hess_dense(xc)
t_build = (time.perf_counter() - t_) / reps_d
t_ = time.perf_counter()
for _ in range(reps):
    Hd @ v
t_dmv = (time.perf_counter() - t_) / reps
macros.update(TimeHessVecFree=sci(t_mf, 1), TimeHessDenseBuild=sci(t_build, 1), TimeHessDenseMatVec=sci(t_dmv, 1),
              TimeHessRatio=fix(t_build / t_mf, 0), HessDiffNorm=sci(np.linalg.norm(Hd @ v - Hv0) / np.linalg.norm(Hv0), 1))

# ----------------------------------------------------------- Question 4: TR-tCG and GD
Ptr = PhaseRetrievalProblem(A, y)
x_tr, Htr = trust_region_tcg(Ptr, x0, Delta0=DELTA0, Delta_bar=DELTA_BAR, rho_prime=RHO_PRIME,
                             gradtol=GRADTOL, kappa=KAPPA, theta=THETA)
Pgd = PhaseRetrievalProblem(A, y)
x_gd, Hgd = gradient_descent(Pgd, x0, gradtol=GRADTOL)
err_tr = np.array([relative_error(x, x_true) for x in Htr['x']])
err_gd = np.array([relative_error(x, x_true) for x in Hgd['x']])
k_tr = len(Htr['f']) - 1
k_gd = len(Hgd['f']) - 1
i_tr = first_below(Htr['gradnorm'], GRADTOL)
i_gd = first_below(Hgd['gradnorm'], GRADTOL)
mv_tr, t_tr = Htr['matvecs'][i_tr], Htr['time'][i_tr]
mv_gd, t_gd = Hgd['matvecs'][i_gd], Hgd['time'][i_gd]
sign = '+' if x_tr @ x_true > 0 else '-'
n_rej = Htr['accepted'][1:].count(False)
n_negc = Htr['tcg_reason'].count('negative_curvature')
n_bdry = Htr['tcg_reason'].count('boundary')
n_tol = Htr['tcg_reason'].count('tolerance')
j_gd_at_tr = int(np.argmax(np.array(Hgd['matvecs']) >= mv_tr))   # first GD iterate that used >= mv_tr products
f_true = P.cost(x_true)
macros.update(
    TRIters=str(i_tr), TRMatvecs=str(mv_tr), TRTime=sci(t_tr, 2), TRFinalGrad=sci(Htr['gradnorm'][-1], 1),
    TRFinalErr=sci(err_tr[-1], 2), TRFinalCost=sci(Htr['f'][-1], 2), TRSign=sign, TRRejected=str(n_rej),
    TRTotalTCG=str(sum(Htr['tcg_iters'])), TRMaxTCG=str(max(Htr['tcg_iters'])),
    TRNegCurv=str(n_negc), TRBoundary=str(n_bdry), TRTolerance=str(n_tol),
    TRDeltaMax=fix(max(Htr['Delta']), 1), TRIterGradBelowEThree=str(first_below(Htr['gradnorm'], 1e-3)),
    GDIters=str(i_gd), GDMatvecs=str(mv_gd), GDTime=sci(t_gd, 2), GDFinalErr=sci(err_gd[i_gd], 2),
    GDGradAtTRBudget=sci(Hgd['gradnorm'][j_gd_at_tr], 1), GDErrAtTRBudget=sci(err_gd[j_gd_at_tr], 1),
    SpeedupMatvecs=fix(mv_gd / mv_tr, 0), SpeedupTime=fix(t_gd / t_tr, 0), SpeedupIters=fix(i_gd / i_tr, 0),
    CostAtTrue=sci(f_true, 2), InitialErr=fix(err_tr[0], 2), InitialGrad=sci(Htr['gradnorm'][0], 1),
    NormXTrue=fix(np.linalg.norm(x_true), 1))
print(f'TR-tCG: {i_tr} iters, {mv_tr} matvecs, {t_tr:.4f}s, err {err_tr[-1]:.2e}, sign {sign}')
print(f'GD    : {i_gd} iters, {mv_gd} matvecs, {t_gd:.3f}s, err {err_gd[i_gd]:.2e}')

fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
its = np.arange(len(Htr['f']))
ax[0].semilogy(its, Htr['gradnorm'], 'o-', ms=3, label=r'$\|\nabla f(x_k)\|$')
ax[0].semilogy(its, err_tr, 's-', ms=3, label=r'$e(x_k)$')
ax[0].set_xlabel('iteration $k$'); ax[0].legend(fontsize=8)
fgap = np.array(Htr['f']) - Htr['f'][-1]
ax[1].semilogy(its[fgap > 0], fgap[fgap > 0], 'o-', ms=3)
ax[1].set_xlabel('iteration $k$'); ax[1].set_title(r'$f(x_k)-f(x_{\mathrm{end}})$', fontsize=9)
ax[2].semilogy(its, Htr['Delta'], 'o-', ms=3, label=r'$\Delta_k$')
acc = np.array(Htr['accepted'])
ax[2].semilogy(its[~acc], np.array(Htr['Delta'])[~acc], 'rx', label='rejected')
ax[2].set_xlabel('iteration $k$'); ax[2].legend(fontsize=8)
for a in ax:
    a.grid(True, which='both', alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_trtcg.pdf')); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
ax[0].loglog(np.array(Hgd['matvecs']) + 1, Hgd['gradnorm'], label='GD + Armijo')
ax[0].loglog(np.array(Htr['matvecs']) + 1, Htr['gradnorm'], 'o-', ms=3, label='TR-tCG')
ax[0].set_xlabel('products with $A$ or $A^\\top$'); ax[0].set_title(r'$\|\nabla f(x_k)\|$', fontsize=9)
ax[1].loglog(np.array(Hgd['matvecs']) + 1, err_gd, label='GD + Armijo')
ax[1].loglog(np.array(Htr['matvecs']) + 1, err_tr, 'o-', ms=3, label='TR-tCG')
ax[1].set_xlabel('products with $A$ or $A^\\top$'); ax[1].set_title('$e(x_k)$', fontsize=9)
ax[2].loglog(np.array(Hgd['time']) + 1e-6, Hgd['gradnorm'], label='GD + Armijo')
ax[2].loglog(np.array(Htr['time']) + 1e-6, Htr['gradnorm'], 'o-', ms=3, label='TR-tCG')
ax[2].set_xlabel('wall-clock time (s)'); ax[2].set_title(r'$\|\nabla f(x_k)\|$', fontsize=9)
for a in ax:
    a.grid(True, which='both', alpha=.3); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_comparison.pdf')); plt.close(fig)

# ---- robustness: same A, y; different random starting points (fairness: same x0 for both)
succ_tr, succ_gd, it_tr, mv_tr_l, mv_gd_l, t_tr_l, t_gd_l, err_final = 0, 0, [], [], [], [], [], []
for s in range(1, N_STARTS + 1):
    xs = np.random.default_rng(1000 + s).standard_normal(N)
    p1, p2 = PhaseRetrievalProblem(A, y), PhaseRetrievalProblem(A, y)
    xa, Ha = trust_region_tcg(p1, xs, Delta0=DELTA0, Delta_bar=DELTA_BAR, rho_prime=RHO_PRIME,
                              gradtol=GRADTOL, kappa=KAPPA, theta=THETA, store_x=False)
    xb, Hb = gradient_descent(p2, xs, gradtol=GRADTOL, store_x=False)
    ea, eb = relative_error(xa, x_true), relative_error(xb, x_true)
    succ_tr += ea < 1e-2; succ_gd += eb < 1e-2
    ia, ib = first_below(Ha['gradnorm'], GRADTOL), first_below(Hb['gradnorm'], GRADTOL)
    it_tr.append(ia); mv_tr_l.append(Ha['matvecs'][ia]); mv_gd_l.append(Hb['matvecs'][ib])
    t_tr_l.append(Ha['time'][ia]); t_gd_l.append(Hb['time'][ib]); err_final.append(ea)
macros.update(MSTRSuccess=str(succ_tr), MSGDSuccess=str(succ_gd),
              MSTRItersMin=str(min(it_tr)), MSTRItersMax=str(max(it_tr)),
              MSTRMatvecsMedian=fix(np.median(mv_tr_l), 0), MSGDMatvecsMedian=fix(np.median(mv_gd_l), 0),
              MSTRTimeMedian=sci(np.median(t_tr_l), 2), MSGDTimeMedian=sci(np.median(t_gd_l), 2),
              MSErrMax=sci(max(err_final), 2))
print(f'multi-start: TR {succ_tr}/{N_STARTS}, GD {succ_gd}/{N_STARTS}')

# ---- conditioning (explains the gap between GD and TR-tCG; discussed in Question 5)
ev_AtA = np.linalg.eigvalsh(A.T @ A)
ev_H = np.linalg.eigvalsh(P.hess_dense(x_tr))
macros.update(EigAtAMax=fix(ev_AtA[-1], 0), EigAtAMin=fix(ev_AtA[0], 2), EigAtAMedian=fix(np.median(ev_AtA), 2),
              CondAtA=fix(ev_AtA[-1] / ev_AtA[0], 0), EigHessMin=sci(ev_H[0], 2), EigHessMax=fix(ev_H[-1], 1),
              CondHess=fix(ev_H[-1] / ev_H[0], 0))
print(f'cond(Hess at solution) = {ev_H[-1] / ev_H[0]:.1f},  lambda_min = {ev_H[0]:.3e}')

# ------------------------------------------------------------------- write macros file
with open(os.path.join(RESULTS, 'numbers.tex'), 'w') as fh:
    fh.write('% generated by code/main.py -- do not edit\n')
    for k, val in macros.items():
        fh.write(rf'\newcommand{{\{k}}}{{{val}\xspace}}' + '\n')
np.savez(os.path.join(RESULTS, 'solution.npz'), x_tr=x_tr, x_gd=x_gd, x_true=x_true)
print('wrote', os.path.abspath(RESULTS))
