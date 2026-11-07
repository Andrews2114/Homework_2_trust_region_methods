# MATH-329 HW2 -- group S

Run everything (tests, derivative checks, TR-tCG vs GD, figures, numbers for the report):

    cd code
    python main.py            # needs numpy, matplotlib; ~20 s on one core

This writes figures, `numbers.tex`, `tests_table.tex`, `test_log.txt` and `solution.npz` into `results/`.

Compile the report (reads `results/`, so run `main.py` first):

    pdflatex S.tex && pdflatex S.tex

Code map: `phase_retrieval.py` (data, cost, gradient, Hessian-vector), `tcg.py` (Question 1),
`trust_region.py` (Question 2), `gradient_descent.py` (baseline), `tests.py`, `main.py`.
The group member names in the header/footer are set in `S.tex` (`\groupmembers`).
