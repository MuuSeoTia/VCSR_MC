#!/usr/bin/env python3
"""
Every kernel vs the rocSPARSE default 
"""
import csv, sys
from analyze import load, median, geo_gstd

BASE = 'rocsparse'
ORDER = ['vcsr_mc', 'vcsr_mc_fp32']
LABEL = {}


def width(path):
    return int(next(csv.DictReader(open(path)))['O'])


for col, what in (('ms_mean', 'MEAN of 10 launches (original repo)'),
                  ('ms_median', 'MEDIAN of 10 launches (paper text)')):
    print(f"\n=== Speedup over rocSPARSE default -- {what} ===")
    print(f"{'O':>5}  {'kernel':<22}{'n':>4}{'geomean':>9}{'gstd':>7}{'median':>8}{'min':>7}{'max':>8}{'wins':>9}{'max rel_l2':>12}")
    pooled = {}
    for path in sys.argv[1:]:
        _, t, e, _ = load(path, col)
        present = {a for m in t for a in t[m]}
        for a in [a for a in ORDER if a in present]:
            sp = [t[m][BASE] / t[m][a] for m in t if t[m].get(BASE, 0) > 0 and t[m].get(a, 0) > 0]
            if not sp:
                continue
            pooled.setdefault(a, []).extend(sp)
            gm, gs = geo_gstd(sp)
            err = max(e[m][a] for m in e if a in e[m])
            print(f"{width(path):>5}  {LABEL.get(a, a):<22}{len(sp):>4}{gm:>8.2f}x{gs:>7.2f}{median(sp):>7.2f}x"
                  f"{min(sp):>6.2f}x{max(sp):>7.1f}x{sum(s > 1 for s in sp):>5}/{len(sp):<3}{err:>12.1e}")
        print()
    if len(sys.argv) > 2:
        for a, sp in pooled.items():
            gm, gs = geo_gstd(sp)
            print(f"{'all':>5}  {LABEL.get(a, a):<22}{len(sp):>4}{gm:>8.2f}x{gs:>7.2f}{median(sp):>7.2f}x"
                  f"{min(sp):>6.2f}x{max(sp):>7.1f}x{sum(s > 1 for s in sp):>5}/{len(sp):<3}")
print("\nPer matrix: median over repetitions (fresh processes). gstd = exp(sigma of log speedup).")
