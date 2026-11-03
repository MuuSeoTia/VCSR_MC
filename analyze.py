#!/usr/bin/env python3
"""Helpers for orig_table.py: per-(matrix, kernel) medians from a run.sh results
CSV, and the geometric mean / geometric standard deviation of speedups."""
import csv, math, collections


def median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def load(path, col='ms'):
    """Per (matrix, algo): median ms over its rows (one per repetition), GFLOP/s from
    that median, and the maximum error."""
    ts, es = collections.defaultdict(lambda: collections.defaultdict(list)), \
             collections.defaultdict(lambda: collections.defaultdict(list))
    fl, nnz = {}, {}
    for r in csv.DictReader(open(path)):
        m = r['mtx'].split('/')[-1].replace('.mtx', '')
        # `col` picks another timing column (ms_mean / ms_median); GFLOP/s is then
        # recomputed from it through the FLOP count, which every row shares.
        ms, gf = float(r[col]), float(r['gflops']) * float(r['ms']) / float(r[col]) if float(r[col]) > 0 else 0.0
        ts[m][r['algo']].append(ms)
        es[m][r['algo']].append(float(r['rel_l2']))
        if ms > 0:
            fl[m] = gf * ms            # FLOP count / 1e6, identical for every row of m
        nnz[m] = int(r['nnz']) if 'nnz' in r and r['nnz'] else 0
    g, t, e = collections.defaultdict(dict), collections.defaultdict(dict), collections.defaultdict(dict)
    for m in ts:
        for a in ts[m]:
            t[m][a] = median(ts[m][a])
            g[m][a] = fl.get(m, 0) / t[m][a] if t[m][a] > 0 else 0
            e[m][a] = max(es[m][a])
    return g, t, e, nnz


def geo_gstd(vals):
    """Geometric mean and geometric standard deviation exp(sigma_log x)."""
    if not vals:
        return float('nan'), float('nan')
    lg = [math.log(v) for v in vals]
    mu = sum(lg) / len(lg)
    sd = math.sqrt(sum((x - mu) ** 2 for x in lg) / len(lg))
    return math.exp(mu), math.exp(sd)
