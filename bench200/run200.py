#!/usr/bin/env python3
"""Run the selected kernels of spmm_vcsr over the matrices listed in a corpus CSV
(bench200/corpus200.csv).

One process per (matrix, algo). Each is passed --mtx/--algo/--csv plus the
arguments in EXTRA, which run.sh sets to --O, --tileK, --segw, --repeat, --warmup
and --stat; anything not in EXTRA stays at the binary's own default.

REPS=<n> (default 1) runs every (matrix, algo group) n times, each in a fresh
process, and tags rows with `rep`. analyze.py takes the per-matrix median, so a
one-off slow process (seen on odyssey: pds-80 30x slow for every kernel in one
process, hipSPARSE on cfd1 at 9.3 ms in one process and 0.11 ms in the next)
cannot decide a matrix's result on its own.

Usage: run200.py <binary> <corpus.csv> <mtx dir> <out.csv>
"""
import csv, io, os, subprocess, sys
from pathlib import Path

ALGOS = os.environ.get('ALGOS', 'rocsparse rocsparse_fp16 vcsr_mc vcsr_mc_fp32').split()

FIELDS = ['gpu', 'sm', 'mtx', 'O', 'segw', 'bundle', 'tileK', 'repeat',
          'algo', 'ms', 'gflops', 'validation', 'rel_l2', 'setup_ms', 'nnz', 'rep',
          'ms_mean', 'ms_median', 'convert_ms', 'upload_ms']
REPS = int(os.environ.get('REPS', '1'))
# Extra command-line arguments appended to every invocation (e.g. an accuracy run:
# EXTRA="--check cpu --lenient --bdist normal --repeat 1 --warmup 0").
EXTRA = os.environ.get('EXTRA', '').split()


def run(binary, corpus, mtxdir, out):
    with out.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for m in csv.DictReader(corpus.open()):
            mtx = mtxdir / f"{m['name']}.mtx"
            if not mtx.is_file():
                print(f"missing {mtx}", file=sys.stderr)
                continue
            for rep, algo in ((r, a) for r in range(REPS) for a in ALGOS):
                cmd = [binary, '--mtx', str(mtx), '--algo', algo, '--csv'] + EXTRA
                try:
                    # capture_output= and text= are 3.7+; the compute nodes run 3.6.
                    r = subprocess.run(cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       universal_newlines=True, timeout=300)
                except subprocess.TimeoutExpired:
                    print(f"timeout {m['name']} {algo} rep{rep}", file=sys.stderr)
                    continue
                if r.returncode != 0:
                    err = (r.stderr or '').strip().splitlines()
                    print(f"fail {m['name']} {algo} rep{rep}: {err[-1] if err else r.returncode}",
                          file=sys.stderr)
                    continue
                for row in csv.DictReader(io.StringIO(r.stdout)):
                    row['nnz'] = m['nnz']
                    row['rep'] = rep
                    w.writerow(row)
                fh.flush()
            print(m['name'], flush=True)


run(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
