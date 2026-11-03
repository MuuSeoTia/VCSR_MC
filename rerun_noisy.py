#!/usr/bin/env python3
"""Rerun matrices whose repetitions disagree, for a run.sh sweep.

A matrix is noisy if either kernel's slowest repetition is more than SPREAD x its
fastest (on odyssey whole processes occasionally run 10-100x slow; when two of three
repetitions are hit, the median is too). Every noisy matrix is rerun with RERUN_REPS
fresh processes under the same protocol and its rows are replaced; the original CSV
is kept as <name>_raw.csv.

Usage: rerun_noisy.py <binary> <corpus.csv> <mtx dir> results_O*.csv
"""
import csv, collections, os, shutil, subprocess, sys
from pathlib import Path

SPREAD, RERUN_REPS = 2.0, 5
binary, corpus, mtxdir = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
here = Path(__file__).resolve().parent
for path in map(Path, sys.argv[4:]):
    raw_path = path.with_name(path.stem + '_raw.csv')
    if not raw_path.exists():
        shutil.copy(path, raw_path)
    rows = list(csv.DictReader(raw_path.open()))
    ms = collections.defaultdict(list)
    for r in rows:
        ms[(r['mtx'].split('/')[-1][:-4], r['algo'])].append(float(r['ms']))
    noisy = sorted({m for (m, _), v in ms.items() if min(v) > 0 and max(v) / min(v) > SPREAD})
    O, tk = rows[0]['O'], rows[0]['tileK']
    print(f"{path.name}: O={O}, {len(noisy)} noisy: {' '.join(noisy)}", flush=True)
    if not noisy:
        continue
    sub = path.with_name(path.stem + '_rerun_corpus.csv')
    with sub.open('w', newline='') as fh:
        reader = csv.DictReader(corpus.open())
        w = csv.DictWriter(fh, fieldnames=reader.fieldnames)
        w.writeheader()
        w.writerows(m for m in reader if m['name'] in noisy)
    out = path.with_name(path.stem + '_rerun.csv')
    # Same protocol as the sweep: its ALGOS and EXTRA, which run.sh exports;
    # --O / --tileK always come from this CSV.
    extra, rest = os.environ.get('EXTRA', '--repeat 10 --warmup 0 --stat median').split(), []
    while extra:
        tok = extra.pop(0)
        if tok in ('--O', '--tileK'):
            extra.pop(0)
        else:
            rest.append(tok)
    env = dict(os.environ, REPS=str(RERUN_REPS), ALGOS=os.environ.get('ALGOS', 'rocsparse,vcsr_mc'),
               EXTRA=' '.join(['--O', O, '--tileK', tk] + rest))
    subprocess.run([sys.executable, str(here / 'bench200/run200.py'), binary, str(sub), mtxdir, str(out)],
                   env=env, check=True, stdout=subprocess.DEVNULL)
    new = list(csv.DictReader(out.open()))
    fields = list(rows[0].keys())
    with path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(r for r in rows if r['mtx'].split('/')[-1][:-4] not in noisy)
        w.writerows(new)
    sub.unlink()
    print(f"  replaced with {len(new)} rows from {RERUN_REPS} reps -> {path.name}", flush=True)
