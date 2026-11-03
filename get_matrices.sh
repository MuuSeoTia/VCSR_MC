#!/bin/bash
# Download the 200 SuiteSparse matrices listed in bench200/corpus200.csv.

set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
CORPUS="${CORPUS:-bench200/corpus200.csv}"
D="${D:-$PWD/matrices200}"
LOG=logs/get_matrices.log
{
  echo "### corpus: $CORPUS   dest: $D   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  mkdir -p "$D/mtx" "$D/.tmp"
  BASE=https://suitesparse-collection-website.herokuapp.com/MM
  get_one() {
    local grp=$1 name=$2
    [ -f "$D/mtx/$name.mtx" ] && return 0
    curl -fsSL --retry 3 --max-time 900 -o "$D/.tmp/$name.tar.gz" "$BASE/$grp/$name.tar.gz" \
      || { echo "FAIL $grp/$name"; return 1; }
    tar -xzf "$D/.tmp/$name.tar.gz" -C "$D/.tmp" 2>/dev/null || { echo "UNTAR $name"; return 1; }

    f="$D/.tmp/$name/$name.mtx"
    [ -f "$f" ] || f=$(find "$D/.tmp/$name" -name '*.mtx' \
                         ! -name '*_b.mtx' ! -name '*_c.mtx' | head -1)
    [ -n "$f" ] && mv "$f" "$D/mtx/$name.mtx" || echo "NOMTX $name"
    rm -rf "$D/.tmp/$name" "$D/.tmp/$name.tar.gz"
  }
  export -f get_one; export D BASE
  tail -n +2 "$CORPUS" | awk -F, '{print $1, $2}' \
    | xargs -P 8 -n 2 bash -c 'get_one "$0" "$1"'
  echo "### downloaded: $(ls "$D"/mtx/*.mtx 2>/dev/null | wc -l) / $(($(wc -l < "$CORPUS")-1))"
  echo "### array-format (bad) files: $(grep -lE '^%%MatrixMarket matrix array' "$D"/mtx/*.mtx 2>/dev/null | wc -l)"
  echo "### size: $(du -shL "$D" | cut -f1)"
} 2>&1 | tee "$LOG"
echo "-> $LOG"
