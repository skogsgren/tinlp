#!/bin/bash
set -e
TMPDIR=$(mktemp -d)
TMPPATH=$(dirname "$TMPDIR")
NAME=$(basename "$TMPDIR")
OUTFILE="./repo.tar.gz"

[ -f "$OUTFILE" ] && trash-put "$OUTFILE"
git ls-files --exclude-standard | xargs -I {} cp --parents {} "$TMPDIR"
git ls-files --exclude-standard -o | xargs -I {} cp --parents {} "$TMPDIR"
tar  cvfz "$OUTFILE" -C "$TMPPATH" "$NAME"
echo exported to "$OUTFILE"
