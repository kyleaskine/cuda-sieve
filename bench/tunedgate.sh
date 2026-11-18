#!/bin/sh
# tunedgate.sh -- saved tuning (tuned.h) on the card: what bench applies,
# where it says so, and that the relations never change.
#
#   BENCH=./bench sh tunedgate.sh        (from bench/; OUT=DIR keeps the runs)
#
# Two jobs: c143 (logI 13, one slab per q, the row tier and its merge on) and
# c183 at --slab-j 4095 with no row tier (five slabs per q, where
# --slab-fill-ahead runs). Every saved-tuning file is written here, keyed on
# this card from bench's own tuned-run: line, and every run's relations are
# byte-compared with the job's run without one. Checks:
#   - no file: says so, and changes nothing
#   - an entry: applied, tagged [tuned] in the banner, recorded in the run
#     log; relations identical to the same flag given explicitly
#   - a speed knob or a flag of its family given by hand sets the whole
#     entry aside (finding 161: the knobs interact)
#   - --no-tuned: "would apply", nothing applied
#   - inapplicable saved values are skipped or turn themselves off, with
#     notes and "saved setting", never a refusal; the plan's slab check
#   - declined (another policy), a bad file, another card's entry, a
#     malformed line beside a good one
#   - on five slabs, a saved --slab-fill-ahead grid runs, tagged, and a
#     hand-given --fill-blocks leaves it at this card's default
# Refuses (exit 2) while an NFS@Home CUDA client or any other process is on
# the card; otherwise exits nonzero if any check failed.
set -u
BENCH=${BENCH:-./bench}
[ -x "$BENCH" ] || { echo "tunedgate: no $BENCH -- run make first"; exit 2; }
if ps -eo args | grep -v grep | grep -E 'ggnfs-sieve-client' | grep -qE 'engine=cuda|cuda-bench'; then
  echo "tunedgate: the NFS@Home CUDA client is running"; exit 2
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
  echo "tunedgate: another process is on the card"; exit 2
fi
if [ -n "${OUT:-}" ]; then O=$OUT; mkdir -p "$O" || exit 2; rm -f "$O"/*.out "$O"/*.rel "$O"/*.log "$O"/*.txt
else O=$(mktemp -d "${TMPDIR:-/tmp}/tunedgate.XXXXXX") || exit 2; trap 'rm -rf "$O"' EXIT; fi
fails=0; checks=0
ok()  { checks=$((checks + 1)); echo "PASS $*"; }
bad() { checks=$((checks + 1)); fails=$((fails + 1)); echo "FAIL $*"; }
# has NAME PATTERN WHAT: grep -E PATTERN in NAME's stdout+stderr
has()  { if grep -qE -- "$2" "$O/$1.out"; then ok "$1: $3"; else bad "$1: $3 (no /$2/ in $O/$1.out)"; fi; }
hasnt() { if grep -qE -- "$2" "$O/$1.out"; then bad "$1: $3 (/$2/ in $O/$1.out)"; else ok "$1: $3"; fi; }
# run NAME REF ARGS...: one run; exit 0 and relations identical to REF's
run() {
  nm=$1; ref=$2; shift 2
  "$BENCH" "$@" --relations "$O/$nm.rel" --log "$O/$nm.log" > "$O/$nm.out" 2>&1; rc=$?
  if [ $rc -ne 0 ]; then bad "$nm: bench exited $rc (see $O/$nm.out)"; return; fi
  if [ "$ref" = - ]; then
    [ -s "$O/$nm.rel" ] && ok "$nm: ran, $(wc -l < "$O/$nm.rel") relations" || bad "$nm: no relations"
  elif cmp -s "$O/$nm.rel" "$O/$ref.rel"; then ok "$nm: relations identical to $ref's"
  else bad "$nm: relations differ from $ref's"; fi
}
# key NAME: this card's key and record from NAME's tuned-run: line, as
# "card|cc|SMs|logI|J|region" and "side|alim|rlim"
key() { sed -n 's/^tuned-run: //p' "$O/$1.out" | awk -F'|' '{ print $1 "|" $2 "|" $3 "|" $4 "|" $5 "|" $6 }'; }
rec() { sed -n 's/^tuned-run: //p' "$O/$1.out" | awk -F'|' '{ print $7 "|" $8 "|" $9 }'; }
HDR='# cuda-sieve tuned v1'

C143="--pipeline --cofactor --poly ../oracle/c143.job --region 13 --maxbits 13 --sq-side 1 --lambda0 1.905 --lambda1 1.94 --qrange 2200000: --logI 13 --J 8192 --nq 30"
C183="--pipeline --cofactor --poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --slab-j 4095 --qrange 130000000: --fill-rows 0 --no-resieve-rows --nq 6"

echo "== c143, one slab per q"
run c143.base - $C143 --tuned "$O/missing.txt"
has c143.base "^tuned: none for .* \(no file .*missing.txt" "no file: says so"
has c143.base "^tuned-run: .*\|nofile\|\|" "no file: tuned-run status nofile"
has c143.base "^tuned-plan: 1 slab per q" "the plan line"
K=$(key c143.base); RC=$(rec c143.base)
[ -n "$K" ] || { echo "tunedgate: no tuned-run line; is $BENCH older than saved tuning?"; exit 1; }
echo "   key $K, record $RC"
side=${RC%%|*}; rest=${RC#*|}
E() { echo "$K|$1|$side|$rest|$2|$3|tunedgate"; }   # nslab policy flags

{ echo "$HDR"; E 1 1 "--fill-blocks 9216"; } > "$O/f.txt"
run c143.saved c143.base $C143 --tuned "$O/f.txt"
has c143.saved "^tuned: applied --fill-blocks 9216 for " "applied"
has c143.saved "^grid: 9216 x [0-9]+ for fill .*\[tuned\]" "tagged [tuned] in the banner"
hasnt c143.saved "this run also sets" "--maxbits, at logI here, is part of the plain base, not an override"
if grep -q "^# tuned .*applied --fill-blocks 9216" "$O/c143.saved.log"; then ok "c143.saved: in the run log"
else bad "c143.saved: no '# tuned' header line in the run log"; fi
run c143.explicit c143.saved $C143 --tuned "$O/missing.txt" --fill-blocks 9216
has c143.explicit "^grid: 9216 x [0-9]+ for fill .*\[--fill-blocks\]" "the explicit flag's tag"

run c143.beats c143.base $C143 --tuned "$O/f.txt" --fill-blocks 2304
has c143.beats "^grid: 2304 x [0-9]+ for fill .*\[--fill-blocks\]" "an explicit --fill-blocks wins"
has c143.beats "^tuned: not using .*f.txt line 2 .*: the command line sets --fill-blocks, and a saved entry applies whole or not at all" "the entry is set aside, saying why"
has c143.beats "^tuned-run: .*\|declined\|\|" "tuned-run: declined, nothing applied"

run c143.nt c143.base $C143 --tuned "$O/f.txt" --no-tuned
has c143.nt "^tuned: would apply \(but --no-tuned\) --fill-blocks 9216" "--no-tuned: would apply"
has c143.nt "^tuned-run: .*\|apply\|--fill-blocks 9216\|" "--no-tuned: tuned-run carries the incumbent"
has c143.nt "^grid: 4608 x [0-9]+ for fill \(absolute, not per SM\)(;|$)" "--no-tuned: the default grid, untagged"

{ echo "$HDR"; E 5 1 "--fill-gate-shift 2 --no-fill-merge --slab-fill-ahead --slab-fill-blocks 384"; } > "$O/g.txt"
run c143.gate c143.base $C143 --tuned "$O/g.txt" --no-fill-gate
has c143.gate "^tuned: not using .*g.txt line 2 .*: the command line sets the fill gate, and a saved entry applies whole" "a family flag (--no-fill-gate) sets the entry aside"
run c143.skip c143.base $C143 --tuned "$O/g.txt" --no-q-pipeline
has c143.skip "not the slab fill ahead: it needs --q-pipeline" "slab fill ahead skipped without the q pipeline, with a note"
has c143.skip "^tuned: applied --fill-gate-shift 2 --no-fill-merge for " "the rest applied"
run c143.sfa1 c143.base $C143 --tuned "$O/g.txt"
has c143.sfa1 "^tuned: applied --fill-gate-shift 2 --no-fill-merge --slab-fill-ahead --slab-fill-blocks 384 for " "all of it applied"
has c143.sfa1 "--slab-fill-ahead: off, because this plan has one slab per q.*; saved setting$" "slab fill ahead off on one slab, saying it was saved"
has c143.sfa1 "^tuned: note: the saved entry was tuned on a plan of 5 slabs per q and this one has 1" "the plan's slab check"

{ echo "$HDR"; E 1 0 "--fill-blocks 9216"; } > "$O/p.txt"
run c143.policy c143.base $C143 --tuned "$O/p.txt"
has c143.policy "^tuned: not using .* policy 0 and this binary's is 1" "another policy declined"
has c143.policy "^grid: 4608 x" "declined: the default grid"

{ echo "# cuda-sieve tuned v9"; E 1 1 "--fill-blocks 9216"; } > "$O/v.txt"
run c143.badfile c143.base $C143 --tuned "$O/v.txt"
has c143.badfile "warning: saved tuning .*v.txt ignored: its first line is not" "a bad file: warned, and the run goes on"

{ echo "$HDR"; echo "SOME OTHER CARD|9.0|132|13|8192|13|1|1|1|1|1|--fill-blocks 9216|x"; } > "$O/o.txt"
run c143.other c143.base $C143 --tuned "$O/o.txt"
has c143.other "^tuned: none for .* in .*o.txt" "another card's entry: none"

{ echo "$HDR"; echo "$K|1|garbage"; E 1 1 "--fill-blocks 9216"; } > "$O/m.txt"
run c143.malformed c143.base $C143 --tuned "$O/m.txt"
has c143.malformed "m.txt:2: .*skipped" "a malformed line warned about"
has c143.malformed "^tuned: applied --fill-blocks 9216 for " "the good entry beside it still applies"

echo "== c183, five slabs per q"
run c183.base - $C183 --tuned "$O/missing.txt"
has c183.base "^tuned-plan: 5 slabs per q" "five slabs"
K=$(key c183.base); RC=$(rec c183.base); side=${RC%%|*}; rest=${RC#*|}
{ echo "$HDR"; E 5 1 "--slab-fill-ahead --slab-fill-blocks 384"; } > "$O/s.txt"
run c183.saved c183.base $C183 --tuned "$O/s.txt"
has c183.saved "^tuned: applied --slab-fill-ahead --slab-fill-blocks 384 for " "applied"
has c183.saved "--slab-fill-ahead: each later slab.* 384 x 32 at the greatest priority \[tuned\].*; saved setting" "the slab fill ahead runs at the saved grid, tagged"
hasnt c183.saved "^tuned: note: the saved entry was tuned on a plan" "no slab-class note when the class matches"
run c183.hand c183.base $C183 --tuned "$O/s.txt" --fill-blocks 9216
has c183.hand "^tuned: not using .*s.txt line 2 .*: the command line sets --fill-blocks" "finding 161's pairing: a hand-given --fill-blocks sets the saved slab fill ahead aside"
has c183.hand "--slab-fill-ahead: each later slab.* 512 x 32 .*; this card's default" "the slab fill ahead at this card's default grid, not the saved one"

echo "tunedgate: $((checks - fails)) of $checks checks passed"
[ $fails -eq 0 ]
