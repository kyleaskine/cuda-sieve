#!/bin/sh
# Survivor-bitmap gate for the packed threshold scan: the standalone harness
# writes k_apply's survbits (--survbits), base (finding 116's fat bench/bench)
# against the experiment build, across the scan's paths -- packed store path
# (16-bit, region >= 6, logI >= 6), the atomicOr fallback (region 4/5, logI 5),
# the 8-bit one-cell loop, odd block sizes, both-even dropping, regions that
# span rows -- on the c183 job, side 1 and side 0.
R=/home/kylea/code/cuda-sieve; W=$R/work/scan-i16; B=$W/bench.f116; X=${X:-$W/exp/bench/bench}
D=$W/sb; mkdir -p "${D:?}"
cd $R/bench || exit 1
J="--poly ../oracle/input.job --fb1 ../oracle/c183.fb1"
n=0; bad=0
case_() {  # name args...
  name=$1; shift
  $B $J "$@" --reps 1 --survbits $D/$name.base.bin > $D/$name.base.out 2>&1; eb=$?
  $X $J "$@" --reps 1 --survbits $D/$name.exp.bin > $D/$name.exp.out 2>&1; ex=$?
  sb=$(grep -h "survivors:" $D/$name.base.out | awk '{print $2}'); sx=$(grep -h "survivors:" $D/$name.exp.out | awk '{print $2}')
  if [ $eb -ne 0 ] || [ $ex -ne 0 ] || [ ! -s $D/$name.base.bin ]; then r="RUN FAILED (exit $eb/$ex)"; bad=$((bad+1))
  elif cmp -s $D/$name.base.bin $D/$name.exp.bin && [ "$sb" = "$sx" ]; then r="IDENTICAL"
  else r="DIFFER"; bad=$((bad+1)); fi
  n=$((n+1))
  printf "  %-22s %-9s survivors %s / %s  bitmap %s B\n" "$name" "$r" "$sb" "$sx" "$(wc -c < $D/$name.base.bin 2>/dev/null)"
  rm -f "${D:?}/$name.base.bin" "${D:?}/$name.exp.bin"
}
case_ s1.default     --logI 15 --side 1
case_ s0.default     --logI 15 --side 0
case_ s1.nbe         --logI 15 --side 1 --not-both-even
case_ s0.nbe         --logI 15 --side 0 --not-both-even
case_ s1.region6     --logI 15 --side 1 --region 6 --not-both-even
case_ s1.thr96       --logI 15 --side 1 --apply-threads 96 --not-both-even
case_ s1.thr32       --logI 15 --side 1 --apply-threads 32
case_ s1.cells8      --logI 15 --side 1 --cells 8 --not-both-even
case_ s1.I12.r12     --logI 12 --J 2048 --side 1 --region 12 --not-both-even
case_ s1.I12.r6      --logI 12 --J 2048 --side 1 --region 6 --not-both-even
case_ s0.I12.r6      --logI 12 --J 2048 --side 0 --region 6 --not-both-even
case_ s1.I12.r5      --logI 12 --J 2048 --side 1 --region 5 --not-both-even
case_ s1.I12.r4      --logI 12 --J 2048 --side 1 --region 4
case_ s1.I6.r6       --logI 6 --J 8192 --side 1 --region 6 --not-both-even
case_ s1.I7.r9.nss   --logI 7 --J 4096 --side 1 --region 9 --no-smallsieve --not-both-even
case_ s1.I6.r8.nss   --logI 6 --J 8192 --side 1 --region 8 --no-smallsieve --not-both-even
case_ s0.I6.r8.nss   --logI 6 --J 8192 --side 0 --region 8 --no-smallsieve --not-both-even
case_ s1.I5.r8.nss   --logI 5 --J 16384 --side 1 --region 8 --no-smallsieve --not-both-even
echo "$n cases, $bad not identical"
