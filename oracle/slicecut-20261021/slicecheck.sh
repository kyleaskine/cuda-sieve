#!/bin/sh
# night.sh's slice check, with the parse fixed: it read the "bucketed entries"
# line instead of the nslice_pow2 line, so it failed on correct values. Same
# check: 2,1147+ recounted from the built source must pad to <= 32 per side.
W=/home/kylea/code/cuda-sieve/work/land127
l=$(grep 'nslice_pow2:' $W/slices.c1147.txt)
p1=$(echo "$l" | sed -n 's/.*side 1 \([0-9]*\), side 0 \([0-9]*\) .*/\1/p')
p0=$(echo "$l" | sed -n 's/.*side 1 \([0-9]*\), side 0 \([0-9]*\) .*/\2/p')
echo "  $l"
if [ "${p1:-999}" -le 32 ] && [ "${p0:-999}" -le 32 ]; then echo "   PASS side 1 $p1, side 0 $p0"; exit 0; fi
echo "   FAIL side 1 ${p1:-?}, side 0 ${p0:-?}"; exit 1
