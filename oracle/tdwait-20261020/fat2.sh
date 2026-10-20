#!/bin/sh
# Rebuild the fat bench/bench (the client's binary) after finding 120's review
# fixes. Writes fat2.build.log, fat2.time and fat2.done.
R=/home/kylea/code/cuda-sieve; W=$R/work/tdwait
cd $R/bench || exit 1
{ time nice make bench > $W/fat2.build.log 2>&1; echo "rc=$?" > $W/fat2.rc; } 2> $W/fat2.time
rc=$(cut -d= -f2 $W/fat2.rc)
echo "fat build exit=$rc $(stat -c %y bench | cut -c1-19)" > $W/fat2.done
[ "$rc" -eq 0 ]
