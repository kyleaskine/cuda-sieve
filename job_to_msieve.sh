#!/usr/bin/env bash
# Create the msieve input files used by the post-processing steps.

set -euo pipefail

usage() {
    echo "Usage: $0 [--force] JOB_FILE [OUTPUT_DIR]" >&2
    echo "       writes msieve.fb and worktodo.ini, and copies msieve.dat" >&2
    echo "       from the job file's directory when present. An existing" >&2
    echo "       OUTPUT_DIR/msieve.dat is kept unless --force is given." >&2
    exit 2
}

force=0
if [[ ${1:-} == --force ]]; then force=1; shift; fi
[[ $# -ge 1 && $# -le 2 ]] || usage

job=$1
[[ -f "$job" ]] || { echo "error: job file not found: $job" >&2; exit 1; }

job_dir=$(cd "$(dirname "$job")" && pwd)
job_file=$(basename "$job")
out_dir=${2:-$job_dir}
mkdir -p "$out_dir"
out_dir=$(cd "$out_dir" && pwd)

# Read one key/value pair from the GGNFS job format. Values are kept as
# strings so that the large decimal N is not rounded by shell arithmetic.
read_key() {
    local key=$1
    awk -v wanted="$key" '
        $1 == wanted ":" { print $2; found = 1; exit }
        END { if (!found) exit 1 }
    ' "$job"
}

required=(n skew Y0 Y1 rlim alim lpbr lpba)
declare -A value
for key in "${required[@]}"; do
    value[$key]=$(read_key "$key") || {
        echo "error: missing '$key:' in $job_file" >&2
        exit 1
    }
done

degree=$(awk '
    $1 ~ /^c[0-9]+:$/ {
        coeff_index = substr($1, 2, length($1) - 2) + 0
        if (!found || coeff_index > degree) degree = coeff_index
        found = 1
    }
    END { if (found) print degree; else exit 1 }
' "$job") || {
    echo "error: no polynomial coefficients (cN:) found in $job_file" >&2
    exit 1
}

is_nonnegative_integer() {
    [[ $1 =~ ^[0-9]+$ ]]
}

for key in rlim alim lpbr lpba; do
    is_nonnegative_integer "${value[$key]}" || {
        echo "error: '$key:' must be a non-negative integer" >&2
        exit 1
    }
done

# Do decimal multiplication as strings so this stays exact beyond awk's
# floating-point integer range (and avoids bash's signed shift limit).
pow2() {
    awk -v exponent="$1" '
        function double_decimal(s,    i, digit, carry, out) {
            carry = 0
            out = ""
            for (i = length(s); i > 0; --i) {
                digit = (substr(s, i, 1) * 2) + carry
                out = (digit % 10) out
                carry = int(digit / 10)
            }
            return (carry ? carry : "") out
        }
        BEGIN {
            result = "1"
            for (i = 0; i < exponent; ++i) result = double_decimal(result)
            print result
        }
    '
}

sr_lpmax=$(pow2 "${value[lpbr]}")
sa_lpmax=$(pow2 "${value[lpba]}")
fb_tmp=$(mktemp "$out_dir/.msieve.fb.XXXXXX")
ini_tmp=$(mktemp "$out_dir/.worktodo.ini.XXXXXX")
cleanup() { rm -f "$fb_tmp" "$ini_tmp"; }
trap cleanup EXIT

{
    printf 'N %s\n' "${value[n]}"
    printf 'SKEW %s\n' "${value[skew]}"
    printf 'R0 %s\n' "${value[Y0]}"
    printf 'R1 %s\n' "${value[Y1]}"
    for ((i = 0; i <= degree; ++i)); do
        # Sparse jobs are valid: an omitted coefficient is zero.
        coefficient=$(read_key "c$i" 2>/dev/null || printf '0')
        printf 'A%s %s\n' "$i" "$coefficient"
    done
    printf 'FRMAX %s\n' "${value[rlim]}"
    printf 'FAMAX %s\n' "${value[alim]}"
    printf 'SRLPMAX %s\n' "$sr_lpmax"
    printf 'SALPMAX %s\n' "$sa_lpmax"
} > "$fb_tmp"
printf '%s\n' "${value[n]}" > "$ini_tmp"

mv "$fb_tmp" "$out_dir/msieve.fb"
mv "$ini_tmp" "$out_dir/worktodo.ini"
trap - EXIT
echo "wrote $out_dir/msieve.fb"
echo "wrote $out_dir/worktodo.ini"

# Relations live next to the job file. Skip when the output directory is
# the same place, since cp refuses to copy a file onto itself.
dat_src="$job_dir/msieve.dat"
if [[ "$job_dir" == "$out_dir" ]]; then
    :
elif [[ -e "$out_dir/msieve.dat" && $force -eq 0 ]]; then
    echo "warning: $out_dir/msieve.dat exists; not overwritten (use --force)" >&2
elif [[ -f "$dat_src" ]]; then
    echo "copying $(du -h "$dat_src" | cut -f1) $dat_src ..."
    dat_tmp=$(mktemp "$out_dir/.msieve.dat.XXXXXX")
    trap 'rm -f "$dat_tmp"' EXIT
    cp "$dat_src" "$dat_tmp"
    mv "$dat_tmp" "$out_dir/msieve.dat"
    trap - EXIT
    echo "copied $out_dir/msieve.dat"
else
    echo "warning: $dat_src not found; msieve.dat not copied" >&2
fi
