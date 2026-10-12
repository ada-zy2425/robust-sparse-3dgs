#!/usr/bin/env bash
set -euo pipefail

# Resume the official archive with independent byte ranges.  The sequential
# downloader's .incoming file is retained as the already-verified prefix.
project_root="${1:?usage: acquire_mipnerf360_parallel.sh PROJECT_ROOT [JOBS]}"
jobs="${2:-8}"
case "$jobs" in
  ''|*[!0-9]*) echo "JOBS must be an integer" >&2; exit 2 ;;
esac
if (( jobs < 1 || jobs > 8 )); then
  echo "JOBS must be between 1 and 8" >&2
  exit 2
fi

url="https://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
expected_bytes=12535427936
expected_md5="cef2ef3aeaf0c062dbe65130bc249870"
data_root="$project_root/data/mipnerf360"
prefix="$data_root/360_v2.zip.incoming"
parts="$data_root/range_parts"
assembled="$data_root/360_v2.zip.assembled"
final="$data_root/360_v2.zip"
mkdir -p "$parts"

if [[ -e "$final" ]]; then
  echo "Refusing to overwrite existing final archive: $final" >&2
  exit 2
fi
if [[ ! -f "$prefix" ]]; then
  echo "Sequential prefix is missing: $prefix" >&2
  exit 2
fi
prefix_bytes="$(stat -c %s "$prefix")"
if (( prefix_bytes <= 0 || prefix_bytes >= expected_bytes )); then
  echo "Invalid prefix size: $prefix_bytes" >&2
  exit 2
fi
free_bytes="$(df -PB1 "$project_root" | awk 'NR==2 {print $4}')"
project_bytes="$(du -sb "$project_root" | awk '{print $1}')"
if (( free_bytes < 80 * 1024 * 1024 * 1024 )); then
  echo "Less than 80 GiB free; refusing data acquisition" >&2
  exit 2
fi
if (( project_bytes + 2 * expected_bytes > 200 * 1024 * 1024 * 1024 )); then
  echo "Projected project size exceeds the 200 GiB budget" >&2
  exit 2
fi

remaining=$((expected_bytes - prefix_bytes))
chunk=$(((remaining + jobs - 1) / jobs))
pids=()
for ((i=0; i<jobs; i++)); do
  start=$((prefix_bytes + i * chunk))
  end=$((start + chunk - 1))
  (( end >= expected_bytes )) && end=$((expected_bytes - 1))
  (( start > end )) && continue
  part="$parts/part.$(printf '%02d' "$i")"
  have=0
  [[ -f "$part" ]] && have="$(stat -c %s "$part")"
  need=$((end - start + 1))
  if (( have > need )); then
    echo "Oversized range part: $part" >&2
    exit 2
  fi
  if (( have == need )); then
    continue
  fi
  next=$((start + have))
  (
    curl --fail --location --retry 5 --retry-delay 5 --max-time 7200 \
      --range "$next-$end" --output - "$url" >> "$part"
    actual="$(stat -c %s "$part")"
    [[ "$actual" == "$need" ]] || {
      echo "Range $i has $actual bytes, expected $need" >&2
      exit 3
    }
  ) >"$parts/part.$(printf '%02d' "$i").log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
(( failed == 0 )) || {
  echo "At least one byte range failed; rerun to resume exact missing suffixes" >&2
  exit 3
}

cp "$prefix" "$assembled"
for ((i=0; i<jobs; i++)); do
  part="$parts/part.$(printf '%02d' "$i")"
  [[ -f "$part" ]] && cat "$part" >> "$assembled"
done
actual_bytes="$(stat -c %s "$assembled")"
[[ "$actual_bytes" == "$expected_bytes" ]] || {
  echo "Assembled size $actual_bytes != $expected_bytes" >&2
  exit 4
}
actual_md5="$(md5sum "$assembled" | awk '{print $1}')"
[[ "$actual_md5" == "$expected_md5" ]] || {
  echo "Assembled MD5 $actual_md5 != $expected_md5" >&2
  exit 4
}
unzip -tq "$assembled"
sha256sum "$assembled" > "$data_root/360_v2.zip.sha256"
mv "$assembled" "$final"
echo "ACQUISITION_PASS $final bytes=$actual_bytes md5=$actual_md5"
