#!/usr/bin/env bash
set -euo pipefail

# Official archive linked by the pinned gsplat downloader. Dataset asset rights
# are treated separately from the MultiNeRF code license; keep RGB private
# until redistribution terms are independently confirmed.
project_root="${1:?usage: acquire_mipnerf360.sh PROJECT_ROOT}"
target_dir="$project_root/data/mipnerf360"
archive="$target_dir/360_v2.zip"
incoming="$archive.incoming"
url="https://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
expected_bytes=12535427936
expected_md5="cef2ef3aeaf0c062dbe65130bc249870"
min_free_bytes=$((80 * 1024 * 1024 * 1024))
project_limit_bytes=$((200 * 1024 * 1024 * 1024))

mkdir -p "$target_dir"
project_bytes="$(du -sb "$project_root" | awk '{print $1}')"
free_bytes="$(df -PB1 "$project_root" | awk 'NR==2 {print $4}')"
if (( project_bytes + expected_bytes > project_limit_bytes )); then
  echo "project disk budget would be exceeded" >&2
  exit 2
fi
if (( free_bytes < min_free_bytes + expected_bytes )); then
  echo "minimum shared-disk free-space reserve would be violated" >&2
  exit 2
fi

if [[ ! -f "$archive" ]]; then
  curl --fail --location --retry 3 --retry-delay 5 --continue-at - \
    --max-time 7200 --output "$incoming" "$url"
  [[ "$(stat -c %s "$incoming")" == "$expected_bytes" ]]
  [[ "$(md5sum "$incoming" | awk '{print $1}')" == "$expected_md5" ]]
  mv "$incoming" "$archive"
fi

[[ "$(stat -c %s "$archive")" == "$expected_bytes" ]]
[[ "$(md5sum "$archive" | awk '{print $1}')" == "$expected_md5" ]]
sha256sum "$archive" > "$archive.sha256"
unzip -tqq "$archive"
echo "$archive"
