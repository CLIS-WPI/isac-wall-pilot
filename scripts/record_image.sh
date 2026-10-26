#!/usr/bin/env bash
# Host-side image provenance. The container cannot inspect its own image.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/preflight

image_id="$(docker image inspect isac-wall-pilot:2.2.0 --format '{{.Id}}')"
image_repo_digests="$(docker image inspect isac-wall-pilot:2.2.0 --format '{{json .RepoDigests}}')"
base_repo_digests="$(docker image inspect nvidia/cuda:12.6.3-devel-ubuntu24.04 --format '{{json .RepoDigests}}')"
base_id="$(docker image inspect nvidia/cuda:12.6.3-devel-ubuntu24.04 --format '{{.Id}}')"

jq -n \
  --arg image_id "$image_id" \
  --argjson image_repo_digests "$image_repo_digests" \
  --arg base_id "$base_id" \
  --argjson base_repo_digests "$base_repo_digests" \
  '{
    image: "isac-wall-pilot:2.2.0",
    image_id: $image_id,
    image_repo_digests: $image_repo_digests,
    base_image: "nvidia/cuda:12.6.3-devel-ubuntu24.04",
    base_image_id: $base_id,
    base_image_repo_digests: $base_repo_digests
  }' > results/preflight/image_info.json

cat results/preflight/image_info.json
