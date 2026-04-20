#!/usr/bin/env bash
# Regenerate Dart gRPC stubs from service/proto/neurons.proto.
# Run from the repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO="$REPO_ROOT/service/proto/neurons.proto"
OUT="$REPO_ROOT/gui/lib/proto"

DART_PLUGIN="$(dart pub global run protoc_plugin --version > /dev/null 2>&1 && \
  which protoc-gen-dart 2>/dev/null || echo "$HOME/.pub-cache/bin/protoc-gen-dart")"

mkdir -p "$OUT"
protoc \
  --proto_path="$REPO_ROOT/service/proto" \
  --dart_out="grpc:$OUT" \
  --plugin="protoc-gen-dart=$DART_PLUGIN" \
  "$PROTO"

echo "Generated Dart stubs in $OUT"
