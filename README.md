# Neurons

[![CI](https://github.com/dexwritescode/neurons/actions/workflows/ci.yml/badge.svg)](https://github.com/dexwritescode/neurons/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A from-scratch LLM inference engine and chat application. Built to understand how large language models actually work at the hardware level — using Metal/MLX, cuBLAS, and flash-attention directly rather than wrapping llama.cpp or Ollama.

---

## What it is

Neurons is a full-stack local AI system:

- **`compute/`** — C++23 inference library. Implements the transformer forward pass from first principles: quantized matmul, RoPE, RMSNorm, KV cache, sampling. Pluggable backends (`ComputeBackend` interface).
- **`service/`** — gRPC inference server (`neurons-service`) + OpenAI-compatible HTTP endpoint. Runs on any machine on your network.
- **`cli/`** — Terminal interface. Chat, download models, manage nodes, start a server.
- **`gui/`** — Flutter macOS app. Chat UI, model browser, multi-node management, live tok/s stats.

The GUI never links C++ directly. Locally it calls `libneurons_core.dylib` over `dart:ffi`; against remote machines it uses gRPC. The same `NeuronsClient` interface covers both.

---

## Feature highlights

| Feature | GUI | CLI | gRPC |
|---|:---:|:---:|:---:|
| Multi-turn chat | ✅ | ✅ | ✅ |
| Streaming generation | ✅ | ✅ | ✅ |
| Live tok/s + token counts | ✅ | ✅ | ✅ |
| Model download from HuggingFace | ✅ | ✅ | ✅ |
| Model search + browser | ✅ | ✅ | ✅ |
| HuggingFace auth (gated models) | ✅ | ✅ | ✅ |
| Sampling params (temp, top-p, top-k, rep-penalty) | ✅ | ✅ | ✅ |
| Multi-session chat history (JSON persistence) | ✅ | ✅ | — |
| Multi-node management | ✅ | ✅ | — |
| OpenAI-compatible HTTP endpoint | — | ✅ | — |
| Remote log streaming | ✅ | — | ✅ |
| MCP server management (add/remove/list/push) | 🚧 | 🚧 | ✅ |
| MCP permission rules (global/session/chat scopes) | 🚧 | 🚧 | ✅ |
| MCP tool approval flow (always_ask / always_allow / always_deny) | 🚧 | 🚧 | ✅ |

---

## Screenshots

<div align="center">
  <img src="https://github.com/user-attachments/assets/03d247c4-c8ba-4dd9-958d-4786a0d4f593" width="32%" />
  <img src="https://github.com/user-attachments/assets/8858675a-c34e-48e9-816a-872612be1794" width="32%" />
  <img src="https://github.com/user-attachments/assets/0084a5d8-cdbb-44c2-a177-fc5e13d80837" width="32%" />
</div>

---

## Supported models

| Family | Example repos | Backend |
|---|---|---|
| Llama 2/3, TinyLlama | `mlx-community/Llama-3.2-3B-Instruct-4bit` | MLX |
| Mistral | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | MLX |
| Qwen2 / Qwen2.5 | `mlx-community/Qwen2.5-7B-Instruct-4bit` | MLX |
| Gemma / Gemma2 / Gemma3 | `mlx-community/gemma-3-1b-it-qat-4bit` | MLX |
| fp16 / bf16 unquantized | any base HuggingFace safetensors repo | MLX |

All models are downloaded directly from HuggingFace in their `mlx-community` MLX-quantized variants for Apple Silicon. CUDA (cuBLAS + flash-attention) and ROCm backends are on the roadmap.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Flutter GUI  (macOS / Windows / Linux / mobile)     │
│  dart:ffi (local) · gRPC (remote nodes)              │
└──────────────────────┬───────────────────────────────┘
                       │ dart:ffi / gRPC
┌──────────────────────▼───────────────────────────────┐
│  libneurons_core  (C FFI surface)                    │
│  NeuronsServiceImpl → LanguageModel::load()          │
└──────────────────────┬───────────────────────────────┘
                       │  LanguageModel (interface)
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    LlamaModel     GemmaModel    (future)
    Llama/Mistral  Gemma 1-3
    Qwen2/2.5      GeGLU/QKV-norm
          └────────────┬────────────┘
                       │  ComputeBackend (interface)
          ┌────────────┼────────────┬──────────────┐
          ▼            ▼            ▼              ▼
    MLXBackend    CUDABackend  ROCmBackend   CPUBackend
    (done)        (roadmap)    (roadmap)     (roadmap)
```

---

## Prerequisites

**macOS (Apple Silicon) — primary platform**

```bash
# Xcode command line tools
xcode-select --install

# Homebrew dependencies
brew install cmake grpc protobuf

# Flutter SDK
# https://docs.flutter.dev/get-started/install/macos
```

**Linux / Windows** — CUDA/ROCm backends are on the roadmap. The gRPC service builds today; MLX inference requires Apple Silicon.

---

## Building

```bash
git clone https://github.com/dexwritescode/neurons.git
cd neurons
```

All C++ + Flutter targets are driven from the root `Makefile`:

Integration tests require model files and skip automatically when absent — see [`docs/models.md`](docs/models.md) for the full list and download commands.

```bash
make help          # list all targets

make all           # build compute + CLI + service
make cli           # CLI only
make service       # gRPC service only
make dylib         # libneurons_core.dylib (Flutter FFI dependency)

make tests         # build and run all C++ tests
make flutter-test  # run Flutter widget + unit tests

make run           # build dylib + launch Flutter app (debug)
make gui           # build dylib + Flutter macOS release app
```

---

## Quick start

### Download and run a model in the terminal

```bash
# Build the CLI
make cli

# Search for models
./build/bin/cli search "qwen 3b"

# Download one
./build/bin/cli download mlx-community/Qwen2.5-3B-Instruct-4bit

# Chat
./build/bin/cli chat mlx-community/Qwen2.5-3B-Instruct-4bit
```

### Run the GUI

```bash
make run
```

The app opens on the Chats screen. Go to **Browse** to search HuggingFace, download a model, then return to **Chats** — the model loads automatically when selected.

### Run as a server (OpenAI-compatible)

```bash
# Start with an HTTP endpoint on port 8080
./build/bin/cli server --http-port 8080 --model mlx-community/Qwen2.5-3B-Instruct-4bit

# Point any OpenAI client at it
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

Works with Cursor's "local model" setting, Continue.dev, and any client that supports the OpenAI chat completions API.

---

## CLI reference

```
neurons chat    <model>          Interactive multi-turn chat
neurons load    <model>          One-shot inference with --prompt
neurons search  <query>          Search HuggingFace
neurons download <repo-id>       Download a model
neurons list                     List local models
neurons server  [--http-port N]  Start gRPC + HTTP server
neurons node    add/remove/list  Manage remote nodes
neurons token   set/clear        HuggingFace auth token
neurons config  show/set         Configuration
```

---

## Remote nodes

Neurons supports connecting multiple machines as inference nodes. Each node runs `neurons-service`; the GUI and CLI connect to all of them and route requests.

```bash
# On the remote machine
neurons server --grpc-port 50051 --http-port 8080

# On your laptop — add the node in the GUI (Nodes tab)
# or via CLI:
neurons node add my-server grpc://192.168.1.10:50051
neurons node use my-server
```

---

## Project layout

```
Neurons/
  compute/    C++ inference library (backends, models, tokenizer, sampler)
  cli/        CLI binary — links compute directly
  service/    gRPC server + OpenAI HTTP server + C FFI surface
  gui/        Flutter macOS app
  models/     HuggingFace client (search, download, metadata)
  Makefile    All build targets
```

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| A–E | ✅ | MLX backend, KV cache, sampling, Llama/Gemma/Qwen/Mistral |
| F | ✅ | Model family support (fp16/bf16, Gemma3, Qwen2.5) |
| G–I | ✅ | gRPC service, Flutter GUI, CLI, OpenAI HTTP, logging |
| J | 🚧 | File attach + RAG (embeddings, sqlite-vec) |
| K | 🚧 | Multi-node: routing, speculative decoding, failover |
| L.1–2 | ✅ | MCP client runtime — stdio/SSE transport, JSON-RPC 2.0, McpManager |
| L.3 | ✅ | MCP gRPC extensions — server/permission RPCs, tool approval flow |
| L.4–6 | 🚧 | MCP GUI — settings, permissions table, live approval prompt |
| L.8 | 🚧 | Built-in MCP servers (filesystem, shell) |
| B/C | 🚧 | CUDA (cuBLAS + flash-attention) and ROCm backends |

---

## Contributing

The project is structured so each layer can be understood and modified independently:

- **Add a new model family** — implement `LanguageModel` in `compute/`, add to the `load()` factory, write an integration test.
- **Add a new backend** — implement `ComputeBackend`, wire into `BackendFactory`.
- **Add a new CLI command** — add a command file in `cli/src/cli/commands/`, register in `main.cpp`.
- **Extend the GUI** — `gui/lib/` is a standard Flutter project; `NeuronsClient` is the interface to mock for tests.

All three interfaces (GUI, CLI, gRPC) must be updated together for any user-facing feature.

---

## License

MIT — see [LICENSE](LICENSE).
