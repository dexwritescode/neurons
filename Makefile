# Neurons — top-level build orchestration
#
# Thin wrapper around CMake + Flutter.  All C++ targets share a single
# CMake build tree rooted at $(BUILD_DIR); Flutter is built separately.
#
# Prerequisites
#   C++:     cmake, a C++23 compiler, brew install grpc (macOS)
#   Flutter: flutter SDK in PATH
#   Protos:  dart pub global activate protoc_plugin  (for `make proto`)

BUILD_DIR  ?= build
BUILD_TYPE ?= Release
JOBS       ?= $(shell sysctl -n hw.logicalcpu 2>/dev/null || nproc)

GUI_DIR      := gui
STAGING_DIR  := $(GUI_DIR)/macos/neurons_core_lib
STAGED_DYLIB := $(STAGING_DIR)/lib/libneurons_core.dylib

CMAKE_CONFIGURE := cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
CMAKE_BUILD     := cmake --build $(BUILD_DIR) -j$(JOBS)

# ── Helpers ──────────────────────────────────────────────────────────────────

.PHONY: _configure
_configure:
	@if [ ! -f "$(BUILD_DIR)/CMakeCache.txt" ]; then \
	    echo "==> Configuring ($(BUILD_TYPE))"; \
	    $(CMAKE_CONFIGURE); \
	fi

# ── Primary targets ──────────────────────────────────────────────────────────

.PHONY: all
all: _configure ## Build all C++ (compute + cli + service)
	$(CMAKE_BUILD)

.PHONY: compute
compute: _configure ## Build compute backend only
	$(CMAKE_BUILD) --target compute-only

.PHONY: cli
cli: _configure ## Build CLI only
	$(CMAKE_BUILD) --target cli-only

.PHONY: service
service: _configure ## Build neurons_service binary only
	$(CMAKE_BUILD) --target service-only

.PHONY: dylib
dylib: _configure ## Build + stage libneurons_core.dylib for Flutter FFI
	$(CMAKE_BUILD) --target neurons_core
	@echo "==> Installing dylib to $(STAGING_DIR)"
	cmake --install $(BUILD_DIR) --component neurons_core --prefix $(STAGING_DIR)
	@for cfg in Debug Release Profile; do \
	    dest="$(GUI_DIR)/build/macos/Build/Products/$$cfg/neurons_gui.app/Contents/MacOS"; \
	    if [ -d "$$dest" ]; then \
	        cp $(STAGED_DYLIB) "$$dest/libneurons_core.dylib"; \
	        echo "==> Copied into $$cfg bundle"; \
	    fi; \
	done

.PHONY: gui
gui: dylib ## Build dylib then Flutter macOS app
	cd $(GUI_DIR) && flutter build macos

.PHONY: run
run: dylib ## Build dylib + launch Flutter app in debug mode
	cd $(GUI_DIR) && flutter run -d macos

# ── Testing ──────────────────────────────────────────────────────────────────

.PHONY: tests
tests: _configure ## Build and run all C++ tests
	$(CMAKE_BUILD) --target all-tests
	ctest --test-dir $(BUILD_DIR) --output-on-failure

.PHONY: test-compute
test-compute: _configure ## Build and run compute tests only
	$(CMAKE_BUILD) --target compute_tests
	ctest --test-dir $(BUILD_DIR) --output-on-failure -R "Compute|compute|Qwen|Llama|Mistral|Tokenizer|SIMD|BPE|Model"

.PHONY: test-debug
test-debug: ## Build + run all unit tests in Debug mode (mirrors CI exactly)
	$(MAKE) _configure BUILD_TYPE=Debug BUILD_DIR=build-debug
	cmake --build build-debug -j$(JOBS) --target all-tests
	ctest --test-dir build-debug --output-on-failure \
	      --label-exclude integration \
	      --timeout 120

.PHONY: flutter-test
flutter-test: ## Run Flutter widget + unit tests
	cd $(GUI_DIR) && flutter test

# ── Code generation ──────────────────────────────────────────────────────────

.PHONY: proto
proto: ## Regenerate Dart gRPC stubs from neurons.proto
	./scripts/gen_proto.sh

# ── Build modes ──────────────────────────────────────────────────────────────

.PHONY: debug
debug: ## Build all C++ in Debug mode (uses build-debug/ directory)
	$(MAKE) all BUILD_TYPE=Debug BUILD_DIR=build-debug

.PHONY: release
release: ## Build all C++ in Release mode (uses build/ directory)
	$(MAKE) all BUILD_TYPE=Release BUILD_DIR=build

# ── Maintenance ──────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove C++ build trees (keeps Flutter build)
	rm -rf $(BUILD_DIR) build-debug

.PHONY: clean-all
clean-all: ## Remove C++ build trees and Flutter build artifacts
	rm -rf $(BUILD_DIR) build-debug
	rm -rf $(GUI_DIR)/build

.PHONY: reconfigure
reconfigure: ## Force CMake reconfigure (e.g. after adding a dependency)
	rm -f $(BUILD_DIR)/CMakeCache.txt
	$(MAKE) _configure

# ── Help ─────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show available targets and their descriptions
	@echo "Usage: make [target] [BUILD_TYPE=Debug|Release] [BUILD_DIR=path] [JOBS=N]"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  \033[1;32m%-16s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@echo "  make                   Build all C++ in Release mode"
	@echo "  make gui               Build dylib + Flutter macOS app"
	@echo "  make run               Build dylib + launch Flutter (debug)"
	@echo "  make tests             Build + run all C++ tests"
	@echo "  make debug             Build all C++ in Debug mode"
	@echo "  make clean && make     Clean rebuild"
	@echo "  make dylib             Rebuild FFI dylib only (Flutter hot-reload cycle)"
	@echo "  make proto             Regenerate Dart stubs after editing neurons.proto"

.DEFAULT_GOAL := help
