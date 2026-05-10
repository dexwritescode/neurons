# neurons_gui

Flutter GUI for the Neurons inference engine. Connects to the `neuron` inference server (gRPC) and provides a local-first chat interface.

## Running the app (macOS)

### First run

**Step 1 — Build and stage the dylib** (from the repo root):

```bash
make dylib
```

This compiles `libneurons_core.dylib` and installs it to `gui/macos/neurons_core_lib/`, where the Xcode build phase picks it up.

**Step 2 — Start the inference server** (from the repo root):

```bash
./build/bin/neuron serve
```

**Step 3 — Run the Flutter app** (from the `gui/` directory):

```bash
flutter run -d macos
```

### Subsequent runs

If you haven't changed any C++ code, skip Step 1:

```bash
./build/bin/neuron serve &
flutter run -d macos
```

### Regenerating proto stubs

If `service/proto/neurons.proto` changes, regenerate the Dart stubs:

```bash
./scripts/gen_proto.sh
```

## Running tests

```bash
flutter test
```