# neurons_gui

Flutter GUI for the Neurons inference engine. Connects to `neurons_service` (gRPC) and provides a local-first chat interface.

## Running the app (macOS)

### First run

**Step 1 — Build and stage the service binary** (from the repo root):

```bash
./scripts/build_service_macos.sh cmake-build-debug
```

Replace `cmake-build-debug` with your CMake build directory if different. This compiles `neurons_service` and installs it to `gui/macos/neurons_service_bin/bin/neurons_service`, where the Xcode build phase picks it up.

**Step 2 — Run the Flutter app** (from the `gui/` directory):

```bash
flutter run -d macos
```

The script also copies the binary directly into the debug app bundle if it already exists. On launch, the app starts the service silently and goes straight to the model picker.

### Subsequent runs

If you haven't changed any C++ code, skip Step 1:

```bash
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