import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:ffi/ffi.dart';
import 'package:fixnum/fixnum.dart' as fixnum;

import 'neurons_client.dart';

// ── Native type aliases ───────────────────────────────────────────────────────

final class NeuronsCoreOpaque extends Opaque {}

typedef _NeuronsCreateNative = Pointer<NeuronsCoreOpaque> Function(
    Pointer<Char> modelsDir);
typedef _NeuronsCreateDart = Pointer<NeuronsCoreOpaque> Function(
    Pointer<Char> modelsDir);

typedef _NeuronsDestroyNative = Void Function(Pointer<NeuronsCoreOpaque> h);
typedef _NeuronsDestroyDart = void Function(Pointer<NeuronsCoreOpaque> h);

typedef _NeuronsInitBackendNative = Int32 Function(
    Pointer<NeuronsCoreOpaque> h, Pointer<Char> err, Int32 errLen);
typedef _NeuronsInitBackendDart = int Function(
    Pointer<NeuronsCoreOpaque> h, Pointer<Char> err, int errLen);

typedef _NeuronsVoidCoreNative = Void Function(Pointer<NeuronsCoreOpaque> h);
typedef _NeuronsVoidCoreDart = void Function(Pointer<NeuronsCoreOpaque> h);

typedef _NeuronsSetHfTokenNative = Void Function(
    Pointer<NeuronsCoreOpaque> h, Pointer<Char> token);
typedef _NeuronsSetHfTokenDart = void Function(
    Pointer<NeuronsCoreOpaque> h, Pointer<Char> token);

typedef _NeuronsGetStatusNative = Pointer<Char> Function(
    Pointer<NeuronsCoreOpaque> h);
typedef _NeuronsGetStatusDart = Pointer<Char> Function(
    Pointer<NeuronsCoreOpaque> h);

typedef _NeuronsListModelsNative = Pointer<Char> Function(
    Pointer<NeuronsCoreOpaque> h);
typedef _NeuronsListModelsDart = Pointer<Char> Function(
    Pointer<NeuronsCoreOpaque> h);

typedef _NeuronsLoadModelNative = Int32 Function(Pointer<NeuronsCoreOpaque> h,
    Pointer<Char> path, Pointer<Char> err, Int32 errLen);
typedef _NeuronsLoadModelDart = int Function(Pointer<NeuronsCoreOpaque> h,
    Pointer<Char> path, Pointer<Char> err, int errLen);

typedef _NeuronsDeleteModelNative = Int32 Function(Pointer<NeuronsCoreOpaque> h,
    Pointer<Char> path, Pointer<Char> err, Int32 errLen);
typedef _NeuronsDeleteModelDart = int Function(Pointer<NeuronsCoreOpaque> h,
    Pointer<Char> path, Pointer<Char> err, int errLen);

typedef _NeuronsSearchNative = Pointer<Char> Function(
    Pointer<NeuronsCoreOpaque> h,
    Pointer<Char> query,
    Int32 limit,
    Pointer<Char> sort,
    Pointer<Char> pipelineTagsJson,
    Pointer<Char> author,
    Pointer<Char> err,
    Int32 errLen);
typedef _NeuronsSearchDart = Pointer<Char> Function(
    Pointer<NeuronsCoreOpaque> h,
    Pointer<Char> query,
    int limit,
    Pointer<Char> sort,
    Pointer<Char> pipelineTagsJson,
    Pointer<Char> author,
    Pointer<Char> err,
    int errLen);

typedef _NeuronsGetModelInfoNative = Pointer<Char> Function(
    Pointer<NeuronsCoreOpaque> h,
    Pointer<Char> repoId,
    Pointer<Char> err,
    Int32 errLen);
typedef _NeuronsGetModelInfoDart = Pointer<Char> Function(
    Pointer<NeuronsCoreOpaque> h,
    Pointer<Char> repoId,
    Pointer<Char> err,
    int errLen);

typedef _NeuronsFreeStringNative = Void Function(Pointer<Char> s);
typedef _NeuronsFreeStringDart = void Function(Pointer<Char> s);

// Token callback: int (*)(const char* token, void* userdata)
typedef _NeuronsTokenCbNative = Int32 Function(
    Pointer<Char> token, Pointer<Void> userdata);

typedef _NeuronsGenerateNative = Int32 Function(
    Pointer<NeuronsCoreOpaque> h,
    Pointer<Char> userPrompt,
    Pointer<Char> historyJson,
    Int32 maxTokens,
    Int32 contextWindow,
    Float temperature,
    Float topP,
    Int32 topK,
    Float repPenalty,
    Pointer<NativeFunction<_NeuronsTokenCbNative>> cb,
    Pointer<Void> userdata,
    Pointer<Char> err,
    Int32 errLen);
typedef _NeuronsGenerateDart = int Function(
    Pointer<NeuronsCoreOpaque> h,
    Pointer<Char> userPrompt,
    Pointer<Char> historyJson,
    int maxTokens,
    int contextWindow,
    double temperature,
    double topP,
    int topK,
    double repPenalty,
    Pointer<NativeFunction<_NeuronsTokenCbNative>> cb,
    Pointer<Void> userdata,
    Pointer<Char> err,
    int errLen);

typedef _NeuronsDownloadCbNative = Int32 Function(
    Int64 bytesDone,
    Int64 bytesTotal,
    Double speedBps,
    Pointer<Char> currentFile,
    Pointer<Void> userdata);

typedef _NeuronsDownloadModelNative = Int32 Function(
    Pointer<NeuronsCoreOpaque> h,
    Pointer<Char> repoId,
    Pointer<NativeFunction<_NeuronsDownloadCbNative>> cb,
    Pointer<Void> userdata,
    Pointer<Char> err,
    Int32 errLen);
typedef _NeuronsDownloadModelDart = int Function(
    Pointer<NeuronsCoreOpaque> h,
    Pointer<Char> repoId,
    Pointer<NativeFunction<_NeuronsDownloadCbNative>> cb,
    Pointer<Void> userdata,
    Pointer<Char> err,
    int errLen);

// ── Library loading ───────────────────────────────────────────────────────────

DynamicLibrary _openNeuronsCore() {
  if (Platform.isMacOS) {
    // In a built app bundle: Contents/Frameworks/libneurons_core.dylib
    // In a debug run: next to the flutter runner executable
    final exe = File(Platform.resolvedExecutable);
    final bundleFrameworks = '${exe.parent.parent.path}/Frameworks/libneurons_core.dylib';
    if (File(bundleFrameworks).existsSync()) {
      return DynamicLibrary.open(bundleFrameworks);
    }
    // Debug: sibling of the executable
    final sibling = '${exe.parent.path}/libneurons_core.dylib';
    if (File(sibling).existsSync()) {
      return DynamicLibrary.open(sibling);
    }
    // Dev fallback: cmake build output.
    // From Contents/MacOS, go 9 levels up to reach the repo root where
    // cmake-build-debug lives (MacOS→Contents→.app→Debug→Products→Build→macos→build→gui→repo).
    final p = exe.parent.path;
    final devBuild = File(
            '$p/../../../../../../../../../cmake-build-debug/service/libneurons_core.dylib')
        .absolute
        .path;
    return DynamicLibrary.open(devBuild);
  }
  if (Platform.isWindows) return DynamicLibrary.open('neurons_core.dll');
  return DynamicLibrary.open('libneurons_core.so');
}

// ── Isolate message types ─────────────────────────────────────────────────────

class _GenArgs {
  _GenArgs({
    required this.libPath,
    required this.coreAddr,
    required this.userPrompt,
    required this.historyJson,
    required this.maxTokens,
    required this.contextWindow,
    required this.temperature,
    required this.topP,
    required this.topK,
    required this.repPenalty,
    required this.sendPort,
  });
  final String libPath;
  final int coreAddr;
  final String userPrompt;
  final String historyJson;
  final int maxTokens;
  final int contextWindow;
  final double temperature;
  final double topP;
  final int topK;
  final double repPenalty;
  final SendPort sendPort;
}

class _DownloadArgs {
  _DownloadArgs({
    required this.libPath,
    required this.coreAddr,
    required this.repoId,
    required this.sendPort,
  });
  final String libPath;
  final int coreAddr;
  final String repoId;
  final SendPort sendPort;
}

// ── Static callback state (one per background isolate) ───────────────────────

SendPort? _genSendPort;
SendPort? _downloadSendPort;

// C token callback — called from within neurons_generate on the isolate thread.
@pragma('vm:entry-point')
int _tokenCb(Pointer<Char> token, Pointer<Void> userdata) {
  _genSendPort?.send(token.cast<Utf8>().toDartString());
  return 0; // 0 = continue
}

// C download callback — called periodically during neurons_download_model.
@pragma('vm:entry-point')
int _downloadCb(int bytesDone, int bytesTotal, double speedBps,
    Pointer<Char> currentFile, Pointer<Void> userdata) {
  _downloadSendPort?.send({
    'bytesDone': bytesDone,
    'bytesTotal': bytesTotal,
    'speedBps': speedBps,
    'currentFile': currentFile.cast<Utf8>().toDartString(),
  });
  return 0;
}

// ── Isolate entry points ──────────────────────────────────────────────────────

void _generateIsolate(_GenArgs args) {
  final lib = DynamicLibrary.open(args.libPath);
  final generateFn = lib.lookupFunction<_NeuronsGenerateNative, _NeuronsGenerateDart>(
      'neurons_generate');

  final core = Pointer<NeuronsCoreOpaque>.fromAddress(args.coreAddr);
  _genSendPort = args.sendPort;

  final cb = NativeCallable<_NeuronsTokenCbNative>.isolateLocal(
    _tokenCb,
    exceptionalReturn: 1,
  );

  using((arena) {
    final promptPtr = args.userPrompt.toNativeUtf8(allocator: arena).cast<Char>();
    final histPtr = args.historyJson.toNativeUtf8(allocator: arena).cast<Char>();
    final errBuf = arena<Char>(512);

    generateFn(
      core,
      promptPtr,
      histPtr,
      args.maxTokens,
      args.contextWindow,
      args.temperature,
      args.topP,
      args.topK,
      args.repPenalty,
      cb.nativeFunction,
      nullptr,
      errBuf,
      512,
    );

    final errStr = errBuf.cast<Utf8>().toDartString();
    if (errStr.isNotEmpty) {
      args.sendPort.send({'error': errStr});
    }
  });

  cb.close();
  args.sendPort.send(null); // signal done
}

void _downloadIsolate(_DownloadArgs args) {
  final lib = DynamicLibrary.open(args.libPath);
  final downloadFn =
      lib.lookupFunction<_NeuronsDownloadModelNative, _NeuronsDownloadModelDart>(
          'neurons_download_model');

  final core = Pointer<NeuronsCoreOpaque>.fromAddress(args.coreAddr);
  _downloadSendPort = args.sendPort;

  final cb = NativeCallable<_NeuronsDownloadCbNative>.isolateLocal(
    _downloadCb,
    exceptionalReturn: 1,
  );

  using((arena) {
    final repoPtr = args.repoId.toNativeUtf8(allocator: arena).cast<Char>();
    final errBuf = arena<Char>(512);

    downloadFn(core, repoPtr, cb.nativeFunction, nullptr, errBuf, 512);

    final errStr = errBuf.cast<Utf8>().toDartString();
    if (errStr.isNotEmpty) {
      args.sendPort.send({'error': errStr});
    }
  });

  cb.close();
  args.sendPort.send(null);
}

// ── FfiNeuronsClient ──────────────────────────────────────────────────────────

/// Local implementation of [NeuronsClient] that loads inference in-process
/// via [dart:ffi] against [libneurons_core]. No subprocess, no gRPC for local.
class FfiNeuronsClient implements NeuronsClient {
  FfiNeuronsClient._({
    required DynamicLibrary lib,
    required String libPath,
    required Pointer<NeuronsCoreOpaque> core,
  })  : _lib = lib,
        _libPath = libPath,
        _core = core;

  final DynamicLibrary _lib;
  final String _libPath;   // needed by background isolates
  final Pointer<NeuronsCoreOpaque> _core;
  Isolate? _activeDownloadIsolate;

  // ── Factory ─────────────────────────────────────────────────────────────────

  /// Create and initialise the FFI client. Must be called from the main thread
  /// (MLX backend requires main-thread initialization).
  static Future<FfiNeuronsClient> create({String? modelsDir}) async {
    final lib = _openNeuronsCore();
    final libPath = _resolveLibPath(lib);

    final createFn =
        lib.lookupFunction<_NeuronsCreateNative, _NeuronsCreateDart>('neurons_create');
    final initFn = lib.lookupFunction<_NeuronsInitBackendNative,
        _NeuronsInitBackendDart>('neurons_init_backend');

    final home = Platform.environment['HOME'] ?? '';
    final dir = modelsDir ?? '$home/.neurons/models';

    late final Pointer<NeuronsCoreOpaque> core;

    using((arena) {
      final dirPtr = dir.toNativeUtf8(allocator: arena).cast<Char>();
      core = createFn(dirPtr);
    });

    if (core == nullptr) throw Exception('neurons_create returned null');

    final errBuf = calloc<Char>(512);
    try {
      final rc = initFn(core, errBuf, 512);
      if (rc != 0) {
        final msg = errBuf.cast<Utf8>().toDartString();
        throw Exception('Backend init failed: $msg');
      }
    } finally {
      calloc.free(errBuf);
    }

    return FfiNeuronsClient._(lib: lib, libPath: libPath, core: core);
  }

  // ── NeuronsClient interface ──────────────────────────────────────────────────

  @override
  Future<StatusResponse> getStatus() async {
    final getStatusFn = _lib.lookupFunction<_NeuronsGetStatusNative,
        _NeuronsGetStatusDart>('neurons_get_status');
    final freeFn = _lib.lookupFunction<_NeuronsFreeStringNative,
        _NeuronsFreeStringDart>('neurons_free_string');

    final jsonPtr = getStatusFn(_core);
    if (jsonPtr == nullptr) return StatusResponse();
    final jsonStr = jsonPtr.cast<Utf8>().toDartString();
    freeFn(jsonPtr);

    final m = json.decode(jsonStr) as Map<String, dynamic>;
    final resp = StatusResponse()
      ..modelLoaded = m['modelLoaded'] as bool? ?? false
      ..modelPath = m['modelPath'] as String? ?? ''
      ..modelType = m['modelType'] as String? ?? ''
      ..backend = m['backend'] as String? ?? ''
      ..vocabSize = (m['vocabSize'] as num? ?? 0).toInt()
      ..numLayers = (m['numLayers'] as num? ?? 0).toInt()
      ..maxPositionEmbeddings = (_protoInt(m['maxPositionEmbeddings']));
    final gpusList = m['gpus'] as List<dynamic>? ?? [];
    for (final g in gpusList) {
      final gm = g as Map<String, dynamic>;
      resp.gpus.add(GpuSlot()
        ..gpuId = gm['gpuId'] as String? ?? '0'
        ..gpuName = gm['gpuName'] as String? ?? ''
        ..vramTotalBytes = fixnum.Int64(_protoInt(gm['vramTotalBytes']))
        ..vramUsedBytes  = fixnum.Int64(_protoInt(gm['vramUsedBytes']))
        ..loadedModel = gm['loadedModel'] as String? ?? ''
        ..modelType   = gm['modelType']   as String? ?? ''
        ..tokPerSec   = (gm['tokPerSec'] as num? ?? 0).toDouble());
    }
    return resp;
  }

  @override
  Future<LoadModelResponse> loadModel(String modelPath) async {
    final loadFn = _lib.lookupFunction<_NeuronsLoadModelNative,
        _NeuronsLoadModelDart>('neurons_load_model');

    final errBuf = calloc<Char>(512);
    try {
      final pathPtr = modelPath.toNativeUtf8().cast<Char>();
      final rc = loadFn(_core, pathPtr, errBuf, 512);
      calloc.free(pathPtr);
      if (rc != 0) {
        final msg = errBuf.cast<Utf8>().toDartString();
        return LoadModelResponse()
          ..success = false
          ..error = msg;
      }
      // Re-read status to get model metadata
      final status = await getStatus();
      return LoadModelResponse()
        ..success = true
        ..modelType = status.modelType
        ..vocabSize = status.vocabSize
        ..numLayers = status.numLayers
        ..maxPositionEmbeddings = status.maxPositionEmbeddings;
    } finally {
      calloc.free(errBuf);
    }
  }

  @override
  Future<UnloadModelResponse> unloadModel() async {
    final unloadFn = _lib.lookupFunction<_NeuronsVoidCoreNative,
        _NeuronsVoidCoreDart>('neurons_unload_model');
    unloadFn(_core);
    return UnloadModelResponse()..success = true;
  }

  @override
  Future<bool> deleteModel(String path) async {
    final deleteFn = _lib.lookupFunction<_NeuronsDeleteModelNative,
        _NeuronsDeleteModelDart>('neurons_delete_model');
    final errBuf = calloc<Char>(512);
    try {
      final pathPtr = path.toNativeUtf8().cast<Char>();
      final rc = deleteFn(_core, pathPtr, errBuf, 512);
      calloc.free(pathPtr);
      return rc == 0;
    } finally {
      calloc.free(errBuf);
    }
  }

  @override
  Future<ListModelsResponse> listModels({String modelsDir = ''}) async {
    final listFn = _lib.lookupFunction<_NeuronsListModelsNative,
        _NeuronsListModelsDart>('neurons_list_models');
    final freeFn = _lib.lookupFunction<_NeuronsFreeStringNative,
        _NeuronsFreeStringDart>('neurons_free_string');

    final jsonPtr = listFn(_core);
    if (jsonPtr == nullptr) return ListModelsResponse();
    final jsonStr = jsonPtr.cast<Utf8>().toDartString();
    freeFn(jsonPtr);

    final m = json.decode(jsonStr) as Map<String, dynamic>;
    final models = (m['models'] as List<dynamic>? ?? []).map((e) {
      final em = e as Map<String, dynamic>;
      return ModelInfo()
        ..name = em['name'] as String? ?? ''
        ..path = em['path'] as String? ?? ''
        ..sizeBytes = fixnum.Int64(_protoInt(em['sizeBytes']));
    }).toList();
    return ListModelsResponse()..models.addAll(models);
  }

  @override
  Stream<GenerateResponse> generate(
    String prompt, {
    List<ChatMessage> history = const [],
    SamplingParams? params,
  }) {
    final ctrl = StreamController<GenerateResponse>();
    final port = ReceivePort();

    final historyJson = json.encode(history
        .map((m) => {'role': m.role, 'content': m.content})
        .toList());

    port.listen((msg) {
      if (msg == null) {
        ctrl.close();
        port.close();
      } else if (msg is String) {
        ctrl.add(GenerateResponse()..token = msg);
      } else if (msg is Map && msg['error'] != null) {
        ctrl.addError(Exception(msg['error'] as String));
        ctrl.close();
        port.close();
      }
    });

    Isolate.spawn(
      _generateIsolate,
      _GenArgs(
        libPath: _libPath,
        coreAddr: _core.address,
        userPrompt: prompt,
        historyJson: historyJson,
        maxTokens: params?.maxTokens.toInt() ?? 200,
        contextWindow: params?.contextWindow.toInt() ?? 0,
        temperature: params?.temperature ?? 0.7,
        topP: params?.topP ?? 0.9,
        topK: params?.topK.toInt() ?? 40,
        repPenalty: params?.repPenalty ?? 1.1,
        sendPort: port.sendPort,
      ),
    );  // background isolate; cancelled via neurons_cancel()

    return ctrl.stream;
  }

  @override
  Future<SearchModelsResponse> searchModels(String query,
      {int limit = 30,
      String sort = 'downloads',
      List<String> pipelineTags = const [],
      String author = ''}) async {
    final searchFn = _lib.lookupFunction<_NeuronsSearchNative,
        _NeuronsSearchDart>('neurons_search_models');
    final freeFn = _lib.lookupFunction<_NeuronsFreeStringNative,
        _NeuronsFreeStringDart>('neurons_free_string');

    final errBuf = calloc<Char>(512);
    final tagsJson = pipelineTags.isEmpty
        ? ''.toNativeUtf8().cast<Char>()
        : json.encode(pipelineTags).toNativeUtf8().cast<Char>();
    try {
      final qPtr      = query.toNativeUtf8().cast<Char>();
      final sortPtr   = sort.toNativeUtf8().cast<Char>();
      final authorPtr = author.toNativeUtf8().cast<Char>();
      final jsonPtr   = searchFn(_core, qPtr, limit, sortPtr, tagsJson, authorPtr, errBuf, 512);
      malloc.free(qPtr);
      malloc.free(sortPtr);
      malloc.free(authorPtr);

      if (jsonPtr == nullptr) {
        final msg = errBuf.cast<Utf8>().toDartString();
        return SearchModelsResponse()..error = msg;
      }
      final jsonStr = jsonPtr.cast<Utf8>().toDartString();
      freeFn(jsonPtr);

      final m = json.decode(jsonStr) as Map<String, dynamic>;
      final results = (m['results'] as List<dynamic>? ?? []).map((e) {
        final em = e as Map<String, dynamic>;
        return HfModelResult()
          ..modelId = em['modelId'] as String? ?? ''
          ..downloads = fixnum.Int64(_protoInt(em['downloads']))
          ..gated = em['gated'] as bool? ?? false;
      }).toList();
      return SearchModelsResponse()..results.addAll(results);
    } finally {
      calloc.free(errBuf);
      malloc.free(tagsJson);
    }
  }

  @override
  Future<GetModelInfoResponse> getModelInfo(String modelId) async {
    final infoFn = _lib.lookupFunction<_NeuronsGetModelInfoNative,
        _NeuronsGetModelInfoDart>('neurons_get_model_info');
    final freeFn = _lib.lookupFunction<_NeuronsFreeStringNative,
        _NeuronsFreeStringDart>('neurons_free_string');

    final errBuf = calloc<Char>(512);
    try {
      final idPtr = modelId.toNativeUtf8().cast<Char>();
      final jsonPtr = infoFn(_core, idPtr, errBuf, 512);
      calloc.free(idPtr);

      if (jsonPtr == nullptr) {
        final msg = errBuf.cast<Utf8>().toDartString();
        return GetModelInfoResponse()..error = msg;
      }
      final jsonStr = jsonPtr.cast<Utf8>().toDartString();
      freeFn(jsonPtr);

      final m = json.decode(jsonStr) as Map<String, dynamic>;
      final files = (m['files'] as List<dynamic>? ?? []).map((e) {
        final em = e as Map<String, dynamic>;
        return HfFileInfo()
          ..filename = em['filename'] as String? ?? ''
          ..sizeBytes = fixnum.Int64(_protoInt(em['sizeBytes']));
      }).toList();
      return GetModelInfoResponse()
        ..modelId = m['modelId'] as String? ?? modelId
        ..totalSizeBytes = fixnum.Int64(_protoInt(m['totalSizeBytes']))
        ..readme = m['readme'] as String? ?? ''
        ..files.addAll(files);
    } finally {
      calloc.free(errBuf);
    }
  }

  @override
  Stream<DownloadProgressResponse> downloadModel(String modelId) {
    final ctrl = StreamController<DownloadProgressResponse>();
    final port = ReceivePort();

    port.listen((msg) {
      if (msg == null) {
        ctrl.close();
        port.close();
      } else if (msg is Map) {
        if (msg['error'] != null) {
          final resp = DownloadProgressResponse()
            ..error = msg['error'] as String
            ..done = true;
          ctrl.add(resp);
          ctrl.close();
          port.close();
        } else {
          final done =
              (msg['bytesDone'] as int) == (msg['bytesTotal'] as int) &&
                  (msg['bytesTotal'] as int) > 0;
          ctrl.add(DownloadProgressResponse()
            ..bytesDownloaded =
                fixnum.Int64(msg['bytesDone'] as int? ?? 0)
            ..totalBytes =
                fixnum.Int64(msg['bytesTotal'] as int? ?? 0)
            ..speedBps = (msg['speedBps'] as double? ?? 0.0)
            ..currentFile = msg['currentFile'] as String? ?? ''
            ..done = done);
        }
      }
    });

    Isolate.spawn(
      _downloadIsolate,
      _DownloadArgs(
        libPath: _libPath,
        coreAddr: _core.address,
        repoId: modelId,
        sendPort: port.sendPort,
      ),
    ).then((iso) => _activeDownloadIsolate = iso);

    return ctrl.stream;
  }

  @override
  Future<CancelDownloadResponse> cancelDownload(String downloadId) async {
    final cancelFn = _lib.lookupFunction<_NeuronsVoidCoreNative,
        _NeuronsVoidCoreDart>('neurons_cancel_download');
    cancelFn(_core);
    _activeDownloadIsolate?.kill(priority: Isolate.immediate);
    _activeDownloadIsolate = null;
    return CancelDownloadResponse()..success = true;
  }

  @override
  Future<void> setHfToken(String token) async {
    final fn = _lib.lookupFunction<_NeuronsSetHfTokenNative,
        _NeuronsSetHfTokenDart>('neurons_set_hf_token');
    final tokenPtr = token.toNativeUtf8().cast<Char>();
    try {
      fn(_core, tokenPtr);
    } finally {
      malloc.free(tokenPtr);
    }
  }

  @override
  Stream<LogEntry> streamLogs({String minLevel = 'INFO'}) =>
      // FFI local node: use gRPC on localhost to stream logs from the service.
      GrpcNeuronsClient(host: 'localhost', port: 50051)
          .streamLogs(minLevel: minLevel);

  /// Cancel an in-progress generation.
  void cancelGeneration() {
    final cancelFn = _lib.lookupFunction<_NeuronsVoidCoreNative,
        _NeuronsVoidCoreDart>('neurons_cancel');
    cancelFn(_core);
  }

  @override
  void close() {
    final destroyFn = _lib.lookupFunction<_NeuronsDestroyNative,
        _NeuronsDestroyDart>('neurons_destroy');
    destroyFn(_core);
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

String _resolveLibPath(DynamicLibrary lib) {
  // DynamicLibrary doesn't expose its path directly; reconstruct from Platform.
  if (Platform.isMacOS) {
    final exe = File(Platform.resolvedExecutable);
    final bundlePath =
        '${exe.parent.parent.path}/Frameworks/libneurons_core.dylib';
    if (File(bundlePath).existsSync()) return bundlePath;
    final siblingPath = '${exe.parent.path}/libneurons_core.dylib';
    if (File(siblingPath).existsSync()) return siblingPath;
    // Dev fallback — same depth as _openNeuronsCore
    final p = exe.parent.path;
    final dev =
        '$p/../../../../../../../../../cmake-build-debug/service/libneurons_core.dylib';
    return File(dev).absolute.path;
  }
  if (Platform.isWindows) return 'neurons_core.dll';
  return 'libneurons_core.so';
}

/// Protobuf JSON serializes int64/uint64 as strings to avoid JS precision loss.
/// This helper accepts both string and int values returned by proto_to_json.
int _protoInt(dynamic v) =>
    v is int ? v : int.tryParse(v?.toString() ?? '0') ?? 0;
