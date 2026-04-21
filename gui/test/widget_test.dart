import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:neurons_gui/services/app_state.dart';
import 'package:neurons_gui/services/neurons_client.dart';
import 'package:neurons_gui/proto/neurons.pb.dart' as proto;
import 'package:fixnum/fixnum.dart';
import 'package:neurons_gui/screens/splash_screen.dart';
import 'package:neurons_gui/screens/model_browser_screen.dart';
import 'package:neurons_gui/widgets/animated_dots.dart';

// ── Fakes ─────────────────────────────────────────────────────────────────────

/// Mixin that stubs out the browser/download methods for test fakes that
/// don't need them.
mixin _NoBrowserMixin implements NeuronsClient {
  @override
  Future<SearchModelsResponse> searchModels(String query,
          {int limit = 30,
          String sort = 'downloads',
          List<String> pipelineTags = const [],
          String author = ''}) async =>
      proto.SearchModelsResponse();

  @override
  Future<GetModelInfoResponse> getModelInfo(String modelId) async =>
      proto.GetModelInfoResponse();

  @override
  Stream<DownloadProgressResponse> downloadModel(String modelId) async* {}

  @override
  Future<CancelDownloadResponse> cancelDownload(String downloadId) async =>
      proto.CancelDownloadResponse();

  @override
  Future<bool> deleteModel(String modelPath) async => true;

  @override
  Future<void> setHfToken(String token) async {}

  @override
  Stream<LogEntry> streamLogs({String minLevel = 'INFO'}) async* {}
}

/// Minimal stub — all methods return empty successful responses.
class _FakeNeuronsClient with _NoBrowserMixin implements NeuronsClient {
  @override
  Future<StatusResponse> getStatus() async => proto.StatusResponse();

  @override
  Future<LoadModelResponse> loadModel(String modelPath) async =>
      proto.LoadModelResponse()
        ..success = false
        ..error = 'no model';

  @override
  Future<UnloadModelResponse> unloadModel() async =>
      proto.UnloadModelResponse()..success = true;

  @override
  Future<ListModelsResponse> listModels({String modelsDir = ''}) async =>
      proto.ListModelsResponse();

  @override
  Stream<GenerateResponse> generate(
    String prompt, {
    List<proto.ChatMessage> history = const [],
    proto.SamplingParams? params,
  }) async* {}

  @override
  void close() {}
}

/// Records the history argument passed to each generate() call.
class _HistoryCapturingClient with _NoBrowserMixin implements NeuronsClient {
  final List<List<proto.ChatMessage>> capturedHistories = [];
  final String replyToken;

  _HistoryCapturingClient({this.replyToken = 'ok'});

  @override
  Future<StatusResponse> getStatus() async => proto.StatusResponse();

  @override
  Future<LoadModelResponse> loadModel(String modelPath) async =>
      proto.LoadModelResponse()..success = false;

  @override
  Future<UnloadModelResponse> unloadModel() async =>
      proto.UnloadModelResponse()..success = true;

  @override
  Future<ListModelsResponse> listModels({String modelsDir = ''}) async =>
      proto.ListModelsResponse();

  @override
  Stream<GenerateResponse> generate(
    String prompt, {
    List<proto.ChatMessage> history = const [],
    proto.SamplingParams? params,
  }) async* {
    capturedHistories.add(List.of(history));
    yield proto.GenerateResponse()..token = replyToken;
    yield proto.GenerateResponse()
      ..done = true
      ..promptTokens = 10
      ..genTokens = 1;
  }

  @override
  void close() {}
}

// ── Widget tests ───────────────────────────────────────────────────────────────

void main() {
  testWidgets('SplashScreen shows spinner when connecting', (tester) async {
    final app = AppState(_FakeNeuronsClient());
    // Default state is disconnected — splash shows connecting spinner.
    await tester.pumpWidget(
      ChangeNotifierProvider<AppState>.value(
        value: app,
        child: const MaterialApp(home: SplashScreen()),
      ),
    );
    expect(find.byType(AnimatedDots), findsOneWidget);
    expect(find.text('Connecting…'), findsOneWidget);
  });

  testWidgets('SplashScreen shows Retry button on error', (tester) async {
    final app = AppState(_FakeNeuronsClient());
    // Simulate an error state.
    app.connectionState = ServiceConnectionState.error;
    app.statusError = 'Connection refused';

    await tester.pumpWidget(
      ChangeNotifierProvider<AppState>.value(
        value: app,
        child: const MaterialApp(home: SplashScreen()),
      ),
    );

    expect(find.text('Connection refused'), findsOneWidget);
    expect(find.text('Retry'), findsOneWidget);
  });

  testWidgets('_AppRouter shows SplashScreen when disconnected', (tester) async {
    final state = AppState(_FakeNeuronsClient());
    await tester.pumpWidget(
      ChangeNotifierProvider<AppState>.value(
        value: state,
        child: MaterialApp(
          home: _AppRouterWrapper(state: state),
        ),
      ),
    );
    // Disconnected → animated dots from SplashScreen
    expect(find.byType(AnimatedDots), findsOneWidget);
  });

  testWidgets('_AppRouter shows Models screen when connected without a model',
      (tester) async {
    final state = AppState(_FakeNeuronsClient());
    await tester.pumpWidget(
      ChangeNotifierProvider<AppState>.value(
        value: state,
        child: MaterialApp(
          home: _AppRouterWrapper(state: state),
        ),
      ),
    );
    // Trigger connect (FakeClient.getStatus() returns no model loaded).
    await state.connect();
    await tester.pumpAndSettle();
    expect(state.connectionState, ServiceConnectionState.connected);
    expect(state.modelLoaded, isFalse);
    expect(find.text('Models'), findsOneWidget);
  });

  // ── Unit tests (no widget required) ─────────────────────────────────────────

  test('AppState.send() passes prior exchange as history on second turn',
      () async {
    final client = _HistoryCapturingClient(replyToken: 'Hi');
    final state = AppState(client);

    await state.send('Hello');
    await state.send('How are you?');

    expect(client.capturedHistories.length, 2);

    // First turn: no history
    expect(client.capturedHistories[0], isEmpty);

    // Second turn: history contains first user + assistant message
    final h = client.capturedHistories[1];
    expect(h.length, 2);
    expect(h[0].role, 'user');
    expect(h[0].content, 'Hello');
    expect(h[1].role, 'assistant');
    expect(h[1].content, 'Hi');
  });

  test('AppState.cancelGeneration() stops an in-progress stream', () async {
    // Client that emits tokens slowly via a stream controller.
    StreamController<proto.GenerateResponse>? ctrl;
    final client = _StreamControllerClient(onGenerate: (controller) {
      ctrl = controller;
    });
    final state = AppState(client);

    // Start generating (don't await — we want to cancel while it's running).
    final sendFuture = state.send('Tell me a story');
    await Future.microtask(() {}); // let send() reach the listen() call
    expect(state.isGenerating, isTrue);

    // Cancel — cancelGeneration() completes the internal completer so send() unblocks.
    await state.cancelGeneration();
    await sendFuture; // should complete now

    expect(state.isGenerating, isFalse);
    ctrl?.close(); // clean up the unused stream controller
  });

  // ── ModelBrowserScreen tests ─────────────────────────────────────────────

  testWidgets('ModelBrowserScreen shows placeholder when no model selected',
      (tester) async {
    final state = AppState(_FakeNeuronsClient());
    await tester.pumpWidget(
      ChangeNotifierProvider.value(
        value: state,
        child: const MaterialApp(home: Scaffold(body: ModelBrowserScreen())),
      ),
    );
    expect(find.text('Select a model to see details'), findsOneWidget);
  });

  testWidgets('ModelBrowserScreen shows search results', (tester) async {
    final client = _BrowserFakeClient(
      searchResult: proto.SearchModelsResponse()
        ..results.addAll([
          proto.HfModelResult()
            ..modelId = 'test/TinyLlama'
            ..downloads = Int64(42000),
          proto.HfModelResult()
            ..modelId = 'test/Mistral'
            ..downloads = Int64(1500000),
        ]),
    );
    final state = AppState(client);

    await tester.pumpWidget(
      ChangeNotifierProvider.value(
        value: state,
        child: const MaterialApp(home: Scaffold(body: ModelBrowserScreen())),
      ),
    );

    // Drive a search through state directly.
    await state.searchModels('test');
    await tester.pumpAndSettle();

    expect(find.text('test/TinyLlama'), findsOneWidget);
    expect(find.text('test/Mistral'), findsOneWidget);
    expect(find.text('↓ 42K'), findsOneWidget);
    expect(find.text('↓ 1.5M'), findsOneWidget);
  });

  test('AppState.send() populates lastPromptTokens and lastGenTokens on completion',
      () async {
    final client = _HistoryCapturingClient(replyToken: 'Hi');
    final state = AppState(client);

    await state.send('Hello');

    expect(state.lastPromptTokens, 10);
    expect(state.lastGenTokens, 1);
    expect(state.isGenerating, isFalse);
  });

  test('AppState.send() increments liveGenTokens during streaming', () async {
    StreamController<proto.GenerateResponse>? ctrl;
    final client = _StreamControllerClient(onGenerate: (c) => ctrl = c);
    final state = AppState(client);

    final sendFuture = state.send('Hello');
    await Future.microtask(() {});

    // Emit a few tokens and verify liveGenTokens tracks them.
    ctrl!.add(proto.GenerateResponse()..token = 'a');
    await Future.microtask(() {});
    expect(state.liveGenTokens, 1);

    ctrl!.add(proto.GenerateResponse()..token = 'b');
    await Future.microtask(() {});
    expect(state.liveGenTokens, 2);

    // Complete generation.
    ctrl!.add(proto.GenerateResponse()..done = true..promptTokens = 5..genTokens = 2);
    ctrl!.close();
    await sendFuture;

    expect(state.isGenerating, isFalse);
    expect(state.lastPromptTokens, 5);
    expect(state.lastGenTokens, 2);
  });

  testWidgets('ModelBrowserScreen shows detail pane after selecting model',
      (tester) async {
    final client = _BrowserFakeClient(
      searchResult: proto.SearchModelsResponse()
        ..results.add(proto.HfModelResult()
          ..modelId = 'test/TinyLlama'
          ..downloads = Int64(1000)),
      modelInfo: proto.GetModelInfoResponse()
        ..modelId = 'test/TinyLlama'
        ..files.addAll([
          proto.HfFileInfo()
            ..filename = 'model.safetensors'
            ..sizeBytes = Int64(1 << 30),
        ])
        ..readme = '# TinyLlama\nSmall model.',
    );
    final state = AppState(client);

    await tester.pumpWidget(
      ChangeNotifierProvider.value(
        value: state,
        child: const MaterialApp(home: Scaffold(body: ModelBrowserScreen())),
      ),
    );

    await state.searchModels('test');
    await tester.pumpAndSettle();

    // Tap the result → loads model info → shows detail pane.
    await tester.tap(find.text('test/TinyLlama'));
    await tester.pumpAndSettle();

    expect(find.textContaining('test/TinyLlama'), findsWidgets);
    expect(find.text('model.safetensors'), findsOneWidget);
    expect(find.text('Download'), findsOneWidget);
  });
}

// ── Browser fake ──────────────────────────────────────────────────────────────

class _BrowserFakeClient with _NoBrowserMixin implements NeuronsClient {
  _BrowserFakeClient({
    this.searchResult,
    this.modelInfo,
  });

  final proto.SearchModelsResponse? searchResult;
  final proto.GetModelInfoResponse? modelInfo;

  @override
  Future<StatusResponse> getStatus() async => proto.StatusResponse();

  @override
  Future<LoadModelResponse> loadModel(String modelPath) async =>
      proto.LoadModelResponse()..success = false;

  @override
  Future<UnloadModelResponse> unloadModel() async =>
      proto.UnloadModelResponse()..success = true;

  @override
  Future<ListModelsResponse> listModels({String modelsDir = ''}) async =>
      proto.ListModelsResponse();

  @override
  Stream<GenerateResponse> generate(
    String prompt, {
    List<proto.ChatMessage> history = const [],
    proto.SamplingParams? params,
  }) async* {}

  @override
  Future<SearchModelsResponse> searchModels(String query,
          {int limit = 30,
          String sort = 'downloads',
          List<String> pipelineTags = const [],
          String author = ''}) async =>
      searchResult ?? proto.SearchModelsResponse();

  @override
  Future<GetModelInfoResponse> getModelInfo(String modelId) async =>
      modelInfo ?? proto.GetModelInfoResponse();

  @override
  Stream<DownloadProgressResponse> downloadModel(String modelId) async* {}

  @override
  Future<CancelDownloadResponse> cancelDownload(String downloadId) async =>
      proto.CancelDownloadResponse();

  @override
  void close() {}
}

// ── Helpers ───────────────────────────────────────────────────────────────────

class _StreamControllerClient with _NoBrowserMixin implements NeuronsClient {
  _StreamControllerClient({required this.onGenerate});
  final void Function(StreamController<proto.GenerateResponse>) onGenerate;

  @override
  Stream<GenerateResponse> generate(
    String prompt, {
    List<proto.ChatMessage> history = const [],
    proto.SamplingParams? params,
  }) {
    final ctrl = StreamController<proto.GenerateResponse>();
    onGenerate(ctrl);
    return ctrl.stream;
  }

  @override
  Future<StatusResponse> getStatus() async => proto.StatusResponse();
  @override
  Future<LoadModelResponse> loadModel(String _) async =>
      proto.LoadModelResponse()..success = false;
  @override
  Future<UnloadModelResponse> unloadModel() async =>
      proto.UnloadModelResponse()..success = true;
  @override
  Future<ListModelsResponse> listModels({String modelsDir = ''}) async =>
      proto.ListModelsResponse();
  @override
  void close() {}
}

// ── AppRouter wrapper for tests ───────────────────────────────────────────────

/// Mirrors [_AppRouter] from main.dart so widget tests don't depend on
/// the real routing tree (which requires a full DynamicLibrary).
class _AppRouterWrapper extends StatelessWidget {
  const _AppRouterWrapper({required this.state});
  final AppState state;

  @override
  Widget build(BuildContext context) {
    final s = context.watch<AppState>();
    return switch (s.connectionState) {
      ServiceConnectionState.connected when s.modelLoaded =>
        const Scaffold(body: Text('Chat')),
      ServiceConnectionState.connected =>
        const Scaffold(body: Text('Models')),
      _ => const SplashScreen(),
    };
  }
}
