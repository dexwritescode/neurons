import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'neurons_client.dart';
import 'chat_repository.dart';
import 'node_repository.dart';
import 'token_repository.dart';
import '../models/node_config.dart';
import '../proto/neurons.pb.dart' as proto;

export 'neurons_client.dart' show GrpcNeuronsClient;
export 'chat_repository.dart' show ChatSession;
export '../models/node_config.dart' show NodeConfig;

enum ServiceConnectionState { disconnected, connecting, connected, error }

/// Per-model inference settings. Applied when calling [AppState.send].
class InferenceSettings {
  InferenceSettings({
    this.maxTokens = 512,
    this.contextWindow = 0,
    this.temperature = 0.7,
    this.topP = 0.9,
    this.topK = 40,
    this.repPenalty = 1.1,
  });

  /// Maximum tokens to generate. Default: 512.
  int maxTokens;

  /// Context window size (tokens). 0 = use the model's built-in maximum.
  int contextWindow;

  double temperature;
  double topP;
  int topK;
  double repPenalty;
}

class ConversationMessage {
  ConversationMessage({required this.role, required this.content});
  final String role; // 'user' | 'assistant'
  String content;
}

class AppState extends ChangeNotifier {
  AppState(NeuronsClient client) : _client = client {
    _loadPrefs();
    _initSessions();
    _initNodes();
    _initHfToken();
  }

  NeuronsClient _client;
  final _chatRepo = ChatRepository();
  final _nodeRepo = NodeRepository();
  final _tokenRepo = TokenRepository();

  static const _kSystemPrompt = 'system_prompt';
  static const _kSettingsPrefix = 'inference_settings_';

  // ── Connection / model state ─────────────────────────────────────────────
  ServiceConnectionState connectionState = ServiceConnectionState.disconnected;
  String? modelPath;
  String? modelType;
  String? backend;
  int vocabSize = 0;
  int numLayers = 0;
  /// The model's absolute maximum context length in tokens (from config).
  /// Used to cap the context-window slider in the load settings sheet.
  int maxPositionEmbeddings = 0;
  bool supportsToolUse = false;
  String? statusError;

  /// OpenAI HTTP server port reported by the active node (0 = not running).
  int httpPort = 0;

  // ── Inference settings ────────────────────────────────────────────────────
  /// Current inference settings; populated with model defaults on load,
  /// then adjusted by the user via the load settings modal.
  InferenceSettings inferenceSettings = InferenceSettings();

  /// Replace inference settings, persist them keyed by the current model path,
  /// and notify listeners. Called from the load settings sheet.
  void applyInferenceSettings(InferenceSettings s) {
    inferenceSettings = s;
    _saveInferenceSettings();
    notifyListeners();
  }

  Future<void> _saveInferenceSettings() async {
    final path = modelPath;
    if (path == null) return;
    final prefs = await SharedPreferences.getInstance();
    final k = _kSettingsPrefix + path;
    await prefs.setInt('${k}_maxTokens', inferenceSettings.maxTokens);
    await prefs.setInt('${k}_contextWindow', inferenceSettings.contextWindow);
    await prefs.setDouble('${k}_temperature', inferenceSettings.temperature);
    await prefs.setDouble('${k}_topP', inferenceSettings.topP);
    await prefs.setInt('${k}_topK', inferenceSettings.topK);
    await prefs.setDouble('${k}_repPenalty', inferenceSettings.repPenalty);
  }

  Future<void> _loadInferenceSettings(String path) async {
    final prefs = await SharedPreferences.getInstance();
    final k = _kSettingsPrefix + path;
    final maxTokens = prefs.getInt('${k}_maxTokens');
    if (maxTokens == null) return; // no saved settings for this model — keep defaults
    inferenceSettings = InferenceSettings(
      maxTokens: maxTokens,
      contextWindow: prefs.getInt('${k}_contextWindow') ?? maxPositionEmbeddings,
      temperature: prefs.getDouble('${k}_temperature') ?? 0.7,
      topP: prefs.getDouble('${k}_topP') ?? 0.9,
      topK: prefs.getInt('${k}_topK') ?? 40,
      repPenalty: prefs.getDouble('${k}_repPenalty') ?? 1.1,
    );
  }

  // ── Model list ────────────────────────────────────────────────────────────
  List<proto.ModelInfo> availableModels = [];
  bool _modelsLoaded = false;

  // ── Model browser ─────────────────────────────────────────────────────────
  List<proto.HfModelResult> searchResults = [];
  bool isSearching = false;
  String? searchError;

  proto.GetModelInfoResponse? selectedModelInfo;
  bool isLoadingModelDetail = false;

  String? downloadingModelId;
  String? _downloadId;
  double? downloadProgress;  // 0.0–1.0 when total known, null = indeterminate
  double downloadSpeedBps = 0.0;
  String? downloadCurrentFile;
  String? downloadError;

  StreamSubscription<DownloadProgressResponse>? _downloadSubscription;

  // ── System prompt ─────────────────────────────────────────────────────────
  String systemPrompt = '';

  Future<void> setSystemPrompt(String value) async {
    systemPrompt = value;
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_kSystemPrompt, value);
  }

  Future<void> _loadPrefs() async {
    final prefs = await SharedPreferences.getInstance();
    systemPrompt = prefs.getString(_kSystemPrompt) ?? '';
    notifyListeners();
  }

  // ── HuggingFace token ─────────────────────────────────────────────────────
  /// Global HF token loaded from Keychain. Null if not set.
  String? hfToken;

  /// Load token from Keychain and push to local FFI client on startup.
  Future<void> _initHfToken() async {
    try {
      final token = await _tokenRepo.loadHfToken();
      if (token != null && token.isNotEmpty) {
        hfToken = token;
        await _client.setHfToken(token);
      }
    } catch (_) {
      // Keychain unavailable (e.g. unit test environment) — proceed without token.
    }
  }

  /// Save a new global HF token and push it to all nodes (respecting per-node overrides).
  Future<void> setGlobalHfToken(String token) async {
    await _tokenRepo.saveHfToken(token);
    hfToken = token;
    notifyListeners();
    await _pushTokenToAllNodes();
  }

  /// Clear the global HF token from Keychain and all nodes that inherit it.
  Future<void> clearGlobalHfToken() async {
    await _tokenRepo.clearHfToken();
    hfToken = null;
    notifyListeners();
    await _pushTokenToAllNodes();
  }

  /// Effective token for a given node: per-node override → global → empty.
  String _effectiveToken(NodeConfig node) =>
      (node.hfToken != null && node.hfToken!.isNotEmpty)
          ? node.hfToken!
          : (hfToken ?? '');

  /// Push the effective token to all connected nodes.
  Future<void> _pushTokenToAllNodes() async {
    for (final node in nodes) {
      if (node.isLocal) {
        await _client.setHfToken(_effectiveToken(node));
      } else {
        // Best-effort: remote nodes may be offline.
        try {
          final remote = GrpcNeuronsClient(host: node.host, port: node.port);
          await remote.setHfToken(_effectiveToken(node));
          remote.close();
        } catch (_) {}
      }
    }
  }

  // ── Node management ───────────────────────────────────────────────────────
  List<NodeConfig> nodes = [];
  String activeNodeId = 'local';

  NodeConfig? get activeNode =>
      nodes.where((n) => n.id == activeNodeId).firstOrNull;

  Future<void> _initNodes() async {
    final remote = await _nodeRepo.loadRemoteNodes();
    nodes = [NodeConfig.local(), ...remote];
    notifyListeners();
  }

  Future<void> addNode(NodeConfig node) async {
    nodes.add(node);
    await _nodeRepo.saveRemoteNodes(nodes);
    notifyListeners();
    // Proactively push token to newly added remote node.
    if (!node.isLocal) {
      try {
        final remote = GrpcNeuronsClient(host: node.host, port: node.port);
        await remote.setHfToken(_effectiveToken(node));
        remote.close();
      } catch (_) {}
    }
  }

  Future<void> updateNode(NodeConfig updated) async {
    final idx = nodes.indexWhere((n) => n.id == updated.id);
    if (idx == -1) return;
    nodes[idx] = updated;
    await _nodeRepo.saveRemoteNodes(nodes);
    notifyListeners();
  }

  Future<void> removeNode(String id) async {
    if (id == 'local') return; // local node cannot be removed
    nodes.removeWhere((n) => n.id == id);
    if (activeNodeId == id) await switchNode('local');
    await _nodeRepo.saveRemoteNodes(nodes);
    notifyListeners();
  }

  /// Switch the active compute node. Swaps _client and refreshes model state.
  Future<void> switchNode(String id) async {
    final target = nodes.where((n) => n.id == id).firstOrNull;
    if (target == null || id == activeNodeId) return;
    _client.close();
    if (target.isLocal) {
      // Local node uses the FFI client that was passed in at construction.
      // Re-create it from the initial client factory isn't wired here yet —
      // for now we signal disconnected and let the user restart if they
      // switched away from local. TODO: store the initial local client.
      connectionState = ServiceConnectionState.disconnected;
    } else {
      _client = GrpcNeuronsClient(host: target.host, port: target.port);
    }
    activeNodeId = id;
    modelPath = null;
    modelType = null;
    backend = null;
    maxPositionEmbeddings = 0;
    notifyListeners();
    // Push effective HF token to the newly active node.
    try { await _client.setHfToken(_effectiveToken(target)); } catch (_) {}
    // Refresh status from the new node
    try {
      final s = await _client.getStatus();
      _applyStatus(s);
      connectionState = ServiceConnectionState.connected;
    } catch (_) {
      connectionState = ServiceConnectionState.error;
    }
    await refreshModels(force: true);
    notifyListeners();
  }

  Future<void> _initSessions() async {
    sessions = await _chatRepo.loadAll();
    if (sessions.isEmpty) {
      sessions.add(_activeSession);
    } else {
      _activeSession = sessions.first;
    }
    notifyListeners();
  }

  // ── Session actions ───────────────────────────────────────────────────────

  /// Start a new empty chat session.
  Future<void> newChat() async {
    await _saveActiveSession();
    _activeSession = ChatSession.create();
    sessions.insert(0, _activeSession);
    generationError = null;
    lastPromptTokens = 0;
    lastGenTokens = 0;
    liveGenTokens = 0;
    _clearUndo();
    notifyListeners();
  }

  /// Switch the active session. Saves the current session first.
  Future<void> switchSession(String id) async {
    if (id == _activeSession.id) return;
    await _saveActiveSession();
    final target = sessions.where((s) => s.id == id).firstOrNull;
    if (target == null) return;
    _activeSession = target;
    generationError = null;
    lastPromptTokens = 0;
    lastGenTokens = 0;
    liveGenTokens = 0;
    _clearUndo();
    notifyListeners();
  }

  /// Delete a session. If it is the active one, switches to newest remaining
  /// or creates a fresh session.
  Future<void> deleteSession(String id) async {
    await _chatRepo.delete(id);
    sessions.removeWhere((s) => s.id == id);
    if (_activeSession.id == id) {
      if (sessions.isEmpty) {
        _activeSession = ChatSession.create();
        sessions.add(_activeSession);
      } else {
        _activeSession = sessions.first;
      }
      generationError = null;
      lastPromptTokens = 0;
      lastGenTokens = 0;
    liveGenTokens = 0;
      _clearUndo();
    }
    notifyListeners();
  }

  /// Rename a session by id and persist the change.
  Future<void> renameSession(String id, String title) async {
    final session = sessions.where((s) => s.id == id).firstOrNull;
    if (session == null) return;
    session.title = title;
    await _chatRepo.save(session);
    notifyListeners();
  }

  /// Persist the active session only if it has messages.
  Future<void> _saveActiveSession() async {
    if (_activeSession.messages.isNotEmpty) {
      await _chatRepo.save(_activeSession);
    }
  }

  // ── Chat sessions ─────────────────────────────────────────────────────────
  /// All known sessions, sorted newest-first.
  List<ChatSession> sessions = [];
  ChatSession _activeSession = ChatSession.create();

  String get activeSessionId => _activeSession.id;

  /// The active session's message list. Widgets read this directly.
  List<ConversationMessage> get messages => _activeSession.messages;

  // ── Undo ──────────────────────────────────────────────────────────────────
  /// Single-level undo snapshot for delete/truncate operations.
  List<ConversationMessage>? _undoSnapshot;
  bool get canUndo => _undoSnapshot != null;

  void undoDelete() {
    final snap = _undoSnapshot;
    if (snap == null) return;
    _undoSnapshot = null;
    _activeSession.messages
      ..clear()
      ..addAll(snap);
    _saveActiveSession();
    notifyListeners();
  }

  void _clearUndo() => _undoSnapshot = null;

  // ── Chat generation state ─────────────────────────────────────────────────
  bool isGenerating = false;
  String? generationError;
  int liveGenTokens = 0;   // increments per token during streaming
  int lastPromptTokens = 0;
  int lastGenTokens = 0;

  // Active generation stream and its completer — kept so they can be cancelled.
  StreamSubscription<GenerateResponse>? _generateSubscription;
  Completer<void>? _generateCompleter;

  bool get modelLoaded => modelPath != null;

  // ── MCP servers ──────────────────────────────────────────────────────────
  List<proto.McpServerConfig> mcpServers = [];

  Future<void> loadMcpServers() async {
    try {
      final resp = await _client.listMcpServers();
      mcpServers = resp.servers.toList();
      notifyListeners();
    } catch (_) {}
  }

  Future<void> addMcpServer(proto.McpServerConfig server) async {
    await _client.addMcpServer(server);
    await loadMcpServers();
  }

  Future<void> removeMcpServer(String name) async {
    await _client.removeMcpServer(name);
    await loadMcpServers();
  }

  // ── Actions ───────────────────────────────────────────────────────────────

  /// Connect to the inference service.
  ///
  /// When [host] and [port] are provided a new [GrpcNeuronsClient] is created
  /// targeting that address. Used by [SplashScreen] to silently connect to the
  /// locally-managed service on its dynamically-assigned port.
  /// When called with no arguments the existing client is reused (remote-node
  /// flow via [ConnectScreen]).
  Future<void> connect({String? host, int? port}) async {
    if (host != null && port != null) {
      _client.close();
      _client = GrpcNeuronsClient(host: host, port: port);
    }
    connectionState = ServiceConnectionState.connecting;
    statusError = null;
    notifyListeners();
    try {
      final status = await _client.getStatus();
      connectionState = ServiceConnectionState.connected;
      _applyStatus(status);
      unawaited(loadMcpServers());
    } catch (e) {
      connectionState = ServiceConnectionState.error;
      statusError = e.toString();
    }
    notifyListeners();
  }

  /// Scan the models directory and update [availableModels].
  /// Subsequent calls are no-ops unless [force] is true — prevents redundant
  /// scans when [ModelPickerScreen] is rebuilt on every tab switch.
  Future<void> refreshModels({bool force = false}) async {
    if (_modelsLoaded && !force) return;
    try {
      final resp = await _client.listModels();
      availableModels = resp.models;
      _modelsLoaded = true;
      notifyListeners();
    } catch (_) {}
  }

  Future<bool> loadModel(String path) async {
    try {
      final resp = await _client.loadModel(path);
      if (resp.success) {
        modelPath      = path;
        modelType      = resp.modelType;
        vocabSize      = resp.vocabSize.toInt();
        numLayers      = resp.numLayers.toInt();
        maxPositionEmbeddings = resp.maxPositionEmbeddings.toInt();
        supportsToolUse = resp.supportsToolUse;
        // Seed context window to max, then overwrite with any previously saved settings.
        inferenceSettings.contextWindow = maxPositionEmbeddings;
        await _loadInferenceSettings(path);
        notifyListeners();
        return true;
      }
      statusError = resp.error;
      notifyListeners();
      return false;
    } catch (e) {
      statusError = e.toString();
      notifyListeners();
      return false;
    }
  }

  Future<bool> unloadModel() async {
    try {
      await _client.unloadModel();
      modelPath       = null;
      modelType       = null;
      backend         = null;
      vocabSize       = 0;
      numLayers       = 0;
      maxPositionEmbeddings = 0;
      supportsToolUse = false;
      notifyListeners();
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<bool> deleteModel(String path) async {
    try {
      final ok = await _client.deleteModel(path);
      if (ok) await refreshModels(force: true);
      return ok;
    } catch (_) {
      return false;
    }
  }

  // ── Browser actions ───────────────────────────────────────────────────────

  Future<void> searchModels(
    String query, {
    String sort = 'downloads',
    List<String> pipelineTags = const [],
    String author = '',
  }) async {
    isSearching = true;
    searchError = null;
    notifyListeners();
    try {
      final resp = await _client.searchModels(
        query,
        sort: sort,
        pipelineTags: pipelineTags,
        author: author,
      );
      searchResults = resp.results;
    } catch (e) {
      searchError = e.toString();
      searchResults = [];
    } finally {
      isSearching = false;
      notifyListeners();
    }
  }

  Future<void> selectModel(String modelId) async {
    selectedModelInfo = null;
    isLoadingModelDetail = true;
    notifyListeners();
    try {
      final resp = await _client.getModelInfo(modelId);
      selectedModelInfo = resp;
    } catch (_) {
      selectedModelInfo = null;
    } finally {
      isLoadingModelDetail = false;
      notifyListeners();
    }
  }

  void clearSelectedModel() {
    selectedModelInfo = null;
    notifyListeners();
  }

  Future<void> startDownload(String modelId) async {
    if (downloadingModelId != null) return;
    downloadingModelId = modelId;
    downloadProgress = null;
    downloadSpeedBps = 0.0;
    downloadCurrentFile = null;
    downloadError = null;
    _downloadId = null;
    notifyListeners();

    try {
      final stream = _client.downloadModel(modelId);
      _downloadSubscription = stream.listen(
        (resp) {
          if (resp.downloadId.isNotEmpty) _downloadId = resp.downloadId;
          downloadCurrentFile = resp.currentFile;
          downloadSpeedBps = resp.speedBps;
          final total = resp.totalBytes.toInt();
          downloadProgress = (total > 0)
              ? (resp.bytesDownloaded.toInt() / total).clamp(0.0, 1.0)
              : null; // indeterminate — HF CDN uses chunked transfer, no Content-Length
          if (resp.done) {
            if (resp.error.isNotEmpty) {
              downloadError = resp.error;
            }
            _finishDownload();
          } else {
            notifyListeners();
          }
        },
        onDone: _finishDownload,
        onError: (Object e) {
          downloadError = e.toString();
          _finishDownload();
        },
        cancelOnError: true,
      );
    } catch (e) {
      downloadError = e.toString();
      _finishDownload();
    }
  }

  void _finishDownload() {
    downloadingModelId = null;
    _downloadId = null;
    _downloadSubscription = null;
    notifyListeners();
    // Force-refresh so the newly downloaded model appears in the list.
    refreshModels(force: true);
  }

  Future<void> cancelDownload() async {
    final id = _downloadId;
    await _downloadSubscription?.cancel();
    _downloadSubscription = null;
    if (id != null) {
      try {
        await _client.cancelDownload(id);
      } catch (_) {}
    }
    downloadingModelId = null;
    _downloadId = null;
    downloadProgress = null;
    downloadCurrentFile = null;
    notifyListeners();
  }

  Future<void> send(String prompt) async {
    if (isGenerating || prompt.trim().isEmpty) return;

    // Build proto history: optional system prompt first, then conversation.
    final history = <proto.ChatMessage>[
      if (systemPrompt.trim().isNotEmpty)
        proto.ChatMessage()
          ..role = 'system'
          ..content = systemPrompt.trim(),
      ...messages.map((m) => proto.ChatMessage()
        ..role = m.role
        ..content = m.content),
    ];

    // Auto-title: use the first user message text (≤50 chars).
    if (_activeSession.title == 'New chat') {
      _activeSession.title =
          prompt.length > 50 ? '${prompt.substring(0, 47)}...' : prompt;
    }

    messages.add(ConversationMessage(role: 'user', content: prompt));
    final assistantMsg = ConversationMessage(role: 'assistant', content: '');
    messages.add(assistantMsg);
    isGenerating = true;
    generationError = null;
    liveGenTokens = 0;
    notifyListeners();

    try {
      final params = proto.SamplingParams()
        ..temperature = inferenceSettings.temperature
        ..topP = inferenceSettings.topP
        ..topK = inferenceSettings.topK
        ..repPenalty = inferenceSettings.repPenalty
        ..maxTokens = inferenceSettings.maxTokens
        ..contextWindow = inferenceSettings.contextWindow;

      final stream = _client.generate(prompt, history: history, params: params);

      _generateCompleter = Completer<void>();
      _generateSubscription = stream.listen(
        (resp) {
          if (resp.done) {
            lastPromptTokens = resp.promptTokens.toInt();
            lastGenTokens    = resp.genTokens.toInt();
            if (resp.error.isNotEmpty) generationError = resp.error;
            notifyListeners();
          } else {
            assistantMsg.content += resp.token;
            liveGenTokens++;
            notifyListeners();
          }
        },
        onDone: () => _generateCompleter?.complete(),
        onError: (Object e) {
          generationError = e.toString();
          _generateCompleter?.complete();
        },
        cancelOnError: true,
      );
      await _generateCompleter!.future;
    } catch (e) {
      generationError = e.toString();
    } finally {
      _generateSubscription = null;
      _generateCompleter = null;
      isGenerating = false;
      notifyListeners();
    }
    // Persist after generation finishes (not during streaming).
    await _saveActiveSession();
  }

  /// Cancel an in-progress generation stream.
  Future<void> cancelGeneration() async {
    await _generateSubscription?.cancel();
    _generateCompleter?.complete(); // unblocks send() which is awaiting the future
    _generateSubscription = null;
    _generateCompleter = null;
    isGenerating = false;
    notifyListeners();
  }

  void clearChat() {
    _activeSession.messages.clear();
    _activeSession.title = 'New chat';
    generationError = null;
    lastPromptTokens = 0;
    lastGenTokens = 0;
    liveGenTokens = 0;
    notifyListeners();
    // Remove the now-empty session file from disk.
    _chatRepo.delete(_activeSession.id);
  }

  /// Delete a message and its paired counterpart:
  /// - user message → also removes the following assistant message
  /// - assistant message → also removes the preceding user message
  void deleteMessagePair(int index) {
    if (index < 0 || index >= messages.length) return;
    _undoSnapshot = List.from(messages);
    final role = messages[index].role;
    if (role == 'user') {
      messages.removeAt(index);
      if (index < messages.length && messages[index].role == 'assistant') {
        messages.removeAt(index);
      }
    } else if (role == 'assistant') {
      messages.removeAt(index);
      if (index > 0 && messages[index - 1].role == 'user') {
        messages.removeAt(index - 1);
      }
    }
    _saveActiveSession();
    notifyListeners();
  }

  /// Remove all messages from [index] to the end of the conversation.
  void truncateFrom(int index) {
    if (index < 0 || index >= messages.length) return;
    _undoSnapshot = List.from(messages);
    messages.removeRange(index, messages.length);
    _saveActiveSession();
    notifyListeners();
  }

  /// Return the full conversation as plain text, suitable for clipboard.
  String chatAsText() {
    final buf = StringBuffer();
    for (final m in messages) {
      final speaker = m.role == 'user' ? 'You' : (modelType ?? 'Assistant');
      buf.writeln('$speaker:');
      buf.writeln(m.content);
      buf.writeln();
    }
    return buf.toString().trimRight();
  }

  void _applyStatus(proto.StatusResponse s) {
    httpPort = s.httpPort.toInt();
    if (s.modelLoaded) {
      modelPath       = s.modelPath;
      modelType       = s.modelType;
      backend         = s.backend;
      vocabSize       = s.vocabSize.toInt();
      numLayers       = s.numLayers.toInt();
      maxPositionEmbeddings = s.maxPositionEmbeddings.toInt();
      supportsToolUse = s.supportsToolUse;
      if (inferenceSettings.contextWindow == 0) {
        inferenceSettings.contextWindow = maxPositionEmbeddings;
      }
    } else {
      modelPath       = null;
      modelType       = null;
      maxPositionEmbeddings = 0;
      supportsToolUse = false;
    }
  }

  @override
  void dispose() {
    _generateSubscription?.cancel();
    _downloadSubscription?.cancel();
    _client.close();
    super.dispose();
  }
}
