import 'package:grpc/grpc.dart';
import '../proto/neurons.pbgrpc.dart';

export '../proto/neurons.pb.dart';
export '../proto/neurons.pbgrpc.dart';

/// Abstract interface for the neurons inference service.
/// Inject this into widgets/screens; mock it in tests.
abstract class NeuronsClient {
  Future<StatusResponse> getStatus();
  Future<LoadModelResponse> loadModel(String modelPath);
  Future<UnloadModelResponse> unloadModel();
  Future<bool> deleteModel(String path);
  Future<ListModelsResponse> listModels({String modelsDir = ''});
  Stream<GenerateResponse> generate(
    String prompt, {
    List<ChatMessage> history = const [],
    SamplingParams? params,
  });

  // Model browser / download
  Future<SearchModelsResponse> searchModels(
    String query, {
    int limit = 30,
    String sort = 'downloads',
    List<String> pipelineTags = const [],
    String author = '',
  });
  Future<GetModelInfoResponse> getModelInfo(String modelId);
  Stream<DownloadProgressResponse> downloadModel(String modelId);
  Future<CancelDownloadResponse> cancelDownload(String downloadId);

  /// Set (or clear) the HuggingFace Bearer token on this client.
  /// Pass an empty string to clear.
  Future<void> setHfToken(String token);

  /// Stream log entries from the service. The stream stays open until cancelled.
  Stream<LogEntry> streamLogs({String minLevel = 'INFO'});

  // ── Tool approval ────────────────────────────────────────────────────────────

  /// Respond to a pending tool approval request (emitted in the generate stream).
  Future<ToolApprovalResult> respondToolApproval(
    String approvalId,
    bool approved, {
    String newPermission = '',
  });

  // ── MCP server management ────────────────────────────────────────────────────

  Future<ListMcpServersResponse> listMcpServers();
  Future<AddMcpServerResponse> addMcpServer(McpServerConfig server);
  Future<RemoveMcpServerResponse> removeMcpServer(String name);
  Future<PushMcpServersResponse> pushMcpServers(List<McpServerConfig> servers);

  // ── MCP permission management ─────────────────────────────────────────────

  Future<ListPermissionRulesResponse> listPermissionRules({String scope = ''});
  Future<SetPermissionRuleResponse> setPermissionRule(PermissionRule rule);
  Future<DeletePermissionRuleResponse> deletePermissionRule(
    String server,
    String tool,
    String scope,
  );

  void close();
}

/// Live gRPC implementation that connects to neurons-service.
class GrpcNeuronsClient implements NeuronsClient {
  GrpcNeuronsClient({String host = 'localhost', int port = 50051})
      : _channel = ClientChannel(
          host,
          port: port,
          options: const ChannelOptions(
            credentials: ChannelCredentials.insecure(),
          ),
        ) {
    _stub = NeuronsInferenceClient(_channel);
  }

  final ClientChannel _channel;
  late final NeuronsInferenceClient _stub;

  @override
  Future<StatusResponse> getStatus() => _stub.getStatus(StatusRequest());

  @override
  Future<LoadModelResponse> loadModel(String modelPath) =>
      _stub.loadModel(LoadModelRequest()..modelPath = modelPath);

  @override
  Future<UnloadModelResponse> unloadModel() =>
      _stub.unloadModel(UnloadModelRequest());

  @override
  Future<bool> deleteModel(String path) =>
      throw UnimplementedError('deleteModel not supported over gRPC');

  @override
  Future<ListModelsResponse> listModels({String modelsDir = ''}) =>
      _stub.listModels(ListModelsRequest()..modelsDir = modelsDir);

  @override
  Stream<GenerateResponse> generate(
    String prompt, {
    List<ChatMessage> history = const [],
    SamplingParams? params,
  }) {
    final req = GenerateRequest()
      ..prompt = prompt
      ..history.addAll(history);
    if (params != null) req.params = params;
    return _stub.generate(req);
  }

  @override
  Future<SearchModelsResponse> searchModels(
    String query, {
    int limit = 30,
    String sort = 'downloads',
    List<String> pipelineTags = const [],
    String author = '',
  }) =>
      _stub.searchModels(SearchModelsRequest()
        ..query = query
        ..limit = limit
        ..sort = sort
        ..pipelineTags.addAll(pipelineTags)
        ..author = author);

  @override
  Future<GetModelInfoResponse> getModelInfo(String modelId) =>
      _stub.getModelInfo(GetModelInfoRequest()..modelId = modelId);

  @override
  Stream<DownloadProgressResponse> downloadModel(String modelId) =>
      _stub.downloadModel(DownloadModelRequest()..modelId = modelId);

  @override
  Future<CancelDownloadResponse> cancelDownload(String downloadId) =>
      _stub.cancelDownload(CancelDownloadRequest()..downloadId = downloadId);

  @override
  Future<void> setHfToken(String token) =>
      _stub.setHfToken(SetHfTokenRequest()..token = token);

  @override
  Stream<LogEntry> streamLogs({String minLevel = 'INFO'}) =>
      _stub.streamLogs(StreamLogsRequest()..minLevel = minLevel);

  // ── Tool approval ────────────────────────────────────────────────────────────

  @override
  Future<ToolApprovalResult> respondToolApproval(
    String approvalId,
    bool approved, {
    String newPermission = '',
  }) =>
      _stub.respondToolApproval(ToolApprovalResponse()
        ..approvalId = approvalId
        ..approved = approved
        ..newPermission = newPermission);

  // ── MCP server management ────────────────────────────────────────────────────

  @override
  Future<ListMcpServersResponse> listMcpServers() =>
      _stub.listMcpServers(ListMcpServersRequest());

  @override
  Future<AddMcpServerResponse> addMcpServer(McpServerConfig server) =>
      _stub.addMcpServer(AddMcpServerRequest()..server = server);

  @override
  Future<RemoveMcpServerResponse> removeMcpServer(String name) =>
      _stub.removeMcpServer(RemoveMcpServerRequest()..name = name);

  @override
  Future<PushMcpServersResponse> pushMcpServers(List<McpServerConfig> servers) =>
      _stub.pushMcpServers(PushMcpServersRequest()..servers.addAll(servers));

  // ── MCP permission management ─────────────────────────────────────────────

  @override
  Future<ListPermissionRulesResponse> listPermissionRules({String scope = ''}) =>
      _stub.listPermissionRules(ListPermissionRulesRequest()..scope = scope);

  @override
  Future<SetPermissionRuleResponse> setPermissionRule(PermissionRule rule) =>
      _stub.setPermissionRule(SetPermissionRuleRequest()..rule = rule);

  @override
  Future<DeletePermissionRuleResponse> deletePermissionRule(
    String server,
    String tool,
    String scope,
  ) =>
      _stub.deletePermissionRule(DeletePermissionRuleRequest()
        ..server = server
        ..tool = tool
        ..scope = scope);

  @override
  void close() => _channel.shutdown();
}
