import 'dart:convert';

enum McpMode { inherit, own }

class NodeConfig {
  NodeConfig({
    required this.id,
    required this.name,
    required this.host,
    required this.port,
    this.isLocal = false,
    this.hfToken,  // null = inherit global token
    this.mcpMode = McpMode.inherit,
  });

  final String id;
  String name;
  String host;
  int port;
  final bool isLocal;
  final String? hfToken;
  final McpMode mcpMode;

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'host': host,
        'port': port,
        'isLocal': isLocal,
        if (hfToken != null) 'hfToken': hfToken,
        'mcpMode': mcpMode.name,
      };

  factory NodeConfig.fromJson(Map<String, dynamic> m) => NodeConfig(
        id: m['id'] as String,
        name: m['name'] as String,
        host: m['host'] as String,
        port: m['port'] as int,
        isLocal: m['isLocal'] as bool? ?? false,
        hfToken: m['hfToken'] as String?,
        mcpMode: McpMode.values.byName(m['mcpMode'] as String? ?? 'inherit'),
      );

  static NodeConfig local() => NodeConfig(
        id: 'local',
        name: 'This Mac',
        host: 'localhost',
        port: 50051,
        isLocal: true,
      );

  static List<NodeConfig> listFromJsonString(String s) {
    final list = json.decode(s) as List<dynamic>;
    return list.map((e) => NodeConfig.fromJson(e as Map<String, dynamic>)).toList();
  }

  static String listToJsonString(List<NodeConfig> nodes) =>
      json.encode(nodes.map((n) => n.toJson()).toList());

  // Use copyWith(hfToken: '') to explicitly clear a per-node token.
  // Use copyWith() without hfToken to leave it unchanged.
  NodeConfig copyWith({
    String? name,
    String? host,
    int? port,
    Object? hfToken = _sentinel,
    McpMode? mcpMode,
  }) =>
      NodeConfig(
        id: id,
        name: name ?? this.name,
        host: host ?? this.host,
        port: port ?? this.port,
        isLocal: isLocal,
        hfToken: hfToken == _sentinel ? this.hfToken : hfToken as String?,
        mcpMode: mcpMode ?? this.mcpMode,
      );

  static const Object _sentinel = Object();
}
