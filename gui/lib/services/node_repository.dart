import 'package:shared_preferences/shared_preferences.dart';

import '../models/node_config.dart';

class NodeRepository {
  static const _kNodes = 'neurons_nodes';

  Future<List<NodeConfig>> loadRemoteNodes() async {
    final prefs = await SharedPreferences.getInstance();
    final s = prefs.getString(_kNodes);
    if (s == null || s.isEmpty) return [];
    try {
      return NodeConfig.listFromJsonString(s);
    } catch (_) {
      return [];
    }
  }

  Future<void> saveRemoteNodes(List<NodeConfig> nodes) async {
    final prefs = await SharedPreferences.getInstance();
    // Never persist the local node — it's always synthesised at runtime.
    final remote = nodes.where((n) => !n.isLocal).toList();
    await prefs.setString(_kNodes, NodeConfig.listToJsonString(remote));
  }
}
