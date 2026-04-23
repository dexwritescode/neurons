import 'dart:async';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/node_config.dart';
import '../proto/neurons.pb.dart' show GpuSlot, McpServerConfig;
import '../services/app_state.dart';
import '../services/neurons_client.dart';
import '../theme/tokens.dart';
import '../widgets/resize_divider.dart';

// ── Live status cache ─────────────────────────────────────────────────────────

enum NodeStatus { online, offline, connecting }

class _LiveStatus {
  const _LiveStatus({
    required this.status,
    this.pingMs,
    this.gpus = const [],
  });
  final NodeStatus status;
  final int? pingMs;
  final List<GpuSlot> gpus;
}

// ── Display model (built from NodeConfig + _LiveStatus) ───────────────────────

class NodeInfo {
  NodeInfo({
    required this.config,
    required this.live,
  });
  final NodeConfig config;
  final _LiveStatus live;

  String get id       => config.id;
  String get name     => config.name;
  String get host     => config.host;
  int    get port     => config.port;
  bool   get isLocal  => config.isLocal;

  NodeStatus get status => live.status;
  int?       get pingMs => live.pingMs;
  List<GpuSlot> get gpus => live.gpus;

  String? get loadedModel {
    if (gpus.isEmpty) return null;
    final m = gpus.first.loadedModel;
    return m.isEmpty ? null : m.split('/').last;
  }
}

// ── Main screen ───────────────────────────────────────────────────────────────

class NodesScreen extends StatefulWidget {
  const NodesScreen({super.key});

  @override
  State<NodesScreen> createState() => _NodesScreenState();
}

class _NodesScreenState extends State<NodesScreen> {
  double _listWidth = 260;
  String? _selectedId;
  bool _showForm = false;
  NodeConfig? _editingNode;

  final Map<String, _LiveStatus> _statusCache = {};
  Timer? _pollTimer;

  static const _listMin = 200.0;
  static const _listMax = 460.0;
  static const _pollInterval = Duration(seconds: 8);

  @override
  void initState() {
    super.initState();
    _selectedId = 'local';
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _pollAll();
      _pollTimer = Timer.periodic(_pollInterval, (_) => _pollAll());
    });
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    super.dispose();
  }

  // Poll all nodes for live status. Local node status comes from AppState.
  Future<void> _pollAll() async {
    if (!mounted) return;
    final state = context.read<AppState>();
    for (final node in state.nodes) {
      if (node.isLocal) {
        // Local node is always online; GPU slots come from AppState's last status
        _statusCache[node.id] = _LiveStatus(
          status: NodeStatus.online,
          pingMs: 0,
          gpus: [], // populated by _applyLocalStatus
        );
      } else {
        _pingRemoteNode(node);
      }
    }
    if (mounted) setState(() {});
  }

  Future<void> _pingRemoteNode(NodeConfig node) async {
    setState(() {
      _statusCache[node.id] = _LiveStatus(
        status: NodeStatus.connecting,
        pingMs: _statusCache[node.id]?.pingMs,
        gpus: _statusCache[node.id]?.gpus ?? [],
      );
    });
    final sw = Stopwatch()..start();
    try {
      final client = GrpcNeuronsClient(host: node.host, port: node.port);
      final resp = await client.getStatus().timeout(const Duration(seconds: 5));
      client.close();
      sw.stop();
      if (mounted) {
        setState(() {
          _statusCache[node.id] = _LiveStatus(
            status: NodeStatus.online,
            pingMs: sw.elapsedMilliseconds,
            gpus: resp.gpus,
          );
        });
      }
    } catch (_) {
      if (mounted) {
        setState(() {
          _statusCache[node.id] = _LiveStatus(
            status: NodeStatus.offline,
            pingMs: null,
            gpus: [],
          );
        });
      }
    }
  }

  List<NodeInfo> _buildDisplayNodes(AppState state) {
    return state.nodes.map((cfg) {
      if (cfg.isLocal) {
        // Local node: status always online; GPUs come from AppState's live status
        final gpus = (state.modelPath?.isNotEmpty == true)
            ? [
                GpuSlot()
                  ..gpuId = '0'
                  ..gpuName = state.backend == 'mlx' ? 'Apple Silicon' : 'CPU'
                  ..loadedModel = state.modelPath ?? ''
                  ..modelType = state.modelType ?? ''
              ]
            : <GpuSlot>[];
        return NodeInfo(
          config: cfg,
          live: _LiveStatus(status: NodeStatus.online, pingMs: 0, gpus: gpus),
        );
      }
      return NodeInfo(
        config: cfg,
        live: _statusCache[cfg.id] ??
            const _LiveStatus(status: NodeStatus.connecting),
      );
    }).toList();
  }

  Future<void> _handleSave(
      String? editId, String name, String host, int port, String? hfToken,
      McpMode mcpMode) async {
    final state = context.read<AppState>();
    final token = (hfToken != null && hfToken.trim().isNotEmpty)
        ? hfToken.trim()
        : null;
    if (editId != null) {
      final existing = state.nodes.where((n) => n.id == editId).firstOrNull;
      if (existing != null) {
        await state.updateNode(existing.copyWith(
            name: name, host: host, port: port,
            hfToken: token, mcpMode: mcpMode));
      }
    } else {
      final node = NodeConfig(
        id: 'node-${DateTime.now().millisecondsSinceEpoch}',
        name: name,
        host: host,
        port: port,
        hfToken: token,
        mcpMode: mcpMode,
      );
      await state.addNode(node);
      setState(() => _selectedId = node.id);
      _pingRemoteNode(node);
    }
    setState(() { _showForm = false; _editingNode = null; });
  }

  Future<void> _handleDelete(String id) async {
    await context.read<AppState>().removeNode(id);
    _statusCache.remove(id);
    if (_selectedId == id) setState(() => _selectedId = 'local');
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final nodes = _buildDisplayNodes(state);
    final selectedNode = nodes.where((n) => n.id == _selectedId).firstOrNull;
    final onlineCount = nodes.where((n) => n.status == NodeStatus.online).length;

    return Stack(
      children: [
        Row(
          children: [
            // ── Left: node list ──────────────────────────────────────────────
            SizedBox(
              width: _listWidth,
              child: Container(
                color: Tokens.surface,
                child: Column(
                  children: [
                    // Header
                    Container(
                      height: 42,
                      padding: const EdgeInsets.symmetric(horizontal: 14),
                      decoration: const BoxDecoration(
                        border: Border(bottom: BorderSide(color: Tokens.glassEdge)),
                      ),
                      child: Row(
                        children: [
                          const Text('NODES',
                              style: TextStyle(
                                fontSize: 10, fontWeight: FontWeight.w700,
                                color: Tokens.textMuted, letterSpacing: 0.8)),
                          const SizedBox(width: 6),
                          Text('· $onlineCount ONLINE',
                              style: const TextStyle(
                                fontSize: 10, fontWeight: FontWeight.w700,
                                color: Tokens.accent, letterSpacing: 0.8)),
                          const Spacer(),
                          GestureDetector(
                            onTap: () => setState(() {
                              _editingNode = null;
                              _showForm = true;
                            }),
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 9, vertical: 4),
                              decoration: BoxDecoration(
                                color: Tokens.accentDim,
                                border: Border.all(
                                    color: Tokens.accent.withAlpha(56)),
                                borderRadius:
                                    BorderRadius.circular(Tokens.radiusInput),
                              ),
                              child: const Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Icon(Icons.add_rounded,
                                      size: 14, color: Tokens.accent),
                                  SizedBox(width: 4),
                                  Text('Add',
                                      style: TextStyle(
                                          fontSize: 12,
                                          fontWeight: FontWeight.w500,
                                          color: Tokens.accent)),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    // Node list
                    Expanded(
                      child: ListView(
                        children: nodes
                            .map((n) => _NodeRow(
                                  node: n,
                                  isActive: state.activeNodeId == n.id,
                                  selected: _selectedId == n.id,
                                  onSelect: () =>
                                      setState(() => _selectedId = n.id),
                                ))
                            .toList(),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            ResizeDivider(
                onDrag: (dx) => setState(() {
                      _listWidth =
                          (_listWidth + dx).clamp(_listMin, _listMax);
                    })),
            // ── Right: detail ────────────────────────────────────────────────
            Expanded(
              child: selectedNode != null
                  ? _NodeDetail(
                      node: selectedNode,
                      state: state,
                      onEdit: (n) => setState(() {
                        _editingNode = n.config;
                        _showForm = true;
                      }),
                      onDelete: _handleDelete,
                      onSwitch: (id) => state.switchNode(id),
                    )
                  : const Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(Icons.hub_outlined,
                              size: 36, color: Tokens.textMuted),
                          SizedBox(height: 12),
                          Text('Select a node',
                              style: TextStyle(
                                  fontSize: 14, color: Tokens.textMuted)),
                        ],
                      ),
                    ),
            ),
          ],
        ),
        // ── Add/edit dialog overlay ──────────────────────────────────────────
        if (_showForm)
          _NodeFormDialog(
            node: _editingNode,
            onSave: _handleSave,
            onClose: () =>
                setState(() { _showForm = false; _editingNode = null; }),
          ),
      ],
    );
  }
}

// ── Node row (left list) ──────────────────────────────────────────────────────

class _NodeRow extends StatefulWidget {
  const _NodeRow({
    required this.node,
    required this.isActive,
    required this.selected,
    required this.onSelect,
  });
  final NodeInfo node;
  final bool isActive;
  final bool selected;
  final VoidCallback onSelect;

  @override
  State<_NodeRow> createState() => _NodeRowState();
}

class _NodeRowState extends State<_NodeRow> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    final n = widget.node;
    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: GestureDetector(
        onTap: widget.onSelect,
        child: AnimatedContainer(
          duration: Tokens.fast,
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 11),
          decoration: BoxDecoration(
            color: widget.selected
                ? Tokens.accentDim
                : _hovered
                    ? Tokens.surfaceElevated
                    : Colors.transparent,
            border: Border(
              left: BorderSide(
                color: widget.selected ? Tokens.accent : Colors.transparent,
                width: 2,
              ),
              bottom: const BorderSide(color: Tokens.glassEdge),
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  _StatusDot(status: n.status),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      n.name,
                      style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w500,
                        color: widget.selected
                            ? Tokens.textPrimary
                            : Tokens.textSecondary,
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  if (widget.isActive)
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 5, vertical: 1),
                      decoration: BoxDecoration(
                        color: Tokens.accentDim,
                        borderRadius: BorderRadius.circular(4),
                        border:
                            Border.all(color: Tokens.accent.withAlpha(60)),
                      ),
                      child: const Text('ACTIVE',
                          style: TextStyle(
                              fontSize: 9,
                              fontWeight: FontWeight.w700,
                              color: Tokens.accent,
                              letterSpacing: 0.4)),
                    )
                  else if (n.pingMs != null)
                    Text('${n.pingMs}ms',
                        style: const TextStyle(
                          fontSize: 11,
                          color: Tokens.textMuted,
                          fontFamily: 'JetBrains Mono',
                        )),
                ],
              ),
              Padding(
                padding: const EdgeInsets.only(left: 16, top: 4),
                child: Row(
                  children: [
                    Flexible(
                      child: Text(
                        '${n.host}:${n.port}',
                        style: const TextStyle(
                          fontSize: 11,
                          color: Tokens.textMuted,
                          fontFamily: 'JetBrains Mono',
                        ),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    if (n.loadedModel != null) ...[
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 6, vertical: 1),
                        decoration: BoxDecoration(
                          color: Tokens.accentDim,
                          borderRadius:
                              BorderRadius.circular(Tokens.radiusPill),
                        ),
                        child: Text(
                          n.loadedModel!.length > 16
                              ? '${n.loadedModel!.substring(0, 14)}…'
                              : n.loadedModel!,
                          style: const TextStyle(
                            fontSize: 10,
                            fontWeight: FontWeight.w500,
                            color: Tokens.accent,
                          ),
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Status dot ────────────────────────────────────────────────────────────────

class _StatusDot extends StatelessWidget {
  const _StatusDot({required this.status});
  final NodeStatus status;

  @override
  Widget build(BuildContext context) {
    final color = switch (status) {
      NodeStatus.online     => Tokens.accent,
      NodeStatus.offline    => Tokens.destructive,
      NodeStatus.connecting => const Color(0xFFFFB84D),
    };
    return Container(
      width: 7,
      height: 7,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: status == NodeStatus.connecting ? Colors.transparent : color,
        border: status == NodeStatus.connecting
            ? Border.all(color: color, width: 1.5)
            : null,
        boxShadow: status == NodeStatus.online
            ? [BoxShadow(color: color.withAlpha(120), blurRadius: 6)]
            : null,
      ),
      child: status == NodeStatus.connecting ? _PulsingDot(color: color) : null,
    );
  }
}

class _PulsingDot extends StatefulWidget {
  const _PulsingDot({required this.color});
  final Color color;

  @override
  State<_PulsingDot> createState() => _PulsingDotState();
}

class _PulsingDotState extends State<_PulsingDot>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..repeat(reverse: true);
  }

  @override
  void dispose() { _ctrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) => FadeTransition(
        opacity: _ctrl,
        child: Container(
          width: 7, height: 7,
          decoration: BoxDecoration(
              shape: BoxShape.circle, color: widget.color),
        ),
      );
}

// ── Node detail pane (right) ──────────────────────────────────────────────────

class _NodeDetail extends StatefulWidget {
  const _NodeDetail({
    required this.node,
    required this.state,
    required this.onEdit,
    required this.onDelete,
    required this.onSwitch,
  });
  final NodeInfo node;
  final AppState state;
  final void Function(NodeInfo) onEdit;
  final void Function(String) onDelete;
  final void Function(String) onSwitch;

  @override
  State<_NodeDetail> createState() => _NodeDetailState();
}

class _NodeDetailState extends State<_NodeDetail> {
  String _detailTab = 'overview';
  String _modelFilter = '';

  static const _tabs = [
    ('overview', 'Overview'),
    ('load',     'Load Model'),
    ('logs',     'Logs'),
  ];

  @override
  void didUpdateWidget(_NodeDetail old) {
    super.didUpdateWidget(old);
    if (old.node.id != widget.node.id) {
      _detailTab = 'overview';
      _modelFilter = '';
    }
  }

  Future<void> _confirmSwitch(BuildContext context, NodeInfo n) async {
    if (widget.state.modelLoaded) {
      final confirmed = await showDialog<bool>(
        context: context,
        builder: (_) => AlertDialog(
          backgroundColor: Tokens.surface,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(Tokens.radiusCard),
            side: const BorderSide(color: Tokens.glassEdge),
          ),
          title: Text('Switch to ${n.name}?',
              style: const TextStyle(
                  color: Tokens.textPrimary, fontSize: 15)),
          content: Text(
              'The current model will be unloaded.',
              style: const TextStyle(
                  color: Tokens.textSecondary, fontSize: 13)),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel',
                  style: TextStyle(color: Tokens.textSecondary)),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Switch'),
            ),
          ],
        ),
      );
      if (confirmed != true) return;
    }
    widget.onSwitch(n.id);
  }

  @override
  Widget build(BuildContext context) {
    final n = widget.node;
    final isActive = widget.state.activeNodeId == n.id;

    return Column(
      children: [
        // Header
        Padding(
          padding: const EdgeInsets.fromLTRB(22, 18, 22, 0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            _StatusDot(status: n.status),
                            const SizedBox(width: 9),
                            Text(n.name,
                                style: const TextStyle(
                                  fontSize: 17,
                                  fontWeight: FontWeight.w600,
                                  color: Tokens.textPrimary,
                                  letterSpacing: -0.3,
                                )),
                          ],
                        ),
                        const SizedBox(height: 4),
                        Row(children: [
                          Text('${n.host}:${n.port}',
                              style: const TextStyle(
                                fontSize: 12,
                                color: Tokens.textSecondary,
                                fontFamily: 'JetBrains Mono',
                              )),
                          if (n.pingMs != null) ...[
                            const Text(' · ',
                                style: TextStyle(color: Tokens.textMuted)),
                            Text('${n.pingMs}ms',
                                style: const TextStyle(
                                  fontSize: 12,
                                  color: Tokens.accent,
                                  fontFamily: 'JetBrains Mono',
                                )),
                          ],
                        ]),
                      ],
                    ),
                  ),
                  // Use / Edit / Remove buttons
                  if (!isActive && n.status == NodeStatus.online)
                    Padding(
                      padding: const EdgeInsets.only(right: 6),
                      child: FilledButton.icon(
                        onPressed: () => _confirmSwitch(context, n),
                        icon: const Icon(Icons.swap_horiz_rounded, size: 14),
                        label: const Text('Use'),
                        style: FilledButton.styleFrom(
                          textStyle: const TextStyle(fontSize: 12),
                          padding: const EdgeInsets.symmetric(
                              horizontal: 10, vertical: 5),
                          minimumSize: Size.zero,
                        ),
                      ),
                    ),
                  if (!n.isLocal) ...[
                    OutlinedButton.icon(
                      onPressed: () => widget.onEdit(n),
                      icon: const Icon(Icons.edit_outlined, size: 13),
                      label: const Text('Edit'),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: Tokens.textSecondary,
                        side: const BorderSide(color: Tokens.glassEdge),
                        textStyle: const TextStyle(fontSize: 12),
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 5),
                        minimumSize: Size.zero,
                      ),
                    ),
                    const SizedBox(width: 6),
                    OutlinedButton.icon(
                      onPressed: () => widget.onDelete(n.id),
                      icon: const Icon(Icons.delete_outline_rounded, size: 13),
                      label: const Text('Remove'),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: Tokens.destructive,
                        side: BorderSide(
                            color: Tokens.destructive.withAlpha(76)),
                        textStyle: const TextStyle(fontSize: 12),
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 5),
                        minimumSize: Size.zero,
                      ),
                    ),
                  ],
                ],
              ),
              const SizedBox(height: 12),
              Row(
                children: _tabs
                    .map((t) => _DetailTab(
                          label: t.$2,
                          active: _detailTab == t.$1,
                          onTap: () =>
                              setState(() => _detailTab = t.$1),
                        ))
                    .toList(),
              ),
            ],
          ),
        ),
        const Divider(height: 1, color: Tokens.glassEdge),
        Expanded(
          child: _detailTab == 'logs'
              ? _LogsTab(node: n)
              : SingleChildScrollView(
                  padding: const EdgeInsets.all(22),
                  child: _detailTab == 'overview'
                      ? _OverviewTab(node: n, state: widget.state)
                      : _LoadModelTab(
                          node: n,
                          state: widget.state,
                          filter: _modelFilter,
                          onFilterChange: (v) =>
                              setState(() => _modelFilter = v)),
                ),
        ),
      ],
    );
  }
}

class _DetailTab extends StatelessWidget {
  const _DetailTab({required this.label, required this.active, required this.onTap});
  final String label;
  final bool active;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          decoration: BoxDecoration(
            border: Border(
              bottom: BorderSide(
                color: active ? Tokens.accent : Colors.transparent,
                width: 2,
              ),
            ),
          ),
          child: Text(label,
              style: TextStyle(
                fontSize: 12,
                fontWeight: active ? FontWeight.w500 : FontWeight.w400,
                color: active ? Tokens.textPrimary : Tokens.textSecondary,
              )),
        ),
      );
}

// ── Logs tab ──────────────────────────────────────────────────────────────────

class _LogsTab extends StatefulWidget {
  const _LogsTab({required this.node});
  final NodeInfo node;
  @override
  State<_LogsTab> createState() => _LogsTabState();
}

class _LogsTabState extends State<_LogsTab> {
  final List<LogEntry> _entries = [];
  StreamSubscription<LogEntry>? _sub;
  final ScrollController _scroll = ScrollController();
  GrpcNeuronsClient? _logClient;

  @override
  void initState() {
    super.initState();
    _startStream();
  }

  @override
  void didUpdateWidget(_LogsTab old) {
    super.didUpdateWidget(old);
    if (old.node.id != widget.node.id) {
      _sub?.cancel();
      _logClient?.close();
      _entries.clear();
      _startStream();
    }
  }

  void _startStream() {
    _logClient = GrpcNeuronsClient(
        host: widget.node.host, port: widget.node.port);
    _sub = _logClient!.streamLogs().listen(
      (entry) {
        if (!mounted) return;
        setState(() {
          _entries.add(entry);
          if (_entries.length > 2000) _entries.removeAt(0);
        });
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (_scroll.hasClients) {
            _scroll.animateTo(_scroll.position.maxScrollExtent,
                duration: const Duration(milliseconds: 100),
                curve: Curves.easeOut);
          }
        });
      },
      onError: (_) {},
    );
  }

  @override
  void dispose() {
    _sub?.cancel();
    _logClient?.close();
    _scroll.dispose();
    super.dispose();
  }

  Color _levelColor(String level) => switch (level) {
        'ERROR' => Tokens.destructive,
        'WARN'  => const Color(0xFFE8A838),
        _       => Tokens.textSecondary,
      };

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // Toolbar
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: Row(
            children: [
              Text('${_entries.length} entries',
                  style: const TextStyle(
                      fontSize: 11, color: Tokens.textMuted)),
              const Spacer(),
              TextButton(
                onPressed: () => setState(() => _entries.clear()),
                child: const Text('Clear',
                    style: TextStyle(fontSize: 11)),
              ),
            ],
          ),
        ),
        const Divider(height: 1, color: Tokens.glassEdge),
        Expanded(
          child: _entries.isEmpty
              ? const Center(
                  child: Text('No logs yet…',
                      style: TextStyle(
                          color: Tokens.textMuted, fontSize: 12)))
              : ListView.builder(
                  controller: _scroll,
                  padding: const EdgeInsets.symmetric(
                      horizontal: 14, vertical: 8),
                  itemCount: _entries.length,
                  itemBuilder: (_, i) {
                    final e = _entries[i];
                    final ts = DateTime.fromMillisecondsSinceEpoch(
                        e.timestampMs.toInt());
                    return Padding(
                      padding: const EdgeInsets.symmetric(vertical: 1),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          SizedBox(
                            width: 58,
                            child: Text(
                              '${ts.hour.toString().padLeft(2, '0')}:'
                              '${ts.minute.toString().padLeft(2, '0')}:'
                              '${ts.second.toString().padLeft(2, '0')}',
                              style: const TextStyle(
                                  fontSize: 10,
                                  fontFamily: 'JetBrainsMono',
                                  color: Tokens.textMuted),
                            ),
                          ),
                          SizedBox(
                            width: 40,
                            child: Text(
                              e.level,
                              style: TextStyle(
                                  fontSize: 10,
                                  fontFamily: 'JetBrainsMono',
                                  color: _levelColor(e.level),
                                  fontWeight: FontWeight.w600),
                            ),
                          ),
                          Expanded(
                            child: Text(
                              e.message,
                              style: const TextStyle(
                                  fontSize: 11,
                                  fontFamily: 'JetBrainsMono',
                                  color: Tokens.textSecondary),
                            ),
                          ),
                        ],
                      ),
                    );
                  },
                ),
        ),
      ],
    );
  }
}

// ── Overview tab ──────────────────────────────────────────────────────────────

class _OverviewTab extends StatelessWidget {
  const _OverviewTab({required this.node, required this.state});
  final NodeInfo node;
  final AppState state;

  @override
  Widget build(BuildContext context) {
    final gpus = node.gpus;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('GPU SLOTS',
            style: TextStyle(
              fontSize: 10, fontWeight: FontWeight.w700,
              color: Tokens.textMuted, letterSpacing: 0.8)),
        const SizedBox(height: 10),
        if (gpus.isEmpty)
          _statusBox(node)
        else
          ...gpus.map((slot) => _GpuSlotCard(slot: slot, isLocal: node.isLocal, state: state)),
        const SizedBox(height: 24),
        const Divider(color: Tokens.glassEdge),
        const SizedBox(height: 16),
        const Text('MCP SERVERS',
            style: TextStyle(
              fontSize: 10, fontWeight: FontWeight.w700,
              color: Tokens.textMuted, letterSpacing: 0.8)),
        const SizedBox(height: 10),
        _McpServersOverview(node: node, state: state),
      ],
    );
  }

  Widget _statusBox(NodeInfo n) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        decoration: BoxDecoration(
          color: Tokens.surfaceElevated,
          borderRadius: BorderRadius.circular(Tokens.radiusInput),
          border: Border.all(color: Tokens.glassEdge),
        ),
        child: Text(
          switch (n.status) {
            NodeStatus.connecting => 'Connecting to node…',
            NodeStatus.offline    => 'Node unreachable',
            NodeStatus.online     => 'No model loaded',
          },
          style: const TextStyle(fontSize: 13, color: Tokens.textMuted),
        ),
      );
}

class _GpuSlotCard extends StatelessWidget {
  const _GpuSlotCard({required this.slot, required this.isLocal, required this.state});
  final GpuSlot slot;
  final bool isLocal;
  final AppState state;

  String _fmtBytes(int b) {
    if (b >= 1 << 30) return '${(b / (1 << 30)).toStringAsFixed(1)} GB';
    if (b >= 1 << 20) return '${(b / (1 << 20)).toStringAsFixed(0)} MB';
    return '$b B';
  }

  @override
  Widget build(BuildContext context) {
    final hasModel = slot.loadedModel.isNotEmpty;
    final total = slot.vramTotalBytes.toInt();
    final used  = slot.vramUsedBytes.toInt();
    final tokS  = slot.tokPerSec;
    final modelName = hasModel ? slot.loadedModel.split('/').last : null;

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Tokens.surfaceElevated,
        borderRadius: BorderRadius.circular(Tokens.radiusInput),
        border: Border.all(
            color: hasModel ? Tokens.accent.withAlpha(56) : Tokens.glassEdge),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // GPU name + slot id
          Row(children: [
            Icon(Icons.memory_rounded, size: 14, color: hasModel ? Tokens.accent : Tokens.textMuted),
            const SizedBox(width: 7),
            Expanded(
              child: Text(slot.gpuName.isEmpty ? 'GPU ${slot.gpuId}' : slot.gpuName,
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: hasModel ? Tokens.textPrimary : Tokens.textSecondary,
                  )),
            ),
            if (total > 0)
              Text(_fmtBytes(total),
                  style: const TextStyle(
                    fontSize: 11, color: Tokens.textMuted,
                    fontFamily: 'JetBrains Mono')),
          ]),
          // VRAM usage bar
          if (total > 0 && used > 0) ...[
            const SizedBox(height: 8),
            ClipRRect(
              borderRadius: BorderRadius.circular(2),
              child: LinearProgressIndicator(
                value: used / total,
                minHeight: 3,
                backgroundColor: Tokens.glassEdge,
                valueColor: AlwaysStoppedAnimation(
                    hasModel ? Tokens.accent : Tokens.textMuted),
              ),
            ),
            const SizedBox(height: 4),
            Row(children: [
              Text('${_fmtBytes(used)} used',
                  style: const TextStyle(
                      fontSize: 10, color: Tokens.textMuted,
                      fontFamily: 'JetBrains Mono')),
              const Spacer(),
              Text('${_fmtBytes(total - used)} free',
                  style: const TextStyle(
                      fontSize: 10, color: Tokens.textMuted,
                      fontFamily: 'JetBrains Mono')),
            ]),
          ],
          // Loaded model
          if (hasModel) ...[
            const SizedBox(height: 10),
            const Divider(height: 1, color: Tokens.glassEdge),
            const SizedBox(height: 10),
            Row(children: [
              Container(
                width: 7, height: 7,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: Tokens.accent,
                  boxShadow: [BoxShadow(
                      color: Tokens.accent.withAlpha(100), blurRadius: 6)],
                ),
              ),
              const SizedBox(width: 9),
              Expanded(
                child: Text(modelName ?? slot.loadedModel,
                    style: const TextStyle(
                      fontSize: 12, fontFamily: 'JetBrains Mono',
                      color: Tokens.accent, fontWeight: FontWeight.w500),
                    overflow: TextOverflow.ellipsis),
              ),
              if (tokS > 0)
                Text('${tokS.toStringAsFixed(1)} tok/s',
                    style: const TextStyle(
                      fontSize: 11, color: Tokens.accent,
                      fontFamily: 'JetBrains Mono')),
              if (isLocal) ...[
                const SizedBox(width: 8),
                OutlinedButton.icon(
                  onPressed: () => state.unloadModel(),
                  icon: const Icon(Icons.eject_rounded, size: 13),
                  label: const Text('Eject'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Tokens.textSecondary,
                    side: const BorderSide(color: Tokens.glassEdge),
                    textStyle: const TextStyle(fontSize: 12),
                    padding: const EdgeInsets.symmetric(
                        horizontal: 10, vertical: 5),
                    minimumSize: Size.zero,
                  ),
                ),
              ],
            ]),
          ],
        ],
      ),
    );
  }
}

// ── Load Model tab ────────────────────────────────────────────────────────────

class _LoadModelTab extends StatefulWidget {
  const _LoadModelTab({
    required this.node,
    required this.state,
    required this.filter,
    required this.onFilterChange,
  });
  final NodeInfo node;
  final AppState state;
  final String filter;
  final ValueChanged<String> onFilterChange;

  @override
  State<_LoadModelTab> createState() => _LoadModelTabState();
}

class _LoadModelTabState extends State<_LoadModelTab> {
  List<dynamic>? _remoteModels;
  bool _loading = false;
  String? _fetchError;

  @override
  void initState() {
    super.initState();
    if (!widget.node.isLocal) _fetchRemoteModels();
  }

  @override
  void didUpdateWidget(_LoadModelTab old) {
    super.didUpdateWidget(old);
    if (old.node.id != widget.node.id && !widget.node.isLocal) {
      setState(() { _remoteModels = null; _fetchError = null; });
      _fetchRemoteModels();
    }
  }

  Future<void> _fetchRemoteModels() async {
    setState(() { _loading = true; _fetchError = null; });
    try {
      final client = GrpcNeuronsClient(
          host: widget.node.host, port: widget.node.port);
      final resp = await client.listModels()
          .timeout(const Duration(seconds: 10));
      client.close();
      if (mounted) setState(() { _remoteModels = resp.models; _loading = false; });
    } catch (e) {
      if (mounted) setState(() {
        _fetchError = e.toString();
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final node = widget.node;
    final state = widget.state;
    final filter = widget.filter;

    if (node.status != NodeStatus.online) {
      return const Padding(
        padding: EdgeInsets.only(top: 32),
        child: Center(
          child: Text('Node must be online to manage models',
              style: TextStyle(fontSize: 13, color: Tokens.textMuted)),
        ),
      );
    }

    final isActive = state.activeNodeId == node.id;

    // Build model list: local from AppState, remote from fetched list.
    final allModels = node.isLocal
        ? state.availableModels
        : (_remoteModels ?? []);

    final models = allModels
        .where((m) =>
            filter.isEmpty ||
            m.name.toLowerCase().contains(filter.toLowerCase()))
        .toList();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Non-active node notice
        if (!isActive)
          Container(
            margin: const EdgeInsets.only(bottom: 14),
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: Tokens.surfaceElevated,
              borderRadius: BorderRadius.circular(Tokens.radiusInput),
              border: Border.all(color: Tokens.glassEdge),
            ),
            child: Row(
              children: [
                const Icon(Icons.info_outline_rounded,
                    size: 14, color: Tokens.textMuted),
                const SizedBox(width: 8),
                const Expanded(
                  child: Text(
                    'Click "Use" to activate this node before loading a model.',
                    style: TextStyle(fontSize: 12, color: Tokens.textMuted),
                  ),
                ),
              ],
            ),
          ),
        TextField(
          onChanged: widget.onFilterChange,
          style: const TextStyle(fontSize: 13, color: Tokens.textPrimary),
          decoration: InputDecoration(
            hintText: 'Filter models on this node…',
            hintStyle:
                const TextStyle(color: Tokens.textMuted, fontSize: 13),
            prefixIcon: const Icon(Icons.search_rounded,
                size: 16, color: Tokens.textMuted),
            filled: true,
            fillColor: Tokens.surfaceElevated,
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(Tokens.radiusInput),
              borderSide: const BorderSide(color: Tokens.glassEdge),
            ),
            enabledBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(Tokens.radiusInput),
              borderSide: const BorderSide(color: Tokens.glassEdge),
            ),
            focusedBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(Tokens.radiusInput),
              borderSide: const BorderSide(color: Tokens.accent),
            ),
            contentPadding:
                const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          ),
        ),
        const SizedBox(height: 14),
        if (_loading)
          const Center(
            child: Padding(
              padding: EdgeInsets.only(top: 24),
              child: CircularProgressIndicator(
                  strokeWidth: 2, color: Tokens.accent),
            ),
          )
        else if (_fetchError != null)
          Center(
            child: Padding(
              padding: const EdgeInsets.only(top: 24),
              child: Column(
                children: [
                  Text(_fetchError!,
                      style: const TextStyle(
                          fontSize: 12, color: Tokens.destructive)),
                  const SizedBox(height: 10),
                  TextButton.icon(
                    onPressed: _fetchRemoteModels,
                    icon: const Icon(Icons.refresh_rounded, size: 14),
                    label: const Text('Retry'),
                  ),
                ],
              ),
            ),
          )
        else if (models.isEmpty)
          Center(
            child: Padding(
              padding: const EdgeInsets.only(top: 24),
              child: Text(
                filter.isNotEmpty
                    ? 'No matching models'
                    : 'No models in ~/.neurons/models/',
                style: const TextStyle(fontSize: 13, color: Tokens.textMuted),
              ),
            ),
          )
        else
          ...models.map((m) {
            final isLoaded = state.modelPath == m.path;
            return _ModelLoadRow(
              name: m.name,
              path: m.path,
              isLoaded: isLoaded,
              canLoad: isActive,
              onLoad: (isLoaded || !isActive) ? null : () => state.loadModel(m.path),
            );
          }),
      ],
    );
  }
}

class _ModelLoadRow extends StatelessWidget {
  const _ModelLoadRow({
    required this.name,
    required this.path,
    required this.isLoaded,
    required this.canLoad,
    required this.onLoad,
  });
  final String name;
  final String path;
  final bool isLoaded;
  final bool canLoad;
  final VoidCallback? onLoad;

  @override
  Widget build(BuildContext context) => Container(
        margin: const EdgeInsets.only(bottom: 6),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: isLoaded ? Tokens.accentDim : Tokens.surfaceElevated,
          borderRadius: BorderRadius.circular(Tokens.radiusInput),
          border: Border.all(
            color: isLoaded ? Tokens.accent.withAlpha(56) : Tokens.glassEdge,
          ),
        ),
        child: Row(
          children: [
            Expanded(
              child: Text(name,
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                    color: isLoaded ? Tokens.accent : Tokens.textPrimary,
                  ),
                  overflow: TextOverflow.ellipsis),
            ),
            if (isLoaded)
              const Text('LOADED',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: Tokens.accent,
                    letterSpacing: 0.4,
                  ))
            else
              FilledButton.icon(
                onPressed: onLoad,
                icon: const Icon(Icons.play_arrow_rounded, size: 14),
                label: const Text('Load'),
                style: FilledButton.styleFrom(
                  textStyle: const TextStyle(fontSize: 12),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                  minimumSize: Size.zero,
                ),
              ),
          ],
        ),
      );
}

// ── Node add/edit dialog ──────────────────────────────────────────────────────

class _NodeFormDialog extends StatefulWidget {
  const _NodeFormDialog({
    required this.node,
    required this.onSave,
    required this.onClose,
  });
  final NodeConfig? node;
  final void Function(
      String? editId, String name, String host, int port, String? hfToken,
      McpMode mcpMode) onSave;
  final VoidCallback onClose;

  @override
  State<_NodeFormDialog> createState() => _NodeFormDialogState();
}

class _NodeFormDialogState extends State<_NodeFormDialog> {
  late final TextEditingController _nameCtrl;
  late final TextEditingController _hostCtrl;
  late final TextEditingController _portCtrl;
  late final TextEditingController _tokenCtrl;
  late bool _useGlobalToken;
  bool _obscureToken = true;
  late McpMode _mcpMode;

  @override
  void initState() {
    super.initState();
    _nameCtrl = TextEditingController(text: widget.node?.name ?? '');
    _hostCtrl = TextEditingController(text: widget.node?.host ?? '');
    _portCtrl = TextEditingController(text: '${widget.node?.port ?? 50051}');
    final existingToken = widget.node?.hfToken;
    _useGlobalToken = existingToken == null || existingToken.isEmpty;
    _tokenCtrl = TextEditingController(text: existingToken ?? '');
    _mcpMode = widget.node?.mcpMode ?? McpMode.inherit;
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _hostCtrl.dispose();
    _portCtrl.dispose();
    _tokenCtrl.dispose();
    super.dispose();
  }

  InputDecoration _inputDec(String hint) => InputDecoration(
        hintText: hint,
        hintStyle: const TextStyle(color: Tokens.textMuted),
        filled: true,
        fillColor: Tokens.surfaceElevated,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(Tokens.radiusInput),
          borderSide: const BorderSide(color: Tokens.glassEdge),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(Tokens.radiusInput),
          borderSide: const BorderSide(color: Tokens.glassEdge),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(Tokens.radiusInput),
          borderSide: const BorderSide(color: Tokens.accent),
        ),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      );

  @override
  Widget build(BuildContext context) {
    final isEdit = widget.node != null;
    return Container(
      color: Colors.black.withAlpha(165),
      child: Center(
        child: Container(
          width: 380,
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            color: Tokens.surface,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: Tokens.glassEdge),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withAlpha(120),
                blurRadius: 48,
                offset: const Offset(0, 24),
              ),
            ],
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(isEdit ? 'Edit Node' : 'Add Node',
                  style: const TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                    color: Tokens.textPrimary,
                  )),
              const SizedBox(height: 20),
              const Text('NAME',
                  style: TextStyle(
                    fontSize: 10, fontWeight: FontWeight.w700,
                    color: Tokens.textMuted, letterSpacing: 0.7)),
              const SizedBox(height: 6),
              TextField(
                controller: _nameCtrl,
                style:
                    const TextStyle(fontSize: 13, color: Tokens.textPrimary),
                decoration: _inputDec('e.g. studio-mac'),
              ),
              const SizedBox(height: 14),
              const Text('HOST',
                  style: TextStyle(
                    fontSize: 10, fontWeight: FontWeight.w700,
                    color: Tokens.textMuted, letterSpacing: 0.7)),
              const SizedBox(height: 6),
              TextField(
                controller: _hostCtrl,
                style:
                    const TextStyle(fontSize: 13, color: Tokens.textPrimary),
                decoration: _inputDec('192.168.1.100'),
              ),
              const SizedBox(height: 14),
              const Text('PORT',
                  style: TextStyle(
                    fontSize: 10, fontWeight: FontWeight.w700,
                    color: Tokens.textMuted, letterSpacing: 0.7)),
              const SizedBox(height: 6),
              TextField(
                controller: _portCtrl,
                keyboardType: TextInputType.number,
                style: const TextStyle(
                    fontSize: 13,
                    color: Tokens.textPrimary,
                    fontFamily: 'JetBrains Mono'),
                decoration: _inputDec('50051'),
              ),
              const SizedBox(height: 20),
              // ── HuggingFace token ──────────────────────────────────────
              const Divider(color: Tokens.glassEdge),
              const SizedBox(height: 14),
              Row(
                children: [
                  const Text('HUGGINGFACE TOKEN',
                      style: TextStyle(
                        fontSize: 10, fontWeight: FontWeight.w700,
                        color: Tokens.textMuted, letterSpacing: 0.7)),
                  const Spacer(),
                  Row(
                    children: [
                      Text('Use global',
                          style: const TextStyle(
                              fontSize: 11, color: Tokens.textSecondary)),
                      const SizedBox(width: 6),
                      Switch(
                        value: _useGlobalToken,
                        onChanged: (v) => setState(() => _useGlobalToken = v),
                        activeColor: Tokens.accent,
                        materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                      ),
                    ],
                  ),
                ],
              ),
              if (!_useGlobalToken) ...[
                const SizedBox(height: 8),
                TextField(
                  controller: _tokenCtrl,
                  obscureText: _obscureToken,
                  style: const TextStyle(
                      fontSize: 13, color: Tokens.textPrimary,
                      fontFamily: 'JetBrains Mono'),
                  decoration: _inputDec('hf_••••••••••••••••••••').copyWith(
                    suffixIcon: IconButton(
                      icon: Icon(
                        _obscureToken
                            ? Icons.visibility_outlined
                            : Icons.visibility_off_outlined,
                        size: 16,
                        color: Tokens.textSecondary,
                      ),
                      onPressed: () =>
                          setState(() => _obscureToken = !_obscureToken),
                    ),
                  ),
                ),
              ] else ...[
                const SizedBox(height: 6),
                const Text('This node will use the global token from Settings.',
                    style: TextStyle(fontSize: 11, color: Tokens.textMuted)),
              ],
              const SizedBox(height: 20),
              const Divider(color: Tokens.glassEdge),
              const SizedBox(height: 14),
              Row(
                children: [
                  const Text('MCP SERVERS',
                      style: TextStyle(
                        fontSize: 10, fontWeight: FontWeight.w700,
                        color: Tokens.textMuted, letterSpacing: 0.7)),
                  const Spacer(),
                  Row(
                    children: [
                      Text(
                        _mcpMode == McpMode.inherit
                            ? 'Inherit from controller'
                            : 'Own config',
                        style: const TextStyle(
                            fontSize: 11, color: Tokens.textSecondary)),
                      const SizedBox(width: 6),
                      Switch(
                        value: _mcpMode == McpMode.inherit,
                        onChanged: (v) => setState(() =>
                            _mcpMode = v ? McpMode.inherit : McpMode.own),
                        activeColor: Tokens.accent,
                        materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                      ),
                    ],
                  ),
                ],
              ),
              const SizedBox(height: 4),
              Text(
                _mcpMode == McpMode.inherit
                    ? 'This node will receive MCP servers from this device on connect.'
                    : 'This node manages its own MCP server list independently.',
                style: const TextStyle(fontSize: 11, color: Tokens.textMuted),
              ),
              const SizedBox(height: 24),
              Row(
                children: [
                  OutlinedButton(
                    onPressed: widget.onClose,
                    style: OutlinedButton.styleFrom(
                      foregroundColor: Tokens.textSecondary,
                      side: const BorderSide(color: Tokens.glassEdge),
                    ),
                    child: const Text('Cancel'),
                  ),
                  const Spacer(),
                  FilledButton(
                    onPressed: () {
                      final name = _nameCtrl.text.trim();
                      final host = _hostCtrl.text.trim();
                      final port =
                          int.tryParse(_portCtrl.text.trim()) ?? 50051;
                      if (name.isEmpty || host.isEmpty) return;
                      final token = _useGlobalToken ? null : _tokenCtrl.text.trim();
                      widget.onSave(widget.node?.id, name, host, port, token, _mcpMode);
                    },
                    child: Text(isEdit ? 'Save' : 'Add Node'),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ── MCP servers overview (inside node Overview tab) ───────────────────────────

class _McpServersOverview extends StatelessWidget {
  const _McpServersOverview({required this.node, required this.state});
  final NodeInfo node;
  final AppState state;

  @override
  Widget build(BuildContext context) {
    if (node.isLocal) {
      final servers = state.mcpServers;
      if (servers.isEmpty) {
        return _infoBox('No MCP servers configured. Add them in Settings → MCP Servers.');
      }
      return Column(
        children: servers.map((s) => _McpServerChip(server: s)).toList(),
      );
    }
    final mode = node.config.mcpMode;
    return _infoBox(
      mode == McpMode.inherit
          ? 'Inherits MCP servers from this device.'
          : 'Configured independently on this node.',
    );
  }

  Widget _infoBox(String text) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: Tokens.surfaceElevated,
          borderRadius: BorderRadius.circular(Tokens.radiusInput),
          border: Border.all(color: Tokens.glassEdge),
        ),
        child: Text(text,
            style: const TextStyle(fontSize: 12, color: Tokens.textMuted)),
      );
}

class _McpServerChip extends StatelessWidget {
  const _McpServerChip({required this.server});
  final McpServerConfig server;

  @override
  Widget build(BuildContext context) {
    final isStdio = server.transport == 'stdio';
    return Container(
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: Tokens.surfaceElevated,
        borderRadius: BorderRadius.circular(Tokens.radiusInput),
        border: Border.all(
            color: server.enabled
                ? Tokens.glassEdge
                : Tokens.glassEdge.withAlpha(80)),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
            decoration: BoxDecoration(
              color: Tokens.accentDim,
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(
              isStdio ? 'STDIO' : 'SSE',
              style: const TextStyle(
                fontSize: 9, fontWeight: FontWeight.w700,
                color: Tokens.accent, letterSpacing: 0.5),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(server.name,
                style: TextStyle(
                  fontSize: 12,
                  color: server.enabled ? Tokens.textPrimary : Tokens.textMuted,
                )),
          ),
          if (!server.enabled)
            const Text('disabled',
                style: TextStyle(fontSize: 10, color: Tokens.textMuted)),
        ],
      ),
    );
  }
}

