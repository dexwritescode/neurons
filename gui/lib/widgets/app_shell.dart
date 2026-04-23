import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import '../services/app_state.dart';
import '../proto/neurons.pb.dart' show McpServerConfig;
import '../screens/chat_screen.dart';
import '../screens/model_picker_screen.dart';
import '../screens/model_browser_screen.dart';
import '../screens/nodes_screen.dart';
import '../theme/tokens.dart';
import '../widgets/neurons_wordmark.dart';
import '../widgets/hf_token_dialog.dart';

// ── Tab enum ──────────────────────────────────────────────────────────────────

enum _Tab { chats, models, browse, nodes }

// ── AppShell ──────────────────────────────────────────────────────────────────

class AppShell extends StatefulWidget {
  const AppShell({super.key});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  _Tab _tab = _Tab.chats;
  bool _settingsOpen = false;
  bool _prevModelLoaded = false;

  void _handleTabChange(_Tab tab) {
    setState(() {
      _settingsOpen = false;
      _tab = tab;
    });
  }

  void _toggleSettings() {
    setState(() => _settingsOpen = !_settingsOpen);
  }

  Widget _buildContent(AppState state) {
    if (_settingsOpen) return const _SettingsPanel();
    return switch (_tab) {
      _Tab.chats => const ChatScreen(),
      _Tab.models => const ModelPickerScreen(),
      _Tab.browse => const ModelBrowserScreen(),
      _Tab.nodes  => const NodesScreen(),
    };
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();

    // Auto-switch to chats when a model finishes loading.
    if (state.modelLoaded && !_prevModelLoaded) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) setState(() { _tab = _Tab.chats; _settingsOpen = false; });
      });
    }
    _prevModelLoaded = state.modelLoaded;

    return CallbackShortcuts(
      bindings: {
        const SingleActivator(LogicalKeyboardKey.keyZ, meta: true): () {
          if (state.canUndo) state.undoDelete();
        },
      },
      child: Focus(
        autofocus: true,
        child: Scaffold(
          backgroundColor: Tokens.background,
          body: Column(
            children: [
              _TopBar(
                tab: _settingsOpen ? null : _tab,
                settingsOpen: _settingsOpen,
                onTabChange: _handleTabChange,
                onSettings: _toggleSettings,
              ),
              const Divider(height: 1, thickness: 1, color: Tokens.glassEdge),
              Expanded(
                child: AnimatedSwitcher(
                  duration: Tokens.normal,
                  switchInCurve: Tokens.curve,
                  child: KeyedSubtree(
                    key: ValueKey(_settingsOpen ? 'settings' : _tab.name),
                    child: _buildContent(state),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Top navigation bar ────────────────────────────────────────────────────────

class _TopBar extends StatelessWidget {
  const _TopBar({
    required this.tab,
    required this.settingsOpen,
    required this.onTabChange,
    required this.onSettings,
  });

  final _Tab? tab;
  final bool settingsOpen;
  final ValueChanged<_Tab> onTabChange;
  final VoidCallback onSettings;

  static const _tabs = [
    (_Tab.chats,  'Chats',  Icons.chat_bubble_outline_rounded),
    (_Tab.models, 'Models', Icons.memory_rounded),
    (_Tab.browse, 'Browse', Icons.travel_explore_rounded),
    (_Tab.nodes,  'Nodes',  Icons.hub_outlined),
  ];

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();

    return Container(
      height: 46,
      color: Tokens.surface,
      child: Row(
        children: [
          // Wordmark
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 16),
            child: NeuronsWordmark(size: 13),
          ),
          // Separator
          Container(width: 1, height: 26, color: Tokens.glassEdge),
          const SizedBox(width: 4),
          // Tab buttons
          for (final (t, label, icon) in _tabs)
            _TopBarTab(
              label: label,
              icon: icon,
              active: tab == t,
              onTap: () => onTabChange(t),
            ),
          const Spacer(),
          // Model pill (when loaded)
          if (state.modelLoaded) ...[
            _ModelStatusPill(state: state),
            const SizedBox(width: 8),
          ],
          // Settings gear
          _SettingsButton(active: settingsOpen, onTap: onSettings),
          const SizedBox(width: 8),
        ],
      ),
    );
  }
}

class _TopBarTab extends StatefulWidget {
  const _TopBarTab({
    required this.label,
    required this.icon,
    required this.active,
    required this.onTap,
  });

  final String label;
  final IconData icon;
  final bool active;
  final VoidCallback onTap;

  @override
  State<_TopBarTab> createState() => _TopBarTabState();
}

class _TopBarTabState extends State<_TopBarTab> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: GestureDetector(
        onTap: widget.onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 13),
          decoration: BoxDecoration(
            border: Border(
              bottom: BorderSide(
                color: widget.active ? Tokens.accent : Colors.transparent,
                width: 2,
              ),
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                widget.icon,
                size: 15,
                color: widget.active
                    ? Tokens.textPrimary
                    : _hovered
                        ? Tokens.textSecondary
                        : Tokens.textMuted,
              ),
              const SizedBox(width: 6),
              Text(
                widget.label,
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: widget.active ? FontWeight.w500 : FontWeight.w400,
                  color: widget.active
                      ? Tokens.textPrimary
                      : _hovered
                          ? Tokens.textSecondary
                          : Tokens.textMuted,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ModelStatusPill extends StatelessWidget {
  const _ModelStatusPill({required this.state});
  final AppState state;

  String get _label {
    final nodeName = state.activeNode?.name ?? 'local';
    final type = state.modelType;
    if (type != null && type.isNotEmpty) return '$type · $nodeName';
    final path = state.modelPath;
    if (path != null) return '${path.split('/').last} · $nodeName';
    return 'Model · $nodeName';
  }

  @override
  Widget build(BuildContext context) {
    return PopupMenuButton<String>(
      offset: const Offset(0, 34),
      color: Tokens.surface,
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(Tokens.radiusInput),
        side: const BorderSide(color: Tokens.glassEdge),
      ),
      onSelected: (_) => state.unloadModel(),
      itemBuilder: (_) => [
        const PopupMenuItem<String>(
          value: 'eject',
          height: 38,
          child: Row(
            children: [
              Icon(Icons.eject_rounded, size: 15, color: Tokens.textSecondary),
              SizedBox(width: 8),
              Text('Eject model',
                  style: TextStyle(fontSize: 13, color: Tokens.textPrimary)),
            ],
          ),
        ),
      ],
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(
          color: Tokens.accentDim,
          borderRadius: BorderRadius.circular(Tokens.radiusPill),
          border: Border.all(color: Tokens.accent.withAlpha(56)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 6,
              height: 6,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Tokens.accent,
                boxShadow: [BoxShadow(color: Tokens.accent.withAlpha(120), blurRadius: 6)],
              ),
            ),
            const SizedBox(width: 7),
            Text(
              _label,
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w500,
                color: Tokens.accent,
                fontFamily: 'JetBrains Mono',
              ),
            ),
            const SizedBox(width: 5),
            const Icon(Icons.expand_more_rounded, size: 13, color: Tokens.accent),
          ],
        ),
      ),
    );
  }
}

class _SettingsButton extends StatelessWidget {
  const _SettingsButton({required this.active, required this.onTap});
  final bool active;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 30,
        height: 30,
        decoration: BoxDecoration(
          color: active ? Tokens.surfaceElevated : Colors.transparent,
          borderRadius: BorderRadius.circular(Tokens.radiusInput),
        ),
        child: Icon(
          Icons.settings_outlined,
          size: 17,
          color: active ? Tokens.textPrimary : Tokens.textMuted,
        ),
      ),
    );
  }
}

// ── Chat empty state ──────────────────────────────────────────────────────────

class _ChatEmptyState extends StatelessWidget {
  const _ChatEmptyState({required this.onBrowse});
  final VoidCallback onBrowse;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Tokens.background,
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const NeuronsWordmark(size: 28),
            const SizedBox(height: 32),
            Text(
              'Load a model to start chatting',
              style: Theme.of(context)
                  .textTheme
                  .bodyMedium
                  ?.copyWith(color: Tokens.textSecondary),
            ),
            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed: onBrowse,
              icon: const Icon(Icons.memory_rounded, size: 16),
              label: const Text('Browse models'),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Settings panel ────────────────────────────────────────────────────────────

class _SettingsPanel extends StatefulWidget {
  const _SettingsPanel();

  @override
  State<_SettingsPanel> createState() => _SettingsPanelState();
}

class _SettingsPanelState extends State<_SettingsPanel> {
  late final TextEditingController _promptCtrl;
  bool _saved = false;

  @override
  void initState() {
    super.initState();
    _promptCtrl = TextEditingController(
        text: context.read<AppState>().systemPrompt);
  }

  @override
  void dispose() {
    _promptCtrl.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    await context.read<AppState>().setSystemPrompt(_promptCtrl.text);
    if (!mounted) return;
    setState(() => _saved = true);
    await Future<void>.delayed(const Duration(milliseconds: 1800));
    if (mounted) setState(() => _saved = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Tokens.background,
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(40),
        child: SizedBox(
          width: 500,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Settings',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                  color: Tokens.textPrimary,
                  letterSpacing: -0.3,
                ),
              ),
              const SizedBox(height: 28),
              const Text('SYSTEM PROMPT',
                  style: TextStyle(
                    fontSize: 10, fontWeight: FontWeight.w700,
                    color: Tokens.textMuted, letterSpacing: 1.1)),
              const SizedBox(height: 10),
              TextField(
                controller: _promptCtrl,
                maxLines: 6,
                minLines: 4,
                style: const TextStyle(fontSize: 13, color: Tokens.textPrimary, height: 1.6),
                decoration: InputDecoration(
                  filled: true,
                  fillColor: Tokens.surface,
                  hintText: 'You are a helpful assistant…',
                  hintStyle: const TextStyle(color: Tokens.textMuted),
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
                  contentPadding: const EdgeInsets.all(14),
                ),
              ),
              const SizedBox(height: 10),
              Align(
                alignment: Alignment.centerRight,
                child: FilledButton.icon(
                  onPressed: _save,
                  icon: _saved
                      ? const Icon(Icons.check_rounded, size: 15)
                      : const SizedBox.shrink(),
                  label: Text(_saved ? 'Saved' : 'Save'),
                  style: _saved
                      ? FilledButton.styleFrom(
                          backgroundColor: Tokens.accentDim,
                          foregroundColor: Tokens.accent,
                        )
                      : null,
                ),
              ),
              const SizedBox(height: 32),
              const Divider(color: Tokens.glassEdge),
              const SizedBox(height: 32),
              _HfTokenSection(),
              const SizedBox(height: 32),
              const Divider(color: Tokens.glassEdge),
              const SizedBox(height: 32),
              _OpenAiSection(),
              const SizedBox(height: 32),
              const Divider(color: Tokens.glassEdge),
              const SizedBox(height: 32),
              const _McpServersSection(),
              const SizedBox(height: 32),
              const Divider(color: Tokens.glassEdge),
              const SizedBox(height: 32),
              const Text('CONNECTION',
                  style: TextStyle(
                    fontSize: 10, fontWeight: FontWeight.w700,
                    color: Tokens.textMuted, letterSpacing: 1.1)),
              const SizedBox(height: 16),
              Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('Host',
                            style: TextStyle(fontSize: 11, color: Tokens.textMuted)),
                        const SizedBox(height: 6),
                        TextField(
                          controller: TextEditingController(text: 'localhost'),
                          style: const TextStyle(fontSize: 13, color: Tokens.textPrimary),
                          decoration: InputDecoration(
                            filled: true,
                            fillColor: Tokens.surface,
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
                            contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 10),
                  SizedBox(
                    width: 90,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('Port',
                            style: TextStyle(fontSize: 11, color: Tokens.textMuted)),
                        const SizedBox(height: 6),
                        TextField(
                          controller: TextEditingController(text: '50051'),
                          style: const TextStyle(fontSize: 13, color: Tokens.textPrimary, fontFamily: 'JetBrains Mono'),
                          decoration: InputDecoration(
                            filled: true,
                            fillColor: Tokens.surface,
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
                            contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                          ),
                        ),
                      ],
                    ),
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

// ── HuggingFace token section (inside Settings) ───────────────────────────────

class _HfTokenSection extends StatelessWidget {
  const _HfTokenSection();

  static String _masked(String token) {
    if (token.length <= 8) return '••••••••';
    return '${token.substring(0, 4)}••••••••${token.substring(token.length - 4)}';
  }

  Future<void> _changeToken(BuildContext context) async {
    final state = context.read<AppState>();
    final token = await showDialog<String>(
      context: context,
      builder: (_) => const HfTokenDialog(),
    );
    if (token == null || token.isEmpty) return;
    await state.setGlobalHfToken(token);
  }

  Future<void> _removeToken(BuildContext context) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: Tokens.surface,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(Tokens.radiusCard),
          side: const BorderSide(color: Tokens.glassEdge),
        ),
        title: const Text('Remove token?',
            style: TextStyle(color: Tokens.textPrimary, fontSize: 15)),
        content: const Text(
            'Gated model downloads will no longer work until you add a token again.',
            style: TextStyle(color: Tokens.textSecondary, fontSize: 13)),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel',
                  style: TextStyle(color: Tokens.textSecondary))),
          TextButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Remove',
                  style: TextStyle(color: Tokens.destructive))),
        ],
      ),
    );
    if (confirmed == true && context.mounted) {
      await context.read<AppState>().clearGlobalHfToken();
    }
  }

  @override
  Widget build(BuildContext context) {
    final token = context.watch<AppState>().hfToken;
    final hasToken = token != null && token.isNotEmpty;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('HUGGINGFACE',
            style: TextStyle(
              fontSize: 10,
              fontWeight: FontWeight.w700,
              color: Tokens.textMuted,
              letterSpacing: 1.1,
            )),
        const SizedBox(height: 12),
        Row(
          children: [
            Icon(
              hasToken ? Icons.lock_open_rounded : Icons.lock_outline_rounded,
              size: 15,
              color: hasToken ? Tokens.accent : Tokens.textMuted,
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                hasToken ? _masked(token) : 'No token set',
                style: TextStyle(
                  fontSize: 13,
                  color: hasToken ? Tokens.textPrimary : Tokens.textMuted,
                  fontFamily: hasToken ? 'JetBrains Mono' : null,
                ),
              ),
            ),
            TextButton(
              onPressed: () => _changeToken(context),
              child: Text(hasToken ? 'Change' : 'Add Token',
                  style: const TextStyle(fontSize: 12, color: Tokens.accent)),
            ),
            if (hasToken) ...[
              const SizedBox(width: 4),
              TextButton(
                onPressed: () => _removeToken(context),
                child: const Text('Remove',
                    style: TextStyle(fontSize: 12, color: Tokens.destructive)),
              ),
            ],
          ],
        ),
        const SizedBox(height: 4),
        const Text(
          'Required for downloading gated models (Meta-Llama, official Mistral, etc.).',
          style: TextStyle(fontSize: 11, color: Tokens.textMuted),
        ),
      ],
    );
  }
}

// ── OpenAI-compatible server section ─────────────────────────────────────────

class _OpenAiSection extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final port = state.httpPort;
    final url = port > 0 ? 'http://localhost:$port/v1' : null;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('OPENAI-COMPATIBLE SERVER',
            style: TextStyle(
                fontSize: 10,
                fontWeight: FontWeight.w700,
                color: Tokens.textMuted,
                letterSpacing: 1.1)),
        const SizedBox(height: 12),
        if (url != null) ...[
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            decoration: BoxDecoration(
              color: Tokens.surface,
              borderRadius: BorderRadius.circular(Tokens.radiusInput),
              border: Border.all(color: Tokens.glassEdge),
            ),
            child: Row(
              children: [
                const Icon(Icons.circle, size: 8, color: Color(0xFF4CAF50)),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(url,
                      style: const TextStyle(
                          fontSize: 12,
                          fontFamily: 'JetBrainsMono',
                          color: Tokens.textPrimary)),
                ),
                IconButton(
                  onPressed: () {
                    final data = ClipboardData(text: url);
                    Clipboard.setData(data);
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                          content: Text('URL copied to clipboard'),
                          duration: Duration(seconds: 2)),
                    );
                  },
                  icon: const Icon(Icons.copy_rounded, size: 15),
                  color: Tokens.textSecondary,
                  visualDensity: VisualDensity.compact,
                  tooltip: 'Copy',
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'Compatible with Cursor, Continue, Open WebUI, and any OpenAI SDK client.',
            style: TextStyle(fontSize: 11, color: Tokens.textMuted),
          ),
        ] else ...[
          const Text(
            'Not running. Start neurons-service with --http-port 8080\nor use: neurons server',
            style: TextStyle(fontSize: 12, color: Tokens.textSecondary),
          ),
        ],
      ],
    );
  }
}

// ── MCP servers section (inside Settings) ────────────────────────────────────

class _McpServersSection extends StatelessWidget {
  const _McpServersSection();

  void _showAddDialog(BuildContext context) {
    showDialog<void>(
      context: context,
      builder: (ctx) => _AddMcpServerDialog(
        onSave: (server) => context.read<AppState>().addMcpServer(server),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final servers = context.watch<AppState>().mcpServers;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Text('MCP SERVERS',
                style: TextStyle(
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                  color: Tokens.textMuted,
                  letterSpacing: 1.1,
                )),
            const Spacer(),
            TextButton.icon(
              onPressed: () => _showAddDialog(context),
              icon: const Icon(Icons.add_rounded, size: 13),
              label: const Text('Add', style: TextStyle(fontSize: 12)),
              style: TextButton.styleFrom(
                foregroundColor: Tokens.accent,
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                minimumSize: Size.zero,
              ),
            ),
          ],
        ),
        const SizedBox(height: 10),
        if (servers.isEmpty)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            decoration: BoxDecoration(
              color: Tokens.surface,
              borderRadius: BorderRadius.circular(Tokens.radiusInput),
              border: Border.all(color: Tokens.glassEdge),
            ),
            child: const Text('No servers configured',
                style: TextStyle(fontSize: 12, color: Tokens.textMuted)),
          )
        else
          ...servers.map((s) => _McpServerRow(server: s)),
        const SizedBox(height: 8),
        const Text(
          'MCP servers give the model access to tools like filesystem, shell, and web search.',
          style: TextStyle(fontSize: 11, color: Tokens.textMuted),
        ),
      ],
    );
  }
}

class _McpServerRow extends StatelessWidget {
  const _McpServerRow({required this.server});
  final McpServerConfig server;

  @override
  Widget build(BuildContext context) {
    final isStdio = server.transport == 'stdio';

    return Container(
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
      decoration: BoxDecoration(
        color: Tokens.surface,
        borderRadius: BorderRadius.circular(Tokens.radiusInput),
        border: Border.all(color: Tokens.glassEdge),
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
                fontSize: 9,
                fontWeight: FontWeight.w700,
                color: Tokens.accent,
                letterSpacing: 0.5,
              ),
            ),
          ),
          const SizedBox(width: 9),
          Expanded(
            child: Text(
              server.name,
              style: const TextStyle(fontSize: 13, color: Tokens.textPrimary),
            ),
          ),
          if (!server.enabled)
            const Padding(
              padding: EdgeInsets.only(right: 8),
              child: Text('disabled',
                  style: TextStyle(fontSize: 11, color: Tokens.textMuted)),
            ),
          IconButton(
            onPressed: () =>
                context.read<AppState>().removeMcpServer(server.name),
            icon: const Icon(Icons.delete_outline_rounded, size: 15),
            color: Tokens.destructive,
            visualDensity: VisualDensity.compact,
            tooltip: 'Remove',
            padding: EdgeInsets.zero,
          ),
        ],
      ),
    );
  }
}

// ── Add MCP server dialog ─────────────────────────────────────────────────────

class _AddMcpServerDialog extends StatefulWidget {
  const _AddMcpServerDialog({required this.onSave});
  final Future<void> Function(McpServerConfig) onSave;

  @override
  State<_AddMcpServerDialog> createState() => _AddMcpServerDialogState();
}

class _AddMcpServerDialogState extends State<_AddMcpServerDialog> {
  final _nameCtrl    = TextEditingController();
  final _commandCtrl = TextEditingController();
  final _argsCtrl    = TextEditingController();
  final _urlCtrl     = TextEditingController();
  String _transport  = 'stdio';
  bool   _enabled    = true;
  bool   _saving     = false;
  String? _error;
  final List<(TextEditingController, TextEditingController)> _envPairs = [];

  @override
  void dispose() {
    _nameCtrl.dispose();
    _commandCtrl.dispose();
    _argsCtrl.dispose();
    _urlCtrl.dispose();
    for (final (k, v) in _envPairs) {
      k.dispose();
      v.dispose();
    }
    super.dispose();
  }

  Future<void> _save() async {
    final name = _nameCtrl.text.trim();
    if (name.isEmpty) {
      setState(() => _error = 'Name is required.');
      return;
    }
    if (_transport == 'stdio' && _commandCtrl.text.trim().isEmpty) {
      setState(() => _error = 'Command is required for stdio transport.');
      return;
    }
    if (_transport == 'sse' && _urlCtrl.text.trim().isEmpty) {
      setState(() => _error = 'URL is required for SSE transport.');
      return;
    }
    final env = <String, String>{
      for (final (k, v) in _envPairs)
        if (k.text.trim().isNotEmpty) k.text.trim(): v.text,
    };
    final args = _argsCtrl.text.trim().isEmpty
        ? <String>[]
        : _argsCtrl.text.trim().split(RegExp(r'\s+')).toList();
    final server = McpServerConfig(
      name: name,
      transport: _transport,
      command: _transport == 'stdio' ? _commandCtrl.text.trim() : '',
      args: _transport == 'stdio' ? args : [],
      url: _transport == 'sse' ? _urlCtrl.text.trim() : '',
      env: env.entries,
      enabled: _enabled,
    );
    setState(() { _saving = true; _error = null; });
    try {
      await widget.onSave(server);
      if (mounted) Navigator.pop(context);
    } catch (e) {
      if (mounted) setState(() { _saving = false; _error = e.toString(); });
    }
  }

  void _addEnvPair() =>
      setState(() => _envPairs.add((TextEditingController(), TextEditingController())));

  void _removeEnvPair(int i) {
    final (k, v) = _envPairs.removeAt(i);
    k.dispose();
    v.dispose();
    setState(() {});
  }

  InputDecoration _dec(String hint) => InputDecoration(
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
    return Dialog(
      backgroundColor: Tokens.surface,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: const BorderSide(color: Tokens.glassEdge),
      ),
      child: SizedBox(
        width: 420,
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Add MCP Server',
                  style: TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                    color: Tokens.textPrimary,
                  )),
              const SizedBox(height: 20),

              // Name
              const Text('NAME',
                  style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700,
                      color: Tokens.textMuted, letterSpacing: 0.7)),
              const SizedBox(height: 6),
              TextField(
                controller: _nameCtrl,
                style: const TextStyle(fontSize: 13, color: Tokens.textPrimary),
                decoration: _dec('e.g. filesystem'),
              ),
              const SizedBox(height: 16),

              // Transport
              const Text('TRANSPORT',
                  style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700,
                      color: Tokens.textMuted, letterSpacing: 0.7)),
              const SizedBox(height: 8),
              SegmentedButton<String>(
                segments: const [
                  ButtonSegment(value: 'stdio', label: Text('stdio')),
                  ButtonSegment(value: 'sse',   label: Text('SSE')),
                ],
                selected: {_transport},
                onSelectionChanged: (s) =>
                    setState(() => _transport = s.first),
                style: ButtonStyle(
                  textStyle: WidgetStateProperty.all(
                      const TextStyle(fontSize: 12)),
                ),
              ),
              const SizedBox(height: 16),

              // stdio fields
              if (_transport == 'stdio') ...[
                const Text('COMMAND',
                    style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700,
                        color: Tokens.textMuted, letterSpacing: 0.7)),
                const SizedBox(height: 6),
                TextField(
                  controller: _commandCtrl,
                  style: const TextStyle(fontSize: 13, color: Tokens.textPrimary,
                      fontFamily: 'JetBrains Mono'),
                  decoration: _dec('npx'),
                ),
                const SizedBox(height: 12),
                const Text('ARGS',
                    style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700,
                        color: Tokens.textMuted, letterSpacing: 0.7)),
                const SizedBox(height: 6),
                TextField(
                  controller: _argsCtrl,
                  style: const TextStyle(fontSize: 13, color: Tokens.textPrimary,
                      fontFamily: 'JetBrains Mono'),
                  decoration: _dec('-y @modelcontextprotocol/server-filesystem /'),
                ),
                const SizedBox(height: 4),
                const Text('Space-separated arguments.',
                    style: TextStyle(fontSize: 11, color: Tokens.textMuted)),
                const SizedBox(height: 4),
              ],

              // sse field
              if (_transport == 'sse') ...[
                const Text('URL',
                    style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700,
                        color: Tokens.textMuted, letterSpacing: 0.7)),
                const SizedBox(height: 6),
                TextField(
                  controller: _urlCtrl,
                  style: const TextStyle(fontSize: 13, color: Tokens.textPrimary,
                      fontFamily: 'JetBrains Mono'),
                  decoration: _dec('http://localhost:3001/sse'),
                ),
                const SizedBox(height: 4),
              ],

              // Env vars
              const SizedBox(height: 12),
              Row(
                children: [
                  const Text('ENV VARS',
                      style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700,
                          color: Tokens.textMuted, letterSpacing: 0.7)),
                  const Spacer(),
                  TextButton.icon(
                    onPressed: _addEnvPair,
                    icon: const Icon(Icons.add_rounded, size: 13),
                    label: const Text('Add', style: TextStyle(fontSize: 11)),
                    style: TextButton.styleFrom(
                      foregroundColor: Tokens.accent,
                      padding: const EdgeInsets.symmetric(
                          horizontal: 6, vertical: 2),
                      minimumSize: Size.zero,
                    ),
                  ),
                ],
              ),
              ..._envPairs.indexed.map(
                  ((int, (TextEditingController, TextEditingController)) e) {
                final i = e.$1;
                final (k, v) = e.$2;
                return Padding(
                  padding: const EdgeInsets.only(top: 6),
                  child: Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: k,
                          style: const TextStyle(fontSize: 12,
                              color: Tokens.textPrimary,
                              fontFamily: 'JetBrains Mono'),
                          decoration: _dec('KEY'),
                        ),
                      ),
                      const SizedBox(width: 6),
                      Expanded(
                        flex: 2,
                        child: TextField(
                          controller: v,
                          style: const TextStyle(fontSize: 12,
                              color: Tokens.textPrimary,
                              fontFamily: 'JetBrains Mono'),
                          decoration: _dec('value'),
                        ),
                      ),
                      IconButton(
                        onPressed: () => _removeEnvPair(i),
                        icon: const Icon(Icons.close_rounded, size: 14),
                        color: Tokens.textMuted,
                        visualDensity: VisualDensity.compact,
                        padding: const EdgeInsets.only(left: 6),
                      ),
                    ],
                  ),
                );
              }),

              const SizedBox(height: 16),
              // Enabled toggle
              Row(
                children: [
                  const Text('ENABLED',
                      style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700,
                          color: Tokens.textMuted, letterSpacing: 0.7)),
                  const Spacer(),
                  Switch(
                    value: _enabled,
                    onChanged: (v) => setState(() => _enabled = v),
                    activeColor: Tokens.accent,
                    materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  ),
                ],
              ),

              if (_error != null) ...[
                const SizedBox(height: 10),
                Text(_error!,
                    style: const TextStyle(
                        fontSize: 12, color: Tokens.destructive)),
              ],
              const SizedBox(height: 24),

              Row(
                children: [
                  OutlinedButton(
                    onPressed: _saving ? null : () => Navigator.pop(context),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: Tokens.textSecondary,
                      side: const BorderSide(color: Tokens.glassEdge),
                    ),
                    child: const Text('Cancel'),
                  ),
                  const Spacer(),
                  FilledButton(
                    onPressed: _saving ? null : _save,
                    child: _saving
                        ? const SizedBox(
                            width: 14,
                            height: 14,
                            child: CircularProgressIndicator(
                                strokeWidth: 2, color: Colors.white))
                        : const Text('Add Server'),
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
