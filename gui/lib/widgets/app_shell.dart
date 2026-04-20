import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import '../services/app_state.dart';
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
