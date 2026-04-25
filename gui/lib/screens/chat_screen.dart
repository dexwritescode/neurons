import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

// ignore_for_file: use_build_context_synchronously

import '../services/app_state.dart';
import '../proto/neurons.pb.dart' show McpServerConfig, ToolApprovalRequest;
import '../theme/tokens.dart';
import '../widgets/blinking_cursor.dart';
import '../widgets/resize_divider.dart';
import '../widgets/token_stats_bar.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _inputCtrl = TextEditingController();
  final _scrollCtrl = ScrollController();

  double _sidebarWidth = 224;
  DateTime? _genStartTime;
  bool _prevGenerating = false;

  static const _sidebarMin = 160.0;
  static const _sidebarMax = 380.0;

  @override
  void dispose() {
    _inputCtrl.dispose();
    _scrollCtrl.dispose();
    super.dispose();
  }

  void _onSidebarDrag(double dx) {
    setState(() {
      _sidebarWidth = (_sidebarWidth + dx).clamp(_sidebarMin, _sidebarMax);
    });
  }

  void _send(AppState state) {
    final text = _inputCtrl.text.trim();
    if (text.isEmpty || state.isGenerating) return;
    _inputCtrl.clear();
    state.send(text);
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollCtrl.hasClients) {
        _scrollCtrl.animateTo(
          _scrollCtrl.position.maxScrollExtent,
          duration: const Duration(milliseconds: 150),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();

    if (state.isGenerating && !_prevGenerating) {
      _genStartTime = null; // reset; will be set on first token
    }
    if (state.isGenerating && state.liveGenTokens == 1 && _genStartTime == null) {
      _genStartTime = DateTime.now(); // start timer on first token, not on prefill start
    }
    _prevGenerating = state.isGenerating;

    if (state.isGenerating) _scrollToBottom();

    return Row(
      children: [
        // ── Session sidebar ──────────────────────────────────────────────────
        SizedBox(
          width: _sidebarWidth,
          child: _SessionSidebar(
            sessions: state.sessions,
            activeId: state.activeSessionId,
            onNewChat: () => state.newChat(),
            onSelect: state.switchSession,
            onDelete: state.deleteSession,
            onRename: state.renameSession,
          ),
        ),
        ResizeDivider(onDrag: _onSidebarDrag),
        // ── Main chat area ───────────────────────────────────────────────────
        Expanded(
          child: Column(
            children: [
              // Error banner
              if (state.generationError != null)
                MaterialBanner(
                  backgroundColor: Tokens.surfaceElevated,
                  dividerColor: Tokens.glassEdge,
                  content: Text(
                    state.generationError!,
                    style: const TextStyle(
                        color: Tokens.destructive, fontSize: 13),
                  ),
                  actions: [
                    TextButton(
                      onPressed: state.clearChat,
                      child: const Text('Dismiss',
                          style: TextStyle(color: Tokens.textSecondary)),
                    ),
                  ],
                ),
              // Message list
              Expanded(
                child: state.messages.isEmpty
                    ? _emptyState(state)
                    : ListView.builder(
                        controller: _scrollCtrl,
                        padding: const EdgeInsets.symmetric(vertical: 8),
                        itemCount: state.messages.length,
                        itemBuilder: (_, i) {
                          final msg = state.messages[i];
                          final isLast = i == state.messages.length - 1;
                          return _MessageRow(
                            message: msg,
                            index: i,
                            isLastMessage: isLast,
                            isGenerating: state.isGenerating,
                            state: state,
                          );
                        },
                      ),
              ),
              // Token stats bar
              if (state.isGenerating || state.lastGenTokens > 0)
                TokenStatsBar(
                  promptTokens: state.lastPromptTokens,
                  genTokens: state.isGenerating
                      ? state.liveGenTokens
                      : state.lastGenTokens,
                  isGenerating: state.isGenerating,
                  genStartTime: _genStartTime,
                ),
              // Approval bar (slides in when a tool call needs approval)
              AnimatedSwitcher(
                duration: Tokens.fast,
                child: state.pendingApproval != null
                    ? _ApprovalBar(
                        key: ValueKey(state.pendingApproval!.approvalId),
                        approval: state.pendingApproval!,
                        onRespond: (approved, newPermission) => state
                            .respondApproval(state.pendingApproval!.approvalId,
                                approved,
                                newPermission: newPermission),
                      )
                    : const SizedBox.shrink(),
              ),
              // Input bar
              _InputBar(
                controller: _inputCtrl,
                isGenerating: state.isGenerating,
                modelLoaded: state.modelLoaded,
                onSend: () => _send(state),
                onStop: state.cancelGeneration,
                supportsToolUse: state.supportsToolUse,
                mcpServers: state.mcpServers,
                activeServerNames: state.activeServerNames,
                onToggleServer: state.toggleActiveServer,
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _emptyState(AppState state) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: Tokens.accentDim,
              shape: BoxShape.circle,
              border: Border.all(color: Tokens.accent.withAlpha(56)),
            ),
            child: const Icon(Icons.chat_bubble_outline_rounded,
                size: 20, color: Tokens.accent),
          ),
          const SizedBox(height: 14),
          Text(
            state.modelLoaded ? 'Start a conversation' : 'Load a model to start chatting',
            style: const TextStyle(fontSize: 14, color: Tokens.textMuted),
          ),
          if (state.modelType != null) ...[
            const SizedBox(height: 4),
            Text(
              '${state.modelType} · local',
              style: const TextStyle(
                fontSize: 12,
                color: Tokens.textMuted,
                fontFamily: 'JetBrains Mono',
              ),
            ),
          ],
        ],
      ),
    );
  }
}

// ── Session sidebar ───────────────────────────────────────────────────────────

class _SessionSidebar extends StatelessWidget {
  const _SessionSidebar({
    required this.sessions,
    required this.activeId,
    required this.onNewChat,
    required this.onSelect,
    required this.onDelete,
    required this.onRename,
  });

  final List<ChatSession> sessions;
  final String activeId;
  final VoidCallback onNewChat;
  final void Function(String) onSelect;
  final void Function(String) onDelete;
  final void Function(String, String) onRename;

  @override
  Widget build(BuildContext context) {
    return Container(
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
                const Text(
                  'CHATS',
                  style: TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                    color: Tokens.textMuted,
                    letterSpacing: 0.8,
                  ),
                ),
                const Spacer(),
                GestureDetector(
                  onTap: onNewChat,
                  child: const Tooltip(
                    message: 'New chat',
                    child: Icon(Icons.edit_outlined,
                        size: 16, color: Tokens.textSecondary),
                  ),
                ),
              ],
            ),
          ),
          // Session list
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.symmetric(vertical: 4),
              itemCount: sessions.length,
              itemBuilder: (_, i) {
                final s = sessions[i];
                return _SessionTile(
                  session: s,
                  isActive: s.id == activeId,
                  onTap: () => onSelect(s.id),
                  onDelete: () => onDelete(s.id),
                  onRename: (title) => onRename(s.id, title),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

class _SessionTile extends StatefulWidget {
  const _SessionTile({
    required this.session,
    required this.isActive,
    required this.onTap,
    required this.onDelete,
    required this.onRename,
  });

  final ChatSession session;
  final bool isActive;
  final VoidCallback onTap;
  final VoidCallback onDelete;
  final void Function(String) onRename;

  @override
  State<_SessionTile> createState() => _SessionTileState();
}

class _SessionTileState extends State<_SessionTile> {
  bool _hovered = false;
  bool _renaming = false;
  late final TextEditingController _renameCtrl;

  @override
  void initState() {
    super.initState();
    _renameCtrl = TextEditingController(text: widget.session.title);
  }

  @override
  void dispose() {
    _renameCtrl.dispose();
    super.dispose();
  }

  void _commitRename() {
    final v = _renameCtrl.text.trim();
    if (v.isNotEmpty) widget.onRename(v);
    setState(() => _renaming = false);
  }

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: GestureDetector(
        onTap: () { if (!_renaming) widget.onTap(); },
        onSecondaryTapUp: (d) => _showContextMenu(context, d.globalPosition),
        child: AnimatedContainer(
          duration: Tokens.fast,
          margin: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
          decoration: BoxDecoration(
            color: widget.isActive
                ? Tokens.accentDim
                : _hovered
                    ? Tokens.surfaceElevated
                    : Colors.transparent,
            borderRadius: BorderRadius.circular(Tokens.radiusCard),
            border: widget.isActive
                ? Border.all(color: Tokens.accent.withAlpha(40))
                : null,
          ),
          child: Row(
            children: [
              Expanded(
                child: _renaming
                    ? TextField(
                        controller: _renameCtrl,
                        autofocus: true,
                        style: const TextStyle(
                            fontSize: 13, color: Tokens.textPrimary),
                        decoration: const InputDecoration.collapsed(hintText: ''),
                        onSubmitted: (_) => _commitRename(),
                        onEditingComplete: _commitRename,
                      )
                    : Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            widget.session.title,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: TextStyle(
                              fontSize: 13,
                              color: widget.isActive
                                  ? Tokens.textPrimary
                                  : Tokens.textSecondary,
                              fontWeight: widget.isActive
                                  ? FontWeight.w500
                                  : FontWeight.normal,
                            ),
                          ),
                          const SizedBox(height: 2),
                          Text(
                            _relativeTime(widget.session.createdAt),
                            style: const TextStyle(
                                fontSize: 11, color: Tokens.textMuted),
                          ),
                        ],
                      ),
              ),
              if (_hovered && !_renaming) ...[
                const SizedBox(width: 4),
                _TileIconBtn(
                  icon: Icons.edit_outlined,
                  tooltip: 'Rename',
                  onTap: () => setState(() {
                    _renaming = true;
                    _renameCtrl.text = widget.session.title;
                  }),
                ),
                _TileIconBtn(
                  icon: Icons.close_rounded,
                  tooltip: 'Delete',
                  onTap: widget.onDelete,
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  void _showContextMenu(BuildContext context, Offset pos) async {
    final result = await showMenu<String>(
      context: context,
      position: RelativeRect.fromLTRB(pos.dx, pos.dy, pos.dx, pos.dy),
      color: Tokens.surfaceElevated,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      items: const [
        PopupMenuItem(value: 'rename',
            child: Text('Rename',
                style: TextStyle(fontSize: 13, color: Tokens.textPrimary))),
        PopupMenuItem(value: 'delete',
            child: Text('Delete',
                style: TextStyle(fontSize: 13, color: Tokens.textPrimary))),
      ],
    );
    if (!mounted) return;
    if (result == 'rename') {
      setState(() { _renaming = true; _renameCtrl.text = widget.session.title; });
    }
    if (result == 'delete') widget.onDelete();
  }
}

class _TileIconBtn extends StatelessWidget {
  const _TileIconBtn({required this.icon, required this.tooltip, required this.onTap});
  final IconData icon;
  final String tooltip;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Tooltip(
        message: tooltip,
        child: Padding(
          padding: const EdgeInsets.all(2),
          child: Icon(icon, size: 13, color: Tokens.textMuted),
        ),
      ),
    );
  }
}

String _relativeTime(DateTime dt) {
  final diff = DateTime.now().difference(dt);
  if (diff.inSeconds < 60) return 'Just now';
  if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
  if (diff.inHours < 24) return '${diff.inHours}h ago';
  if (diff.inDays == 1) return 'Yesterday';
  if (diff.inDays < 7) return '${diff.inDays}d ago';
  return '${dt.month}/${dt.day}/${dt.year}';
}

// ── Message row ───────────────────────────────────────────────────────────────

class _MessageRow extends StatefulWidget {
  const _MessageRow({
    required this.message,
    required this.index,
    required this.isLastMessage,
    required this.isGenerating,
    required this.state,
  });

  final ConversationMessage message;
  final int index;
  final bool isLastMessage;
  final bool isGenerating;
  final AppState state;

  @override
  State<_MessageRow> createState() => _MessageRowState();
}

class _MessageRowState extends State<_MessageRow> {
  bool _hovered = false;
  bool _thinkExpanded = true;  // starts expanded; auto-collapses when generation ends

  @override
  void didUpdateWidget(covariant _MessageRow oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.isGenerating && !widget.isGenerating && _thinkExpanded) {
      setState(() => _thinkExpanded = false);
    }
  }

  // Splits content into (thinking, answer) around the first <think>...</think> block.
  // Returns empty thinking string when no think block is present.
  static ({String thinking, String answer}) _parseThinking(String content) {
    const open = '<think>';
    const close = '</think>';
    final openIdx = content.indexOf(open);
    if (openIdx == -1) return (thinking: '', answer: content);
    final closeIdx = content.indexOf(close, openIdx);
    if (closeIdx == -1) {
      // Still generating inside the think block
      return (thinking: content.substring(openIdx + open.length), answer: '');
    }
    return (
      thinking: content.substring(openIdx + open.length, closeIdx).trim(),
      answer: content.substring(closeIdx + close.length).trimLeft(),
    );
  }

  void _copyMessage() =>
      Clipboard.setData(ClipboardData(text: widget.message.content));

  void _showContextMenu(BuildContext context, Offset pos) {
    if (widget.isGenerating) return;
    showMenu<String>(
      context: context,
      position: RelativeRect.fromLTRB(pos.dx, pos.dy, pos.dx + 1, pos.dy + 1),
      items: const [
        PopupMenuItem(value: 'copy',
            child: Row(children: [Icon(Icons.copy_rounded, size: 14), SizedBox(width: 8), Text('Copy')])),
        PopupMenuItem(value: 'delete',
            child: Row(children: [Icon(Icons.delete_outline_rounded, size: 14), SizedBox(width: 8), Text('Delete pair')])),
        PopupMenuItem(value: 'truncate',
            child: Row(children: [Icon(Icons.content_cut_rounded, size: 14), SizedBox(width: 8), Text('Truncate here')])),
      ],
    ).then((value) {
      if (!mounted) return;
      switch (value) {
        case 'copy':     _copyMessage();
        case 'delete':   widget.state.deleteMessagePair(widget.index);
        case 'truncate': widget.state.truncateFrom(widget.index);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final isUser = widget.message.role == 'user';
    final showActions = _hovered && !widget.isGenerating;

    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: GestureDetector(
        onSecondaryTapUp: (d) => _showContextMenu(context, d.globalPosition),
        child: isUser
            ? _buildUser(showActions)
            : _buildAssistant(showActions),
      ),
    );
  }

  Widget _buildUser(bool showActions) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              AnimatedOpacity(
                duration: Tokens.fast,
                opacity: showActions ? 1.0 : 0.0,
                child: Row(children: [
                  _MsgIconBtn(icon: Icons.copy_rounded, tooltip: 'Copy', onTap: _copyMessage),
                  _MsgIconBtn(icon: Icons.delete_outline_rounded, tooltip: 'Delete pair',
                      onTap: () => widget.state.deleteMessagePair(widget.index)),
                  _MsgIconBtn(icon: Icons.content_cut_rounded, tooltip: 'Truncate here',
                      onTap: () => widget.state.truncateFrom(widget.index)),
                  const SizedBox(width: 6),
                ]),
              ),
              const Text('You',
                  style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: Tokens.accent)),
            ],
          ),
          const SizedBox(height: 5),
          SelectableText(
            widget.message.content,
            textAlign: TextAlign.right,
            style: const TextStyle(
                fontSize: 14, color: Tokens.textPrimary, height: 1.65),
          ),
        ],
      ),
    );
  }

  Widget _buildAssistant(bool showActions) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 10),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 22,
            height: 22,
            decoration: BoxDecoration(
              color: Tokens.accentDim,
              borderRadius: BorderRadius.circular(5),
              border: Border.all(color: Tokens.accent.withAlpha(56)),
            ),
            child: const Icon(Icons.memory_outlined, size: 12, color: Tokens.accent),
          ),
          const SizedBox(width: 13),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(
                      widget.state.modelType ?? 'Assistant',
                      style: const TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                          color: Tokens.textSecondary),
                    ),
                    const Spacer(),
                    AnimatedOpacity(
                      duration: Tokens.fast,
                      opacity: showActions ? 1.0 : 0.0,
                      child: Row(children: [
                        _MsgIconBtn(icon: Icons.copy_rounded, tooltip: 'Copy', onTap: _copyMessage),
                        _MsgIconBtn(icon: Icons.delete_outline_rounded, tooltip: 'Delete pair',
                            onTap: () => widget.state.deleteMessagePair(widget.index)),
                        _MsgIconBtn(icon: Icons.content_cut_rounded, tooltip: 'Truncate here',
                            onTap: () => widget.state.truncateFrom(widget.index)),
                      ]),
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                Builder(builder: (context) {
                  final parsed = _parseThinking(widget.message.content);
                  final hasThink = parsed.thinking.isNotEmpty;

                  return Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      if (hasThink) ...[
                        GestureDetector(
                          onTap: () => setState(() => _thinkExpanded = !_thinkExpanded),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(Icons.psychology_outlined,
                                  size: 13, color: Tokens.textSecondary),
                              const SizedBox(width: 4),
                              Text('Thinking',
                                  style: const TextStyle(
                                      fontSize: 11,
                                      color: Tokens.textSecondary,
                                      fontWeight: FontWeight.w500)),
                              const SizedBox(width: 3),
                              Icon(
                                _thinkExpanded
                                    ? Icons.keyboard_arrow_up_rounded
                                    : Icons.keyboard_arrow_down_rounded,
                                size: 15,
                                color: Tokens.textSecondary,
                              ),
                            ],
                          ),
                        ),
                        if (_thinkExpanded) ...[
                          const SizedBox(height: 8),
                          SelectableText(
                            parsed.thinking,
                            style: const TextStyle(
                                fontSize: 13,
                                color: Tokens.textMuted,
                                height: 1.65),
                          ),
                          const SizedBox(height: 10),
                        ] else
                          const SizedBox(height: 6),
                      ],
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.end,
                        children: [
                          Flexible(
                            child: SelectableText(
                              widget.message.toolCalls.isNotEmpty
                                  ? widget.message.content
                                  : (parsed.answer.isEmpty ? '…' : parsed.answer),
                              style: const TextStyle(
                                  fontSize: 14,
                                  color: Tokens.textPrimary,
                                  height: 1.72),
                            ),
                          ),
                          if (widget.isLastMessage &&
                              widget.isGenerating &&
                              widget.message.toolCalls.isEmpty) ...[
                            const SizedBox(width: 2),
                            const BlinkingCursor(),
                          ],
                        ],
                      ),
                    ],
                  );
                }),
                if (widget.message.toolCalls.isNotEmpty) ...[
                  ...() {
                    final pending = widget.state.pendingApproval;
                    return widget.message.toolCalls.map((r) {
                      final isWaiting = pending != null &&
                          r.resultJson == null &&
                          r.server == pending.server &&
                          r.tool == pending.tool;
                      return _ToolCallBlock(
                          record: r, isWaitingApproval: isWaiting);
                    });
                  }(),
                  if (widget.isLastMessage && widget.isGenerating)
                    const Padding(
                      padding: EdgeInsets.only(top: 6),
                      child: BlinkingCursor(),
                    ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _MsgIconBtn extends StatelessWidget {
  const _MsgIconBtn({required this.icon, required this.tooltip, required this.onTap});
  final IconData icon;
  final String tooltip;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Tooltip(
        message: tooltip,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 3, vertical: 2),
          child: Icon(icon, size: 13, color: Tokens.textMuted),
        ),
      ),
    );
  }
}

// ── Input bar ─────────────────────────────────────────────────────────────────

class _InputBar extends StatelessWidget {
  const _InputBar({
    required this.controller,
    required this.isGenerating,
    required this.modelLoaded,
    required this.onSend,
    required this.onStop,
    this.supportsToolUse = false,
    this.mcpServers = const [],
    this.activeServerNames = const {},
    this.onToggleServer,
  });

  final TextEditingController controller;
  final bool isGenerating;
  final bool modelLoaded;
  final VoidCallback onSend;
  final VoidCallback onStop;
  final bool supportsToolUse;
  final List<McpServerConfig> mcpServers;
  final Set<String> activeServerNames;
  final void Function(String)? onToggleServer;

  void _showToolsSheet(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Tokens.surfaceElevated,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(14)),
      ),
      builder: (_) => Consumer<AppState>(
        builder: (ctx, state, _) => _ToolsSheet(
          mcpServers: state.mcpServers,
          activeServerNames: state.activeServerNames,
          onToggle: state.toggleActiveServer,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final showToolsButton = supportsToolUse && mcpServers.isNotEmpty;
    final activeCount = activeServerNames.intersection(
        mcpServers.map((s) => s.name).toSet()).length;

    return Container(
      decoration: const BoxDecoration(
        color: Tokens.surface,
        border: Border(top: BorderSide(color: Tokens.glassEdge)),
      ),
      padding: const EdgeInsets.fromLTRB(14, 10, 14, 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          if (showToolsButton) ...[
            GestureDetector(
              onTap: () => _showToolsSheet(context),
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: activeCount > 0 ? Tokens.accentDim : Tokens.surfaceElevated,
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(
                    color: activeCount > 0
                        ? Tokens.accent.withAlpha(80)
                        : Tokens.glassEdge,
                  ),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.build_outlined, size: 11,
                        color: activeCount > 0 ? Tokens.accent : Tokens.textMuted),
                    const SizedBox(width: 5),
                    Text(
                      activeCount > 0 ? '$activeCount tool${activeCount == 1 ? '' : 's'} active' : 'Tools',
                      style: TextStyle(
                        fontSize: 11,
                        color: activeCount > 0 ? Tokens.accent : Tokens.textMuted,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 8),
          ],
          Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Expanded(
            child: Focus(
              onKeyEvent: (_, event) {
                if (event is KeyDownEvent &&
                    event.logicalKey == LogicalKeyboardKey.enter &&
                    !HardwareKeyboard.instance.isShiftPressed) {
                  if (controller.text.trim().isNotEmpty && !isGenerating) {
                    onSend();
                    return KeyEventResult.handled;
                  }
                }
                return KeyEventResult.ignored;
              },
              child: TextField(
                controller: controller,
                maxLines: null,
                minLines: 1,
                enabled: modelLoaded && !isGenerating,
                textInputAction: TextInputAction.newline,
                style: const TextStyle(
                    fontSize: 14, color: Tokens.textPrimary, height: 1.55),
                decoration: InputDecoration(
                  hintText: modelLoaded ? 'Message…' : 'Load a model to start chatting',
                  hintStyle:
                      const TextStyle(color: Tokens.textMuted, fontSize: 14),
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
                      const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
                ),
              ),
            ),
          ),
          const SizedBox(width: 8),
          ListenableBuilder(
            listenable: controller,
            builder: (_, __) {
              final hasText = controller.text.trim().isNotEmpty;
              return AnimatedSwitcher(
                duration: Tokens.fast,
                child: isGenerating
                    ? GestureDetector(
                        key: const ValueKey('stop'),
                        onTap: onStop,
                        child: Container(
                          width: 38,
                          height: 38,
                          decoration: BoxDecoration(
                            color: Tokens.surfaceElevated,
                            borderRadius:
                                BorderRadius.circular(Tokens.radiusInput),
                            border: Border.all(
                                color: Tokens.destructive.withAlpha(100)),
                          ),
                          child: const Icon(Icons.stop_rounded,
                              size: 16, color: Tokens.destructive),
                        ),
                      )
                    : GestureDetector(
                        key: const ValueKey('send'),
                        onTap: hasText ? onSend : null,
                        child: Container(
                          width: 38,
                          height: 38,
                          decoration: BoxDecoration(
                            color: hasText
                                ? Tokens.accent
                                : Tokens.surfaceElevated,
                            borderRadius:
                                BorderRadius.circular(Tokens.radiusInput),
                          ),
                          child: Icon(Icons.arrow_upward_rounded,
                              size: 16,
                              color: hasText
                                  ? Tokens.background
                                  : Tokens.textMuted),
                        ),
                      ),
              );
            },
          ),
        ],
          ),
        ],
      ),
    );
  }
}

// ── Tool approval bar ─────────────────────────────────────────────────────────

class _ApprovalBar extends StatelessWidget {
  const _ApprovalBar({
    super.key,
    required this.approval,
    required this.onRespond,
  });
  final ToolApprovalRequest approval;
  final void Function(bool approved, String newPermission) onRespond;

  String get _argsPreview {
    final raw = approval.argsJson;
    if (raw.isEmpty) return '';
    return raw.length > 80 ? '${raw.substring(0, 80)}…' : raw;
  }

  @override
  Widget build(BuildContext context) {
    final isDestructive = approval.destructive;
    final barColor = isDestructive
        ? Tokens.destructive.withAlpha(18)
        : Tokens.accentDim;
    final borderColor = isDestructive
        ? Tokens.destructive.withAlpha(80)
        : Tokens.accent.withAlpha(60);

    return Container(
      margin: const EdgeInsets.fromLTRB(12, 0, 12, 8),
      padding: const EdgeInsets.fromLTRB(14, 10, 10, 10),
      decoration: BoxDecoration(
        color: barColor,
        borderRadius: BorderRadius.circular(Tokens.radiusInput),
        border: Border.all(color: borderColor),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Tool identity row
          Row(
            children: [
              Icon(
                Icons.build_circle_outlined,
                size: 13,
                color: isDestructive ? Tokens.destructive : Tokens.accent,
              ),
              const SizedBox(width: 6),
              Expanded(
                child: RichText(
                  text: TextSpan(
                    style: const TextStyle(fontSize: 12, color: Tokens.textSecondary),
                    children: [
                      TextSpan(
                        text: approval.server,
                        style: TextStyle(
                          fontWeight: FontWeight.w600,
                          color: isDestructive ? Tokens.destructive : Tokens.accent,
                        ),
                      ),
                      const TextSpan(text: ' · '),
                      TextSpan(
                        text: approval.tool,
                        style: const TextStyle(
                          color: Tokens.textPrimary,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              if (isDestructive)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
                  decoration: BoxDecoration(
                    color: Tokens.destructive.withAlpha(30),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: const Text('DESTRUCTIVE',
                      style: TextStyle(
                          fontSize: 9,
                          fontWeight: FontWeight.w700,
                          color: Tokens.destructive,
                          letterSpacing: 0.4)),
                ),
            ],
          ),
          if (_argsPreview.isNotEmpty) ...[
            const SizedBox(height: 5),
            Text(
              _argsPreview,
              style: const TextStyle(
                  fontSize: 11,
                  color: Tokens.textMuted,
                  fontFamily: 'JetBrains Mono'),
            ),
          ],
          const SizedBox(height: 10),
          // Action buttons
          Row(
            children: [
              // Approve (primary unless destructive)
              _ApproveButton(
                label: 'Approve',
                primary: !isDestructive,
                destructive: isDestructive,
                onTap: () => onRespond(true, ''),
              ),
              const SizedBox(width: 6),
              _ApproveButton(
                label: 'Approve for session',
                onTap: () => onRespond(true, 'allow_session'),
              ),
              const SizedBox(width: 6),
              _ApproveButton(
                label: 'Always approve',
                onTap: () => onRespond(true, 'always_allow'),
              ),
              const Spacer(),
              _ApproveButton(
                label: 'Deny',
                destructive: true,
                onTap: () => onRespond(false, ''),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _ApproveButton extends StatelessWidget {
  const _ApproveButton({
    required this.label,
    required this.onTap,
    this.primary = false,
    this.destructive = false,
  });
  final String label;
  final VoidCallback onTap;
  final bool primary;
  final bool destructive;

  @override
  Widget build(BuildContext context) {
    final fg = destructive
        ? Tokens.destructive
        : primary
            ? Tokens.background
            : Tokens.textSecondary;
    final bg = destructive
        ? Tokens.destructive.withAlpha(20)
        : primary
            ? Tokens.accent
            : Tokens.surface;
    final border = destructive
        ? Tokens.destructive.withAlpha(80)
        : primary
            ? Colors.transparent
            : Tokens.glassEdge;

    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: bg,
          borderRadius: BorderRadius.circular(6),
          border: Border.all(color: border),
        ),
        child: Text(label,
            style: TextStyle(
              fontSize: 12,
              fontWeight: primary ? FontWeight.w600 : FontWeight.w400,
              color: fg,
            )),
      ),
    );
  }
}

// ── Tools sheet ───────────────────────────────────────────────────────────────

class _ToolsSheet extends StatelessWidget {
  const _ToolsSheet({
    required this.mcpServers,
    required this.activeServerNames,
    required this.onToggle,
  });
  final List<McpServerConfig> mcpServers;
  final Set<String> activeServerNames;
  final void Function(String) onToggle;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(18, 16, 18, 8),
            child: Row(
              children: [
                const Icon(Icons.build_outlined, size: 14, color: Tokens.textMuted),
                const SizedBox(width: 8),
                const Text(
                  'TOOLS FOR THIS CHAT',
                  style: TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                    color: Tokens.textMuted,
                    letterSpacing: 0.8,
                  ),
                ),
              ],
            ),
          ),
          const Divider(height: 1, color: Tokens.glassEdge),
          ...mcpServers.map((server) {
            final active = activeServerNames.contains(server.name);
            return SwitchListTile(
              value: active,
              onChanged: server.enabled ? (v) => onToggle(server.name) : null,
              title: Text(
                server.name,
                style: TextStyle(
                  fontSize: 13,
                  color: server.enabled ? Tokens.textPrimary : Tokens.textMuted,
                ),
              ),
              subtitle: Text(
                server.transport.toUpperCase(),
                style: const TextStyle(fontSize: 11, color: Tokens.textMuted),
              ),
              activeColor: Tokens.accent,
              dense: true,
              contentPadding: const EdgeInsets.symmetric(horizontal: 18),
            );
          }),
          const SizedBox(height: 8),
        ],
      ),
    );
  }
}

// ── Tool call block ───────────────────────────────────────────────────────────

class _ToolCallBlock extends StatefulWidget {
  const _ToolCallBlock({required this.record, this.isWaitingApproval = false});
  final ToolCallRecord record;
  final bool isWaitingApproval;

  @override
  State<_ToolCallBlock> createState() => _ToolCallBlockState();
}

class _ToolCallBlockState extends State<_ToolCallBlock> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final r = widget.record;
    final isDone = r.resultJson != null;
    final isWaiting = widget.isWaitingApproval;

    return Container(
      margin: const EdgeInsets.only(top: 6),
      decoration: BoxDecoration(
        color: isWaiting
            ? Tokens.destructive.withAlpha(10)
            : Tokens.surfaceElevated,
        borderRadius: BorderRadius.circular(6),
        border: Border.all(
          color: isWaiting ? Tokens.destructive.withAlpha(60) : Tokens.glassEdge,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          GestureDetector(
            onTap: isDone ? () => setState(() => _expanded = !_expanded) : null,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
              child: Row(
                children: [
                  Icon(
                    r.error
                        ? Icons.error_outline_rounded
                        : isDone
                            ? Icons.check_circle_outline_rounded
                            : isWaiting
                                ? Icons.pending_outlined
                                : Icons.sync_rounded,
                    size: 12,
                    color: r.error
                        ? Tokens.destructive
                        : isDone
                            ? Tokens.accent
                            : isWaiting
                                ? Tokens.destructive.withAlpha(180)
                                : Tokens.textMuted,
                  ),
                  const SizedBox(width: 6),
                  Text(
                    '${r.server} · ${r.tool}',
                    style: const TextStyle(
                      fontSize: 11,
                      color: Tokens.textSecondary,
                      fontFamily: 'JetBrains Mono',
                    ),
                  ),
                  if (isWaiting) ...[
                    const SizedBox(width: 8),
                    Text(
                      'Awaiting approval…',
                      style: TextStyle(
                        fontSize: 10,
                        color: Tokens.destructive.withAlpha(180),
                        fontStyle: FontStyle.italic,
                      ),
                    ),
                  ] else if (isDone && !r.error) ...[
                    const SizedBox(width: 6),
                    Text(
                      '${r.elapsedMs}ms',
                      style: const TextStyle(fontSize: 10, color: Tokens.textMuted),
                    ),
                  ],
                  const Spacer(),
                  if (isDone)
                    Icon(
                      _expanded ? Icons.expand_less : Icons.expand_more,
                      size: 13,
                      color: Tokens.textMuted,
                    ),
                ],
              ),
            ),
          ),
          if (_expanded && isDone) ...[
            const Divider(height: 1, color: Tokens.glassEdge),
            Padding(
              padding: const EdgeInsets.all(10),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (r.argsJson.isNotEmpty) ...[
                    const Text('Args',
                        style: TextStyle(
                            fontSize: 10,
                            color: Tokens.textMuted,
                            fontWeight: FontWeight.w600)),
                    const SizedBox(height: 3),
                    SelectableText(
                      r.argsJson,
                      style: const TextStyle(
                          fontSize: 11,
                          color: Tokens.textPrimary,
                          fontFamily: 'JetBrains Mono',
                          height: 1.5),
                    ),
                    const SizedBox(height: 8),
                  ],
                  const Text('Result',
                      style: TextStyle(
                          fontSize: 10,
                          color: Tokens.textMuted,
                          fontWeight: FontWeight.w600)),
                  const SizedBox(height: 3),
                  SelectableText(
                    r.resultJson ?? '',
                    style: TextStyle(
                      fontSize: 11,
                      color: r.error ? Tokens.destructive : Tokens.textPrimary,
                      fontFamily: 'JetBrains Mono',
                      height: 1.5,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }
}
