import 'package:fixnum/fixnum.dart';
import 'package:flutter/material.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';

import '../services/app_state.dart';
import '../proto/neurons.pb.dart' as proto;
import '../theme/tokens.dart';
import '../theme/app_theme.dart';
import '../widgets/glass_card.dart';
import '../widgets/glass_input.dart';
import '../widgets/model_capabilities.dart';
import '../widgets/resize_divider.dart';
import '../widgets/hf_token_dialog.dart';

class ModelBrowserScreen extends StatefulWidget {
  const ModelBrowserScreen({super.key});

  @override
  State<ModelBrowserScreen> createState() => _ModelBrowserScreenState();
}

enum _SizeFilter { all, small, medium, large }

extension _SizeFilterExt on _SizeFilter {
  String get label => switch (this) {
        _SizeFilter.all    => 'All',
        _SizeFilter.small  => '≤3B',
        _SizeFilter.medium => '3–7B',
        _SizeFilter.large  => '7B+',
      };
}

// Available pipeline tag filters shown in the checklist dropdown.
const _kPipelineTags = [
  'text-generation',
  'text2text-generation',
  'question-answering',
  'summarization',
  'translation',
  'fill-mask',
  'image-text-to-text',
];

// Sort options: (display label, API value)
const _kSortOptions = [
  ('Trending',  'trending'),
  ('Downloads', 'downloads'),
  ('Likes',     'likes'),
  ('Newest',    'lastModified'),
];

// Extract parameter count in billions from a model ID string.
double? _parseParamB(String modelId) {
  final m = RegExp(r'(\d+(?:\.\d+)?)\s*[Bb](?!\w)').firstMatch(modelId);
  if (m == null) return null;
  return double.parse(m.group(1)!);
}

bool _matchesSize(String modelId, _SizeFilter filter) {
  if (filter == _SizeFilter.all) return true;
  final b = _parseParamB(modelId);
  if (b == null) return true; // unknown size → show in all filters
  return switch (filter) {
    _SizeFilter.all    => true,
    _SizeFilter.small  => b <= 3,
    _SizeFilter.medium => b > 3 && b <= 7,
    _SizeFilter.large  => b > 7,
  };
}

class _ModelBrowserScreenState extends State<ModelBrowserScreen> {
  final _searchCtrl = TextEditingController();
  double _listWidth = 320;
  _SizeFilter _sizeFilter = _SizeFilter.all;

  // Search filter state
  String _sort = 'downloads';
  final Set<String> _selectedTags = {};
  // Non-empty once the user has submitted at least one search; used to decide
  // whether sort/tag changes should auto-re-search.
  String _lastQuery = '';

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
  }

  void _search(AppState state) {
    final q = _searchCtrl.text.trim().isNotEmpty
        ? _searchCtrl.text.trim()
        : _lastQuery;
    if (q.isEmpty) return;
    _lastQuery = q;
    state.searchModels(
      q,
      sort: _sort,
      pipelineTags: _selectedTags.toList(),
    );
  }

  // Build a set of "org/model" strings from local available models.
  Set<String> _downloadedIds(AppState state) {
    return state.availableModels.map((m) {
      final parts = m.path.split('/');
      return parts.length >= 2
          ? '${parts[parts.length - 2]}/${parts.last}'
          : parts.last;
    }).toSet();
  }

  String _formatDownloads(Int64 n) {
    final v = n.toInt();
    if (v >= 1000000) return '↓ ${(v / 1000000).toStringAsFixed(1)}M';
    if (v >= 1000) return '↓ ${(v / 1000).toStringAsFixed(0)}K';
    return '↓ $v';
  }

  String _formatBytes(Int64 bytes) {
    final b = bytes.toInt();
    if (b >= 1 << 30) return '${(b / (1 << 30)).toStringAsFixed(2)} GB';
    if (b >= 1 << 20) return '${(b / (1 << 20)).toStringAsFixed(0)} MB';
    if (b >= 1 << 10) return '${(b / (1 << 10)).toStringAsFixed(0)} KB';
    return '$b B';
  }

  Int64 _totalSize(List<proto.HfFileInfo> files) =>
      files.fold(Int64.ZERO, (acc, f) => acc + f.sizeBytes);

  String _formatSpeed(double bps) {
    if (bps >= 1 << 20) return '${(bps / (1 << 20)).toStringAsFixed(1)} MB/s';
    if (bps >= 1 << 10) return '${(bps / (1 << 10)).toStringAsFixed(0)} KB/s';
    return '${bps.toStringAsFixed(0)} B/s';
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final info = state.selectedModelInfo;
    final downloaded = _downloadedIds(state);
    final filtered = state.searchResults
        .where((r) => _matchesSize(r.modelId, _sizeFilter))
        .toList();

    return Row(
      children: [
        // ── Left: search + results ────────────────────────────────────────
        SizedBox(
          width: _listWidth,
          child: Column(
            children: [
              // Search bar
              Container(
                padding: const EdgeInsets.all(12),
                decoration: const BoxDecoration(
                  color: Tokens.surface,
                  border: Border(bottom: BorderSide(color: Tokens.glassEdge)),
                ),
                child: Column(
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: GlassInput(
                            controller: _searchCtrl,
                            hintText: 'HuggingFace repo (e.g. meta-llama/…)',
                            textInputAction: TextInputAction.search,
                            onSubmitted: (_) => _search(state),
                          ),
                        ),
                        const SizedBox(width: 8),
                        GestureDetector(
                          onTap: state.isSearching ? null : () => _search(state),
                          child: Container(
                            width: 36,
                            height: 36,
                            decoration: BoxDecoration(
                              color: Tokens.accent,
                              borderRadius: BorderRadius.circular(Tokens.radiusInput),
                            ),
                            child: Center(
                              child: state.isSearching
                                  ? const SizedBox(
                                      width: 18, height: 18,
                                      child: CircularProgressIndicator(
                                          strokeWidth: 2, color: Tokens.background),
                                    )
                                  : const Icon(Icons.search_rounded,
                                      size: 18, color: Tokens.background),
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    // ── Sort dropdown + pipeline filter ───────────────────
                    Row(
                      children: [
                        // Sort dropdown
                        _SortDropdown(
                          value: _sort,
                          onChanged: (v) {
                            setState(() => _sort = v);
                            if (_lastQuery.isNotEmpty) _search(state);
                          },
                        ),
                        const SizedBox(width: 6),
                        // Pipeline filter dropdown button
                        _PipelineFilterButton(
                          selectedTags: _selectedTags,
                          onChanged: (tag, selected) {
                            setState(() {
                              if (selected) {
                                _selectedTags.add(tag);
                              } else {
                                _selectedTags.remove(tag);
                              }
                            });
                            if (_lastQuery.isNotEmpty) _search(state);
                          },
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              // Size filter chips
              if (state.searchResults.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.fromLTRB(10, 8, 10, 0),
                  child: SingleChildScrollView(
                    scrollDirection: Axis.horizontal,
                    child: Row(
                      children: _SizeFilter.values.map((f) {
                        final active = _sizeFilter == f;
                        return Padding(
                          padding: const EdgeInsets.only(right: 6),
                          child: GestureDetector(
                            onTap: () => setState(() => _sizeFilter = f),
                            child: AnimatedContainer(
                              duration: const Duration(milliseconds: 150),
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 10, vertical: 4),
                              decoration: BoxDecoration(
                                color: active
                                    ? Tokens.accentDim
                                    : Tokens.surfaceElevated,
                                borderRadius: BorderRadius.circular(20),
                                border: Border.all(
                                  color: active
                                      ? Tokens.accent.withAlpha(80)
                                      : Tokens.glassEdge,
                                ),
                              ),
                              child: Text(
                                f.label,
                                style: TextStyle(
                                  fontSize: 11,
                                  fontWeight: active
                                      ? FontWeight.w600
                                      : FontWeight.w400,
                                  color: active
                                      ? Tokens.accent
                                      : Tokens.textMuted,
                                ),
                              ),
                            ),
                          ),
                        );
                      }).toList(),
                    ),
                  ),
                ),
              if (state.searchError != null)
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  child: Text(state.searchError!,
                      style: const TextStyle(
                          color: Tokens.destructive, fontSize: 12)),
                ),
              Expanded(
                child: _SearchResults(
                  results: filtered,
                  downloadedIds: downloaded,
                  selectedModelId: info?.modelId,
                  downloadingModelId: state.downloadingModelId,
                  formatDownloads: _formatDownloads,
                  onSelect: (id) => state.selectModel(id),
                ),
              ),
            ],
          ),
        ),
        ResizeDivider(
          onDrag: (dx) => setState(() {
            _listWidth = (_listWidth + dx).clamp(220.0, 520.0);
          }),
        ),
        // ── Right: detail pane ────────────────────────────────────────────
          Expanded(
            child: info == null && !state.isLoadingModelDetail
                ? const Center(
                    child: Text(
                      'Select a model to see details',
                      style: TextStyle(
                        color: Tokens.textSecondary,
                        fontSize: 14,
                      ),
                    ),
                  )
                : _DetailPane(
                    info: info,
                    loading: state.isLoadingModelDetail,
                    downloadingModelId: state.downloadingModelId,
                    downloadProgress: state.downloadProgress,
                    downloadSpeedBps: state.downloadSpeedBps,
                    downloadCurrentFile: state.downloadCurrentFile,
                    downloadError: state.downloadError,
                    totalSize:
                        info != null ? _totalSize(info.files) : Int64.ZERO,
                    formatBytes: _formatBytes,
                    formatSpeed: _formatSpeed,
                    onClose: () => state.clearSelectedModel(),
                    onDownload: () async {
                      final modelId = info!.modelId;
                      final result = state.searchResults
                          .where((r) => r.modelId == modelId)
                          .firstOrNull;
                      final isGated = result?.gated ?? false;
                      final hasToken = state.hfToken != null &&
                          state.hfToken!.isNotEmpty;
                      if (isGated && !hasToken) {
                        final token = await showDialog<String>(
                          context: context,
                          builder: (_) => const HfTokenDialog(),
                        );
                        if (token == null || token.isEmpty) return;
                        await state.setGlobalHfToken(token);
                      }
                      state.startDownload(modelId);
                    },
                    onCancel: () => state.cancelDownload(),
                  ),
        ),
      ],
    );
  }
}

// ── Search results list ────────────────────────────────────────────────────────

class _SearchResults extends StatelessWidget {
  const _SearchResults({
    required this.results,
    required this.downloadedIds,
    required this.selectedModelId,
    required this.downloadingModelId,
    required this.formatDownloads,
    required this.onSelect,
  });

  final List<proto.HfModelResult> results;
  final Set<String> downloadedIds;
  final String? selectedModelId;
  final String? downloadingModelId;
  final String Function(Int64) formatDownloads;
  final void Function(String) onSelect;

  @override
  Widget build(BuildContext context) {
    if (results.isEmpty) return const SizedBox.shrink();

    return ListView.builder(
      itemCount: results.length,
      itemBuilder: (context, i) {
        final m = results[i];
        final isSelected = m.modelId == selectedModelId;
        final isDownloading = m.modelId == downloadingModelId;
        final isDownloaded = downloadedIds.contains(m.modelId);

        return AnimatedContainer(
          duration: Tokens.normal,
          decoration: BoxDecoration(
            color: isSelected ? Tokens.accentDim : Colors.transparent,
            border: Border(
              left: BorderSide(
                color: isSelected ? Tokens.accent : Colors.transparent,
                width: 2,
              ),
            ),
          ),
          child: InkWell(
            onTap: () => onSelect(m.modelId),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
              child: Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          m.modelId,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                            color: Tokens.textPrimary,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Row(
                          children: [
                            Text(formatDownloads(m.downloads),
                                style: AppTheme.monoStyle(fontSize: 11)),
                            if (isDownloaded) ...[
                              const SizedBox(width: 6),
                              Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 5, vertical: 1),
                                decoration: BoxDecoration(
                                  color: Tokens.accentDim,
                                  borderRadius: BorderRadius.circular(4),
                                  border: Border.all(
                                      color: Tokens.accent.withAlpha(60)),
                                ),
                                child: const Text('✓ Downloaded',
                                    style: TextStyle(
                                        fontSize: 10,
                                        color: Tokens.accent,
                                        fontWeight: FontWeight.w500)),
                              ),
                            ],
                            if (m.gated) ...[
                              const SizedBox(width: 6),
                              const Icon(Icons.lock_outline_rounded,
                                  size: 11, color: Tokens.textSecondary),
                            ],
                          ],
                        ),
                        const SizedBox(height: 4),
                        // Capability icons (icon-only to fit the compact card)
                        Builder(builder: (_) {
                          final caps = inferCapabilities(m.modelId, '');
                          return Row(
                            children: caps.map((c) => Padding(
                              padding: const EdgeInsets.only(right: 5),
                              child: Icon(c.icon, size: 12,
                                  color: Tokens.textMuted),
                            )).toList(),
                          );
                        }),
                      ],
                    ),
                  ),
                  if (isDownloading)
                    const SizedBox(
                      width: 18, height: 18,
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Tokens.accent),
                    )
                  else if (isSelected)
                    const Icon(Icons.chevron_right,
                        size: 18, color: Tokens.accent),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

// ── Detail pane ────────────────────────────────────────────────────────────────

class _DetailPane extends StatefulWidget {
  const _DetailPane({
    required this.info,
    required this.loading,
    required this.downloadingModelId,
    required this.downloadProgress,
    required this.downloadSpeedBps,
    required this.downloadCurrentFile,
    required this.downloadError,
    required this.totalSize,
    required this.formatBytes,
    required this.formatSpeed,
    required this.onClose,
    required this.onDownload,
    required this.onCancel,
  });

  final proto.GetModelInfoResponse? info;
  final bool loading;
  final String? downloadingModelId;
  final double? downloadProgress;
  final double downloadSpeedBps;
  final String? downloadCurrentFile;
  final String? downloadError;
  final Int64 totalSize;
  final String Function(Int64) formatBytes;
  final String Function(double) formatSpeed;
  final VoidCallback onClose;
  final VoidCallback onDownload;
  final VoidCallback onCancel;

  @override
  State<_DetailPane> createState() => _DetailPaneState();
}

class _DetailPaneState extends State<_DetailPane> {
  bool _readmeExpanded = false;

  static const _readmePreviewLength = 3000;

  static const _sectionLabelStyle = TextStyle(
    fontSize: 11,
    fontWeight: FontWeight.w600,
    color: Tokens.textSecondary,
    letterSpacing: 1.2,
  );

  @override
  Widget build(BuildContext context) {
    final info = widget.info;

    if (widget.loading) {
      return const Center(
        child: CircularProgressIndicator(
          strokeWidth: 2,
          color: Tokens.accent,
        ),
      );
    }

    if (info == null) return const SizedBox.shrink();

    final isDownloadingThis = widget.downloadingModelId == info.modelId;
    final isDownloadingOther = widget.downloadingModelId != null &&
        widget.downloadingModelId != info.modelId;

    final readme = info.readme;
    final showExpandLink = readme.length > _readmePreviewLength;
    final displayedReadme = (showExpandLink && !_readmeExpanded)
        ? '${readme.substring(0, _readmePreviewLength)}…'
        : readme;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // ── Header ────────────────────────────────────────────────────────
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 8, 4),
          child: Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      info.modelId,
                      style: const TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w600,
                        color: Tokens.textPrimary,
                      ),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                    if (widget.totalSize > Int64.ZERO)
                      Text(
                        widget.formatBytes(widget.totalSize),
                        style: AppTheme.monoStyle(fontSize: 12),
                      ),
                  ],
                ),
              ),
              IconButton(
                icon: const Icon(Icons.close_rounded),
                onPressed: widget.onClose,
                tooltip: 'Close',
              ),
            ],
          ),
        ),
        const Divider(height: 1),
        // ── Scrollable content ────────────────────────────────────────────
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Capability chips
                Builder(builder: (_) {
                  final caps = inferCapabilities(info.modelId, '');
                  if (caps.isEmpty) return const SizedBox.shrink();
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: Wrap(
                      spacing: 6,
                      runSpacing: 6,
                      children: caps.map((c) => CapChip(cap: c)).toList(),
                    ),
                  );
                }),
                // Files
                if (info.files.isNotEmpty) ...[
                  const Text('FILES', style: _sectionLabelStyle),
                  const SizedBox(height: 8),
                  GlassCard(
                    padding: const EdgeInsets.all(12),
                    child: Column(
                      children: info.files.map(
                        (f) => Padding(
                          padding: const EdgeInsets.symmetric(vertical: 4),
                          child: Row(
                            children: [
                              Expanded(
                                child: Text(
                                  f.filename,
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                  style: const TextStyle(
                                    fontSize: 13,
                                    color: Tokens.textPrimary,
                                  ),
                                ),
                              ),
                              const SizedBox(width: 12),
                              Text(
                                widget.formatBytes(f.sizeBytes),
                                style: AppTheme.monoStyle(fontSize: 11),
                              ),
                            ],
                          ),
                        ),
                      ).toList(),
                    ),
                  ),
                  const SizedBox(height: 16),
                ],
                // README
                if (readme.isNotEmpty) ...[
                  const Text('README', style: _sectionLabelStyle),
                  const SizedBox(height: 8),
                  MarkdownBody(
                    data: displayedReadme,
                    shrinkWrap: true,
                    styleSheet: MarkdownStyleSheet(
                      p: const TextStyle(
                        color: Tokens.textPrimary,
                        fontSize: 14,
                        height: 1.6,
                      ),
                      h1: const TextStyle(
                        color: Tokens.textPrimary,
                        fontSize: 18,
                        fontWeight: FontWeight.w600,
                      ),
                      h2: const TextStyle(
                        color: Tokens.textPrimary,
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                      h3: const TextStyle(
                        color: Tokens.textSecondary,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                      code: GoogleFonts.jetBrainsMono(
                        fontSize: 12,
                        color: Tokens.accent,
                        backgroundColor: Tokens.surfaceElevated,
                      ),
                      codeblockDecoration: BoxDecoration(
                        color: Tokens.surfaceElevated,
                        borderRadius:
                            BorderRadius.circular(Tokens.radiusInput),
                        border: Border.all(color: Tokens.glassEdge),
                      ),
                      a: const TextStyle(color: Tokens.accent),
                    ),
                  ),
                  if (showExpandLink)
                    GestureDetector(
                      onTap: () =>
                          setState(() => _readmeExpanded = !_readmeExpanded),
                      child: Padding(
                        padding: const EdgeInsets.only(top: 8),
                        child: Text(
                          _readmeExpanded ? 'Show less ▲' : 'Show more ▼',
                          style: const TextStyle(
                            color: Tokens.accent,
                            fontSize: 13,
                          ),
                        ),
                      ),
                    ),
                ],
              ],
            ),
          ),
        ),
        // ── Download bar ──────────────────────────────────────────────────
        const Divider(height: 1),
        Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (isDownloadingThis) ...[
                LinearProgressIndicator(
                  value: widget.downloadProgress,
                  backgroundColor: Tokens.surfaceElevated,
                  valueColor:
                      const AlwaysStoppedAnimation(Tokens.accent),
                  minHeight: 3,
                  borderRadius: BorderRadius.circular(2),
                ),
                const SizedBox(height: 6),
                Row(
                  children: [
                    if (widget.downloadProgress != null)
                      Text(
                        '${(widget.downloadProgress! * 100).toStringAsFixed(0)}%',
                        style: AppTheme.monoStyle(color: Tokens.accent),
                      ),
                    if (widget.downloadSpeedBps > 0) ...[
                      const SizedBox(width: 8),
                      Text(
                        widget.formatSpeed(widget.downloadSpeedBps),
                        style: AppTheme.monoStyle(color: Tokens.accent),
                      ),
                    ],
                    const Spacer(),
                    if (widget.downloadCurrentFile != null)
                      Flexible(
                        child: Text(
                          widget.downloadCurrentFile!,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: AppTheme.monoStyle(
                              color: Tokens.textMuted),
                        ),
                      ),
                  ],
                ),
                const SizedBox(height: 8),
              ],
              if (widget.downloadError != null)
                Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Text(
                    widget.downloadError!,
                    style: const TextStyle(
                      color: Tokens.destructive,
                      fontSize: 12,
                    ),
                  ),
                ),
              if (isDownloadingThis) ...[
                OutlinedButton(
                  onPressed: widget.onCancel,
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Tokens.destructive,
                    side: const BorderSide(color: Tokens.destructive),
                  ),
                  child: const Text('Cancel'),
                ),
              ] else ...[
                FilledButton(
                  onPressed: isDownloadingOther ? null : widget.onDownload,
                  child: const Text('Download'),
                ),
              ],
            ],
          ),
        ),
      ],
    );
  }
}

// ── Pipeline filter dropdown button ───────────────────────────────────────────

class _PipelineFilterButton extends StatelessWidget {
  const _PipelineFilterButton({
    required this.selectedTags,
    required this.onChanged,
  });

  final Set<String> selectedTags;
  final void Function(String tag, bool selected) onChanged;

  String get _label {
    if (selectedTags.isEmpty) return 'Any type';
    if (selectedTags.length == 1) return selectedTags.first;
    return '${selectedTags.length} types';
  }

  @override
  Widget build(BuildContext context) {
    return PopupMenuButton<String>(
      tooltip: 'Filter by pipeline type',
      offset: const Offset(0, 32),
      color: Tokens.surfaceElevated,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(Tokens.radiusCard),
        side: const BorderSide(color: Tokens.glassEdge),
      ),
      itemBuilder: (_) => _kPipelineTags.map((tag) {
        return PopupMenuItem<String>(
          value: tag,
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 2),
          child: StatefulBuilder(
            builder: (ctx, setMenuState) {
              final checked = selectedTags.contains(tag);
              return Row(
                children: [
                  SizedBox(
                    width: 20,
                    height: 20,
                    child: Checkbox(
                      value: checked,
                      onChanged: (v) {
                        onChanged(tag, v ?? false);
                        setMenuState(() {});
                      },
                      activeColor: Tokens.accent,
                      side: const BorderSide(color: Tokens.textMuted),
                      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(tag,
                      style: const TextStyle(
                          fontSize: 12, color: Tokens.textPrimary)),
                ],
              );
            },
          ),
        );
      }).toList(),
      onSelected: (tag) {
        onChanged(tag, !selectedTags.contains(tag));
      },
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
        decoration: BoxDecoration(
          color: selectedTags.isNotEmpty
              ? Tokens.accentDim
              : Tokens.surfaceElevated,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: selectedTags.isNotEmpty
                ? Tokens.accent.withAlpha(80)
                : Tokens.glassEdge,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(_label,
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: selectedTags.isNotEmpty
                      ? FontWeight.w600
                      : FontWeight.w400,
                  color: selectedTags.isNotEmpty
                      ? Tokens.accent
                      : Tokens.textMuted,
                )),
            const SizedBox(width: 3),
            Icon(
              Icons.arrow_drop_down,
              size: 14,
              color: selectedTags.isNotEmpty
                  ? Tokens.accent
                  : Tokens.textMuted,
            ),
          ],
        ),
      ),
    );
  }
}

// ── Sort dropdown ─────────────────────────────────────────────────────────────

class _SortDropdown extends StatelessWidget {
  const _SortDropdown({required this.value, required this.onChanged});

  final String value;
  final void Function(String) onChanged;

  String get _label =>
      _kSortOptions.firstWhere((o) => o.$2 == value,
          orElse: () => _kSortOptions.first).$1;

  @override
  Widget build(BuildContext context) {
    return PopupMenuButton<String>(
      tooltip: 'Sort by',
      offset: const Offset(0, 32),
      color: Tokens.surfaceElevated,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(Tokens.radiusCard),
        side: const BorderSide(color: Tokens.glassEdge),
      ),
      onSelected: onChanged,
      itemBuilder: (_) => _kSortOptions.map((opt) {
        final (label, val) = opt;
        final active = val == value;
        return PopupMenuItem<String>(
          value: val,
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          child: Row(
            children: [
              SizedBox(
                width: 16,
                child: active
                    ? const Icon(Icons.check_rounded,
                        size: 14, color: Tokens.accent)
                    : null,
              ),
              const SizedBox(width: 6),
              Text(label,
                  style: TextStyle(
                    fontSize: 12,
                    color: active ? Tokens.accent : Tokens.textPrimary,
                    fontWeight:
                        active ? FontWeight.w600 : FontWeight.normal,
                  )),
            ],
          ),
        );
      }).toList(),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
        decoration: BoxDecoration(
          color: Tokens.surfaceElevated,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: Tokens.glassEdge),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.sort_rounded, size: 12, color: Tokens.textMuted),
            const SizedBox(width: 4),
            Text(_label,
                style: const TextStyle(
                    fontSize: 11, color: Tokens.textMuted)),
            const SizedBox(width: 3),
            const Icon(Icons.arrow_drop_down,
                size: 14, color: Tokens.textMuted),
          ],
        ),
      ),
    );
  }
}
