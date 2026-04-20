import 'package:fixnum/fixnum.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../services/app_state.dart';
import '../theme/app_theme.dart';
import '../theme/tokens.dart';
import '../widgets/resize_divider.dart';
import '../widgets/model_capabilities.dart';

// ── Public entry point ────────────────────────────────────────────────────────

class ModelPickerScreen extends StatefulWidget {
  /// Called after the model is loaded and settings are applied. When shown as
  /// a modal the caller passes Navigator.pop here so the sheet closes itself.
  const ModelPickerScreen({super.key, this.onLoaded});

  final VoidCallback? onLoaded;

  @override
  State<ModelPickerScreen> createState() => _ModelPickerScreenState();
}

class _ModelPickerScreenState extends State<ModelPickerScreen> {
  final _filterCtrl = TextEditingController();
  String _filter = '';
  String? _selectedPath;
  bool _loading = false;
  double _listWidth = 270;

  // Pre-load settings — initialised from saved prefs when a model is selected.
  int _contextWindow = 8192;
  int _maxTokens = 512;
  double _temperature = 0.7;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final state = context.read<AppState>();
      state.refreshModels();
      // Pre-select the currently loaded model so its detail shows by default.
      if (state.modelPath != null) {
        setState(() {
          _selectedPath = state.modelPath;
          _syncSettings(state);
        });
      }
    });
  }

  @override
  void dispose() {
    _filterCtrl.dispose();
    super.dispose();
  }

  void _selectModel(AppState state, String path) {
    setState(() {
      _selectedPath = path;
      _syncSettings(state);
    });
  }

  void _syncSettings(AppState state) {
    final s = state.inferenceSettings;
    final maxCtx = state.maxPositionEmbeddings > 0
        ? state.maxPositionEmbeddings
        : 32768;
    _contextWindow = s.contextWindow > 0
        ? s.contextWindow.clamp(512, maxCtx)
        : maxCtx.clamp(512, maxCtx);
    _maxTokens = s.maxTokens.clamp(64, 4096);
    _temperature = s.temperature;
  }

  Future<void> _confirmDelete(AppState state, String path) async {
    final name = path.split('/').last;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: Tokens.surface,
        title: const Text('Delete model?',
            style: TextStyle(color: Tokens.textPrimary)),
        content: Text(
          'This will permanently delete "$name" from disk.',
          style: const TextStyle(color: Tokens.textSecondary),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(backgroundColor: Colors.red.shade700),
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (confirmed == true && mounted) {
      await state.deleteModel(path);
      setState(() => _selectedPath = null);
    }
  }

  Future<void> _load(AppState state, String path) async {
    setState(() => _loading = true);
    final ok = await state.loadModel(path);
    if (!mounted) return;
    setState(() => _loading = false);
    if (ok) {
      // Apply the settings chosen in the detail pane immediately — no second sheet.
      state.applyInferenceSettings(InferenceSettings(
        maxTokens: _maxTokens,
        contextWindow: _contextWindow,
        temperature: _temperature,
        topP: state.inferenceSettings.topP,
        topK: state.inferenceSettings.topK,
        repPenalty: state.inferenceSettings.repPenalty,
      ));
      widget.onLoaded?.call();
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final allModels = state.availableModels;
    final models = _filter.isEmpty
        ? allModels
        : allModels
            .where((m) =>
                m.name.toLowerCase().contains(_filter) ||
                m.path.toLowerCase().contains(_filter))
            .toList();

    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // ── Left: model list ──────────────────────────────────────────────
        SizedBox(
          width: _listWidth,
          child: Column(
            children: [
              // Search bar
              Padding(
                padding: const EdgeInsets.all(12),
                child: TextField(
                  controller: _filterCtrl,
                  onChanged: (v) => setState(() => _filter = v.toLowerCase()),
                  style: const TextStyle(fontSize: 13, color: Tokens.textPrimary),
                  decoration: InputDecoration(
                    hintText: 'Filter models…',
                    hintStyle: const TextStyle(fontSize: 13, color: Tokens.textMuted),
                    prefixIcon: const Icon(Icons.search_rounded, size: 16, color: Tokens.textMuted),
                    contentPadding: const EdgeInsets.symmetric(vertical: 8),
                    filled: true,
                    fillColor: Tokens.surfaceElevated,
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(6),
                      borderSide: const BorderSide(color: Tokens.glassEdge),
                    ),
                    enabledBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(6),
                      borderSide: const BorderSide(color: Tokens.glassEdge),
                    ),
                    focusedBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(6),
                      borderSide: const BorderSide(color: Tokens.accent),
                    ),
                  ),
                ),
              ),
              // Model rows
              Expanded(
                child: models.isEmpty
                    ? _EmptyList(hasFilter: _filter.isNotEmpty)
                    : ListView.builder(
                        itemCount: models.length,
                        itemBuilder: (_, i) {
                          final m = models[i];
                          return _ModelRow(
                            name: m.name,
                            path: m.path,
                            sizeBytes: m.sizeBytes,
                            isSelected: _selectedPath == m.path,
                            isLoaded: state.modelPath == m.path,
                            onTap: () => _selectModel(state, m.path),
                          );
                        },
                      ),
              ),
            ],
          ),
        ),

        ResizeDivider(
          onDrag: (dx) => setState(() {
            _listWidth = (_listWidth + dx).clamp(180.0, 480.0);
          }),
        ),

        // ── Right: detail pane ──────────────────────────────────────────
        Expanded(
          child: _selectedPath == null
              ? const _NoSelection()
              : _ModelDetailPane(
                  path: _selectedPath!,
                  isLoaded: state.modelPath == _selectedPath,
                  isLoading: _loading,
                  contextWindow: _contextWindow,
                  maxTokens: _maxTokens,
                  temperature: _temperature,
                  maxCtx: state.maxPositionEmbeddings > 0
                      ? state.maxPositionEmbeddings
                      : 32768,
                  onContextWindowChanged: (v) =>
                      setState(() => _contextWindow = v),
                  onMaxTokensChanged: (v) =>
                      setState(() => _maxTokens = v),
                  onTemperatureChanged: (v) =>
                      setState(() => _temperature = v),
                  onLoad: () => _load(state, _selectedPath!),
                  onEject: () => state.unloadModel(),
                  onDelete: () => _confirmDelete(state, _selectedPath!),
                ),
        ),
      ],
    );
  }
}

// ── Model row (left pane) ─────────────────────────────────────────────────────

class _ModelRow extends StatelessWidget {
  const _ModelRow({
    required this.name,
    required this.path,
    required this.sizeBytes,
    required this.isSelected,
    required this.isLoaded,
    required this.onTap,
  });

  final String name;
  final String path;
  final Int64 sizeBytes;
  final bool isSelected;
  final bool isLoaded;
  final VoidCallback onTap;

  String _fmtSize(Int64 b) {
    final v = b.toInt();
    if (v >= 1 << 30) return '${(v / (1 << 30)).toStringAsFixed(1)} GB';
    if (v >= 1 << 20) return '${(v / (1 << 20)).toStringAsFixed(0)} MB';
    return '${(v / (1 << 10)).toStringAsFixed(0)} KB';
  }

  @override
  Widget build(BuildContext context) {
    final caps = _inferCapabilities(name, '');
    return AnimatedContainer(
      duration: Tokens.normal,
      curve: Tokens.curve,
      decoration: BoxDecoration(
        color: isSelected ? Tokens.surfaceElevated : Colors.transparent,
        border: Border(
          left: BorderSide(
            color: isLoaded
                ? Tokens.accent
                : isSelected
                    ? Tokens.accent.withAlpha(80)
                    : Colors.transparent,
            width: 2,
          ),
        ),
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: onTap,
          splashColor: Tokens.accentDim,
          highlightColor: Tokens.accentDim.withAlpha(40),
          child: Padding(
            padding: const EdgeInsets.symmetric(
                horizontal: 14, vertical: 10),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        name,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w500,
                          color: isSelected
                              ? Tokens.textPrimary
                              : Tokens.textSecondary,
                        ),
                      ),
                      const SizedBox(height: 3),
                      Row(
                        children: [
                          Text(
                            _fmtSize(sizeBytes),
                            style: AppTheme.monoStyle(fontSize: 11),
                          ),
                          const SizedBox(width: 6),
                          ...caps.map((c) => Padding(
                                padding: const EdgeInsets.only(right: 4),
                                child: Icon(c.icon,
                                    size: 12, color: Tokens.textMuted),
                              )),
                        ],
                      ),
                    ],
                  ),
                ),
                if (isLoaded)
                  const Icon(Icons.circle, size: 7, color: Tokens.accent),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ── Model detail pane (right pane) ────────────────────────────────────────────

class _ModelDetailPane extends StatelessWidget {
  const _ModelDetailPane({
    required this.path,
    required this.isLoaded,
    required this.isLoading,
    required this.contextWindow,
    required this.maxTokens,
    required this.temperature,
    required this.maxCtx,
    required this.onContextWindowChanged,
    required this.onMaxTokensChanged,
    required this.onTemperatureChanged,
    required this.onLoad,
    required this.onEject,
    required this.onDelete,
  });

  final String path;
  final bool isLoaded;
  final bool isLoading;
  final int contextWindow;
  final int maxTokens;
  final double temperature;
  final int maxCtx;
  final ValueChanged<int> onContextWindowChanged;
  final ValueChanged<int> onMaxTokensChanged;
  final ValueChanged<double> onTemperatureChanged;
  final VoidCallback onLoad;
  final VoidCallback onEject;
  final VoidCallback onDelete;

  String get _modelName {
    final parts = path.split('/');
    // Return "org/model-name" from the path
    return parts.length >= 2
        ? '${parts[parts.length - 2]}/${parts.last}'
        : parts.last;
  }

  @override
  Widget build(BuildContext context) {
    final caps = _inferCapabilities(path.split('/').last, '');
    final maxTokensCeil = (contextWindow - 64).clamp(64, 4096);

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── Name ───────────────────────────────────────────────────────────
          Text(
            _modelName,
            style: const TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w600,
              color: Tokens.textPrimary,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            path,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            style: AppTheme.monoStyle(fontSize: 10),
          ),
          const SizedBox(height: 14),

          // ── Capability chips ────────────────────────────────────────────────
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: caps.map((c) => _CapChip(cap: c)).toList(),
          ),
          const SizedBox(height: 20),
          const Divider(color: Tokens.glassEdge),
          const SizedBox(height: 20),

          // ── Load settings ───────────────────────────────────────────────────
          const Text(
            'LOAD SETTINGS',
            style: TextStyle(
              fontSize: 10,
              fontWeight: FontWeight.w600,
              letterSpacing: 1.0,
              color: Tokens.textMuted,
            ),
          ),
          const SizedBox(height: 16),

          _SliderRow(
            label: 'Context window',
            displayValue: _fmtCtx(contextWindow),
            value: contextWindow.toDouble().clamp(512.0, maxCtx.toDouble()),
            min: 512,
            max: maxCtx.toDouble(),
            divisions: ((maxCtx - 512) / 512).round().clamp(4, 120),
            onChanged: (v) => onContextWindowChanged(v.round()),
          ),
          const SizedBox(height: 18),

          _SliderRow(
            label: 'Max output tokens',
            displayValue: '$maxTokens',
            value: maxTokens.toDouble().clamp(64.0, maxTokensCeil.toDouble()),
            min: 64,
            max: maxTokensCeil.toDouble(),
            divisions: ((maxTokensCeil - 64) / 64).round().clamp(4, 63),
            onChanged: (v) => onMaxTokensChanged(v.round()),
          ),
          const SizedBox(height: 18),

          _SliderRow(
            label: 'Temperature',
            displayValue: temperature.toStringAsFixed(2),
            value: temperature,
            min: 0.0,
            max: 2.0,
            divisions: 40,
            onChanged: (v) =>
                onTemperatureChanged((v * 20).roundToDouble() / 20),
          ),
          const SizedBox(height: 28),

          // ── Action buttons ──────────────────────────────────────────────────
          Row(
            children: [
              // Delete — disabled while model is loaded
              OutlinedButton.icon(
                onPressed: isLoaded ? null : onDelete,
                icon: const Icon(Icons.delete_outline_rounded, size: 14),
                label: const Text('Delete'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: isLoaded ? Tokens.textMuted : Colors.red.shade300,
                  side: BorderSide(
                    color: isLoaded ? Tokens.glassEdge : Colors.red.shade800,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              if (isLoaded) ...[
                OutlinedButton.icon(
                  onPressed: onEject,
                  icon: const Icon(Icons.eject_rounded, size: 14),
                  label: const Text('Eject'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Tokens.textSecondary,
                    side: const BorderSide(color: Tokens.glassEdge),
                  ),
                ),
                const SizedBox(width: 8),
              ],
              const Spacer(),
              FilledButton.icon(
                onPressed: (isLoaded || isLoading) ? null : onLoad,
                icon: isLoading
                    ? const SizedBox(
                        width: 14,
                        height: 14,
                        child: CircularProgressIndicator(
                            strokeWidth: 2, color: Colors.black),
                      )
                    : Icon(
                        isLoaded
                            ? Icons.check_circle_rounded
                            : Icons.play_arrow_rounded,
                        size: 16,
                      ),
                label: Text(isLoaded
                    ? 'Loaded'
                    : isLoading
                        ? 'Loading…'
                        : 'Load'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// _Cap, _CapChip, _inferCapabilities → see widgets/model_capabilities.dart
// Local aliases for backward-compat within this file:
typedef _Cap = ModelCap;
final _inferCapabilities = inferCapabilities;
typedef _CapChip = CapChip;

// ── Slider row ────────────────────────────────────────────────────────────────

class _SliderRow extends StatelessWidget {
  const _SliderRow({
    required this.label,
    required this.displayValue,
    required this.value,
    required this.min,
    required this.max,
    required this.divisions,
    required this.onChanged,
  });

  final String label;
  final String displayValue;
  final double value;
  final double min;
  final double max;
  final int divisions;
  final ValueChanged<double> onChanged;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(label,
                style: const TextStyle(
                    fontSize: 12, color: Tokens.textSecondary)),
            const Spacer(),
            Text(displayValue,
                style: const TextStyle(
                  fontSize: 12,
                  color: Tokens.textPrimary,
                  fontWeight: FontWeight.w500,
                )),
          ],
        ),
        const SizedBox(height: 4),
        SliderTheme(
          data: SliderTheme.of(context).copyWith(
            activeTrackColor: Tokens.accent,
            thumbColor: Tokens.accent,
            inactiveTrackColor: Tokens.surfaceElevated,
            overlayColor: Tokens.accentDim,
            trackHeight: 3,
            thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 6),
            overlayShape: const RoundSliderOverlayShape(overlayRadius: 14),
          ),
          child: Slider(
            value: value.clamp(min, max),
            min: min,
            max: max,
            divisions: divisions,
            onChanged: onChanged,
          ),
        ),
      ],
    );
  }
}

// ── Placeholder states ────────────────────────────────────────────────────────

class _NoSelection extends StatelessWidget {
  const _NoSelection();

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.memory_outlined, size: 36, color: Tokens.textMuted),
          SizedBox(height: 12),
          Text(
            'Select a model',
            style: TextStyle(fontSize: 13, color: Tokens.textMuted),
          ),
        ],
      ),
    );
  }
}

class _EmptyList extends StatelessWidget {
  const _EmptyList({required this.hasFilter});
  final bool hasFilter;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            hasFilter
                ? Icons.search_off_rounded
                : Icons.folder_open_rounded,
            size: 36,
            color: Tokens.textMuted,
          ),
          const SizedBox(height: 10),
          Text(
            hasFilter ? 'No matches' : 'No models found',
            style: const TextStyle(fontSize: 13, color: Tokens.textMuted),
          ),
          if (!hasFilter) ...[
            const SizedBox(height: 4),
            const Text(
              '~/.neurons/models/',
              style: TextStyle(fontSize: 11, color: Tokens.textMuted),
            ),
          ],
        ],
      ),
    );
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

String _fmtCtx(int tokens) {
  if (tokens >= 1000) {
    final k = tokens / 1000;
    return k == k.truncateToDouble()
        ? '${k.toInt()}K'
        : '${k.toStringAsFixed(1)}K';
  }
  return '$tokens';
}
