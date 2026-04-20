import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../services/app_state.dart';
import '../theme/tokens.dart';

// ── Public entry point ────────────────────────────────────────────────────────

/// Show the post-load inference settings sheet.
/// Call this immediately after a successful [AppState.loadModel] to let the
/// user tune context window, max tokens, and temperature before chatting.
Future<void> showLoadSettingsSheet(BuildContext context) {
  return showModalBottomSheet<void>(
    context: context,
    backgroundColor: Tokens.surface,
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
    ),
    isScrollControlled: true,
    builder: (_) => const _LoadSettingsSheet(),
  );
}

// ── Sheet widget ──────────────────────────────────────────────────────────────

class _LoadSettingsSheet extends StatefulWidget {
  const _LoadSettingsSheet();

  @override
  State<_LoadSettingsSheet> createState() => _LoadSettingsSheetState();
}

class _LoadSettingsSheetState extends State<_LoadSettingsSheet> {
  late int _contextWindow;
  late int _maxTokens;
  late double _temperature;

  @override
  void initState() {
    super.initState();
    final state = context.read<AppState>();
    final maxCtx =
        state.maxPositionEmbeddings > 0 ? state.maxPositionEmbeddings : 32768;
    _contextWindow =
        state.inferenceSettings.contextWindow.clamp(512, maxCtx);
    _maxTokens =
        state.inferenceSettings.maxTokens.clamp(64, (_contextWindow - 64).clamp(64, 4096));
    _temperature = state.inferenceSettings.temperature;
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final maxCtx =
        state.maxPositionEmbeddings > 0 ? state.maxPositionEmbeddings : 32768;

    final maxTokensCeil = (_contextWindow - 64).clamp(64, 4096);

    return Padding(
      padding: EdgeInsets.fromLTRB(
          24, 24, 24, 24 + MediaQuery.of(context).viewInsets.bottom),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── Header ─────────────────────────────────────────────────────────
          Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              _TypeChip(state.modelType ?? 'model'),
              const SizedBox(width: 10),
              Text(
                'ready',
                style: const TextStyle(fontSize: 12, color: Tokens.accent),
              ),
              const SizedBox(width: 4),
              const Icon(Icons.check_circle_rounded,
                  color: Tokens.accent, size: 14),
              const Spacer(),
              Text(
                '${_formatCtx(maxCtx)} max',
                style: const TextStyle(
                    fontSize: 11, color: Tokens.textMuted),
              ),
            ],
          ),
          const SizedBox(height: 8),
          const Divider(color: Tokens.glassEdge),
          const SizedBox(height: 16),

          // ── Context window ─────────────────────────────────────────────────
          _SliderSetting(
            label: 'Context window',
            displayValue: _formatCtx(_contextWindow),
            value: _contextWindow.toDouble(),
            min: 512,
            max: maxCtx.toDouble(),
            divisions: ((maxCtx - 512) / 512).round().clamp(4, 120),
            onChanged: (v) => setState(() {
              _contextWindow = v.round();
              // Keep max tokens within the new context budget.
              final newCeil = (_contextWindow - 64).clamp(64, 4096);
              if (_maxTokens > newCeil) _maxTokens = newCeil;
            }),
          ),
          const SizedBox(height: 20),

          // ── Max output tokens ──────────────────────────────────────────────
          _SliderSetting(
            label: 'Max output tokens',
            displayValue: '$_maxTokens',
            value: _maxTokens.toDouble(),
            min: 64,
            max: maxTokensCeil.toDouble(),
            divisions: ((maxTokensCeil - 64) / 64).round().clamp(4, 63),
            onChanged: (v) => setState(() => _maxTokens = v.round()),
          ),
          const SizedBox(height: 20),

          // ── Temperature ────────────────────────────────────────────────────
          _SliderSetting(
            label: 'Temperature',
            displayValue: _temperature.toStringAsFixed(2),
            value: _temperature,
            min: 0.0,
            max: 2.0,
            divisions: 40,
            onChanged: (v) =>
                setState(() => _temperature = (v * 20).round() / 20),
          ),
          const SizedBox(height: 28),

          // ── Action button ──────────────────────────────────────────────────
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              icon: const Icon(Icons.chat_bubble_outline_rounded, size: 16),
              label: const Text('Start chatting'),
              onPressed: () {
                state.applyInferenceSettings(InferenceSettings(
                  maxTokens: _maxTokens,
                  contextWindow: _contextWindow,
                  temperature: _temperature,
                  topP: state.inferenceSettings.topP,
                  topK: state.inferenceSettings.topK,
                  repPenalty: state.inferenceSettings.repPenalty,
                ));
                Navigator.of(context).pop();
              },
            ),
          ),
          const SizedBox(height: 4),
        ],
      ),
    );
  }
}

// ── Slider row ────────────────────────────────────────────────────────────────

class _SliderSetting extends StatelessWidget {
  const _SliderSetting({
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
            Text(
              label,
              style: const TextStyle(
                  fontSize: 12, color: Tokens.textSecondary),
            ),
            const Spacer(),
            Text(
              displayValue,
              style: const TextStyle(
                fontSize: 12,
                color: Tokens.textPrimary,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
        const SizedBox(height: 6),
        SliderTheme(
          data: SliderTheme.of(context).copyWith(
            activeTrackColor: Tokens.accent,
            thumbColor: Tokens.accent,
            inactiveTrackColor: Tokens.surfaceElevated,
            overlayColor: Tokens.accentDim,
            trackHeight: 3,
            thumbShape:
                const RoundSliderThumbShape(enabledThumbRadius: 6),
            overlayShape:
                const RoundSliderOverlayShape(overlayRadius: 14),
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

// ── Model type chip ───────────────────────────────────────────────────────────

class _TypeChip extends StatelessWidget {
  const _TypeChip(this.label);
  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding:
          const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: Tokens.accentDim,
        borderRadius: BorderRadius.circular(Tokens.radiusPill),
        border: Border.all(color: Tokens.accent.withAlpha(60)),
      ),
      child: Text(
        label,
        style: const TextStyle(
          fontSize: 12,
          color: Tokens.accent,
          fontWeight: FontWeight.w500,
        ),
      ),
    );
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

String _formatCtx(int tokens) {
  if (tokens >= 1000) {
    final k = tokens / 1000;
    return k == k.truncateToDouble()
        ? '${k.toInt()}K'
        : '${k.toStringAsFixed(1)}K';
  }
  return '$tokens';
}
