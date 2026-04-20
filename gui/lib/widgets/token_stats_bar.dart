import 'dart:async';

import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../theme/tokens.dart';

/// Displays live tok/s during generation and a summary prompt/gen count
/// after generation completes.  Refreshes itself every 250 ms while
/// generating so the tok/s counter ticks smoothly without waiting for
/// AppState notifications.
class TokenStatsBar extends StatefulWidget {
  const TokenStatsBar({
    super.key,
    required this.promptTokens,
    required this.genTokens,
    required this.isGenerating,
    this.genStartTime,
  });

  final int promptTokens;
  final int genTokens;
  final bool isGenerating;
  final DateTime? genStartTime;

  @override
  State<TokenStatsBar> createState() => _TokenStatsBarState();
}

class _TokenStatsBarState extends State<TokenStatsBar> {
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    if (widget.isGenerating) _startTimer();
  }

  @override
  void didUpdateWidget(TokenStatsBar old) {
    super.didUpdateWidget(old);
    if (widget.isGenerating && !old.isGenerating) {
      _startTimer();
    } else if (!widget.isGenerating && old.isGenerating) {
      _stopTimer();
    }
  }

  @override
  void dispose() {
    _stopTimer();
    super.dispose();
  }

  void _startTimer() {
    _timer ??= Timer.periodic(const Duration(milliseconds: 250), (_) {
      if (mounted) setState(() {});
    });
  }

  void _stopTimer() {
    _timer?.cancel();
    _timer = null;
  }

  String _buildLabel() {
    if (widget.isGenerating &&
        widget.genStartTime != null &&
        widget.genTokens > 0) {
      final elapsed =
          DateTime.now().difference(widget.genStartTime!).inMilliseconds;
      final tokPerSec =
          elapsed > 0 ? (widget.genTokens * 1000 / elapsed) : 0.0;
      return '${widget.genTokens} tokens · ${tokPerSec.toStringAsFixed(1)} tok/s';
    }
    if (!widget.isGenerating && widget.genTokens > 0) {
      return '${widget.promptTokens} prompt · ${widget.genTokens} generated';
    }
    return '';
  }

  @override
  Widget build(BuildContext context) {
    final label = _buildLabel();
    if (label.isEmpty) return const SizedBox.shrink();
    return Padding(
      padding: const EdgeInsets.symmetric(
          horizontal: Tokens.sp16, vertical: Tokens.sp4),
      child: Align(
        alignment: Alignment.centerRight,
        child: Text(label, style: AppTheme.monoStyle()),
      ),
    );
  }
}
