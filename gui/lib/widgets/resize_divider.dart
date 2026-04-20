import 'package:flutter/material.dart';
import '../theme/tokens.dart';

/// Thin draggable vertical divider for resizable split panes.
class ResizeDivider extends StatefulWidget {
  const ResizeDivider({super.key, required this.onDrag});
  final ValueChanged<double> onDrag;

  @override
  State<ResizeDivider> createState() => _ResizeDividerState();
}

class _ResizeDividerState extends State<ResizeDivider> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      cursor: SystemMouseCursors.resizeColumn,
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onHorizontalDragUpdate: (d) => widget.onDrag(d.delta.dx),
        child: AnimatedContainer(
          duration: Tokens.fast,
          width: 4,
          color: _hovered ? Tokens.accent.withAlpha(56) : Tokens.glassEdge,
        ),
      ),
    );
  }
}
