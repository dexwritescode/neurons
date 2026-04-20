import 'package:flutter/material.dart';
import '../theme/tokens.dart';

class AnimatedDots extends StatefulWidget {
  const AnimatedDots({super.key});
  @override
  State<AnimatedDots> createState() => _AnimatedDotsState();
}

class _AnimatedDotsState extends State<AnimatedDots>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 900))
      ..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Row(mainAxisSize: MainAxisSize.min, children: [
      for (int i = 0; i < 3; i++) ...[
        if (i > 0) const SizedBox(width: 6),
        FadeTransition(
          opacity: CurvedAnimation(
            parent: _controller,
            curve: Interval(i * 0.2, i * 0.2 + 0.5, curve: Curves.easeInOut),
          ),
          child: Container(
            width: 6,
            height: 6,
            decoration: BoxDecoration(
                color: Tokens.accent,
                borderRadius: BorderRadius.circular(3)),
          ),
        ),
      ],
    ]);
  }
}
