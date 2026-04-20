import 'package:flutter/material.dart';
import '../theme/tokens.dart';

class ModelPill extends StatelessWidget {
  const ModelPill({super.key, required this.label, this.onTap});

  final String label;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: Tokens.sp12, vertical: Tokens.sp4),
        decoration: BoxDecoration(
          color: Tokens.accentDim,
          borderRadius: BorderRadius.circular(Tokens.radiusPill),
          border: Border.all(color: Tokens.accent.withAlpha(76)),
        ),
        child: Text(
          label,
          style: const TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: Tokens.accent,
          ),
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
      ),
    );
  }
}
