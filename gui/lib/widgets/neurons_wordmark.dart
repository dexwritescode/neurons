import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/tokens.dart';

class NeuronsWordmark extends StatelessWidget {
  const NeuronsWordmark({super.key, this.size = 20.0});

  final double size;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Container(
          width: size * 0.4,
          height: size * 0.4,
          decoration: BoxDecoration(
            color: Tokens.accent,
            borderRadius: BorderRadius.circular(size * 0.08),
          ),
        ),
        SizedBox(width: size * 0.3),
        Text(
          'neurons',
          style: GoogleFonts.inter(
            fontSize: size,
            fontWeight: FontWeight.w600,
            color: Tokens.textPrimary,
            letterSpacing: -0.5,
          ),
        ),
      ],
    );
  }
}
