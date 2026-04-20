import 'package:flutter/material.dart';

abstract final class Tokens {
  // ── Palette ──────────────────────────────────────────────────────────────
  static const Color background      = Color(0xFF0C0E12);
  static const Color surface         = Color(0xFF14171E);
  static const Color surfaceElevated = Color(0xFF1C1F28);
  static const Color glassEdge       = Color(0x12FFFFFF);
  static const Color accent          = Color(0xFF4DFFB4);
  static const Color accentDim       = Color(0x1F4DFFB4);
  static const Color textPrimary     = Color(0xFFE8EAF0);
  static const Color textSecondary   = Color(0xFF7B8399);
  static const Color textMuted       = Color(0xFF44495A);
  static const Color destructive     = Color(0xFFFF5F5F);

  // ── Spacing ───────────────────────────────────────────────────────────────
  static const double sp4  = 4.0;
  static const double sp6  = 6.0;
  static const double sp8  = 8.0;
  static const double sp12 = 12.0;
  static const double sp16 = 16.0;
  static const double sp20 = 20.0;
  static const double sp24 = 24.0;
  static const double sp32 = 32.0;

  // ── Border radius ─────────────────────────────────────────────────────────
  static const double radiusCard  = 10.0;
  static const double radiusInput = 6.0;
  static const double radiusPill  = 20.0;

  // ── Animation ─────────────────────────────────────────────────────────────
  static const Duration fast   = Duration(milliseconds: 150);
  static const Duration normal = Duration(milliseconds: 200);
  static const Curve    curve  = Curves.easeOut;
}
