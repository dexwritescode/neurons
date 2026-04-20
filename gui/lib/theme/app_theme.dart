import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'tokens.dart';

abstract final class AppTheme {
  static ThemeData dark() {
    final base = ThemeData.dark(useMaterial3: true);

    final textTheme = GoogleFonts.interTextTheme(base.textTheme).copyWith(
      displayLarge:   GoogleFonts.inter(fontSize: 32, fontWeight: FontWeight.w600, color: Tokens.textPrimary),
      headlineMedium: GoogleFonts.inter(fontSize: 20, fontWeight: FontWeight.w600, color: Tokens.textPrimary),
      titleMedium:    GoogleFonts.inter(fontSize: 15, fontWeight: FontWeight.w600, color: Tokens.textPrimary),
      bodyLarge:      GoogleFonts.inter(fontSize: 14, fontWeight: FontWeight.w400, color: Tokens.textPrimary),
      bodyMedium:     GoogleFonts.inter(fontSize: 13, fontWeight: FontWeight.w400, color: Tokens.textPrimary),
      bodySmall:      GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w400, color: Tokens.textSecondary),
      labelSmall:     GoogleFonts.inter(fontSize: 11, fontWeight: FontWeight.w500, color: Tokens.textMuted),
    );

    final colorScheme = const ColorScheme.dark().copyWith(
      surface:                 Tokens.background,
      surfaceContainerHighest: Tokens.surfaceElevated,
      primary:                 Tokens.accent,
      onPrimary:               Tokens.background,
      secondary:               Tokens.accentDim,
      onSecondary:             Tokens.accent,
      error:                   Tokens.destructive,
      onSurface:               Tokens.textPrimary,
      onSurfaceVariant:        Tokens.textSecondary,
      outline:                 Tokens.textMuted,
      outlineVariant:          Tokens.glassEdge,
    );

    return base.copyWith(
      colorScheme:             colorScheme,
      scaffoldBackgroundColor: Tokens.background,
      textTheme:               textTheme,
      appBarTheme: AppBarTheme(
        backgroundColor:  Tokens.surface,
        surfaceTintColor: Colors.transparent,
        shadowColor:      Colors.transparent,
        elevation:        0,
        titleTextStyle:   GoogleFonts.inter(fontSize: 15, fontWeight: FontWeight.w600, color: Tokens.textPrimary),
        iconTheme:        const IconThemeData(color: Tokens.textSecondary),
        actionsIconTheme: const IconThemeData(color: Tokens.textSecondary),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled:         true,
        fillColor:      Tokens.surfaceElevated,
        hintStyle:      GoogleFonts.inter(fontSize: 14, color: Tokens.textMuted),
        contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(Tokens.radiusInput),
          borderSide:   const BorderSide(color: Tokens.glassEdge),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(Tokens.radiusInput),
          borderSide:   const BorderSide(color: Tokens.accent, width: 1.5),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(Tokens.radiusInput),
          borderSide:   const BorderSide(color: Tokens.destructive),
        ),
        focusedErrorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(Tokens.radiusInput),
          borderSide:   const BorderSide(color: Tokens.destructive, width: 1.5),
        ),
        disabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(Tokens.radiusInput),
          borderSide:   BorderSide(color: Tokens.glassEdge.withAlpha(80)),
        ),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: Tokens.accent,
          foregroundColor: Tokens.background,
          textStyle:       GoogleFonts.inter(fontSize: 14, fontWeight: FontWeight.w600),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(Tokens.radiusInput)),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: Tokens.textSecondary,
          side:            const BorderSide(color: Tokens.glassEdge),
          textStyle:       GoogleFonts.inter(fontSize: 14, fontWeight: FontWeight.w500),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(Tokens.radiusInput)),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        ),
      ),
      iconButtonTheme: IconButtonThemeData(
        style: IconButton.styleFrom(foregroundColor: Tokens.textSecondary),
      ),
      dividerTheme: const DividerThemeData(color: Tokens.glassEdge, thickness: 1, space: 1),
      listTileTheme: const ListTileThemeData(tileColor: Colors.transparent),
      progressIndicatorTheme: const ProgressIndicatorThemeData(color: Tokens.accent),
      scrollbarTheme: ScrollbarThemeData(
        thumbColor: WidgetStateProperty.all(Tokens.textMuted),
      ),
    );
  }

  /// JetBrains Mono style for metadata, stats, paths, file sizes.
  static TextStyle monoStyle({
    double fontSize = 12,
    FontWeight fontWeight = FontWeight.w500,
    Color color = Tokens.textMuted,
  }) =>
      GoogleFonts.jetBrainsMono(fontSize: fontSize, fontWeight: fontWeight, color: color);
}
