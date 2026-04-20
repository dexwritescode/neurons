import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../services/app_state.dart';
import '../theme/tokens.dart';
import '../widgets/animated_dots.dart';
import '../widgets/glass_card.dart';
import '../widgets/neurons_wordmark.dart';

/// Shown when [AppState] is in a connecting or error state.
/// Used by [_AppRouter] in main.dart as a fallback while status is resolving.
class SplashScreen extends StatelessWidget {
  const SplashScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final appState = context.watch<AppState>();
    final isError =
        appState.connectionState == ServiceConnectionState.error;

    return Scaffold(
      backgroundColor: Tokens.background,
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const NeuronsWordmark(size: 24),
            const SizedBox(height: 32),
            if (!isError) ...[
              const AnimatedDots(),
              const SizedBox(height: 12),
              const Text(
                'Connecting…',
                style: TextStyle(
                  color: Tokens.textSecondary,
                  fontSize: 14,
                ),
              ),
            ] else ...[
              GlassCard(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      appState.statusError ?? 'Connection failed',
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        color: Tokens.destructive,
                        fontSize: 14,
                      ),
                    ),
                    const SizedBox(height: 12),
                    FilledButton(
                      onPressed: () => context.read<AppState>().connect(),
                      child: const Text('Retry'),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
