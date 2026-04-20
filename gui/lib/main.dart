import 'dart:io';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';

import 'services/app_state.dart';
import 'services/ffi_neurons_client.dart';
import 'screens/splash_screen.dart';
import 'screens/connect_screen.dart';
import 'screens/model_picker_screen.dart';
import 'screens/model_browser_screen.dart';
import 'theme/app_theme.dart';
import 'theme/tokens.dart';
import 'routing/glass_page_route.dart';
import 'widgets/animated_dots.dart';
import 'widgets/app_shell.dart';
import 'widgets/glass_card.dart';
import 'widgets/neurons_wordmark.dart';

void main() {
  GoogleFonts.config.allowRuntimeFetching = false;
  runApp(const NeuronsApp());
}

class NeuronsApp extends StatefulWidget {
  const NeuronsApp({super.key});

  @override
  State<NeuronsApp> createState() => _NeuronsAppState();
}

class _NeuronsAppState extends State<NeuronsApp> {
  AppState? _appState;
  String? _initError;

  @override
  void initState() {
    super.initState();
    _init();
  }

  /// Creates the FFI client and connects. Runs on the main thread (MLX requires it).
  Future<void> _init() async {
    await _bootstrapDataDirs();
    try {
      final client = await FfiNeuronsClient.create();
      final state = AppState(client);
      await state.connect();
      if (mounted) setState(() => _appState = state);
    } catch (e) {
      if (mounted) setState(() => _initError = e.toString());
    }
  }

  /// Ensures ~/.neurons/{models,chats} exist on first launch.
  Future<void> _bootstrapDataDirs() async {
    try {
      final home = Platform.environment['HOME'] ?? Platform.environment['USERPROFILE'];
      if (home == null) return;
      await Directory('$home/.neurons/models').create(recursive: true);
      await Directory('$home/.neurons/chats').create(recursive: true);
    } catch (_) {}
  }

  @override
  void dispose() {
    _appState?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final appState = _appState;

    // FFI backend still initialising — show a minimal splash outside any provider.
    if (appState == null) {
      return MaterialApp(
        title: 'Neurons',
        theme: AppTheme.dark(),
        themeMode: ThemeMode.dark,
        home: _InitSplash(error: _initError),
      );
    }

    return ChangeNotifierProvider.value(
      value: appState,
      child: MaterialApp(
        title: 'Neurons',
        theme: AppTheme.dark(),
        themeMode: ThemeMode.dark,
        home: const _AppRouter(),
        onGenerateRoute: (settings) {
          final routes = <String, Widget>{
            '/models': const ModelPickerScreen(),
            '/connect': const ConnectScreen(),
            '/browse': const ModelBrowserScreen(),
          };
          final page = routes[settings.name];
          if (page != null) return GlassPageRoute(page: page, settings: settings);
          return null;
        },
      ),
    );
  }
}

/// Shown while [FfiNeuronsClient] is initialising (before any Provider is set up).
class _InitSplash extends StatelessWidget {
  const _InitSplash({this.error});
  final String? error;

  @override
  Widget build(BuildContext context) {
    final err = error;
    return Scaffold(
      backgroundColor: Tokens.background,
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const NeuronsWordmark(size: 24),
            const SizedBox(height: 32),
            if (err == null) ...[
              const AnimatedDots(),
              const SizedBox(height: 12),
              const Text(
                'Initialising Neurons…',
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
                      'Initialisation failed:\n$err',
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        color: Tokens.destructive,
                        fontSize: 14,
                      ),
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

/// Routes between the loading splash and the main app shell based on
/// connection state. Once connected, [AppShell] owns all further navigation.
class _AppRouter extends StatelessWidget {
  const _AppRouter();

  @override
  Widget build(BuildContext context) {
    final appState = context.watch<AppState>();
    return AnimatedSwitcher(
      duration: Tokens.normal,
      switchInCurve: Tokens.curve,
      child: KeyedSubtree(
        key: ValueKey(appState.connectionState.name),
        child: switch (appState.connectionState) {
          ServiceConnectionState.connected => const AppShell(),
          _ => const SplashScreen(),
        },
      ),
    );
  }
}
