import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../services/app_state.dart';
import '../theme/tokens.dart';
import '../widgets/glass_card.dart';
import '../widgets/glass_input.dart';
import '../widgets/neurons_wordmark.dart';

class ConnectScreen extends StatefulWidget {
  const ConnectScreen({super.key});

  @override
  State<ConnectScreen> createState() => _ConnectScreenState();
}

class _ConnectScreenState extends State<ConnectScreen> {
  final _hostController = TextEditingController(text: 'localhost');
  final _portController = TextEditingController(text: '50051');

  @override
  void dispose() {
    _hostController.dispose();
    _portController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<AppState>();
    final connecting =
        state.connectionState == ServiceConnectionState.connecting;

    return Scaffold(
      backgroundColor: Tokens.background,
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 400),
          child: GlassCard(
            padding: const EdgeInsets.all(32),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Center(child: NeuronsWordmark(size: 22)),
                const SizedBox(height: 8),
                const Text(
                  'Connect to inference service',
                  style: TextStyle(
                    color: Tokens.textSecondary,
                    fontSize: 14,
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 32),
                GlassInput(
                  controller: _hostController,
                  hintText: 'Host',
                  labelText: 'Host',
                  enabled: !connecting,
                ),
                const SizedBox(height: 12),
                GlassInput(
                  controller: _portController,
                  hintText: 'Port',
                  labelText: 'Port',
                  keyboardType: TextInputType.number,
                  enabled: !connecting,
                ),
                if (state.statusError != null) ...[
                  const SizedBox(height: 12),
                  Text(
                    state.statusError!,
                    style: const TextStyle(
                      color: Tokens.destructive,
                      fontSize: 12,
                    ),
                  ),
                ],
                const SizedBox(height: 24),
                FilledButton(
                  onPressed: connecting ? null : () => state.connect(),
                  child: connecting
                      ? const SizedBox(
                          height: 20,
                          width: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Tokens.background,
                          ),
                        )
                      : const Text('Connect'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
