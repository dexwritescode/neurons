import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme/tokens.dart';
import '../theme/app_theme.dart';

/// Dialog that prompts the user to enter their HuggingFace access token.
/// Returns the token string, or null if the user cancelled.
class HfTokenDialog extends StatefulWidget {
  const HfTokenDialog({super.key});

  @override
  State<HfTokenDialog> createState() => _HfTokenDialogState();
}

class _HfTokenDialogState extends State<HfTokenDialog> {
  final _ctrl = TextEditingController();
  bool _obscure = true;
  bool _saving = false;

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  void _submit() {
    final token = _ctrl.text.trim();
    if (token.isEmpty) return;
    Navigator.of(context).pop(token);
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: Tokens.surface,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(Tokens.radiusCard),
        side: const BorderSide(color: Tokens.glassEdge),
      ),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: SizedBox(
          width: 400,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header
              Row(
                children: [
                  const Icon(Icons.lock_outline_rounded,
                      size: 18, color: Tokens.accent),
                  const SizedBox(width: 8),
                  const Text(
                    'HuggingFace Token Required',
                    style: TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w600,
                      color: Tokens.textPrimary,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              const Text(
                'This model requires a HuggingFace account with approved access. '
                'Enter your access token to continue.',
                style: TextStyle(fontSize: 13, color: Tokens.textSecondary),
              ),
              const SizedBox(height: 16),

              // Token field
              TextField(
                controller: _ctrl,
                autofocus: true,
                obscureText: _obscure,
                onSubmitted: (_) => _submit(),
                style: const TextStyle(
                    fontSize: 13, color: Tokens.textPrimary,
                    fontFamily: 'JetBrains Mono'),
                decoration: InputDecoration(
                  hintText: 'hf_••••••••••••••••••••••••••••••••••••••',
                  hintStyle: const TextStyle(
                      color: Tokens.textMuted, fontFamily: null),
                  filled: true,
                  fillColor: Tokens.surfaceElevated,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(Tokens.radiusInput),
                    borderSide: const BorderSide(color: Tokens.glassEdge),
                  ),
                  enabledBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(Tokens.radiusInput),
                    borderSide: const BorderSide(color: Tokens.glassEdge),
                  ),
                  focusedBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(Tokens.radiusInput),
                    borderSide: const BorderSide(color: Tokens.accent),
                  ),
                  contentPadding: const EdgeInsets.symmetric(
                      horizontal: 12, vertical: 10),
                  suffixIcon: IconButton(
                    icon: Icon(
                      _obscure
                          ? Icons.visibility_outlined
                          : Icons.visibility_off_outlined,
                      size: 16,
                      color: Tokens.textSecondary,
                    ),
                    onPressed: () => setState(() => _obscure = !_obscure),
                  ),
                ),
              ),
              const SizedBox(height: 8),

              // Get token link
              GestureDetector(
                onTap: () {
                  Clipboard.setData(const ClipboardData(
                      text: 'https://huggingface.co/settings/tokens'));
                },
                child: Row(
                  children: [
                    const Icon(Icons.open_in_new_rounded,
                        size: 12, color: Tokens.textMuted),
                    const SizedBox(width: 4),
                    Text(
                      'Get a token at huggingface.co/settings/tokens',
                      style: AppTheme.monoStyle(
                          fontSize: 11, color: Tokens.textMuted),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),

              // Buttons
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  TextButton(
                    onPressed: () => Navigator.of(context).pop(null),
                    child: const Text('Cancel',
                        style: TextStyle(color: Tokens.textSecondary)),
                  ),
                  const SizedBox(width: 8),
                  FilledButton(
                    onPressed: _saving ? null : _submit,
                    style: FilledButton.styleFrom(
                      backgroundColor: Tokens.accent,
                      foregroundColor: Colors.black,
                      shape: RoundedRectangleBorder(
                          borderRadius:
                              BorderRadius.circular(Tokens.radiusInput)),
                    ),
                    child: const Text('Save Token',
                        style: TextStyle(fontWeight: FontWeight.w600)),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
