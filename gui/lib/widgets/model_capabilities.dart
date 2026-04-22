import 'package:flutter/material.dart';
import '../theme/tokens.dart';

enum ModelCap { chat, reasoning, vision, code, toolUse }

extension ModelCapExt on ModelCap {
  IconData get icon => switch (this) {
        ModelCap.chat      => Icons.chat_bubble_outline_rounded,
        ModelCap.reasoning => Icons.lightbulb_outline_rounded,
        ModelCap.vision    => Icons.visibility_outlined,
        ModelCap.code      => Icons.code_rounded,
        ModelCap.toolUse   => Icons.build_outlined,
      };

  String get label => switch (this) {
        ModelCap.chat      => 'Chat',
        ModelCap.reasoning => 'Reasoning',
        ModelCap.vision    => 'Vision',
        ModelCap.code      => 'Code',
        ModelCap.toolUse   => 'Tool use',
      };
}

/// Infer model capabilities from name + model_type heuristics.
List<ModelCap> inferCapabilities(String modelName, String modelType) {
  final n = modelName.toLowerCase();
  final caps = <ModelCap>[ModelCap.chat];

  // Reasoning / chain-of-thought models
  if (n.contains('qwq') || n.contains('thinking') || n.contains('-r1') ||
      n.contains('deepseek-r') || n.contains('marco-o1')) {
    caps.add(ModelCap.reasoning);
  }

  // Vision / multimodal
  if (n.contains('-vl') || n.contains('vision') || n.contains('llava') ||
      n.contains('qwen-vl') || n.contains('pixtral') ||
      n.contains('llava') || modelType == 'gemma3') {
    caps.add(ModelCap.vision);
  }

  // Code-focused models
  if (n.contains('code') || n.contains('coder') || n.contains('starcoder') ||
      n.contains('codestral') || n.contains('devstral')) {
    caps.add(ModelCap.code);
  }

  // Tool / function calling.
  // modelType wins when available (authoritative from C++ after load).
  // Fall back to name heuristics for unloaded / browse-screen models.
  final llama31 = RegExp(r'llama-?3\.?[1-9]');
  final qwen2   = RegExp(r'qwen-?2');
  final qwen3   = RegExp(r'qwen-?3');
  final bool toolByType = modelType == 'qwen2' ||
      modelType == 'qwen3' ||
      modelType == 'mistral' ||
      modelType == 'llama';
  final bool toolByName =
      n.contains('tool') || n.contains('function') ||
      n.contains('hermes') || n.contains('functionary') ||
      n.contains('xlam') || n.contains('nexusraven') ||
      // Mistral / Mixtral / Ministral instruct variants
      (n.contains('instruct') &&
          (n.contains('mistral') || n.contains('mixtral') || n.contains('ministral'))) ||
      // Llama 3.1+ instruct (not Llama 2, not base Llama 3.0)
      (llama31.hasMatch(n) && n.contains('instruct')) ||
      // Qwen 2+ / 3 instruct
      ((qwen2.hasMatch(n) || qwen3.hasMatch(n)) && n.contains('instruct')) ||
      // DeepSeek V2/V3 chat (not R1 reasoning)
      (n.contains('deepseek') && !n.contains('-r1') && !n.contains('r2'));
  if (toolByType || toolByName) caps.add(ModelCap.toolUse);

  return caps;
}

class CapChip extends StatelessWidget {
  const CapChip({super.key, required this.cap});
  final ModelCap cap;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: Tokens.surfaceElevated,
        borderRadius: BorderRadius.circular(Tokens.radiusPill),
        border: Border.all(color: Tokens.glassEdge),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(cap.icon, size: 12, color: Tokens.textSecondary),
          const SizedBox(width: 4),
          Text(cap.label,
              style: const TextStyle(
                  fontSize: 11, color: Tokens.textSecondary)),
        ],
      ),
    );
  }
}
