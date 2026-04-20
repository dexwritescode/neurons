import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class GlassInput extends StatelessWidget {
  const GlassInput({
    super.key,
    this.controller,
    this.hintText,
    this.labelText,
    this.enabled = true,
    this.maxLines = 1,
    this.minLines,
    this.onSubmitted,
    this.onChanged,
    this.keyboardType,
    this.textInputAction,
    this.focusNode,
    this.autofocus = false,
    this.suffix,
    this.inputFormatters,
  });

  final TextEditingController? controller;
  final String? hintText;
  final String? labelText;
  final bool enabled;
  final int? maxLines;
  final int? minLines;
  final ValueChanged<String>? onSubmitted;
  final ValueChanged<String>? onChanged;
  final TextInputType? keyboardType;
  final TextInputAction? textInputAction;
  final FocusNode? focusNode;
  final bool autofocus;
  final Widget? suffix;
  final List<TextInputFormatter>? inputFormatters;

  @override
  Widget build(BuildContext context) {
    return TextField(
      controller: controller,
      enabled: enabled,
      maxLines: maxLines,
      minLines: minLines,
      onSubmitted: onSubmitted,
      onChanged: onChanged,
      keyboardType: keyboardType,
      textInputAction: textInputAction,
      focusNode: focusNode,
      autofocus: autofocus,
      inputFormatters: inputFormatters,
      decoration: InputDecoration(
        hintText: hintText,
        labelText: labelText,
        suffixIcon: suffix,
      ),
    );
  }
}
