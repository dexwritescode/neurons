import 'package:flutter/material.dart';
import '../theme/tokens.dart';

class GlassPageRoute extends PageRouteBuilder {
  GlassPageRoute({required Widget page, super.settings})
      : super(
          transitionDuration: Tokens.normal,
          reverseTransitionDuration: Tokens.fast,
          pageBuilder: (context, animation, secondaryAnimation) => page,
          transitionsBuilder: (context, animation, secondaryAnimation, child) {
            final curved =
                CurvedAnimation(parent: animation, curve: Tokens.curve);
            return FadeTransition(
              opacity: curved,
              child: SlideTransition(
                position: Tween<Offset>(
                        begin: const Offset(0.04, 0), end: Offset.zero)
                    .animate(curved),
                child: child,
              ),
            );
          },
        );
}
