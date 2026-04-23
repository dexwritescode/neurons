import 'dart:convert';
import 'dart:io';

import 'app_state.dart';

// ── ChatSession ───────────────────────────────────────────────────────────────

class ChatSession {
  ChatSession({
    required this.id,
    required this.title,
    required this.createdAt,
    List<ConversationMessage>? messages,
    Set<String>? activeServerNames,
  })  : messages = messages ?? [],
        activeServerNames = activeServerNames ?? {};

  /// Create a fresh, empty session with a generated id.
  factory ChatSession.create() => ChatSession(
        id: _newId(),
        title: 'New chat',
        createdAt: DateTime.now(),
      );

  factory ChatSession.fromJson(Map<String, dynamic> json) => ChatSession(
        id: json['id'] as String,
        title: json['title'] as String? ?? 'Chat',
        createdAt: DateTime.parse(json['createdAt'] as String),
        messages: (json['messages'] as List<dynamic>? ?? [])
            .map((m) => ConversationMessage(
                  role: m['role'] as String,
                  content: m['content'] as String,
                ))
            .toList(),
        activeServerNames: Set<String>.from(
            (json['activeServerNames'] as List<dynamic>? ?? [])
                .cast<String>()),
      );

  String id;
  String title;
  final DateTime createdAt;
  final List<ConversationMessage> messages;
  final Set<String> activeServerNames;

  Map<String, dynamic> toJson() => {
        'id': id,
        'title': title,
        'createdAt': createdAt.toIso8601String(),
        'messages': messages
            .map((m) => {'role': m.role, 'content': m.content})
            .toList(),
        'activeServerNames': activeServerNames.toList(),
      };

  static String _newId() =>
      DateTime.now().microsecondsSinceEpoch.toRadixString(16);
}

// ── ChatRepository ────────────────────────────────────────────────────────────

class ChatRepository {
  static Directory get _dir {
    final home = Platform.environment['HOME'] ?? '';
    return Directory('$home/.neurons/chats');
  }

  /// Load all persisted sessions, sorted newest-first.
  Future<List<ChatSession>> loadAll() async {
    final dir = _dir;
    if (!await dir.exists()) return [];
    final sessions = <ChatSession>[];
    await for (final entity in dir.list()) {
      if (entity is File && entity.path.endsWith('.json')) {
        try {
          final contents = await entity.readAsString();
          final decoded = jsonDecode(contents) as Map<String, dynamic>;
          sessions.add(ChatSession.fromJson(decoded));
        } catch (_) {
          // Skip corrupted files silently.
        }
      }
    }
    sessions.sort((a, b) => b.createdAt.compareTo(a.createdAt));
    return sessions;
  }

  /// Persist a session. Only call when messages is non-empty.
  Future<void> save(ChatSession session) async {
    final dir = _dir;
    if (!await dir.exists()) await dir.create(recursive: true);
    final file = File('${dir.path}/${session.id}.json');
    await file.writeAsString(jsonEncode(session.toJson()));
  }

  /// Delete a persisted session file. Safe to call if the file doesn't exist.
  Future<void> delete(String id) async {
    final file = File('${_dir.path}/$id.json');
    if (await file.exists()) await file.delete();
  }
}
