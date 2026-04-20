import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class TokenRepository {
  static const _kHfToken = 'neurons_hf_token';

  final FlutterSecureStorage _storage;

  TokenRepository({FlutterSecureStorage? storage})
      : _storage = storage ?? const FlutterSecureStorage();

  Future<String?> loadHfToken() => _storage.read(key: _kHfToken);

  Future<void> saveHfToken(String token) =>
      _storage.write(key: _kHfToken, value: token);

  Future<void> clearHfToken() => _storage.delete(key: _kHfToken);
}
