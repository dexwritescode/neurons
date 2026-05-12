#pragma once

// C bindings for the tokenizers_sys Rust crate (HuggingFace tokenizers 0.23.1).
// Generated reference: cbindgen --config cbindgen.toml --crate tokenizers_sys
// This header is committed so consumers do not need a Rust toolchain at read time.

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TokenizerHandle TokenizerHandle;

/// Load a tokenizer from a tokenizer.json path. Returns NULL on error.
TokenizerHandle* tokenizer_from_file(const char* path);

/// Free a tokenizer returned by tokenizer_from_file.
void tokenizer_free(TokenizerHandle* handle);

/// Encode text to token ids. Writes min(count, capacity) ids into ids_out.
/// Returns the actual token count (may exceed capacity), or -1 on error.
/// If the return value exceeds capacity, resize and call again.
int32_t tokenizer_encode(const TokenizerHandle* handle,
                         const char* text,
                         bool add_special_tokens,
                         int32_t* ids_out,
                         int32_t capacity);

/// Decode token ids to text. Returns a malloc'd UTF-8 string.
/// Free with tokenizer_free_string(). Returns NULL on error.
char* tokenizer_decode(const TokenizerHandle* handle,
                       const int32_t* ids,
                       int32_t ids_len,
                       bool skip_special_tokens);

/// Free a string returned by tokenizer_decode or tokenizer_id_to_token.
void tokenizer_free_string(char* s);

/// Return the vocabulary size (including added tokens).
int32_t tokenizer_vocab_size(const TokenizerHandle* handle);

/// Look up a token string in the vocabulary. Returns -1 if not found.
int32_t tokenizer_token_to_id(const TokenizerHandle* handle, const char* token);

/// Look up a token id. Returns a malloc'd token string.
/// Free with tokenizer_free_string(). Returns NULL if id is not in vocabulary.
char* tokenizer_id_to_token(const TokenizerHandle* handle, int32_t id);

#ifdef __cplusplus
}
#endif
