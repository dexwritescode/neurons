#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a neurons session (owns backend + service impl).
typedef struct NeuronsCore NeuronsCore;

// ── Lifecycle ────────────────────────────────────────────────────────────────

/// Create a session. models_dir is the directory where models are stored
/// (e.g. ~/.neurons/models). Returns NULL on allocation failure.
NeuronsCore* neurons_create(const char* models_dir);

/// Destroy a session and free all resources. Safe to call with NULL.
void neurons_destroy(NeuronsCore* h);

/// Initialise the compute backend (MLX on Apple Silicon, CPU otherwise).
/// Must be called from the main thread before any other operation.
/// Returns 0 on success; writes a NUL-terminated error into err (up to err_len bytes).
int neurons_init_backend(NeuronsCore* h, char* err, int err_len);

// ── HuggingFace auth ─────────────────────────────────────────────────────────

/// Set (or clear) the HuggingFace Bearer token for gated model access.
/// Pass an empty string or NULL to clear. Safe to call at any time.
void neurons_set_hf_token(NeuronsCore* h, const char* token);

// ── Model management ─────────────────────────────────────────────────────────

/// Load a model from path. Returns 0 on success.
int  neurons_load_model(NeuronsCore* h, const char* path,
                        char* err, int err_len);

/// Unload the currently loaded model.
void neurons_unload_model(NeuronsCore* h);

/// Delete a model directory from disk. Returns -1 if the model is currently
/// loaded (eject first) or if deletion fails. Returns 0 on success.
int  neurons_delete_model(NeuronsCore* h, const char* path,
                          char* err, int err_len);

/// Returns a JSON string describing current status (model_loaded, model_path,
/// model_type, backend, vocab_size, num_layers). Caller must free with
/// neurons_free_string().
char* neurons_get_status(NeuronsCore* h);

/// Returns a JSON array of {path, name, model_type} objects for all models
/// found in models_dir. Caller must free with neurons_free_string().
char* neurons_list_models(NeuronsCore* h);

// ── Generation ───────────────────────────────────────────────────────────────

/// Called from a worker thread for each decoded token.
/// token  — NUL-terminated UTF-8 text of the token.
/// Return 0 to continue, non-zero to cancel generation.
typedef int (*NeuronsTokenCb)(const char* token, void* userdata);

/// Run inference. Blocks until generation is complete or cancelled.
/// user_prompt  — the latest user message (raw text, not yet formatted).
/// history_json — nullable JSON array of {"role":"user|assistant|system",
///                "content":"..."} objects representing prior turns. The C
///                layer applies the model-specific chat template (same logic
///                as the gRPC Generate RPC) so behaviour is consistent.
/// max_tokens   — 0 means use model default (200).
/// Returns 0 on success, non-zero on error (error written into err).
/// Must be called from a background thread — never from the main/UI thread.
int  neurons_generate(NeuronsCore*   h,
                      const char*    user_prompt,
                      const char*    history_json,
                      int            max_tokens,
                      int            context_window,
                      float          temperature,
                      float          top_p,
                      int            top_k,
                      float          rep_penalty,
                      NeuronsTokenCb cb,
                      void*          userdata,
                      char* err, int err_len);

/// Signal a running neurons_generate() to stop after the current token.
/// Safe to call from any thread.
void neurons_cancel(NeuronsCore* h);

// ── Model browser / download ─────────────────────────────────────────────────

/// Search HuggingFace for models matching query. Returns JSON or NULL on error.
/// sort: "downloads" | "likes" | "trending" | "lastModified" (NULL → "downloads")
/// pipeline_tags_json: JSON array e.g. "[\"text-generation\"]" (NULL/empty → no filter)
/// author: org filter e.g. "mlx-community" (NULL/empty → all)
/// Caller must free with neurons_free_string().
char* neurons_search_models(NeuronsCore* h, const char* query, int limit,
                            const char* sort,
                            const char* pipeline_tags_json,
                            const char* author,
                            char* err, int err_len);

/// Fetch metadata for a single HuggingFace repo. Returns JSON or NULL.
/// Caller must free with neurons_free_string().
char* neurons_get_model_info(NeuronsCore* h, const char* repo_id,
                             char* err, int err_len);

/// Called periodically during download.
/// bytes_done / bytes_total — progress counters (-1 if unknown).
/// current_file — name of file being downloaded (NUL-terminated).
/// Return 0 to continue, non-zero to cancel.
typedef int (*NeuronsDownloadCb)(int64_t     bytes_done,
                                 int64_t     bytes_total,
                                 double      speed_bps,
                                 const char* current_file,
                                 void*       userdata);

/// Download a model from HuggingFace. Blocks until done or cancelled.
/// Must be called from a background thread.
/// Returns 0 on success.
int  neurons_download_model(NeuronsCore* h, const char* repo_id,
                            NeuronsDownloadCb cb, void* userdata,
                            char* err, int err_len);

/// Cancel a running neurons_download_model().
void neurons_cancel_download(NeuronsCore* h);

// ── Utilities ────────────────────────────────────────────────────────────────

/// Free a string returned by any neurons_* function.
void neurons_free_string(char* s);

#ifdef __cplusplus
} // extern "C"
#endif
