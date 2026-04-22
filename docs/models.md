# Test Models

Integration tests skip automatically (via `GTEST_SKIP`) when a model is absent —
you don't need all of them. Download only the families you want to test.

```bash
# Tokenizer, forward-pass, and attention trace tests
neurons download mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit

# fp16 / bf16 integration tests
neurons download TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Mistral integration + BPE regression test
neurons download mlx-community/Mistral-7B-Instruct-v0.3-8bit

# Qwen2.5 integration tests (falls back to 1.5B if 3B absent)
neurons download mlx-community/Qwen2.5-3B-Instruct-4bit
neurons download mlx-community/Qwen2.5-1.5B-Instruct-4bit

# Llama 3.1 integration tests
neurons download mlx-community/Llama-3.1-8B-Instruct-4bit

# Gemma 3 integration tests
neurons download mlx-community/gemma-3-1b-it-qat-4bit
```
