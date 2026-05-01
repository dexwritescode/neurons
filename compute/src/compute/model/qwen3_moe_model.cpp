#include "qwen3_moe_model.h"
#include "model_loader.h"
#include "simple_bpe_tokenizer.h"
#include "../core/compute_backend.h"
#include <algorithm>
#include <cmath>

namespace compute {

// ── Factory ──────────────────────────────────────────────────────────────────

Result<Qwen3MoeModel> Qwen3MoeModel::from_model_dir(
    const std::filesystem::path& model_dir,
    ComputeBackend*              backend)
{
    auto model_result = ModelLoader::load_model(model_dir, backend);
    if (!model_result) return std::unexpected(model_result.error());

    auto& [config, weights] = *model_result;

    auto tokenizer_result = SimpleBpeTokenizer::from_model_dir(model_dir);
    if (!tokenizer_result) return std::unexpected(tokenizer_result.error());

    return Qwen3MoeModel(
        std::move(config),
        std::move(*tokenizer_result),
        std::move(weights),
        backend);
}

Qwen3MoeModel::Qwen3MoeModel(
    ModelConfig                             config,
    SimpleBpeTokenizer                      tokenizer,
    std::unordered_map<std::string, Tensor> weights,
    ComputeBackend*                         backend)
    : Qwen3MoeModelBase(std::move(config), std::move(tokenizer), std::move(weights))
    , backend_(backend)
{}

// ── LanguageModel interface ───────────────────────────────────────────────────

Result<std::vector<float>> Qwen3MoeModel::prefill(const std::vector<int>& prompt_ids) {
    if (prompt_ids.empty())
        return std::unexpected(Error{ErrorCode::InvalidInput, "prefill: prompt_ids cannot be empty"});
    reset_cache();
    kv_cache_.assign(config_.num_hidden_layers, LayerKVCache{});
    ssm_cache_.assign(config_.num_hidden_layers, SsmState{});
    auto result = forward_impl(prompt_ids, 0, &kv_cache_);
    if (result) cache_position_ = prompt_ids.size();
    return result;
}

Result<std::vector<float>> Qwen3MoeModel::decode(int token_id) {
    if (cache_position_ == 0)
        return std::unexpected(Error{ErrorCode::InvalidInput, "decode: must call prefill() first"});

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
    if (backend_->type() == BackendType::MLX) {
        if (!mlx_state_) {
            initialize_mlx_state();
            build_mlx_compile_fn();
        }

        auto& st = *mlx_state_;
        const int n_full = (int)st.kv_keys.size();
        const int n_ssm  = (int)st.ssm_conv.size();

        // Pack inputs: [token_id, kv_k_0, kv_v_0, ..., ssm_conv_0, ssm_rec_0, ..., pos]
        std::vector<mx::array> inputs;
        inputs.reserve(1 + 2*n_full + 2*n_ssm + 1);

        int32_t tid = static_cast<int32_t>(token_id);
        inputs.push_back(mx::array(&tid, {1}, mx::int32));
        for (int fi = 0; fi < n_full; ++fi) {
            inputs.push_back(st.kv_keys[fi]);
            inputs.push_back(st.kv_vals[fi]);
        }
        for (int li = 0; li < n_ssm; ++li) {
            inputs.push_back(st.ssm_conv[li]);
            inputs.push_back(st.ssm_rec[li]);
        }
        int32_t pos_val = static_cast<int32_t>(cache_position_);
        inputs.push_back(mx::array(&pos_val, {1}, mx::int32));

        // Run compiled decode step
        auto outputs = st.compiled_fn(inputs);

        // Evaluate everything together (one Metal submit for the whole token)
        mx::eval(outputs);

        // Unpack outputs: [logits, kv_k_0, kv_v_0, ..., ssm_conv_0, ssm_rec_0, ...]
        const mx::array& logits_arr = outputs[0];
        for (int fi = 0; fi < n_full; ++fi) {
            st.kv_keys[fi] = outputs[1 + 2*fi];
            st.kv_vals[fi] = outputs[1 + 2*fi + 1];
        }
        for (int li = 0; li < n_ssm; ++li) {
            st.ssm_conv[li] = outputs[1 + 2*n_full + 2*li];
            st.ssm_rec[li]  = outputs[1 + 2*n_full + 2*li + 1];
        }

        ++cache_position_;

        // Extract logits to CPU
        size_t vocab = config_.vocab_size;
        std::vector<float> result(vocab);
        auto flat = mx::reshape(logits_arr, {(int)vocab});
        mx::eval(flat);
        std::copy(flat.data<float>(), flat.data<float>() + vocab, result.data());
        return result;
    }
#endif

    auto result = forward_impl({token_id}, static_cast<int>(cache_position_), &kv_cache_);
    if (result) ++cache_position_;
    return result;
}

void Qwen3MoeModel::reset_cache() {
    kv_cache_.clear();
    ssm_cache_.clear();
    cache_position_ = 0;
#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
    mlx_state_.reset();
#endif
}

// ── Embedding ─────────────────────────────────────────────────────────────────

Result<Tensor> Qwen3MoeModel::embedding(const std::vector<int>& token_ids) {
    if (token_ids.empty())
        return std::unexpected(Error{ErrorCode::InvalidInput, "Empty token_ids"});

    if (!dequantized_embed_tokens_.has_value()) {
        auto w = get_weight("language_model.model.embed_tokens.weight");
        if (!w) return std::unexpected(w.error());

        auto s_it = weights_.find("language_model.model.embed_tokens.scales");
        if (s_it != weights_.end()) {
            auto b_it = weights_.find("language_model.model.embed_tokens.biases");
            if (b_it == weights_.end())
                return std::unexpected(Error{ErrorCode::InvalidModel, "embed_tokens.biases missing"});
            int gs   = config_.quantization ? config_.quantization->group_size : 64;
            int bits = infer_quant_bits(*w, s_it->second);
            auto deq = backend_->dequantize(*w, s_it->second, b_it->second, gs, bits);
            if (!deq) return std::unexpected(deq.error());
            dequantized_embed_tokens_ = std::move(*deq);
        } else {
            dequantized_embed_tokens_ = std::move(*w);
        }
    }

    const auto& ew = *dequantized_embed_tokens_;
    const size_t vocab_size  = ew.shape()[0];
    const size_t hidden_size = ew.shape()[1];

    std::vector<Tensor> rows;
    rows.reserve(token_ids.size());
    for (int id : token_ids) {
        if (id < 0 || static_cast<size_t>(id) >= vocab_size)
            return std::unexpected(Error{ErrorCode::InvalidInput,
                "Token ID " + std::to_string(id) + " out of range"});
        auto row = backend_->slice(ew, id, id + 1, 0);
        if (!row) return std::unexpected(row.error());
        auto vec = backend_->reshape(*row, {1, hidden_size});
        if (!vec) return std::unexpected(vec.error());
        rows.push_back(*vec);
    }

    return backend_->concatenate(rows, 0);
}

// ── Linear projection ─────────────────────────────────────────────────────────

Result<Tensor> Qwen3MoeModel::linear(const Tensor& input, const std::string& weight_key) {
    auto w = get_weight(weight_key + ".weight");
    if (!w) return std::unexpected(w.error());

    auto s_it = weights_.find(weight_key + ".scales");
    if (s_it != weights_.end()) {
        int gs   = config_.quantization ? config_.quantization->group_size : 64;
        int bits = infer_quant_bits(*w, s_it->second);
        auto b_it = weights_.find(weight_key + ".biases");
        const Tensor* biases = (b_it != weights_.end()) ? &b_it->second : nullptr;
        return backend_->quantized_matmul(input, *w, s_it->second, biases, true, gs, bits, "affine");
    }

    auto w_t = backend_->swapaxes(*w, 0, 1);
    if (!w_t) return std::unexpected(w_t.error());
    return backend_->matmul(input, *w_t);
}

// ── Expert linear (slice from 3-D weight bank) ───────────────────────────────

Result<Tensor> Qwen3MoeModel::expert_linear(
    const Tensor& input, const std::string& weight_key, int expert_idx)
{
    auto w3d = get_weight(weight_key + ".weight");
    if (!w3d) return std::unexpected(w3d.error());

    // Slice expert e: [E, out, in_packed] → [1, out, in_packed] → [out, in_packed]
    auto w_e3 = backend_->slice(*w3d, expert_idx, expert_idx + 1, 0);
    if (!w_e3) return std::unexpected(w_e3.error());
    auto w_e = backend_->reshape(*w_e3, {w3d->shape()[1], w3d->shape()[2]});
    if (!w_e) return std::unexpected(w_e.error());

    auto s_it = weights_.find(weight_key + ".scales");
    if (s_it != weights_.end()) {
        auto s_e3 = backend_->slice(s_it->second, expert_idx, expert_idx + 1, 0);
        if (!s_e3) return std::unexpected(s_e3.error());
        auto s_e = backend_->reshape(*s_e3,
            {s_it->second.shape()[1], s_it->second.shape()[2]});
        if (!s_e) return std::unexpected(s_e.error());

        int gs   = config_.quantization ? config_.quantization->group_size : 64;
        int bits = infer_quant_bits(*w_e, *s_e);

        auto b_it = weights_.find(weight_key + ".biases");
        if (b_it != weights_.end()) {
            auto b_e3 = backend_->slice(b_it->second, expert_idx, expert_idx + 1, 0);
            if (!b_e3) return std::unexpected(b_e3.error());
            auto b_e = backend_->reshape(*b_e3,
                {b_it->second.shape()[1], b_it->second.shape()[2]});
            if (!b_e) return std::unexpected(b_e.error());
            return backend_->quantized_matmul(input, *w_e, *s_e, &*b_e, true, gs, bits, "affine");
        }
        return backend_->quantized_matmul(input, *w_e, *s_e, nullptr, true, gs, bits, "affine");
    }

    auto w_t = backend_->swapaxes(*w_e, 0, 1);
    if (!w_t) return std::unexpected(w_t.error());
    return backend_->matmul(input, *w_t);
}

// ── MoE MLP ───────────────────────────────────────────────────────────────────

Result<Tensor> Qwen3MoeModel::moe_mlp(const Tensor& input, int layer_idx) {
    const std::string pfx = "language_model.model.layers." + std::to_string(layer_idx) + ".mlp.";
    const std::string sw  = pfx + "switch_mlp.";

    size_t num_experts = config_.num_experts.value_or(256);
    size_t top_k       = config_.num_experts_per_tok.value_or(8);
    size_t seq         = input.shape()[0];
    size_t hidden      = input.shape()[1];
    int    gs          = config_.quantization ? config_.quantization->group_size : 64;

    // Router gate logits [seq, num_experts] — computed once, shared by both paths
    auto gate_logits = linear(input, pfx + "gate");
    if (!gate_logits) return std::unexpected(gate_logits.error());

    // ── Switch expert computation ─────────────────────────────────────────────
    auto switch_out = [&]() -> Result<Tensor> {
    if (seq == 1) {
        // ── GPU-side decode routing (zero CPU-GPU syncs) ──────────────────────
        // All routing and expert computation stays lazy on GPU.
        // topk_indices on 1-D logit vector → no extract() needed.

        auto logits_1d = backend_->reshape(*gate_logits, {num_experts});
        if (!logits_1d) return std::unexpected(logits_1d.error());

        // Top-k expert indices as GPU tensor [top_k]
        auto topk_idx = backend_->topk_indices(*logits_1d, (int)top_k, 0);
        if (!topk_idx) return std::unexpected(topk_idx.error());

        // Routing scores: softmax of top-k logits (= renormalized top-k probs)
        auto topk_logits = backend_->take(*logits_1d, *topk_idx, 0);
        if (!topk_logits) return std::unexpected(topk_logits.error());
        auto scores_1d = backend_->softmax(*topk_logits, -1);
        if (!scores_1d) return std::unexpected(scores_1d.error());
        auto normed_scores = backend_->reshape(*scores_1d, {1, top_k}); // [1, top_k]
        if (!normed_scores) return std::unexpected(normed_scores.error());

        const size_t k = top_k;

        auto gw3 = get_weight(sw + "gate_proj.weight"); if (!gw3) return std::unexpected(gw3.error());
        auto gs3 = get_weight(sw + "gate_proj.scales"); if (!gs3) return std::unexpected(gs3.error());
        auto uw3 = get_weight(sw + "up_proj.weight");   if (!uw3) return std::unexpected(uw3.error());
        auto us3 = get_weight(sw + "up_proj.scales");   if (!us3) return std::unexpected(us3.error());
        auto dw3 = get_weight(sw + "down_proj.weight"); if (!dw3) return std::unexpected(dw3.error());
        auto ds3 = get_weight(sw + "down_proj.scales"); if (!ds3) return std::unexpected(ds3.error());

        auto gb_it = weights_.find(sw + "gate_proj.biases");
        auto ub_it = weights_.find(sw + "up_proj.biases");
        auto db_it = weights_.find(sw + "down_proj.biases");

        int bits = infer_quant_bits(*gw3, *gs3);

        // Gather k expert weight slices via GPU-tensor indices (one take per tensor)
        auto gw_k = backend_->take(*gw3, *topk_idx, 0); if (!gw_k) return std::unexpected(gw_k.error());
        auto gs_k = backend_->take(*gs3, *topk_idx, 0); if (!gs_k) return std::unexpected(gs_k.error());
        auto uw_k = backend_->take(*uw3, *topk_idx, 0); if (!uw_k) return std::unexpected(uw_k.error());
        auto us_k = backend_->take(*us3, *topk_idx, 0); if (!us_k) return std::unexpected(us_k.error());
        auto dw_k = backend_->take(*dw3, *topk_idx, 0); if (!dw_k) return std::unexpected(dw_k.error());
        auto ds_k = backend_->take(*ds3, *topk_idx, 0); if (!ds_k) return std::unexpected(ds_k.error());

        // Stack gate + up weights: [k, out, in_p] → [k*out, in_p]
        const size_t out_e  = gw3->shape()[1];
        const size_t in_p   = gw3->shape()[2];
        const size_t gss    = gs3->shape()[2];

        auto gw2 = backend_->reshape(*gw_k, {k * out_e, in_p});  if (!gw2) return std::unexpected(gw2.error());
        auto gs2 = backend_->reshape(*gs_k, {k * out_e, gss});   if (!gs2) return std::unexpected(gs2.error());
        auto uw2 = backend_->reshape(*uw_k, {k * out_e, in_p});  if (!uw2) return std::unexpected(uw2.error());
        auto us2 = backend_->reshape(*us_k, {k * out_e, gss});   if (!us2) return std::unexpected(us2.error());

        std::optional<Tensor> gb2, ub2;
        if (gb_it != weights_.end()) {
            auto gb_k = backend_->take(gb_it->second, *topk_idx, 0); if (!gb_k) return std::unexpected(gb_k.error());
            auto r = backend_->reshape(*gb_k, {k * out_e, gb_it->second.shape()[2]}); if (!r) return std::unexpected(r.error());
            gb2 = std::move(*r);
        }
        if (ub_it != weights_.end()) {
            auto ub_k = backend_->take(ub_it->second, *topk_idx, 0); if (!ub_k) return std::unexpected(ub_k.error());
            auto r = backend_->reshape(*ub_k, {k * out_e, ub_it->second.shape()[2]}); if (!r) return std::unexpected(r.error());
            ub2 = std::move(*r);
        }

        // Single quantized_matmul for all k gate projections: [1,hidden] × [k*out,in_p]^T → [1, k*out]
        auto gate_k = backend_->quantized_matmul(input, *gw2, *gs2, gb2 ? &*gb2 : nullptr, true, gs, bits, "affine");
        if (!gate_k) return std::unexpected(gate_k.error());
        auto up_k = backend_->quantized_matmul(input, *uw2, *us2, ub2 ? &*ub2 : nullptr, true, gs, bits, "affine");
        if (!up_k) return std::unexpected(up_k.error());

        // Reshape [1, k*out] → [k, out] for per-expert activation
        auto gate_ke = backend_->reshape(*gate_k, {k, out_e}); if (!gate_ke) return std::unexpected(gate_ke.error());
        auto up_ke   = backend_->reshape(*up_k,   {k, out_e}); if (!up_ke)   return std::unexpected(up_ke.error());

        auto act  = backend_->silu(*gate_ke);          if (!act)  return std::unexpected(act.error());
        auto h_ke = backend_->multiply(*act, *up_ke);  if (!h_ke) return std::unexpected(h_ke.error()); // [k, out]

        // Down projection: k matmuls (each expert has different h_ke[i]).
        // Weights already gathered contiguously in dw_k/ds_k → sequential slices = cache-friendly.
        const size_t down_out = dw3->shape()[1];
        const size_t int_p    = dw3->shape()[2];
        const size_t dss      = ds3->shape()[2];

        std::optional<Tensor> db_k;
        if (db_it != weights_.end()) {
            auto r = backend_->take(db_it->second, *topk_idx, 0);
            if (!r) return std::unexpected(r.error());
            db_k = std::move(*r);
        }

        std::vector<Tensor> down_outs;
        down_outs.reserve(k);
        for (size_t i = 0; i < k; ++i) {
            auto h_i   = backend_->slice(*h_ke, (int)i, (int)i + 1, 0); if (!h_i)   return std::unexpected(h_i.error());

            auto dw_i3 = backend_->slice(*dw_k,  (int)i, (int)i + 1, 0); if (!dw_i3) return std::unexpected(dw_i3.error());
            auto dw_i  = backend_->reshape(*dw_i3, {down_out, int_p});    if (!dw_i)  return std::unexpected(dw_i.error());

            auto dsi_3 = backend_->slice(*ds_k,  (int)i, (int)i + 1, 0); if (!dsi_3) return std::unexpected(dsi_3.error());
            auto dsi   = backend_->reshape(*dsi_3, {down_out, dss});      if (!dsi)   return std::unexpected(dsi.error());

            std::optional<Tensor> db_i;
            if (db_k) {
                auto dbi_3 = backend_->slice(*db_k, (int)i, (int)i + 1, 0); if (!dbi_3) return std::unexpected(dbi_3.error());
                auto dbi   = backend_->reshape(*dbi_3, {down_out, db_it->second.shape()[2]}); if (!dbi) return std::unexpected(dbi.error());
                db_i = std::move(*dbi);
            }

            auto out_i = backend_->quantized_matmul(
                *h_i, *dw_i, *dsi, db_i ? &*db_i : nullptr, true, gs, bits, "affine");
            if (!out_i) return std::unexpected(out_i.error());
            down_outs.push_back(std::move(*out_i));
        }

        // Concatenate [k × [1, hidden]] → [k, hidden], then score-weighted sum via matmul
        auto down_all = backend_->concatenate(down_outs, 0); // [k, hidden]
        if (!down_all) return std::unexpected(down_all.error());

        // normed_scores [1, top_k] @ down_all [top_k, hidden] → [1, hidden] (all GPU)
        return backend_->matmul(*normed_scores, *down_all);

    } else {
        // ── CPU-side prefill routing (seq > 1) ────────────────────────────────
        // Extract + CPU top-k is acceptable for prefill (one-time cost per prompt).
        auto gate_probs = backend_->softmax(*gate_logits, -1);
        if (!gate_probs) return std::unexpected(gate_probs.error());

        std::vector<float> probs_cpu(seq * num_experts);
        {
            auto flat = backend_->reshape(*gate_probs, {seq * num_experts});
            if (!flat) return std::unexpected(flat.error());
            auto ex = backend_->extract(*flat, probs_cpu);
            if (!ex) return std::unexpected(ex.error());
        }

        struct ES { int idx; float score; };
        std::vector<std::vector<ES>> selected(seq);
        for (size_t t = 0; t < seq; ++t) {
            float* row = probs_cpu.data() + t * num_experts;
            std::vector<ES> cands(num_experts);
            for (size_t e = 0; e < num_experts; ++e) cands[e] = {(int)e, row[e]};
            std::partial_sort(cands.begin(), cands.begin() + top_k, cands.end(),
                              [](const ES& a, const ES& b){ return a.score > b.score; });
            selected[t].assign(cands.begin(), cands.begin() + top_k);
            float sum = 0.0f;
            for (auto& es : selected[t]) sum += es.score;
            if (sum > 1e-8f)
                for (auto& es : selected[t]) es.score /= sum;
        }

        // Sequential per-token expert loop
        std::vector<Tensor> token_outs;
        token_outs.reserve(seq);
        for (size_t t = 0; t < seq; ++t) {
            auto tok3 = backend_->slice(input, (int)t, (int)t + 1, 0);
            if (!tok3) return std::unexpected(tok3.error());
            auto tok = backend_->reshape(*tok3, {1, hidden});
            if (!tok) return std::unexpected(tok.error());

            std::optional<Tensor> acc;
            for (auto& es : selected[t]) {
                auto g_out = expert_linear(*tok, sw + "gate_proj", es.idx);
                if (!g_out) return std::unexpected(g_out.error());
                auto u_out = expert_linear(*tok, sw + "up_proj", es.idx);
                if (!u_out) return std::unexpected(u_out.error());
                auto act = backend_->silu(*g_out);
                if (!act) return std::unexpected(act.error());
                auto h = backend_->multiply(*act, *u_out);
                if (!h) return std::unexpected(h.error());
                auto out = expert_linear(*h, sw + "down_proj", es.idx);
                if (!out) return std::unexpected(out.error());

                Tensor score_t = backend_->create_tensor(
                    std::span<const float>(&es.score, 1), {1});
                auto scaled = backend_->multiply(*out, score_t);
                if (!scaled) return std::unexpected(scaled.error());

                if (!acc) { acc = std::move(*scaled); }
                else {
                    auto added = backend_->add(*acc, *scaled);
                    if (!added) return std::unexpected(added.error());
                    acc = std::move(*added);
                }
            }
            if (!acc) {
                std::vector<float> z(hidden, 0.0f);
                acc = backend_->create_tensor(z, {1, hidden});
            }
            token_outs.push_back(std::move(*acc));
        }

        return (token_outs.size() == 1)
            ? Result<Tensor>{token_outs[0]}
            : backend_->concatenate(token_outs, 0);
    }
    }();

    if (!switch_out) return std::unexpected(switch_out.error());

    // Shared expert: standard SwiGLU MLP
    auto se_g = linear(input, pfx + "shared_expert.gate_proj");
    if (!se_g) return std::unexpected(se_g.error());
    auto se_u = linear(input, pfx + "shared_expert.up_proj");
    if (!se_u) return std::unexpected(se_u.error());
    auto se_act = backend_->silu(*se_g);
    if (!se_act) return std::unexpected(se_act.error());
    auto se_h = backend_->multiply(*se_act, *se_u);
    if (!se_h) return std::unexpected(se_h.error());
    auto se_out = linear(*se_h, pfx + "shared_expert.down_proj");
    if (!se_out) return std::unexpected(se_out.error());

    // Shared expert gate (scalar sigmoid): [seq, 1] * [seq, hidden]
    auto seg = linear(input, pfx + "shared_expert_gate");
    if (!seg) return std::unexpected(seg.error());
    auto seg_sig = backend_->sigmoid(*seg);
    if (!seg_sig) return std::unexpected(seg_sig.error());
    auto se_gated = backend_->multiply(*se_out, *seg_sig);
    if (!se_gated) return std::unexpected(se_gated.error());

    return backend_->add(*switch_out, *se_gated);
}

// ── Full attention block ──────────────────────────────────────────────────────

Result<Tensor> Qwen3MoeModel::full_attention_block(
    const Tensor& input, int layer_idx, int position_offset, LayerKVCache* cache)
{
    const std::string pfx = "language_model.model.layers." + std::to_string(layer_idx) + ".self_attn.";

    const size_t seq      = input.shape()[0];
    const size_t n_heads  = config_.num_attention_heads;
    const size_t n_kv     = config_.num_key_value_heads;
    const size_t head_dim = config_.effective_head_dim();
    const float  scale    = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Partial RoPE dims
    float prf = config_.partial_rotary_factor.value_or(0.25f);
    int rope_dims = static_cast<int>(head_dim * prf);

    // q_proj: [seq, n_heads * head_dim * 2] (includes output gate)
    auto q_raw = linear(input, pfx + "q_proj");
    if (!q_raw) return std::unexpected(q_raw.error());

    // Split q_proj output into q and gate (along last axis, each head has head_dim*2)
    // Reshape to [seq, n_heads, head_dim*2] then slice
    auto q_4d = backend_->reshape(*q_raw, {seq, n_heads, head_dim * 2});
    if (!q_4d) return std::unexpected(q_4d.error());
    auto q_h = backend_->slice(*q_4d, 0, (int)head_dim, 2);
    if (!q_h) return std::unexpected(q_h.error());
    auto gate_h = backend_->slice(*q_4d, (int)head_dim, (int)(head_dim * 2), 2);
    if (!gate_h) return std::unexpected(gate_h.error());
    // gate: [seq, n_heads, head_dim] → flatten to [seq, n_heads*head_dim]
    auto gate = backend_->reshape(*gate_h, {seq, n_heads * head_dim});
    if (!gate) return std::unexpected(gate.error());
    // q: [seq, n_heads, head_dim] → swapaxes → [n_heads, seq, head_dim]
    auto qt = backend_->swapaxes(*q_h, 0, 1);
    if (!qt) return std::unexpected(qt.error());

    // k, v: standard projections
    auto k_raw = linear(input, pfx + "k_proj");
    if (!k_raw) return std::unexpected(k_raw.error());
    auto v_raw = linear(input, pfx + "v_proj");
    if (!v_raw) return std::unexpected(v_raw.error());

    auto k3 = backend_->reshape(*k_raw, {seq, n_kv, head_dim});
    if (!k3) return std::unexpected(k3.error());
    auto kt = backend_->swapaxes(*k3, 0, 1);
    if (!kt) return std::unexpected(kt.error());

    auto v3 = backend_->reshape(*v_raw, {seq, n_kv, head_dim});
    if (!v3) return std::unexpected(v3.error());
    auto vt = backend_->swapaxes(*v3, 0, 1);
    if (!vt) return std::unexpected(vt.error());

    // QK norms
    {
        auto w = weights_.find(pfx + "q_norm.weight");
        if (w != weights_.end()) {
            auto f = backend_->reshape(*qt, {n_heads * seq, head_dim});
            if (!f) return std::unexpected(f.error());
            auto n = backend_->rms_norm(*f, w->second, config_.rms_norm_eps);
            if (!n) return std::unexpected(n.error());
            qt = backend_->reshape(*n, {n_heads, seq, head_dim});
            if (!qt) return std::unexpected(qt.error());
        }
    }
    {
        auto w = weights_.find(pfx + "k_norm.weight");
        if (w != weights_.end()) {
            auto f = backend_->reshape(*kt, {n_kv * seq, head_dim});
            if (!f) return std::unexpected(f.error());
            auto n = backend_->rms_norm(*f, w->second, config_.rms_norm_eps);
            if (!n) return std::unexpected(n.error());
            kt = backend_->reshape(*n, {n_kv, seq, head_dim});
            if (!kt) return std::unexpected(kt.error());
        }
    }

    // RoPE (partial)
    auto q_rope = backend_->rope(*qt, rope_dims, config_.rope_theta, position_offset);
    if (!q_rope) return std::unexpected(q_rope.error());
    auto k_rope = backend_->rope(*kt, rope_dims, config_.rope_theta, position_offset);
    if (!k_rope) return std::unexpected(k_rope.error());

    // KV cache
    Tensor full_k = *k_rope;
    Tensor full_v = *vt;
    std::string attn_mask = "causal";

    if (cache) {
        if (!cache->valid) {
            cache->keys   = *k_rope;
            cache->values = *vt;
            cache->valid  = true;
        } else {
            auto cat_k = backend_->concatenate({*cache->keys,   *k_rope}, 1);
            if (!cat_k) return std::unexpected(cat_k.error());
            auto cat_v = backend_->concatenate({*cache->values, *vt},    1);
            if (!cat_v) return std::unexpected(cat_v.error());
            cache->keys   = *cat_k;
            cache->values = *cat_v;
            full_k = *cat_k;
            full_v = *cat_v;
        }
        if (seq == 1) attn_mask = "";
    }

    // SDPA
    auto attn_out = backend_->scaled_dot_product_attention(
        *q_rope, full_k, full_v, scale, attn_mask);
    if (!attn_out) return std::unexpected(attn_out.error());

    // Reshape output: [n_heads, seq, head_dim] → [seq, n_heads*head_dim]
    auto attn_t    = backend_->swapaxes(*attn_out, 0, 1);
    if (!attn_t) return std::unexpected(attn_t.error());
    auto attn_flat = backend_->reshape(*attn_t, {seq, n_heads * head_dim});
    if (!attn_flat) return std::unexpected(attn_flat.error());

    // Output gate: attn_flat * sigmoid(gate)
    auto gate_sig = backend_->sigmoid(*gate);
    if (!gate_sig) return std::unexpected(gate_sig.error());
    auto gated = backend_->multiply(*attn_flat, *gate_sig);
    if (!gated) return std::unexpected(gated.error());

    return linear(*gated, pfx + "o_proj");
}

// ── GatedDeltaNet SSM block ───────────────────────────────────────────────────

Result<Tensor> Qwen3MoeModel::linear_attention_block(const Tensor& input, int layer_idx) {
    const std::string pfx =
        "language_model.model.layers." + std::to_string(layer_idx) + ".linear_attn.";

    // SSM dimensions from config (with sensible defaults)
    const size_t Hv          = config_.linear_num_value_heads.value_or(32);
    const size_t Dv          = config_.linear_value_head_dim.value_or(128);
    const size_t Hk          = config_.linear_num_key_heads.value_or(Hv);
    const size_t Dk          = config_.linear_key_head_dim.value_or(128);
    const size_t kernel_size = config_.linear_conv_kernel_dim.value_or(4);
    const size_t key_dim     = Hk * Dk;
    const size_t value_dim   = Hv * Dv;
    const size_t conv_dim    = key_dim * 2 + value_dim;

    const size_t seq    = input.shape()[0];
    const size_t hidden = input.shape()[1];

    // Linear projections
    auto qkv = linear(input, pfx + "in_proj_qkv");  // [seq, conv_dim]
    if (!qkv) return std::unexpected(qkv.error());
    auto z = linear(input, pfx + "in_proj_z");      // [seq, Hv*Dv]
    if (!z) return std::unexpected(z.error());
    auto b = linear(input, pfx + "in_proj_b");      // [seq, Hv]
    if (!b) return std::unexpected(b.error());
    auto a = linear(input, pfx + "in_proj_a");      // [seq, Hv]
    if (!a) return std::unexpected(a.error());

    // Model weights (plain floats)
    auto A_log_w   = get_weight(pfx + "A_log");         // [Hv]
    if (!A_log_w) return std::unexpected(A_log_w.error());
    auto dt_bias_w = get_weight(pfx + "dt_bias");       // [Hv]
    if (!dt_bias_w) return std::unexpected(dt_bias_w.error());
    auto conv1d_w  = get_weight(pfx + "conv1d.weight"); // [conv_dim, kernel_size, 1]
    if (!conv1d_w) return std::unexpected(conv1d_w.error());
    auto norm_w    = get_weight(pfx + "norm.weight");   // [Dv]
    if (!norm_w) return std::unexpected(norm_w.error());

    // Initialize SSM state if needed
    SsmState& state = ssm_cache_[layer_idx];
    if (!state.valid) {
        std::vector<float> czeros((kernel_size - 1) * conv_dim, 0.0f);
        state.conv_state = backend_->create_tensor(czeros, {kernel_size - 1, conv_dim});
        std::vector<float> rzeros(Hv * Dv * Dk, 0.0f);
        state.rec_state = backend_->create_tensor(rzeros, {Hv, Dv, Dk});
        state.valid = true;
    }

    // Causal conv: concat [conv_state, qkv] along time axis, then apply depthwise conv1d
    // conv_state: [kernel-1, conv_dim], qkv: [seq, conv_dim]
    auto conv_input = backend_->concatenate({*state.conv_state, *qkv}, 0);
    if (!conv_input) return std::unexpected(conv_input.error());

    // Save updated conv state (last kernel-1 rows of conv_input)
    {
        size_t total = kernel_size - 1 + seq;
        auto ns = backend_->slice(*conv_input, (int)(total - (kernel_size - 1)), (int)total, 0);
        if (!ns) return std::unexpected(ns.error());
        state.conv_state = *ns;
    }

    // Depthwise conv1d then silu: output [seq, conv_dim]
    auto conv_raw = backend_->conv1d(*conv_input, *conv1d_w, 1, 0, (int)conv_dim);
    if (!conv_raw) return std::unexpected(conv_raw.error());
    auto conv_out = backend_->silu(*conv_raw);
    if (!conv_out) return std::unexpected(conv_out.error());

    // Split conv output into q, k, v along feature axis
    auto q_flat = backend_->slice(*conv_out, 0,           (int)key_dim,           1);
    if (!q_flat) return std::unexpected(q_flat.error());
    auto k_flat = backend_->slice(*conv_out, (int)key_dim, (int)(2 * key_dim),    1);
    if (!k_flat) return std::unexpected(k_flat.error());
    auto v_flat = backend_->slice(*conv_out, (int)(2*key_dim), (int)conv_dim,     1);
    if (!v_flat) return std::unexpected(v_flat.error());

    // Reshape to head format
    auto q_3d = backend_->reshape(*q_flat, {seq, Hk, Dk});
    if (!q_3d) return std::unexpected(q_3d.error());
    auto k_3d = backend_->reshape(*k_flat, {seq, Hk, Dk});
    if (!k_3d) return std::unexpected(k_3d.error());
    auto v_3d = backend_->reshape(*v_flat, {seq, Hv, Dv});
    if (!v_3d) return std::unexpected(v_3d.error());
    auto z_3d = backend_->reshape(*z,      {seq, Hv, Dv});
    if (!z_3d) return std::unexpected(z_3d.error());

    // RMSNorm q and k (unit weight) then scale by 1/Dk and 1/sqrt(Dk)
    std::vector<float> ones_Dk_v(Dk, 1.0f);
    Tensor ones_Dk = backend_->create_tensor(ones_Dk_v, {Dk});

    {
        auto f = backend_->reshape(*q_3d, {seq * Hk, Dk});
        if (!f) return std::unexpected(f.error());
        auto n = backend_->rms_norm(*f, ones_Dk, 1e-6f);
        if (!n) return std::unexpected(n.error());
        q_3d = backend_->reshape(*n, {seq, Hk, Dk});
        if (!q_3d) return std::unexpected(q_3d.error());
    }
    {
        auto f = backend_->reshape(*k_3d, {seq * Hk, Dk});
        if (!f) return std::unexpected(f.error());
        auto n = backend_->rms_norm(*f, ones_Dk, 1e-6f);
        if (!n) return std::unexpected(n.error());
        k_3d = backend_->reshape(*n, {seq, Hk, Dk});
        if (!k_3d) return std::unexpected(k_3d.error());
    }

    // Scale: q *= 1/Dk, k *= 1/sqrt(Dk)
    float inv_scale   = 1.0f / std::sqrt(static_cast<float>(Dk));
    float q_scale_val = inv_scale * inv_scale;
    Tensor q_scale_t = backend_->create_tensor(
        std::span<const float>(&q_scale_val, 1), {1});
    Tensor k_scale_t = backend_->create_tensor(
        std::span<const float>(&inv_scale, 1), {1});

    auto q_sc = backend_->multiply(*q_3d, q_scale_t);
    if (!q_sc) return std::unexpected(q_sc.error());
    auto k_sc = backend_->multiply(*k_3d, k_scale_t);
    if (!k_sc) return std::unexpected(k_sc.error());

    // Repeat q and k along head axis if Hv > Hk
    if (Hv > Hk) {
        int rf = static_cast<int>(Hv / Hk);
        auto qr = backend_->repeat(*q_sc, rf, 1);
        if (!qr) return std::unexpected(qr.error());
        q_sc = qr;
        auto kr = backend_->repeat(*k_sc, rf, 1);
        if (!kr) return std::unexpected(kr.error());
        k_sc = kr;
    }
    // q_sc, k_sc: [seq, Hv, Dk]

    // Compute per-position decay (g) and input gate (beta) for all positions at once
    // g = exp(-exp(A_log) * softplus(a + dt_bias))  →  [seq, Hv]
    // beta = sigmoid(b)                              →  [seq, Hv]
    auto a_biased = backend_->add(*a, *dt_bias_w);      // [seq, Hv] + [Hv] → [seq, Hv]
    if (!a_biased) return std::unexpected(a_biased.error());
    auto sp = backend_->softplus(*a_biased);             // [seq, Hv]
    if (!sp) return std::unexpected(sp.error());
    auto exp_A = backend_->exp(*A_log_w);               // [Hv]
    if (!exp_A) return std::unexpected(exp_A.error());
    auto neg_pre = backend_->multiply(*exp_A, *sp);     // [seq, Hv]
    if (!neg_pre) return std::unexpected(neg_pre.error());
    {
        float zero = 0.0f;
        Tensor zt = backend_->create_tensor(std::span<const float>(&zero, 1), {1});
        auto neg = backend_->subtract(zt, *neg_pre);    // [seq, Hv] (negate)
        if (!neg) return std::unexpected(neg.error());
        neg_pre = neg;
    }
    auto g_seq    = backend_->exp(*neg_pre);            // [seq, Hv]
    if (!g_seq) return std::unexpected(g_seq.error());
    auto beta_seq = backend_->sigmoid(*b);              // [seq, Hv]
    if (!beta_seq) return std::unexpected(beta_seq.error());

    // Sequential SSM recurrence (one step per time position)
    std::vector<Tensor> ys;
    ys.reserve(seq);

    for (size_t t = 0; t < seq; ++t) {
        // Extract position t from [seq, Hv, Dk] tensors
        auto qt3 = backend_->slice(*q_sc, (int)t, (int)t+1, 0);
        if (!qt3) return std::unexpected(qt3.error());
        auto q_t = backend_->reshape(*qt3, {Hv, Dk});
        if (!q_t) return std::unexpected(q_t.error());

        auto kt3 = backend_->slice(*k_sc, (int)t, (int)t+1, 0);
        if (!kt3) return std::unexpected(kt3.error());
        auto k_t = backend_->reshape(*kt3, {Hv, Dk});
        if (!k_t) return std::unexpected(k_t.error());

        auto vt3 = backend_->slice(*v_3d, (int)t, (int)t+1, 0);
        if (!vt3) return std::unexpected(vt3.error());
        auto v_t = backend_->reshape(*vt3, {Hv, Dv});
        if (!v_t) return std::unexpected(v_t.error());

        auto gt3 = backend_->slice(*g_seq, (int)t, (int)t+1, 0);
        if (!gt3) return std::unexpected(gt3.error());
        auto g_t = backend_->reshape(*gt3, {Hv});
        if (!g_t) return std::unexpected(g_t.error());

        auto bt3 = backend_->slice(*beta_seq, (int)t, (int)t+1, 0);
        if (!bt3) return std::unexpected(bt3.error());
        auto beta_t = backend_->reshape(*bt3, {Hv});
        if (!beta_t) return std::unexpected(beta_t.error());

        // SSM step on state: [Hv, Dv, Dk]
        Tensor& h = *state.rec_state;

        // 1. Decay: h = h * g_t  (g_t broadcast [Hv] → [Hv,1,1])
        auto g_exp = backend_->reshape(*g_t, {Hv, 1, 1});
        if (!g_exp) return std::unexpected(g_exp.error());
        auto h_dec = backend_->multiply(h, *g_exp);       // [Hv, Dv, Dk]
        if (!h_dec) return std::unexpected(h_dec.error());

        // 2. kv_mem = h_dec @ k_t  →  [Hv, Dv]
        auto k_col = backend_->reshape(*k_t, {Hv, Dk, 1});
        if (!k_col) return std::unexpected(k_col.error());
        auto kv3d = backend_->matmul(*h_dec, *k_col);     // [Hv, Dv, 1]
        if (!kv3d) return std::unexpected(kv3d.error());
        auto kv_mem = backend_->reshape(*kv3d, {Hv, Dv});
        if (!kv_mem) return std::unexpected(kv_mem.error());

        // 3. delta = (v_t - kv_mem) * beta_t  →  [Hv, Dv]
        auto err = backend_->subtract(*v_t, *kv_mem);
        if (!err) return std::unexpected(err.error());
        auto beta_exp = backend_->reshape(*beta_t, {Hv, 1});
        if (!beta_exp) return std::unexpected(beta_exp.error());
        auto delta = backend_->multiply(*err, *beta_exp);  // [Hv, Dv]
        if (!delta) return std::unexpected(delta.error());

        // 4. state += outer(delta, k_t)  →  [Hv, Dv, Dk]
        auto k_row = backend_->reshape(*k_t,   {Hv, 1,  Dk});
        if (!k_row) return std::unexpected(k_row.error());
        auto d_col = backend_->reshape(*delta, {Hv, Dv, 1});
        if (!d_col) return std::unexpected(d_col.error());
        auto outer = backend_->multiply(*d_col, *k_row);   // [Hv, Dv, Dk]
        if (!outer) return std::unexpected(outer.error());
        auto h_new = backend_->add(*h_dec, *outer);
        if (!h_new) return std::unexpected(h_new.error());

        // 5. y = h_new @ q_t  →  [Hv, Dv]
        auto q_col = backend_->reshape(*q_t, {Hv, Dk, 1});
        if (!q_col) return std::unexpected(q_col.error());
        auto y3d = backend_->matmul(*h_new, *q_col);       // [Hv, Dv, 1]
        if (!y3d) return std::unexpected(y3d.error());
        auto y_t = backend_->reshape(*y3d, {Hv, Dv});
        if (!y_t) return std::unexpected(y_t.error());

        state.rec_state = *h_new;
        ys.push_back(*y_t);
    }

    // Stack outputs: [seq, Hv, Dv]
    std::vector<Tensor> ys_batched;
    ys_batched.reserve(seq);
    for (auto& y : ys) {
        auto yr = backend_->reshape(y, {1, Hv, Dv});
        if (!yr) return std::unexpected(yr.error());
        ys_batched.push_back(*yr);
    }
    Result<Tensor> out_3d = (ys_batched.size() == 1)
        ? Result<Tensor>{ys_batched[0]}
        : backend_->concatenate(ys_batched, 0);
    if (!out_3d) return std::unexpected(out_3d.error());

    // RMSNormGated: norm(out, norm.weight, eps) * silu(z)
    // Flatten [seq, Hv, Dv] → [seq*Hv, Dv] for rms_norm
    auto out_flat = backend_->reshape(*out_3d, {seq * Hv, Dv});
    if (!out_flat) return std::unexpected(out_flat.error());
    auto out_normed = backend_->rms_norm(*out_flat, *norm_w, config_.rms_norm_eps);
    if (!out_normed) return std::unexpected(out_normed.error());
    auto out_n3d = backend_->reshape(*out_normed, {seq, Hv, Dv});
    if (!out_n3d) return std::unexpected(out_n3d.error());

    auto z_silu = backend_->silu(*z_3d);
    if (!z_silu) return std::unexpected(z_silu.error());
    auto gated = backend_->multiply(*out_n3d, *z_silu);     // [seq, Hv, Dv]
    if (!gated) return std::unexpected(gated.error());

    // Reshape and out_proj: [seq, Hv*Dv] → [seq, hidden]
    auto gated_flat = backend_->reshape(*gated, {seq, Hv * Dv});
    if (!gated_flat) return std::unexpected(gated_flat.error());

    return linear(*gated_flat, pfx + "out_proj");
}

// ── Forward pass ──────────────────────────────────────────────────────────────

Result<std::vector<float>> Qwen3MoeModel::forward_impl(
    const std::vector<int>&    input_ids,
    int                        position_offset,
    std::vector<LayerKVCache>* cache_vec)
{
    auto hidden = embedding(input_ids);
    if (!hidden) return std::unexpected(hidden.error());

    const int full_attn_interval = 4;  // (layer_idx + 1) % 4 == 0 → full attention

    for (int i = 0; i < static_cast<int>(config_.num_hidden_layers); ++i) {
        const std::string lpfx =
            "language_model.model.layers." + std::to_string(i) + ".";

        // Pre-attention RMSNorm
        auto pre_norm_w = get_weight(lpfx + "input_layernorm.weight");
        if (!pre_norm_w) return std::unexpected(pre_norm_w.error());
        auto normed = backend_->rms_norm(*hidden, *pre_norm_w, config_.rms_norm_eps);
        if (!normed) return std::unexpected(normed.error());

        // Attention (SSM or full)
        bool is_linear = (i + 1) % full_attn_interval != 0;
        Result<Tensor> attn_out{std::unexpected(Error{ErrorCode::NotImplemented, "unset"})};
        if (is_linear) {
            attn_out = linear_attention_block(*normed, i);
        } else {
            LayerKVCache* kvc = (cache_vec && i < (int)cache_vec->size())
                                ? &(*cache_vec)[i] : nullptr;
            attn_out = full_attention_block(*normed, i, position_offset, kvc);
        }
        if (!attn_out) return std::unexpected(attn_out.error());

        auto residual1 = backend_->add(*hidden, *attn_out);
        if (!residual1) return std::unexpected(residual1.error());

        // Post-attention RMSNorm
        auto post_norm_w = get_weight(lpfx + "post_attention_layernorm.weight");
        if (!post_norm_w) return std::unexpected(post_norm_w.error());
        auto normed2 = backend_->rms_norm(*residual1, *post_norm_w, config_.rms_norm_eps);
        if (!normed2) return std::unexpected(normed2.error());

        // MoE MLP (all layers)
        auto mlp_out = moe_mlp(*normed2, i);
        if (!mlp_out) return std::unexpected(mlp_out.error());

        auto result = backend_->add(*residual1, *mlp_out);
        if (!result) return std::unexpected(result.error());
        hidden = std::move(result);
    }

    // Final RMSNorm
    auto norm_w = get_weight("language_model.model.norm.weight");
    if (!norm_w) return std::unexpected(norm_w.error());
    auto normed = backend_->rms_norm(*hidden, *norm_w, config_.rms_norm_eps);
    if (!normed) return std::unexpected(normed.error());

    // LM head
    auto logits = linear(*normed, "language_model.lm_head");
    if (!logits) return std::unexpected(logits.error());

    // Extract last token's logits
    const size_t seq_len    = input_ids.size();
    const size_t vocab_size = config_.vocab_size;

    auto last = backend_->slice(*logits, (int)(seq_len - 1), (int)seq_len, 0);
    if (!last) return std::unexpected(last.error());
    auto flat = backend_->reshape(*last, {vocab_size});
    if (!flat) return std::unexpected(flat.error());

    std::vector<float> result(vocab_size);
    auto ex = backend_->extract(*flat, result);
    if (!ex) return std::unexpected(ex.error());
    return result;
}


// ═══════════════════════════════════════════════════════════════════════════════
// MLX-native compiled decode path (Apple Silicon only)
// ═══════════════════════════════════════════════════════════════════════════════
#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

namespace {

using WM = std::unordered_map<std::string, mx::array>;

// ── Fused GatedDeltaNet decode kernel ─────────────────────────────────────────
// Metal source for a single decode step (B=1, T=1 specialisation).
// Template params: InT (activation dtype), StT (state dtype), Dk, Dv, Hk, Hv.
// Inputs:  q[1,Hk,Dk], k[1,Hk,Dk], v[1,Hv,Dv], g[1,Hv], beta[1,Hv],
//          state_in[Hv,Dv,Dk]
// Outputs: y[Hv,Dv], state_out[Hv,Dv,Dk]
// Grid=(32,Dv,Hv), threadgroup=(32,4,1)
// Ported from mlx_lm/models/gated_delta.py (_make_gated_delta_kernel).
static constexpr const char* kGatedDeltaDecodeSource = R"msl(
    auto hv_idx = thread_position_in_grid.z;  // B=1: n==hv_idx
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    auto q_    = q    + hk_idx * Dk;
    auto k_    = k    + hk_idx * Dk;
    auto v_    = v    + hv_idx * Dv;
    y         += hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    auto i_state = state_in  + (hv_idx * Dv + dv_idx) * Dk;
    auto o_state = state_out + (hv_idx * Dv + dv_idx) * Dk;

    float s[n_per_t];
    for (int i = 0; i < n_per_t; ++i)
        s[i] = (float)i_state[n_per_t * dk_idx + i];

    float g_val    = (float)g[hv_idx];
    float beta_val = (float)beta[hv_idx];

    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
        s[i] *= g_val;
        kv_mem += s[i] * (float)k_[n_per_t * dk_idx + i];
    }
    kv_mem = simd_sum(kv_mem);

    float delta = ((float)v_[dv_idx] - kv_mem) * beta_val;

    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
        s[i] += (float)k_[n_per_t * dk_idx + i] * delta;
        out  += s[i] * (float)q_[n_per_t * dk_idx + i];
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0)
        y[dv_idx] = (InT)out;

    for (int i = 0; i < n_per_t; ++i)
        o_state[n_per_t * dk_idx + i] = (StT)s[i];
)msl";

// Lazily-initialised kernel (thread-safe C++11 static init).
static const mx::fast::CustomKernelFunction& gated_delta_kernel() {
    static const auto fn = mx::fast::metal_kernel(
        "neurons_gated_delta_decode",
        {"q", "k", "v", "g", "beta", "state_in"},
        {"y", "state_out"},
        kGatedDeltaDecodeSource);
    return fn;
}

// Infer quantization bits from weight/scales shapes.
static int mlx_bits(const mx::array& w, const mx::array& s, int gs) {
    double ratio = static_cast<double>(w.shape().back()) /
                   static_cast<double>(s.shape().back());
    int bits = static_cast<int>(std::round(32.0 * ratio / static_cast<double>(gs)));
    return (bits > 0) ? bits : 4;
}

// Linear projection: quantized (group-wise affine) or plain matmul.
static mx::array mlx_lin(const mx::array& x, const std::string& key, const WM& wm, int gs) {
    const mx::array& w = wm.at(key + ".weight");
    auto sit = wm.find(key + ".scales");
    if (sit != wm.end()) {
        int bits = mlx_bits(w, sit->second, gs);
        std::optional<mx::array> bias;
        auto bit = wm.find(key + ".biases");
        if (bit != wm.end()) bias = bit->second;
        return mx::quantized_matmul(x, w, sit->second, bias, true, gs, bits, "affine");
    }
    return mx::matmul(x, mx::swapaxes(w, 0, 1));
}

// Full-attention (GQA) decode step for a single token.
// KV cache grows each step via concatenate; returned cache has shape [n_kv, T+1, head_dim].
static std::tuple<mx::array, mx::array, mx::array>
mlx_full_attn_step(const mx::array& x, int layer_idx,
                    const mx::array& kv_k, const mx::array& kv_v,
                    const mx::array& pos,
                    const WM& wm, int gs,
                    size_t n_heads, size_t n_kv, size_t head_dim,
                    float rope_theta, int rope_dims, float rms_eps)
{
    const std::string pfx = "language_model.model.layers." +
                             std::to_string(layer_idx) + ".self_attn.";

    // Q projection includes output gate: [1, n_heads * head_dim * 2]
    auto q_raw = mlx_lin(x, pfx + "q_proj", wm, gs);
    auto k_raw = mlx_lin(x, pfx + "k_proj", wm, gs);  // [1, n_kv * head_dim]
    auto v_raw = mlx_lin(x, pfx + "v_proj", wm, gs);

    // Split Q → query [1, n_heads, head_dim] and gate [1, n_heads*head_dim]
    // Reshape to [1, n_heads, 2, head_dim] then take along axis 2 (avoids mx::slice).
    auto q_gate = mx::reshape(q_raw, {1, (int)n_heads, 2, (int)head_dim});
    auto q_h    = mx::take(q_gate, mx::array(0, mx::int32), 2);  // [1, n_heads, head_dim]
    auto gate_h = mx::take(q_gate, mx::array(1, mx::int32), 2);  // [1, n_heads, head_dim]
    auto gate   = mx::reshape(gate_h, {1, (int)(n_heads * head_dim)});

    // K, V: [1, n_kv, head_dim] → [n_kv, 1, head_dim]
    auto kt = mx::swapaxes(mx::reshape(k_raw, {1, (int)n_kv, (int)head_dim}), 0, 1);
    auto vt = mx::swapaxes(mx::reshape(v_raw, {1, (int)n_kv, (int)head_dim}), 0, 1);

    // Transpose Q for norm/rope: [1, n_heads, head_dim] → [n_heads, 1, head_dim]
    auto qt = mx::swapaxes(q_h, 0, 1);

    // Per-head QK norms (if present)
    {
        auto it = wm.find(pfx + "q_norm.weight");
        if (it != wm.end()) {
            auto f = mx::reshape(qt, {(int)n_heads, (int)head_dim});
            qt = mx::reshape(mx::fast::rms_norm(f, it->second, rms_eps),
                             {(int)n_heads, 1, (int)head_dim});
        }
    }
    {
        auto it = wm.find(pfx + "k_norm.weight");
        if (it != wm.end()) {
            auto f = mx::reshape(kt, {(int)n_kv, (int)head_dim});
            kt = mx::reshape(mx::fast::rms_norm(f, it->second, rms_eps),
                             {(int)n_kv, 1, (int)head_dim});
        }
    }

    // Partial RoPE: fast::rope requires 4D [1, heads, seq, head_dim]
    auto qt4 = mx::reshape(qt, {1, (int)n_heads, 1, (int)head_dim});
    auto kt4 = mx::reshape(kt, {1, (int)n_kv, 1, (int)head_dim});
    auto q_rope = mx::reshape(mx::fast::rope(qt4, rope_dims, false, rope_theta, 1.0f, pos),
                              {(int)n_heads, 1, (int)head_dim});
    auto k_rope = mx::reshape(mx::fast::rope(kt4, rope_dims, false, rope_theta, 1.0f, pos),
                              {(int)n_kv, 1, (int)head_dim});

    // Append new K/V to growing cache — [n_kv, T, head_dim] → [n_kv, T+1, head_dim]
    auto new_k = mx::concatenate({kv_k, k_rope}, 1);
    auto new_v = mx::concatenate({kv_v, vt},     1);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto q4 = mx::reshape(q_rope, {1, (int)n_heads, 1,  (int)head_dim});
    auto k4 = mx::reshape(new_k,  {1, (int)n_kv,   -1,  (int)head_dim});  // -1 infers T+1
    auto v4 = mx::reshape(new_v,  {1, (int)n_kv,   -1,  (int)head_dim});
    auto attn4 = mx::fast::scaled_dot_product_attention(q4, k4, v4, scale, "");

    // [1, n_heads, 1, head_dim] → [1, n_heads*head_dim]
    auto attn_flat = mx::reshape(attn4, {1, (int)(n_heads * head_dim)});

    // Output gate (from q_proj second half)
    auto gated = attn_flat * mx::sigmoid(gate);
    auto out   = mlx_lin(gated, pfx + "o_proj", wm, gs);

    return {out, new_k, new_v};
}

// GatedDeltaNet SSM decode step for a single token.
// Returns: (attn_output [1,hidden], updated_conv_state, updated_rec_state)
static std::tuple<mx::array, mx::array, mx::array>
mlx_ssm_step(const mx::array& x, int layer_idx,
              const mx::array& conv_state, const mx::array& rec_state,
              const WM& wm, int gs,
              size_t Hv, size_t Dv, size_t Hk, size_t Dk,
              size_t kernel_size, size_t conv_dim,
              float rms_eps)
{
    const std::string pfx = "language_model.model.layers." +
                             std::to_string(layer_idx) + ".linear_attn.";

    // Input projections
    auto qkv = mlx_lin(x, pfx + "in_proj_qkv", wm, gs);  // [1, conv_dim]
    auto z   = mlx_lin(x, pfx + "in_proj_z",   wm, gs);  // [1, Hv*Dv]
    auto b   = mlx_lin(x, pfx + "in_proj_b",   wm, gs);  // [1, Hv]
    auto a   = mlx_lin(x, pfx + "in_proj_a",   wm, gs);  // [1, Hv]

    const mx::array& A_log   = wm.at(pfx + "A_log");         // [Hv]
    const mx::array& dt_bias = wm.at(pfx + "dt_bias");       // [Hv]
    const mx::array& conv1d_w = wm.at(pfx + "conv1d.weight"); // [conv_dim, kernel_size, 1]
    const mx::array& norm_w  = wm.at(pfx + "norm.weight");   // [Dv]

    // Causal conv: concat [conv_state, qkv] along time → [kernel_size, conv_dim]
    auto conv_input = mx::concatenate({conv_state, qkv}, 0);

    // New conv state = last (kernel_size-1) rows: take conv_state[1:] + qkv.
    // Uses take + concat to avoid mx::slice (not supported in shapeless compile).
    auto tail_idx = mx::arange(1, (int)(kernel_size - 1), 1, mx::int32);  // [ks-2]
    auto conv_tail = mx::take(conv_state, tail_idx, 0);                    // [ks-2, conv_dim]
    auto new_conv = mx::concatenate({conv_tail, qkv}, 0);                  // [ks-1, conv_dim]

    // Depthwise conv1d: [kernel_size, conv_dim] → [1, 1, conv_dim] → squeeze → [1, conv_dim]
    auto ci4 = mx::reshape(conv_input, {1, (int)kernel_size, (int)conv_dim});
    auto conv_raw = mx::squeeze(mx::conv1d(ci4, conv1d_w, 1, 0, 1, (int)conv_dim), 0);
    // conv_raw: [1, conv_dim]
    auto conv_out = mx::sigmoid(conv_raw) * conv_raw;  // SiLU: x * sigmoid(x)

    // Split conv_out into Q [1,key_dim], K [1,key_dim], V [1,val_dim].
    // Use take with arange indices to avoid mx::slice (not shapeless-compatible).
    const size_t key_dim = Hk * Dk;
    const size_t val_dim = Hv * Dv;
    auto q_idx  = mx::arange(0,                (int)key_dim,       1, mx::int32);
    auto k_idx  = mx::arange((int)key_dim,     (int)(2*key_dim),   1, mx::int32);
    auto v_idx  = mx::arange((int)(2*key_dim), (int)conv_dim,      1, mx::int32);
    auto q_flat = mx::take(conv_out, q_idx, 1);  // [1, key_dim]
    auto k_flat = mx::take(conv_out, k_idx, 1);  // [1, key_dim]
    auto v_flat = mx::take(conv_out, v_idx, 1);  // [1, val_dim]

    // Reshape to head format
    auto q_3d = mx::reshape(q_flat, {1, (int)Hk, (int)Dk});
    auto k_3d = mx::reshape(k_flat, {1, (int)Hk, (int)Dk});
    auto v_3d = mx::reshape(v_flat, {1, (int)Hv, (int)Dv});
    auto z_3d = mx::reshape(z,      {1, (int)Hv, (int)Dv});

    // Per-head RMSNorm on Q and K (unit weight = ones)
    mx::array ones_Dk = mx::ones({(int)Dk}, mx::float32);
    {
        auto f = mx::reshape(q_3d, {(int)Hk, (int)Dk});
        q_3d = mx::reshape(mx::fast::rms_norm(f, ones_Dk, 1e-6f), {1, (int)Hk, (int)Dk});
    }
    {
        auto f = mx::reshape(k_3d, {(int)Hk, (int)Dk});
        k_3d = mx::reshape(mx::fast::rms_norm(f, ones_Dk, 1e-6f), {1, (int)Hk, (int)Dk});
    }

    // Scale: q *= 1/Dk, k *= 1/sqrt(Dk)
    float inv_sqrt_Dk = 1.0f / std::sqrt(static_cast<float>(Dk));
    float q_scale_val = inv_sqrt_Dk * inv_sqrt_Dk;  // = 1/Dk
    q_3d = q_3d * mx::array(q_scale_val);
    k_3d = k_3d * mx::array(inv_sqrt_Dk);

    // Compute decay g and input gate beta for this token
    // g = exp(-exp(A_log) * softplus(a + dt_bias))  [1, Hv]
    auto a_biased = a + dt_bias;
    auto sp       = mx::log(mx::exp(a_biased) + mx::array(1.0f));  // softplus
    auto g_seq    = mx::exp(-(mx::exp(A_log) * sp));               // [1, Hv] float32
    auto beta_seq = mx::astype(mx::sigmoid(b), mx::float32);       // [1, Hv] float32

    // ── Fused GatedDeltaNet recurrence (single Metal dispatch) ───────────────
    // q_3d:[1,Hk,Dk], k_3d:[1,Hk,Dk], v_3d:[1,Hv,Dv]  (bfloat16)
    // g_seq:[1,Hv], beta_seq:[1,Hv]                     (float32)
    // rec_state:[Hv,Dv,Dk]                              (float32)
    // Kernel handles GQA repeat internally via hk_idx = hv_idx / (Hv/Hk).
    using TA = mx::fast::TemplateArg;
    auto gd = gated_delta_kernel()(
        {q_3d, k_3d, v_3d, g_seq, beta_seq, rec_state},
        {mx::Shape{(int)Hv, (int)Dv}, mx::Shape{(int)Hv, (int)Dv, (int)Dk}},
        {mx::bfloat16, mx::float32},
        {32, (int)Dv, (int)Hv},
        {32, 4, 1},
        {{"InT", TA{mx::bfloat16}}, {"StT", TA{mx::float32}},
         {"Dk",  TA{(int)Dk}},      {"Dv",  TA{(int)Dv}},
         {"Hk",  TA{(int)Hk}},      {"Hv",  TA{(int)Hv}}},
        std::nullopt, false, mx::Device::gpu);

    auto y_t  = gd[0];   // [Hv, Dv]
    auto h_new = gd[1];  // [Hv, Dv, Dk]

    // RMSNorm + silu gate
    auto out_normed = mx::fast::rms_norm(y_t, norm_w, rms_eps);             // [Hv, Dv]
    auto gated = mx::reshape(out_normed, {1, (int)Hv, (int)Dv}) *
                 mx::sigmoid(z_3d) * z_3d;  // silu(z) = z * sigmoid(z)
    auto gated_flat = mx::reshape(gated, {1, (int)(Hv * Dv)});
    auto out = mlx_lin(gated_flat, pfx + "out_proj", wm, gs);

    return {out, new_conv, h_new};
}

// MoE MLP decode step (seq=1, all-GPU routing).
static mx::array mlx_moe_mlp_step(const mx::array& x, int layer_idx,
                                    const WM& wm, int gs,
                                    size_t num_experts, size_t top_k)
{
    const std::string pfx = "language_model.model.layers." +
                             std::to_string(layer_idx) + ".mlp.";
    const std::string sw  = pfx + "switch_mlp.";

    // Router logits [1, num_experts] → [num_experts]
    auto gate_logits = mlx_lin(x, pfx + "gate", wm, gs);
    auto logits_1d   = mx::reshape(gate_logits, {(int)num_experts});

    // Top-k expert indices (GPU-side): argsort descending, take first k
    auto sorted_idx = mx::argsort(mx::negative(logits_1d), 0);
    auto range_arr  = mx::arange(0, (int)top_k, 1, mx::int32);
    auto topk_idx   = mx::take(sorted_idx, range_arr, 0);  // [top_k]

    // Routing scores: softmax over top-k logits, shape [1, top_k]
    auto topk_logits = mx::take(logits_1d, topk_idx, 0);
    auto scores      = mx::reshape(mx::softmax(topk_logits, -1), {1, (int)top_k});

    // Gather top-k slices from 3-D expert weight banks [E, out, in_p]
    const mx::array& gw3 = wm.at(sw + "gate_proj.weight");
    const mx::array& gs3 = wm.at(sw + "gate_proj.scales");
    const mx::array& uw3 = wm.at(sw + "up_proj.weight");
    const mx::array& us3 = wm.at(sw + "up_proj.scales");
    const mx::array& dw3 = wm.at(sw + "down_proj.weight");
    const mx::array& ds3 = wm.at(sw + "down_proj.scales");

    int bits = mlx_bits(gw3, gs3, gs);
    const size_t k       = top_k;
    const size_t out_e   = gw3.shape()[1];
    const size_t in_p    = gw3.shape()[2];
    const size_t gss     = gs3.shape()[2];
    const size_t down_out = dw3.shape()[1];
    const size_t int_p    = dw3.shape()[2];
    const size_t dss      = ds3.shape()[2];

    // Batched gate + up projections (reshape gathered weights to 2D, one matmul each)
    auto gw_k = mx::take(gw3, topk_idx, 0);  // [k, out_e, in_p]
    auto gs_k = mx::take(gs3, topk_idx, 0);
    auto uw_k = mx::take(uw3, topk_idx, 0);
    auto us_k = mx::take(us3, topk_idx, 0);
    auto dw_k = mx::take(dw3, topk_idx, 0);
    auto ds_k = mx::take(ds3, topk_idx, 0);

    auto gw2 = mx::reshape(gw_k, {(int)(k*out_e), (int)in_p});
    auto gs2 = mx::reshape(gs_k, {(int)(k*out_e), (int)gss});
    auto uw2 = mx::reshape(uw_k, {(int)(k*out_e), (int)in_p});
    auto us2 = mx::reshape(us_k, {(int)(k*out_e), (int)gss});

    std::optional<mx::array> gb2, ub2;
    {
        auto gb_it = wm.find(sw + "gate_proj.biases");
        if (gb_it != wm.end()) {
            auto gb_k = mx::take(gb_it->second, topk_idx, 0);
            gb2 = mx::reshape(gb_k, {(int)(k*out_e), (int)gb_it->second.shape()[2]});
        }
    }
    {
        auto ub_it = wm.find(sw + "up_proj.biases");
        if (ub_it != wm.end()) {
            auto ub_k = mx::take(ub_it->second, topk_idx, 0);
            ub2 = mx::reshape(ub_k, {(int)(k*out_e), (int)ub_it->second.shape()[2]});
        }
    }

    auto gate_k = mx::quantized_matmul(x, gw2, gs2, gb2, true, gs, bits, "affine");
    auto up_k   = mx::quantized_matmul(x, uw2, us2, ub2, true, gs, bits, "affine");

    auto gate_ke = mx::reshape(gate_k, {(int)k, (int)out_e});
    auto up_ke   = mx::reshape(up_k,   {(int)k, (int)out_e});
    auto h_ke    = mx::sigmoid(gate_ke) * gate_ke * up_ke;  // SiLU(gate) * up

    // Batched down projection: [k, 1, out_e] @ [k, down_out, int_p]^T → [k, 1, down_out]
    // mx::quantized_matmul supports x.ndim()>2 via broadcast_arrays (ops.cpp:4516).
    std::optional<mx::array> db_k;
    {
        auto db_it = wm.find(sw + "down_proj.biases");
        if (db_it != wm.end())
            db_k = mx::take(db_it->second, topk_idx, 0);  // [k, down_out, db_sz]
    }
    int d_bits = mlx_bits(dw_k, ds_k, gs);
    auto h_k3    = mx::reshape(h_ke, {(int)k, 1, (int)out_e});
    auto down_k3 = mx::quantized_matmul(h_k3, dw_k, ds_k, db_k, true, gs, d_bits, "affine");
    auto down_2d = mx::reshape(down_k3, {(int)k, (int)down_out});
    auto switch_out = mx::matmul(scores, down_2d);              // [1, k] @ [k, down_out] = [1, hidden]

    // Shared expert (always active SwiGLU MLP)
    auto se_g   = mlx_lin(x, pfx + "shared_expert.gate_proj", wm, gs);
    auto se_u   = mlx_lin(x, pfx + "shared_expert.up_proj",   wm, gs);
    auto se_h   = mx::sigmoid(se_g) * se_g * se_u;  // SwiGLU
    auto se_out = mlx_lin(se_h, pfx + "shared_expert.down_proj", wm, gs);

    // Shared expert gate: scalar sigmoid applied to shared expert output
    auto seg     = mlx_lin(x, pfx + "shared_expert_gate", wm, gs);  // [1, 1] or [1, hidden]
    auto se_gated = se_out * mx::sigmoid(seg);

    return switch_out + se_gated;
}

} // anonymous namespace

// ── initialize_mlx_state ─────────────────────────────────────────────────────

void Qwen3MoeModel::initialize_mlx_state() {
    mlx_state_.emplace();
    auto& st = *mlx_state_;

    const int n = (int)config_.num_hidden_layers;
    for (int i = 0; i < n; ++i) {
        bool is_ssm = (i + 1) % 4 != 0;
        if (is_ssm) {
            if (i < (int)ssm_cache_.size() && ssm_cache_[i].valid) {
                st.ssm_conv.push_back(ssm_cache_[i].conv_state->to_mlx());
                st.ssm_rec.push_back(ssm_cache_[i].rec_state->to_mlx());
            } else {
                // Prefill must have run first; this path should not be reached.
                size_t Hv = config_.linear_num_value_heads.value_or(32);
                size_t Dv = config_.linear_value_head_dim.value_or(128);
                size_t Hk = config_.linear_num_key_heads.value_or(Hv);
                size_t Dk = config_.linear_key_head_dim.value_or(128);
                size_t ks = config_.linear_conv_kernel_dim.value_or(4);
                size_t cd = Hk*Dk*2 + Hv*Dv;
                st.ssm_conv.push_back(mx::zeros({(int)(ks-1),(int)cd}, mx::float32));
                st.ssm_rec.push_back(mx::zeros({(int)Hv,(int)Dv,(int)Dk}, mx::float32));
            }
        } else {
            size_t nkv = config_.num_key_value_heads;
            size_t hd  = config_.effective_head_dim();
            if (i < (int)kv_cache_.size() && kv_cache_[i].valid) {
                auto pk = kv_cache_[i].keys->to_mlx();
                auto pv = kv_cache_[i].values->to_mlx();
                mx::eval(pk); mx::eval(pv);
                st.kv_keys.push_back(std::move(pk));
                st.kv_vals.push_back(std::move(pv));
            } else {
                // Empty cache — prefill produces the first K/V on first decode step.
                st.kv_keys.push_back(mx::zeros({(int)nkv, 0, (int)hd}, mx::bfloat16));
                st.kv_vals.push_back(mx::zeros({(int)nkv, 0, (int)hd}, mx::bfloat16));
            }
        }
    }
}

// ── build_mlx_compile_fn ─────────────────────────────────────────────────────

void Qwen3MoeModel::build_mlx_compile_fn() {
    // Snapshot all weights as mx::array (reference-counted, no data copy)
    WM wm;
    wm.reserve(weights_.size());
    for (auto& [name, tensor] : weights_)
        wm.emplace(name, tensor.to_mlx());

    // Pre-materialise the embedding matrix so compile treats it as a constant
    mx::array embed_mat = dequantized_embed_tokens_
        ? dequantized_embed_tokens_->to_mlx()
        : wm.at("language_model.model.embed_tokens.weight");
    mx::eval(embed_mat);

    // Config snapshot
    const int    n_layers   = (int)config_.num_hidden_layers;
    const size_t n_heads    = config_.num_attention_heads;
    const size_t n_kv       = config_.num_key_value_heads;
    const size_t head_dim   = config_.effective_head_dim();
    const size_t hidden     = config_.hidden_size;
    const size_t vocab_size = config_.vocab_size;
    const float  rms_eps    = config_.rms_norm_eps;
    const float  rope_theta = config_.rope_theta;
    const int    rope_dims  = (int)(head_dim * config_.partial_rotary_factor.value_or(0.25f));
    const int    gs         = config_.quantization ? config_.quantization->group_size : 64;
    const size_t num_experts = config_.num_experts.value_or(256);
    const size_t top_k       = config_.num_experts_per_tok.value_or(8);
    const size_t Hv = config_.linear_num_value_heads.value_or(32);
    const size_t Dv = config_.linear_value_head_dim.value_or(128);
    const size_t Hk = config_.linear_num_key_heads.value_or(Hv);
    const size_t Dk = config_.linear_key_head_dim.value_or(128);
    const size_t ks = config_.linear_conv_kernel_dim.value_or(4);
    const size_t cd = Hk*Dk*2 + Hv*Dv;

    int n_full = 0, n_ssm = 0;
    for (int i = 0; i < n_layers; ++i)
        ((i+1) % 4 == 0) ? ++n_full : ++n_ssm;

    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> fn =
        [wm          = std::move(wm),
         embed_mat   = std::move(embed_mat),
         n_layers, n_full, n_ssm,
         n_heads, n_kv, head_dim, hidden, vocab_size,
         rms_eps, rope_theta, rope_dims, gs,
         num_experts, top_k,
         Hv, Dv, Hk, Dk, ks, cd]
        (const std::vector<mx::array>& inputs) mutable -> std::vector<mx::array>
    {
        // ── Unpack inputs ────────────────────────────────────────────────────
        // [0]: token_id  [1]  int32
        // [1 .. 2*n_full]: kv_k_fi, kv_v_fi
        // [1+2*n_full .. 1+2*n_full+2*n_ssm]: ssm_conv_li, ssm_rec_li
        // [last]: pos  scalar int32
        const mx::array& token_id = inputs[0];

        // Use reserve+push_back — mx::array has no default constructor
        std::vector<mx::array> kv_k, kv_v;
        kv_k.reserve(n_full); kv_v.reserve(n_full);
        for (int fi = 0; fi < n_full; ++fi) {
            kv_k.push_back(inputs[1 + 2*fi]);
            kv_v.push_back(inputs[1 + 2*fi + 1]);
        }
        std::vector<mx::array> ssm_conv, ssm_rec;
        ssm_conv.reserve(n_ssm); ssm_rec.reserve(n_ssm);
        for (int li = 0; li < n_ssm; ++li) {
            ssm_conv.push_back(inputs[1 + 2*n_full + 2*li]);
            ssm_rec.push_back(inputs[1 + 2*n_full + 2*li + 1]);
        }
        const mx::array& pos = inputs[1 + 2*n_full + 2*n_ssm];

        // ── Embedding ────────────────────────────────────────────────────────
        mx::array hidden_arr = mx::take(embed_mat, token_id, 0);   // [1, hidden]
        hidden_arr = mx::reshape(hidden_arr, {1, (int)hidden});

        // ── Layer loop ───────────────────────────────────────────────────────
        // Output state vectors — filled via push_back in layer order
        std::vector<mx::array> out_kv_k, out_kv_v;
        std::vector<mx::array> out_ssm_conv, out_ssm_rec;
        out_kv_k.reserve(n_full);    out_kv_v.reserve(n_full);
        out_ssm_conv.reserve(n_ssm); out_ssm_rec.reserve(n_ssm);
        int fi = 0, li = 0;

        for (int i = 0; i < n_layers; ++i) {
            const std::string lpfx = "language_model.model.layers." +
                                      std::to_string(i) + ".";
            bool is_ssm_layer = (i + 1) % 4 != 0;

            // Pre-attention RMSNorm
            auto normed = mx::fast::rms_norm(
                hidden_arr, wm.at(lpfx + "input_layernorm.weight"), rms_eps);

            mx::array attn_out = [&]() -> mx::array {
                if (is_ssm_layer) {
                    auto [y, nc, nr] = mlx_ssm_step(
                        normed, i, ssm_conv[li], ssm_rec[li],
                        wm, gs, Hv, Dv, Hk, Dk, ks, cd, rms_eps);
                    out_ssm_conv.push_back(nc);
                    out_ssm_rec.push_back(nr);
                    ++li;
                    return y;
                } else {
                    auto [y, nk, nv] = mlx_full_attn_step(
                        normed, i, kv_k[fi], kv_v[fi], pos,
                        wm, gs, n_heads, n_kv, head_dim,
                        rope_theta, rope_dims, rms_eps);
                    out_kv_k.push_back(nk);
                    out_kv_v.push_back(nv);
                    ++fi;
                    return y;
                }
            }();

            hidden_arr = hidden_arr + attn_out;

            // Post-attention RMSNorm + MoE MLP
            auto normed2 = mx::fast::rms_norm(
                hidden_arr, wm.at(lpfx + "post_attention_layernorm.weight"), rms_eps);
            hidden_arr = hidden_arr + mlx_moe_mlp_step(
                normed2, i, wm, gs, num_experts, top_k);
        }

        // ── Final norm + LM head ─────────────────────────────────────────────
        hidden_arr = mx::fast::rms_norm(
            hidden_arr, wm.at("language_model.model.norm.weight"), rms_eps);
        hidden_arr = mlx_lin(hidden_arr, "language_model.lm_head", wm, gs);
        mx::array logits = mx::reshape(hidden_arr, {(int)vocab_size});

        // ── Pack outputs ─────────────────────────────────────────────────────
        std::vector<mx::array> outputs;
        outputs.reserve(1 + 2*n_full + 2*n_ssm);
        outputs.push_back(logits);
        for (int f = 0; f < n_full; ++f) {
            outputs.push_back(out_kv_k[f]);
            outputs.push_back(out_kv_v[f]);
        }
        for (int l = 0; l < n_ssm; ++l) {
            outputs.push_back(out_ssm_conv[l]);
            outputs.push_back(out_ssm_rec[l]);
        }
        return outputs;
    };

    mlx_state_->compiled_fn = std::move(fn);
    mlx_state_->fn_ready    = true;
}

#endif // MLX_BACKEND_ENABLED

} // namespace compute
