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

    auto result = forward_impl({token_id}, static_cast<int>(cache_position_), &kv_cache_);
    if (result) ++cache_position_;
    return result;
}

void Qwen3MoeModel::reset_cache() {
    kv_cache_.clear();
    ssm_cache_.clear();
    cache_position_ = 0;
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

} // namespace compute
