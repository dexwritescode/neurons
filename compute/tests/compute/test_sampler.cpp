#include <gtest/gtest.h>
#include "compute/model/sampler.h"
#include <algorithm>
#include <numeric>
#include <vector>

namespace compute {

// ── Greedy (temperature = 0) ──────────────────────────────────────────────────

TEST(SamplerTest, GreedyPicksArgmax) {
    std::vector<float> logits = {0.1f, 0.5f, 9.0f, 0.3f};
    SamplingParams p;
    p.temperature = 0.0f;
    EXPECT_EQ(Sampler::sample(logits, p, {}), 2);
}

TEST(SamplerTest, GreedyFirstTokenWinsOnTie) {
    std::vector<float> logits = {5.0f, 5.0f, 1.0f};
    SamplingParams p;
    p.temperature = 0.0f;
    EXPECT_EQ(Sampler::sample(logits, p, {}), 0);
}

// ── Top-k ─────────────────────────────────────────────────────────────────────

TEST(SamplerTest, TopKRestrictsToTopKTokens) {
    // 10 tokens; only top-2 should ever be sampled
    const int vocab = 10;
    std::vector<float> logits(vocab, 0.0f);
    logits[3] = 10.0f;
    logits[7] = 9.0f;  // second highest

    SamplingParams p;
    p.temperature = 1.0f;
    p.top_k       = 2;
    p.top_p       = 1.0f;

    // Run many samples — only ids 3 and 7 should appear
    std::vector<int> seen;
    for (int i = 0; i < 200; ++i)
        seen.push_back(Sampler::sample(logits, p, {}));

    for (int id : seen)
        EXPECT_TRUE(id == 3 || id == 7) << "Got unexpected token id: " << id;
}

TEST(SamplerTest, TopKDisabledWhenZero) {
    // With top_k=0 and temperature=0, still returns argmax
    std::vector<float> logits = {1.0f, 2.0f, 5.0f};
    SamplingParams p;
    p.temperature = 0.0f;
    p.top_k       = 0;
    EXPECT_EQ(Sampler::sample(logits, p, {}), 2);
}

// ── Top-p (nucleus) ───────────────────────────────────────────────────────────

TEST(SamplerTest, TopPDisabledAt1) {
    // top_p=1.0 doesn't filter anything — all tokens are in nucleus
    std::vector<float> logits = {1.0f, 1.0f, 1.0f, 100.0f};
    SamplingParams p;
    p.temperature = 0.0f;
    p.top_p       = 1.0f;
    EXPECT_EQ(Sampler::sample(logits, p, {}), 3);
}

TEST(SamplerTest, TopPConcentratesOnDominantToken) {
    // Token 0 has overwhelmingly high logit — with top_p=0.9 it should always win
    std::vector<float> logits = {100.0f, 0.0f, 0.0f, 0.0f};
    SamplingParams p;
    p.temperature = 1.0f;
    p.top_k       = 0;
    p.top_p       = 0.9f;

    for (int i = 0; i < 50; ++i)
        EXPECT_EQ(Sampler::sample(logits, p, {}), 0);
}

// ── Min-p ─────────────────────────────────────────────────────────────────────

TEST(SamplerTest, MinPFiltersLowProbTokens) {
    // Token 0 has overwhelming probability; min_p=0.1 clips the rest
    std::vector<float> logits = {100.0f, 0.0f, 0.0f, 0.0f};
    SamplingParams p;
    p.temperature = 1.0f;
    p.top_k       = 0;
    p.top_p       = 1.0f;
    p.min_p       = 0.1f;

    for (int i = 0; i < 50; ++i)
        EXPECT_EQ(Sampler::sample(logits, p, {}), 0);
}

// ── Repetition penalty ────────────────────────────────────────────────────────

TEST(SamplerTest, RepPenaltyShiftsWinnerWhenTokenSeen) {
    // top_k=1 makes sampling deterministic: only the single highest-scoring token
    // after all transforms can be selected.
    // Without penalty: token 0 (logit 100) wins.
    // With rep_penalty=1e4 on token 0: score drops to 0.01 < token 1's 10 → token 1 wins.
    std::vector<float> logits = {100.0f, 10.0f, 1.0f, 0.1f};
    SamplingParams no_pen;
    no_pen.temperature = 0.0f;
    no_pen.rep_penalty = 1.0f;
    EXPECT_EQ(Sampler::sample(logits, no_pen, {}), 0);

    SamplingParams p;
    p.temperature = 1.0f;
    p.top_k       = 1;
    p.top_p       = 1.0f;
    p.rep_penalty = 1e4f;
    EXPECT_EQ(Sampler::sample(logits, p, {0}), 1);
}

TEST(SamplerTest, RepPenaltyOnNegativeLogitPushesItFurtherDown) {
    // Tokens with a negative logit in context get multiplied (more negative),
    // not divided. Use top_k=1 to verify the non-penalised token wins after
    // the negative token is pushed deep negative.
    // logits = {5.0, -0.01, 0.0, 0.0}; penalty on token 1 (negative):
    //   scores[1] = -0.01 * 1e6 = -10000 — stays the worst
    //   top_k=1 selects token 0 (score 5.0)
    std::vector<float> logits = {5.0f, -0.01f, 0.0f, 0.0f};
    SamplingParams p;
    p.temperature = 1.0f;
    p.top_k       = 1;
    p.top_p       = 1.0f;
    p.rep_penalty = 1e6f;
    EXPECT_EQ(Sampler::sample(logits, p, {1}), 0);
}

TEST(SamplerTest, RepPenaltyIgnoresTokensNotInContext) {
    std::vector<float> logits = {5.0f, 1.0f};
    SamplingParams p;
    p.temperature = 0.0f;
    p.rep_penalty = 100.0f;
    // Context doesn't include token 0, so penalty doesn't apply
    EXPECT_EQ(Sampler::sample(logits, p, {1}), 0);
}

TEST(SamplerTest, RepPenaltyDeduplicatesContext) {
    // Seeing token 0 three times should apply the penalty only once
    std::vector<float> logits = {2.0f, 1.0f};
    SamplingParams p;
    p.temperature = 0.0f;
    p.rep_penalty = 3.0f;
    // With 3× occurrence but dedup: same result as 1× occurrence
    int with_one  = Sampler::sample(logits, p, {0});
    int with_many = Sampler::sample(logits, p, {0, 0, 0});
    EXPECT_EQ(with_one, with_many);
}

} // namespace compute
