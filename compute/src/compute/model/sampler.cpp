#include "sampler.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_set>

namespace compute {

int Sampler::sample(
    const std::vector<float>& logits,
    const SamplingParams&     params,
    const std::vector<int>&   context_ids)
{
    const size_t vocab_size = logits.size();

    // Greedy — bypass all sampling logic
    if (params.temperature == 0.0f) {
        return static_cast<int>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    std::vector<float> scores(logits.begin(), logits.end());

    // Repetition penalty: apply once per UNIQUE token already generated.
    // Using once-per-occurrence would compound the penalty exponentially
    // (e.g. "Paris" appearing 3× gets ÷1.3³ ≈ ÷2.2), causing output to spiral.
    // HuggingFace RepetitionPenaltyLogitsProcessor uses the same unique-token semantics.
    if (params.rep_penalty != 1.0f) {
        std::unordered_set<int> seen;
        for (int id : context_ids) {
            if (id >= 0 && static_cast<size_t>(id) < vocab_size && seen.insert(id).second) {
                scores[id] = scores[id] > 0.0f
                    ? scores[id] / params.rep_penalty
                    : scores[id] * params.rep_penalty;
            }
        }
    }

    // Temperature
    for (float& s : scores) s /= params.temperature;

    // Top-k: mask everything outside the k highest logits
    if (params.top_k > 0 && params.top_k < vocab_size) {
        std::vector<float> sorted = scores;
        std::nth_element(sorted.begin(), sorted.begin() + params.top_k, sorted.end(),
                         std::greater<float>());
        const float threshold = sorted[params.top_k - 1];
        for (float& s : scores) {
            if (s < threshold) s = -std::numeric_limits<float>::infinity();
        }
    }

    // Softmax
    float max_val = *std::max_element(scores.begin(), scores.end());
    std::vector<float> probs(vocab_size);
    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(scores[i] - max_val);
        sum += probs[i];
    }
    for (float& p : probs) p /= sum;

    // Top-p (nucleus): keep the smallest set of tokens whose cumulative prob >= top_p
    if (params.top_p < 1.0f) {
        std::vector<size_t> idx(vocab_size);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
            return probs[a] > probs[b];
        });
        float cumsum = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            cumsum += probs[idx[i]];
            if (cumsum > params.top_p) {
                for (size_t j = i + 1; j < vocab_size; ++j) probs[idx[j]] = 0.0f;
                break;
            }
        }
        sum = 0.0f;
        for (float p : probs) sum += p;
        for (float& p : probs) p /= sum;
    }

    // Min-p: clip tokens whose prob < min_p * max_prob, then renormalise
    if (params.min_p > 0.0f) {
        float max_prob = *std::max_element(probs.begin(), probs.end());
        float threshold = params.min_p * max_prob;
        for (float& p : probs) if (p < threshold) p = 0.0f;
        sum = 0.0f;
        for (float p : probs) sum += p;
        for (float& p : probs) p /= sum;
    }

    // Multinomial sample
    std::random_device rd;
    std::mt19937 rng(rd());
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

} // namespace compute
