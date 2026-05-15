#include <gtest/gtest.h>
#include "compute/model/hf_tokenizer.h"
#include "test_config.h"
#include <filesystem>

namespace fs = std::filesystem;

namespace compute {

// ── Fixture ───────────────────────────────────────────────────────────────────
// All tests require the TinyLlama model on disk; skip automatically if absent.

class HFTokenizerTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        model_dir_ = TINYLLAMA_MODEL_DIR;
        if (!fs::exists(model_dir_)) {
            skip_reason_ = "Model not found: " + model_dir_.string();
            return;
        }
        auto result = HFTokenizer::from_model_dir(model_dir_);
        if (!result) {
            skip_reason_ = "Tokenizer load failed: " + result.error().message;
            return;
        }
        tok_ = std::make_unique<HFTokenizer>(std::move(*result));
    }

    static void TearDownTestSuite() { tok_.reset(); }

    void SetUp() override {
        if (!skip_reason_.empty()) GTEST_SKIP() << skip_reason_;
    }

    static fs::path                    model_dir_;
    static std::string                 skip_reason_;
    static std::unique_ptr<HFTokenizer> tok_;
};

fs::path                    HFTokenizerTest::model_dir_;
std::string                 HFTokenizerTest::skip_reason_;
std::unique_ptr<HFTokenizer> HFTokenizerTest::tok_;

// ── Encode / decode roundtrip ─────────────────────────────────────────────────

TEST_F(HFTokenizerTest, EncodeProducesNonEmptyIds) {
    const auto ids = tok_->encode("Hello, world!", /*add_special_tokens=*/false);
    EXPECT_FALSE(ids.empty());
}

TEST_F(HFTokenizerTest, DecodeRoundtrip) {
    const std::string text = "The capital of France is Paris.";
    const auto ids = tok_->encode(text, /*add_special_tokens=*/false);
    ASSERT_FALSE(ids.empty());
    const std::string decoded = tok_->decode(ids, /*skip_special_tokens=*/true);
    EXPECT_EQ(decoded, text);
}

TEST_F(HFTokenizerTest, AddSpecialTokensInsertsBoS) {
    const auto with    = tok_->encode("Hi", /*add_special_tokens=*/true);
    const auto without = tok_->encode("Hi", /*add_special_tokens=*/false);
    EXPECT_GT(with.size(), without.size());
    // BOS token is prepended
    EXPECT_EQ(with.front(), tok_->bos_token_id());
}

TEST_F(HFTokenizerTest, SkipSpecialTokensStripsBoS) {
    const auto ids = tok_->encode("Hi", /*add_special_tokens=*/true);
    ASSERT_FALSE(ids.empty());
    const std::string with_skip    = tok_->decode(ids, /*skip_special_tokens=*/true);
    const std::string without_skip = tok_->decode(ids, /*skip_special_tokens=*/false);
    // Skipping removes the BOS marker; not-skipping keeps it
    EXPECT_LT(with_skip.size(), without_skip.size());
}

// ── find_token_id / get_token_string ─────────────────────────────────────────

TEST_F(HFTokenizerTest, FindTokenIdForBosToken) {
    const int bos = tok_->bos_token_id();
    ASSERT_GE(bos, 0);
    const std::string bos_str = tok_->get_token_string(bos);
    EXPECT_EQ(tok_->find_token_id(bos_str), bos);
}

TEST_F(HFTokenizerTest, FindTokenIdUnknownReturnsNegativeOne) {
    EXPECT_EQ(tok_->find_token_id("ZZZDEFINITELYNOTINVOCABZZZ"), -1);
}

// ── Metadata ─────────────────────────────────────────────────────────────────

TEST_F(HFTokenizerTest, VocabSizeIsReasonable) {
    EXPECT_GT(tok_->vocab_size(), 100u);
    EXPECT_LT(tok_->vocab_size(), 300000u);
}

TEST_F(HFTokenizerTest, SpecialTokenIdsAreValid) {
    EXPECT_GE(tok_->bos_token_id(), 0);
    EXPECT_GE(tok_->eos_token_id(), 0);
}

// ── Error path ────────────────────────────────────────────────────────────────

TEST(HFTokenizerErrorTest, MissingDirReturnsError) {
    const auto result = HFTokenizer::from_model_dir("/nonexistent/path/to/model");
    EXPECT_FALSE(result.has_value());
}

} // namespace compute
