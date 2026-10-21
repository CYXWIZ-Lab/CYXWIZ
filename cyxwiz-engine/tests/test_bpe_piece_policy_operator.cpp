#include <catch2/catch_test_macros.hpp>
#include "core/node_executors/text_tokenizer_operator.h"
#include <cyxwiz/tokenizer.h>
#include <arrow/api.h>
#include <filesystem>
#include <sstream>
#include <arrow/util/key_value_metadata.h>

TEST_CASE("Arrow tokenizer fits reloads and rejects conflicting BPE piece policies") {
    const auto path = std::filesystem::temp_directory_path() / "cyxwiz-bpe-policy-regression.vocab";
    struct Cleanup { std::filesystem::path p; ~Cleanup() { std::error_code ec; std::filesystem::remove(p,ec); } } cleanup{path};
    std::filesystem::remove(path);
    arrow::StringBuilder builder;
    REQUIRE(builder.Append(" word word").ok());
    REQUIRE(builder.Append("  word\ttruth\n").ok());
    std::shared_ptr<arrow::Array> array;
    REQUIRE(builder.Finish(&array).ok());
    const auto input=arrow::Table::Make(arrow::schema({arrow::field("text",arrow::utf8())}),{array});
    std::map<std::string,std::string> params{{"text_col","text"},{"tokenizer_type","3"},
        {"lowercase","false"},{"min_word_freq","1"},{"max_vocab_size","300"},
        {"max_length","32"},{"vocab_file",path.string()},{"vocab_build_if_missing","true"},
        {"bpe_piece_policy","leading_space_v2"}};
    std::string error;
    cyxwiz::TextTokenizerOperator fit;
    REQUIRE(fit.Configure(params,error));
    auto fitted=fit.Apply(input);
    INFO(fitted.status().ToString()); REQUIRE(fitted.ok());
    cyxwiz::Vocabulary vocab;
    REQUIRE(vocab.LoadFromFile(path.string()));
    REQUIRE(vocab.GetBPEPiecePolicy()==cyxwiz::ByteBPEPiecePolicy::LeadingSpaceV2);
    cyxwiz::Tokenizer t(cyxwiz::TokenizerType::ByteBPE);
    t.SetVocabulary(vocab); t.SetPadding(false); t.SetTruncation(false);
    REQUIRE(t.Encode(" word").size()==1);
    params["vocab_build_if_missing"]="false";
    cyxwiz::TextTokenizerOperator load;
    REQUIRE(load.Configure(params,error));
    auto loaded=load.Apply(input); REQUIRE(loaded.ok());
    REQUIRE(loaded.ValueOrDie()->Equals(*fitted.ValueOrDie()));
    params.erase("bpe_piece_policy"); // Omitted options must respect artifact policy.
    REQUIRE(load.Configure(params,error));
    loaded=load.Apply(input); REQUIRE(loaded.ok());
    REQUIRE(loaded.ValueOrDie()->Equals(*fitted.ValueOrDie()));
    params["bpe_piece_policy"]="whitespace_v1";
    REQUIRE(load.Configure(params,error));
    loaded=load.Apply(input); REQUIRE_FALSE(loaded.ok());
    REQUIRE(loaded.status().ToString().find("piece policy differs")!=std::string::npos);
    params["output_mode"]="decode";
    REQUIRE(load.Configure(params,error));
    loaded=load.Apply(fitted.ValueOrDie()); REQUIRE_FALSE(loaded.ok());
    params["bpe_piece_policy"]="leading_space_v2";
    REQUIRE(load.Configure(params,error));
    loaded=load.Apply(fitted.ValueOrDie()); REQUIRE(loaded.ok());
    auto text=std::static_pointer_cast<arrow::StringArray>(loaded.ValueOrDie()->GetColumnByName("decoded_text")->chunk(0));
    REQUIRE(text->GetString(0)==" word word");
    REQUIRE(text->GetString(1)=="  word\ttruth\n");
    params["bpe_piece_policy"]="unsupported";
    REQUIRE_FALSE(load.Configure(params,error));
    params["bpe_piece_policy"]="leading_space_v2"; params["tokenizer_type"]="1";
    REQUIRE_FALSE(load.Configure(params,error));
}

TEST_CASE("Arrow BPE starting units fit reload decode and reject explicit mismatch") {
    const auto path = std::filesystem::temp_directory_path() / "cyxwiz-bpe-character-regression.vocab";
    struct Cleanup { std::filesystem::path p; ~Cleanup() { std::error_code ec; std::filesystem::remove(p,ec); } } cleanup{path};
    std::filesystem::remove(path);
    arrow::StringBuilder builder;
    REQUIRE(builder.Append(" caf\xc3\xa9 caf\xc3\xa9").ok());
    std::shared_ptr<arrow::Array> array;
    REQUIRE(builder.Finish(&array).ok());
    auto input = arrow::Table::Make(arrow::schema({arrow::field("text", arrow::utf8())}), {array});
    std::map<std::string,std::string> p{{"text_col","text"},{"tokenizer_type","3"},
        {"max_length","32"},{"max_vocab_size","30"},{"min_word_freq","1"},
        {"bpe_initial_unit","unicode_character"},{"bpe_piece_policy","leading_space_v2"},
        {"vocab_file",path.string()},{"vocab_build_if_missing","true"}};
    std::string error;
    cyxwiz::TextTokenizerOperator op;
    REQUIRE(op.Configure(p,error));
    auto fitted=op.Apply(input); INFO(fitted.status().ToString()); REQUIRE(fitted.ok());
    cyxwiz::Vocabulary vocab;
    REQUIRE(vocab.LoadFromFile(path.string()));
    REQUIRE(vocab.GetBPEInitialUnit() == cyxwiz::BPEInitialUnit::UnicodeCharacter);
    p["vocab_build_if_missing"]="false";
    p.erase("bpe_initial_unit");
    REQUIRE(op.Configure(p,error));
    auto reload=op.Apply(input); REQUIRE(reload.ok());
    REQUIRE(reload.ValueOrDie()->Equals(*fitted.ValueOrDie()));
    p["output_mode"]="decode";
    REQUIRE(op.Configure(p,error));
    auto decoded=op.Apply(fitted.ValueOrDie()); REQUIRE(decoded.ok());
    auto text=std::static_pointer_cast<arrow::StringArray>(decoded.ValueOrDie()->GetColumnByName("decoded_text")->chunk(0));
    REQUIRE(text->GetString(0)==" caf\xc3\xa9 caf\xc3\xa9");
    p["bpe_initial_unit"]="byte"; p["max_vocab_size"]="300";
    REQUIRE(op.Configure(p,error));
    auto mismatch=op.Apply(fitted.ValueOrDie()); REQUIRE_FALSE(mismatch.ok());
    REQUIRE(mismatch.status().ToString().find("initial unit differs")!=std::string::npos);
    p["bpe_initial_unit"]="unicode_character";
    p["output_mode"]="roundtrip";
    REQUIRE(op.Configure(p,error));
    REQUIRE(op.Apply(input).ok());
    arrow::StringBuilder document_builder, split_builder;
    REQUIRE(document_builder.Append("doc1").ok());
    REQUIRE(split_builder.Append("validation").ok());
    std::shared_ptr<arrow::Array> documents, splits;
    REQUIRE(document_builder.Finish(&documents).ok());
    REQUIRE(split_builder.Finish(&splits).ok());
    auto window_input=arrow::Table::Make(arrow::schema({arrow::field("text",arrow::utf8()),
        arrow::field("document_id",arrow::utf8()),arrow::field("split",arrow::utf8())}), {array,documents,splits});
    p["output_mode"]="causal_windows";
    p["document_id_col"]="document_id"; p["split_col"]="split";
    REQUIRE(op.Configure(p,error));
    auto windows=op.Apply(window_input); INFO(windows.status().ToString()); REQUIRE(windows.ok());
    REQUIRE(windows.ValueOrDie()->num_rows()==1);
    auto artifact=windows.ValueOrDie()->schema()->metadata()->Get("cyxwiz.token_windows.vocabulary");
    REQUIRE(artifact.ok());
    std::istringstream serialized(artifact.ValueOrDie());
    cyxwiz::Vocabulary window_vocabulary;
    REQUIRE(window_vocabulary.LoadFromStream(serialized));
    REQUIRE(window_vocabulary.GetBPEInitialUnit()==cyxwiz::BPEInitialUnit::UnicodeCharacter);
    REQUIRE(window_vocabulary.GetWords()==vocab.GetWords());
    p["bpe_initial_unit"]="invalid";
    REQUIRE_FALSE(op.Configure(p,error));
    p["bpe_initial_unit"]="unicode_character"; p.erase("bpe_piece_policy"); p["tokenizer_type"]="1";
    REQUIRE_FALSE(op.Configure(p,error));
}
