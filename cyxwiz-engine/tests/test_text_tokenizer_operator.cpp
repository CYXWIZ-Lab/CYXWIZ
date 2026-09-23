#include "../src/core/node_executors/text_tokenizer_operator.h"

#include <arrow/api.h>
#include <parquet/arrow/reader.h>
#include <arrow/io/file.h>
#include <cyxwiz/tokenizer.h>
#include <cstdlib>
#include <sstream>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::shared_ptr<arrow::Array> FinishStringArray(
    const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    for (const auto& value : values) {
        auto st = builder.Append(value);
        Check(st.ok(), st.ToString());
    }
    std::shared_ptr<arrow::Array> array;
    auto st = builder.Finish(&array);
    Check(st.ok(), st.ToString());
    return array;
}

float ReadFloatValue(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& column_name,
    int64_t row) {

    auto column = table->GetColumnByName(column_name);
    Check(column != nullptr, "missing float column " + column_name);
    Check(column->num_chunks() == 1, "expected one float chunk");
    auto array = std::static_pointer_cast<arrow::FloatArray>(column->chunk(0));
    return array->Value(row);
}

std::string ReadStringValue(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& column_name,
    int64_t row) {

    auto column = table->GetColumnByName(column_name);
    Check(column != nullptr, "missing string column " + column_name);
    Check(column->num_chunks() == 1, "expected one string chunk");
    auto array = std::static_pointer_cast<arrow::StringArray>(column->chunk(0));
    return array->GetString(row);
}

bool ReadBoolValue(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& column_name,
    int64_t row) {

    auto column = table->GetColumnByName(column_name);
    Check(column != nullptr, "missing bool column " + column_name);
    Check(column->num_chunks() == 1, "expected one bool chunk");
    auto array = std::static_pointer_cast<arrow::BooleanArray>(column->chunk(0));
    return array->Value(row);
}

std::shared_ptr<arrow::Table> ReadParquetTable(const std::filesystem::path& path) {
    auto maybe_file = arrow::io::ReadableFile::Open(path.string());
    Check(maybe_file.ok(), maybe_file.status().ToString());
    auto maybe_reader = parquet::arrow::OpenFile(*maybe_file, arrow::default_memory_pool());
    Check(maybe_reader.ok(), maybe_reader.status().ToString());
    std::shared_ptr<arrow::Table> table;
    auto status = (*maybe_reader)->ReadTable(&table);
    Check(status.ok(), status.ToString());
    return table;
}

void RunOptionalExternalRoundTrip() {
    const char* parquet_env = std::getenv("CYXWIZ_TEXT_TOKENIZER_ROUNDTRIP_PARQUETS");
    const char* vocab_env = std::getenv("CYXWIZ_TEXT_TOKENIZER_ROUNDTRIP_VOCAB");
    if (!parquet_env || !vocab_env || std::string(parquet_env).empty() ||
        std::string(vocab_env).empty()) {
        return;
    }

    const std::string text_col = std::getenv("CYXWIZ_TEXT_TOKENIZER_ROUNDTRIP_TEXT_COL")
        ? std::getenv("CYXWIZ_TEXT_TOKENIZER_ROUNDTRIP_TEXT_COL")
        : "text";
    std::stringstream paths(parquet_env);
    std::string path;
    int checked_files = 0;
    int64_t checked_rows = 0;
    while (std::getline(paths, path, ';')) {
        if (path.empty()) continue;
        auto table = ReadParquetTable(path);
        cyxwiz::TextTokenizerOperator op;
        std::map<std::string, std::string> params = {
            {"text_col", text_col},
            {"tokenizer_type", "3"},
            {"lowercase", "false"},
            {"max_length", "1"},
            {"min_word_freq", "1"},
            {"max_vocab_size", "8192"},
            {"vocab_file", vocab_env},
            {"vocab_build_if_missing", "false"},
            {"output_mode", "roundtrip"},
        };
        std::string error;
        Check(op.Configure(params, error), error);
        auto result = op.Apply(table);
        Check(result.ok(), result.status().ToString());
        auto output = result.ValueOrDie();
        Check(output->num_rows() == table->num_rows(), "external roundtrip row count mismatch");
        auto ok_col = output->GetColumnByName("roundtrip_ok");
        Check(ok_col != nullptr, "external roundtrip missing roundtrip_ok");
        for (int64_t row = 0; row < output->num_rows(); ++row) {
            Check(ReadBoolValue(output, "roundtrip_ok", row),
                  "external ByteBPE roundtrip mismatch at row " + std::to_string(row) +
                  " in " + path);
        }
        Check(op.GetLastVocabSize() == 8192, "external ByteBPE vocab size should be 8192");
        ++checked_files;
        checked_rows += output->num_rows();
    }
    Check(checked_files > 0, "external roundtrip env listed no parquet files");
    std::cout << "External ByteBPE roundtrip checked " << checked_rows
              << " rows from " << checked_files << " file(s)\n";
}

} // namespace

int main() {
    auto text = FinishStringArray({
        "Small text sample",
        "Another small sample",
        "Text pipelines should tokenize",
    });
    auto label = FinishStringArray({"positive", "positive", "negative"});

    auto schema = arrow::schema({
        arrow::field("text", arrow::utf8()),
        arrow::field("label", arrow::utf8()),
    });
    auto input = arrow::Table::Make(schema, {text, label}, 3);

    cyxwiz::TextTokenizerOperator op;
    std::map<std::string, std::string> params = {
        {"text_col", "text"},
        {"label_col", "label"},
        {"tokenizer_type", "1"},
        {"max_length", "4"},
        {"lowercase", "true"},
        {"min_word_freq", "1"},
        {"max_vocab_size", "100"},
    };

    std::string error;
    Check(op.Configure(params, error), error);

    std::vector<cyxwiz::PipelineOperatorProgress> progress_events;
    op.SetProgressCallback(
        [&](const cyxwiz::PipelineOperatorProgress& event) {
            progress_events.push_back(event);
        });
    auto result = op.Apply(input);
    Check(result.ok(), result.status().ToString());
    Check(!progress_events.empty(),
          "TextTokenizer should emit materialization progress events");
    Check(progress_events.front().stage == "TextTokenizer memory preflight",
          "TextTokenizer first progress event should be memory preflight");
    Check(progress_events.front().status == "running",
          "safe TextTokenizer preflight should stay in running status");
    Check(progress_events.front().memory_risk_level == "safe",
          "safe TextTokenizer preflight should report safe risk");
    Check(progress_events.front().estimated_memory_bytes >
              3ULL * 5ULL * static_cast<uint64_t>(sizeof(float)),
          "TextTokenizer preflight should include peak allocation overhead");
    Check(progress_events.front().message.find("Suggestion:") !=
              std::string::npos,
          "TextTokenizer preflight message should include mitigation guidance");
    Check(op.GetLastVocabSize() > 0,
          "operator should report trained vocabulary size");

    auto output = result.ValueOrDie();
    Check(output != nullptr, "output table is null");
    Check(output->num_rows() == 3, "expected 3 output rows");
    Check(output->num_columns() == 5, "expected 4 token columns plus y");

    for (int i = 0; i < 4; ++i) {
        const std::string name = "tok_" + std::to_string(i);
        auto column = output->GetColumnByName(name);
        Check(column != nullptr, "missing column " + name);
        Check(column->type()->id() == arrow::Type::FLOAT,
              name + " should be float32");
    }

    auto y = output->GetColumnByName("y");
    Check(y != nullptr, "missing y column");
    Check(y->type()->id() == arrow::Type::INT32, "y should be int32");

    const auto vocab_file =
        std::filesystem::temp_directory_path() /
        "cyxwiz_text_tokenizer_operator_vocab.txt";
    {
        std::ofstream out(vocab_file);
        Check(out.good(), "failed to create vocab file");
        out << "[PAD]\n[UNK]\n[BOS]\n[EOS]\nsmall\n";
    }

    cyxwiz::TextTokenizerOperator file_vocab_op;
    params["vocab_file"] = vocab_file.string();
    Check(file_vocab_op.Configure(params, error), error);
    auto file_vocab_result = file_vocab_op.Apply(input);
    Check(file_vocab_result.ok(), file_vocab_result.status().ToString());
    auto file_vocab_output = file_vocab_result.ValueOrDie();
    Check(file_vocab_op.GetLastVocabSize() == 5,
          "operator should report loaded vocabulary size");
    Check(ReadFloatValue(file_vocab_output, "tok_0", 0) == 4.0f,
          "known vocab token should use loaded vocabulary index");
    Check(ReadFloatValue(file_vocab_output, "tok_1", 0) == 1.0f,
          "unknown vocab token should use loaded UNK index");
    std::filesystem::remove(vocab_file);

    cyxwiz::TextTokenizerOperator pad_value_op;
    params.erase("vocab_file");
    params["max_length"] = "5";
    params["pad_value"] = "9";
    Check(pad_value_op.Configure(params, error), error);
    auto pad_value_result = pad_value_op.Apply(input);
    Check(pad_value_result.ok(), pad_value_result.status().ToString());
    auto pad_value_output = pad_value_result.ValueOrDie();
    Check(ReadFloatValue(pad_value_output, "tok_3", 0) == 9.0f,
          "custom pad_value should replace tokenizer PAD ids");
    Check(ReadFloatValue(pad_value_output, "tok_4", 0) == 9.0f,
          "custom pad_value should fill all padded positions");

    const auto missing_vocab_file =
        std::filesystem::temp_directory_path() /
        "cyxwiz_text_tokenizer_operator_built_vocab.txt";
    std::filesystem::remove(missing_vocab_file);

    cyxwiz::TextTokenizerOperator strict_missing_op;
    params["vocab_file"] = missing_vocab_file.string();
    params.erase("vocab_build_if_missing");
    Check(strict_missing_op.Configure(params, error), error);
    auto strict_missing_result = strict_missing_op.Apply(input);
    Check(!strict_missing_result.ok(),
          "missing strict vocab_file should fail without build-if-missing");

    cyxwiz::TextTokenizerOperator build_vocab_op;
    params["vocab_build_if_missing"] = "true";
    Check(build_vocab_op.Configure(params, error), error);
    auto build_vocab_result = build_vocab_op.Apply(input);
    Check(build_vocab_result.ok(), build_vocab_result.status().ToString());
    Check(std::filesystem::exists(missing_vocab_file),
          "build-if-missing should write vocab file");
    Check(build_vocab_op.GetLastVocabSize() > 4,
          "built vocabulary should include corpus tokens");
    std::filesystem::remove(missing_vocab_file);

    cyxwiz::TextTokenizerOperator character_op;
    params.erase("vocab_file");
    params.erase("vocab_build_if_missing");
    params.erase("pad_value");

    cyxwiz::TextTokenizerOperator unsupported_sentencepiece;
    params["tokenizer_type"] = "5";
    Check(!unsupported_sentencepiece.Configure(params, error),
          "SentencePiece BPE should fail clearly when provider is not enabled");
    Check(error.find("SentencePiece tokenizer support is not enabled") != std::string::npos,
          "SentencePiece unsupported error should be actionable");
    params["tokenizer_type"] = "6";
    Check(!unsupported_sentencepiece.Configure(params, error),
          "SentencePiece Unigram should fail clearly when provider is not enabled");
    Check(error.find("SentencePiece tokenizer support is not enabled") != std::string::npos,
          "SentencePiece Unigram unsupported error should be actionable");
    params["tokenizer_type"] = "1";

    params["tokenizer_type"] = "2";
    params["max_length"] = "3";
    params["max_vocab_size"] = "20";
    Check(character_op.Configure(params, error), error);
    auto character_result = character_op.Apply(input);
    Check(character_result.ok(), character_result.status().ToString());
    auto character_output = character_result.ValueOrDie();
    Check(character_op.GetLastVocabSize() == 20,
          "character vocabulary should honor total max vocab size including specials");
    Check(ReadFloatValue(character_output, "tok_0", 0) != 1.0f,
          "character tokenizer should train character tokens, not word-only UNKs");

    // Reversible byte coverage and persistence: Character currently tokenizes bytes.
    cyxwiz::Tokenizer bytes(cyxwiz::TokenizerType::Character);
    bytes.SetLowercase(false);
    bytes.SetPadding(false);
    bytes.SetTruncation(false);
    std::string all_bytes;
    for (int value = 0; value < 256; ++value) all_bytes.push_back(static_cast<char>(value));
    bytes.Train({all_bytes}, 1, -1);
    const auto byte_ids = bytes.Encode(all_bytes);
    Check(bytes.Decode(byte_ids) == all_bytes, "Character decode must preserve all known bytes");
    Check(bytes.Decode(bytes.Encode("ab")) == "ab", "Character decode must not add spaces");
    Check(bytes.Decode(bytes.Encode("")) == "", "Empty character text must round-trip");
    const auto byte_vocab_file = std::filesystem::temp_directory_path() / "cyxwiz_byte_vocab_roundtrip.txt";
    Check(bytes.GetVocabulary().SaveToFile(byte_vocab_file.string()), "save byte vocabulary");
    cyxwiz::Vocabulary restored;
    Check(restored.LoadFromFile(byte_vocab_file.string()), "load byte vocabulary");
    Check(restored.GetWords() == bytes.GetVocabulary().GetWords(), "preserve exact byte token order");
    bytes.SetVocabulary(restored);
    Check(bytes.Encode(all_bytes) == byte_ids, "vocabulary reload must preserve byte IDs");
    std::filesystem::remove(byte_vocab_file);

    cyxwiz::Vocabulary legacy;
    legacy.SetVocabulary({"hello", "world"});
    std::ostringstream legacy_stream;
    Check(legacy.SaveToStream(legacy_stream), "save ordinary vocabulary");
    Check(legacy_stream.str() == "[PAD]\n[UNK]\n[BOS]\n[EOS]\nhello\nworld\n",
          "ordinary vocabularies retain the legacy line format");
    std::istringstream crlf("[PAD]\r\n[UNK]\r\n[BOS]\r\n[EOS]\r\nhello\r\nworld\r\n");
    std::vector<std::string> parsed;
    Check(cyxwiz::Vocabulary::ReadTokens(crlf, parsed) && parsed == legacy.GetWords(),
          "legacy CRLF vocabularies remain readable");
    cyxwiz::Tokenizer words(cyxwiz::TokenizerType::Whitespace);
    words.SetVocabulary(legacy);
    Check(words.Decode({4, 5}) == "hello world", "word decoding retains separators");


    cyxwiz::Vocabulary wp_vocab;
    wp_vocab.SetVocabulary({"play", "##ing", "##ed"});
    cyxwiz::Tokenizer wordpiece(cyxwiz::TokenizerType::WordPiece);
    wordpiece.SetLowercase(true);
    wordpiece.SetPadding(false);
    wordpiece.SetTruncation(false);
    wordpiece.SetVocabulary(wp_vocab);
    const auto wp_ids = wordpiece.Encode("PLAYING played");
    Check(wp_ids.size() == 4, "WordPiece should segment two words into four pieces");
    Check(wp_ids[0] == wp_vocab.WordToIndex("play") &&
              wp_ids[1] == wp_vocab.WordToIndex("##ing") &&
              wp_ids[2] == wp_vocab.WordToIndex("play") &&
              wp_ids[3] == wp_vocab.WordToIndex("##ed"),
          "WordPiece should use greedy prefix plus ## continuation IDs");
    Check(wordpiece.Decode(wp_ids) == "playing played",
          "WordPiece decode should join ## continuation pieces");
    const auto wp_unknown = wordpiece.Encode("player");
    Check(wp_unknown.size() == 1 && wp_unknown[0] == wp_vocab.UnkIndex(),
          "WordPiece should emit UNK when a word cannot be fully segmented");
    for (const std::string malformed : {
         "#cyxwiz-vocabulary-hex-v2\n1\n61\n", // Unknown version
         "#cyxwiz-vocabulary-hex-v1\n2\n61\n", // Truncation
         "#cyxwiz-vocabulary-hex-v1\n1\n6z\n", // Invalid hex
         "#cyxwiz-vocabulary-hex-v1\n1\n6\n",  // Odd hex length
         "#cyxwiz-vocabulary-hex-v1\n2\n61\n61\n", // Duplicate
         "#cyxwiz-vocabulary-hex-v1\n1\n61\n62\n"}) { // Trailing token
        std::istringstream invalid(malformed);
        parsed = {"preserved"};
        Check(!cyxwiz::Vocabulary::ReadTokens(invalid, parsed), "reject malformed byte vocabulary");
        Check(parsed == std::vector<std::string>{"preserved"}, "parse failure preserves destination");
    }
    {
        std::ofstream invalid(byte_vocab_file, std::ios::binary);
        invalid << "#cyxwiz-vocabulary-hex-v1\n2\n61\n";
    }
    const auto unchanged_words = restored.GetWords();
    Check(!restored.LoadFromFile(byte_vocab_file.string()), "reject truncated file");
    Check(restored.GetWords() == unchanged_words, "failed load preserves active vocabulary");
    std::filesystem::remove(byte_vocab_file);

    // Existing Arrow operator: fit/save, then a separate load-only transform.
    auto multiline = arrow::Table::Make(arrow::schema({arrow::field("text", arrow::utf8())}),
        {FinishStringArray({std::string("A\nB\t") + "\xC3\xA9"})});
    std::map<std::string, std::string> byte_params = {
        {"text_col", "text"}, {"tokenizer_type", "2"}, {"lowercase", "false"},
        {"max_length", "8"}, {"min_word_freq", "1"}, {"max_vocab_size", "512"},
        {"vocab_file", byte_vocab_file.string()}, {"vocab_build_if_missing", "true"}};
    cyxwiz::TextTokenizerOperator fit_bytes;
    Check(fit_bytes.Configure(byte_params, error), error);
    auto fitted = fit_bytes.Apply(multiline);
    Check(fitted.ok(), fitted.status().ToString());
    byte_params["vocab_build_if_missing"] = "false";
    cyxwiz::TextTokenizerOperator load_bytes;
    Check(load_bytes.Configure(byte_params, error), error);
    auto loaded = load_bytes.Apply(multiline);
    Check(loaded.ok(), loaded.status().ToString());
    Check(fitted.ValueOrDie()->Equals(*loaded.ValueOrDie()), "fit and reload token rows must match exactly");
    Check(restored.LoadFromFile(byte_vocab_file.string()), "load operator vocabulary for decoding");
    bytes.SetVocabulary(restored);
    std::vector<int> row_ids;
    for (int i = 0; i < 8; ++i) row_ids.push_back(static_cast<int>(ReadFloatValue(loaded.ValueOrDie(), "tok_" + std::to_string(i), 0)));
    Check(bytes.Decode(row_ids) == std::string("A\nB\t") + "\xC3\xA9", "operator output decodes to original UTF-8 text");
    std::filesystem::remove(byte_vocab_file);

    // BPE reuses this same Arrow operator and persisted artifact on load-only runs.
    byte_params["tokenizer_type"] = "3";
    byte_params["vocab_build_if_missing"] = "true";
    byte_params["max_vocab_size"] = "280";
    auto bpe_input = arrow::Table::Make(arrow::schema({arrow::field("text", arrow::utf8())}),
        {FinishStringArray({"abab", "AB\n", "abab"})});
    cyxwiz::TextTokenizerOperator bpe_fit;
    Check(bpe_fit.Configure(byte_params,error),error);
    auto bpe_fitted=bpe_fit.Apply(bpe_input);
    Check(bpe_fitted.ok(),bpe_fitted.status().ToString());
    byte_params["vocab_build_if_missing"] = "false";
    cyxwiz::TextTokenizerOperator bpe_load;
    Check(bpe_load.Configure(byte_params,error),error);
    auto bpe_loaded=bpe_load.Apply(bpe_input);
    Check(bpe_loaded.ok(),bpe_loaded.status().ToString());
    Check(bpe_fitted.ValueOrDie()->Equals(*bpe_loaded.ValueOrDie()),"BPE fit/load Arrow rows equal");
    cyxwiz::Tokenizer bpe(cyxwiz::TokenizerType::ByteBPE);
    Check(bpe.GetVocabulary().LoadFromFile(byte_vocab_file.string()),"BPE saved artifact loads");
    row_ids.clear();
    for(int i=0;i<8;++i) row_ids.push_back(static_cast<int>(ReadFloatValue(bpe_loaded.ValueOrDie(),"tok_"+std::to_string(i),1)));
    Check(bpe.Decode(row_ids)=="AB\n","BPE Arrow rows retain case and newline");
    std::map<std::string, std::string> decode_params = byte_params;
    decode_params["output_mode"] = "decode";
    decode_params["vocab_build_if_missing"] = "false";
    cyxwiz::TextTokenizerOperator bpe_decode_wide;
    Check(bpe_decode_wide.Configure(decode_params,error),error);
    auto bpe_decoded_wide = bpe_decode_wide.Apply(bpe_loaded.ValueOrDie());
    Check(bpe_decoded_wide.ok(), bpe_decoded_wide.status().ToString());
    Check(ReadStringValue(bpe_decoded_wide.ValueOrDie(), "decoded_text", 1) == "AB\n",
          "BPE decode mode should decode wide tok_* columns");
    Check(ReadStringValue(bpe_decoded_wide.ValueOrDie(), "token_ids", 1).find(' ') !=
              std::string::npos,
          "BPE decode mode should report normalized token id text");
    auto id_text_input = arrow::Table::Make(
        arrow::schema({arrow::field("token_ids", arrow::utf8())}),
        {FinishStringArray({ReadStringValue(bpe_decoded_wide.ValueOrDie(), "token_ids", 1)})});
    cyxwiz::TextTokenizerOperator bpe_decode_ids;
    Check(bpe_decode_ids.Configure(decode_params,error),error);
    auto bpe_decoded_ids = bpe_decode_ids.Apply(id_text_input);
    Check(bpe_decoded_ids.ok(), bpe_decoded_ids.status().ToString());
    Check(ReadStringValue(bpe_decoded_ids.ValueOrDie(), "decoded_text", 0) == "AB\n",
          "BPE decode mode should decode token_ids text column");
    auto list_values = std::make_shared<arrow::Int64Builder>();
    arrow::ListBuilder list_builder(arrow::default_memory_pool(), list_values);
    Check(list_builder.Append().ok(), "append token id list row");
    for (int id : row_ids) {
        Check(list_values->Append(static_cast<int64_t>(id)).ok(), "append list token id");
    }
    std::shared_ptr<arrow::Array> list_array;
    Check(list_builder.Finish(&list_array).ok(), "finish token id list array");
    auto id_list_input = arrow::Table::Make(
        arrow::schema({arrow::field("token_ids", arrow::list(arrow::int64()))}),
        {list_array});
    cyxwiz::TextTokenizerOperator bpe_decode_list;
    Check(bpe_decode_list.Configure(decode_params,error),error);
    auto bpe_decoded_list = bpe_decode_list.Apply(id_list_input);
    Check(bpe_decoded_list.ok(), bpe_decoded_list.status().ToString());
    Check(ReadStringValue(bpe_decoded_list.ValueOrDie(), "decoded_text", 0) == "AB\n",
          "BPE decode mode should decode list<int64> token_ids column");
    std::map<std::string, std::string> roundtrip_params = byte_params;
    roundtrip_params["output_mode"] = "roundtrip";
    roundtrip_params["vocab_build_if_missing"] = "false";
    cyxwiz::TextTokenizerOperator bpe_roundtrip;
    Check(bpe_roundtrip.Configure(roundtrip_params,error),error);
    auto bpe_roundtrip_result = bpe_roundtrip.Apply(bpe_input);
    Check(bpe_roundtrip_result.ok(), bpe_roundtrip_result.status().ToString());
    Check(ReadStringValue(bpe_roundtrip_result.ValueOrDie(), "decoded_text", 1) == "AB\n",
          "BPE roundtrip should decode sample text");
    Check(ReadBoolValue(bpe_roundtrip_result.ValueOrDie(), "roundtrip_ok", 1),
          "BPE roundtrip should report exact success for case/newline sample");

    auto long_roundtrip_input = arrow::Table::Make(
        arrow::schema({arrow::field("text", arrow::utf8())}),
        {FinishStringArray({"abab AB\nabab AB"})});
    std::map<std::string, std::string> long_roundtrip_params = byte_params;
    long_roundtrip_params["output_mode"] = "roundtrip";
    long_roundtrip_params["max_length"] = "2";
    long_roundtrip_params["vocab_build_if_missing"] = "false";
    cyxwiz::TextTokenizerOperator long_roundtrip;
    Check(long_roundtrip.Configure(long_roundtrip_params,error),error);
    auto long_roundtrip_result = long_roundtrip.Apply(long_roundtrip_input);
    Check(long_roundtrip_result.ok(), long_roundtrip_result.status().ToString());
    Check(ReadBoolValue(long_roundtrip_result.ValueOrDie(), "roundtrip_ok", 0),
          "BPE roundtrip inspection should not truncate by max_length");

    const auto wp_vocab_file = std::filesystem::temp_directory_path() /
        "cyxwiz_wordpiece_operator_vocab.txt";
    std::filesystem::remove(wp_vocab_file);
    std::map<std::string, std::string> wp_params = {
        {"text_col", "text"}, {"tokenizer_type", "4"}, {"lowercase", "true"},
        {"max_length", "8"}, {"min_word_freq", "1"}, {"max_vocab_size", "64"},
        {"vocab_file", wp_vocab_file.string()}, {"vocab_build_if_missing", "true"}};
    auto wp_input = arrow::Table::Make(arrow::schema({arrow::field("text", arrow::utf8())}),
        {FinishStringArray({"playing played", "playful playing"})});
    cyxwiz::TextTokenizerOperator wp_fit;
    Check(wp_fit.Configure(wp_params, error), error);
    auto wp_fitted = wp_fit.Apply(wp_input);
    Check(wp_fitted.ok(), wp_fitted.status().ToString());
    Check(std::filesystem::exists(wp_vocab_file), "WordPiece fit should save vocabulary artifact");
    wp_params["vocab_build_if_missing"] = "false";
    cyxwiz::TextTokenizerOperator wp_load;
    Check(wp_load.Configure(wp_params, error), error);
    auto wp_loaded = wp_load.Apply(wp_input);
    Check(wp_loaded.ok(), wp_loaded.status().ToString());
    Check(wp_fitted.ValueOrDie()->Equals(*wp_loaded.ValueOrDie()),
          "WordPiece fit/load Arrow rows equal");
    cyxwiz::Tokenizer wp_loaded_tokenizer(cyxwiz::TokenizerType::WordPiece);
    wp_loaded_tokenizer.SetLowercase(true);
    Check(wp_loaded_tokenizer.GetVocabulary().LoadFromFile(wp_vocab_file.string()),
          "WordPiece saved artifact loads");
    row_ids.clear();
    for (int i = 0; i < 8; ++i) {
        row_ids.push_back(static_cast<int>(ReadFloatValue(wp_loaded.ValueOrDie(), "tok_" + std::to_string(i), 0)));
    }
    Check(wp_loaded_tokenizer.Decode(row_ids) == "playing played",
          "WordPiece operator output should decode continuation pieces");
    std::map<std::string, std::string> wp_roundtrip_params = wp_params;
    wp_roundtrip_params["output_mode"] = "roundtrip";
    cyxwiz::TextTokenizerOperator wp_roundtrip;
    Check(wp_roundtrip.Configure(wp_roundtrip_params, error), error);
    auto wp_roundtrip_result = wp_roundtrip.Apply(wp_input);
    Check(wp_roundtrip_result.ok(), wp_roundtrip_result.status().ToString());
    Check(ReadBoolValue(wp_roundtrip_result.ValueOrDie(), "roundtrip_ok", 0),
          "WordPiece roundtrip should preserve lowercase training sample text");
    std::filesystem::remove(wp_vocab_file);
    byte_params["tokenizer_type"] = "1";
    cyxwiz::TextTokenizerOperator wrong_strategy;
    Check(wrong_strategy.Configure(byte_params,error),error);
    Check(!wrong_strategy.Apply(bpe_input).ok(),"operator rejects BPE artifact with word strategy");
    byte_params["tokenizer_type"] = "3";
    byte_params["lowercase"] = "true";
    Check(!bpe_load.Configure(byte_params,error),"operator rejects BPE lowercasing");
    std::filesystem::remove(byte_vocab_file);

    RunOptionalExternalRoundTrip();

    std::cout << "TextTokenizerOperator Arrow path passed\n";
    return 0;
}
