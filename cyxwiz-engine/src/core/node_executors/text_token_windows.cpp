#include "text_tokenizer_operator.h"
#include "../materialization_memory_guard.h"
#include <cyxwiz/tokenizer.h>
#include <arrow/api.h>
#include <arrow/util/key_value_metadata.h>
#include <filesystem>
#include <algorithm>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace cyxwiz {
arrow::Status TextTokenizerOperator::ValidateWindowInput(const std::shared_ptr<arrow::Table>& input) const {
    if (input->schema()->HasDistinctFieldNames() == false)
        return arrow::Status::Invalid("Token windows require distinct column names");
    for (const auto* name : {"token_ids", "__window_index", "__token_start", "__token_end", "__valid_targets"})
        if (input->GetColumnByName(name)) return arrow::Status::Invalid("Token window output name collision: ", name);
    auto documents = input->GetColumnByName(document_id_col_);
    auto splits = input->GetColumnByName(split_col_);
    auto texts = input->GetColumnByName(text_col_);
    if (!documents || !splits || !texts) return arrow::Status::Invalid("Token windows require text, document ID and split columns");
    auto string_type = [](const auto& col) { return col->type()->id() == arrow::Type::STRING || col->type()->id() == arrow::Type::LARGE_STRING; };
    if (!string_type(documents) || !string_type(splits)) return arrow::Status::Invalid("Document IDs and split labels must be strings");
    const bool fitting = vocab_file_.empty() || !std::filesystem::exists(vocab_file_);
    std::unordered_set<std::string> seen;
    for (int64_t r=0; r<input->num_rows(); ++r) {
        ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        ARROW_ASSIGN_OR_RAISE(auto id, documents->GetScalar(r));
        ARROW_ASSIGN_OR_RAISE(auto split, splits->GetScalar(r));
        ARROW_ASSIGN_OR_RAISE(auto text, texts->GetScalar(r));
        if (!id->is_valid || id->ToString().empty() || !seen.insert(id->ToString()).second)
            return arrow::Status::Invalid("Document IDs must be nonempty, nonnull and unique within the input table");
        if (!text->is_valid) return arrow::Status::Invalid("Null document text is not allowed");
        const auto role=split->ToString();
        if (!split->is_valid || (role!="train" && role!="validation" && role!="dev" && role!="test" && role!="inference"))
            return arrow::Status::Invalid("Unsupported document split label: ", role);
        if (fitting && role!="train") return arrow::Status::Invalid("Fit vocabulary on train documents only; load the saved vocabulary for other splits");
    }
    return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Table>> TextTokenizerOperator::BuildTokenWindows(
    const std::shared_ptr<arrow::Table>& input, const std::vector<std::string>& texts, Tokenizer& tokenizer) {
    tokenizer.SetPadding(false);
    tokenizer.SetTruncation(false);
    tokenizer.SetAddBos(false);
    tokenizer.SetAddEos(true);
    const uint64_t context=static_cast<uint64_t>(max_length_);
    // Conservative preflight: no tokenizer emits more tokens than source bytes
    // plus EOS. Include repeated provenance values and builder growth.
    uint64_t row_bound=0, metadata_bytes=0;
    std::vector<int> retained;
    for (int col=0;col<input->num_columns();++col)
        if (input->field(col)->name()!=text_col_) retained.push_back(col);
    for (int64_t r=0;r<input->num_rows();++r) {
        ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        const uint64_t bytes=texts[static_cast<size_t>(r)].size();
        const uint64_t count=bytes/context+(bytes%context!=0);
        if (!CheckedAddU64(row_bound,count,row_bound)) return arrow::Status::CapacityError("Token window row count overflow");
        for (int col:retained) {
            ARROW_ASSIGN_OR_RAISE(auto value,input->column(col)->GetScalar(r));
            uint64_t repeated=0;
            if (!CheckedMulU64(value->ToString().size()+32,count,repeated) || !CheckedAddU64(metadata_bytes,repeated,metadata_bytes))
                return arrow::Status::CapacityError("Token window provenance size overflow");
        }
    }
    auto estimate=EstimateDenseMaterializationMemory(row_bound,context+6,8);
    uint64_t extra=SaturatingScaleBytes(metadata_bytes,3.0);
    if (!CheckedAddU64(estimate.estimated_peak_bytes,extra,estimate.estimated_peak_bytes))
        return arrow::Status::CapacityError("Token window memory estimate overflow");
    const auto decision=EvaluateMaterializationMemory(estimate,GetMaterializationMemoryContext());
    if (decision.blocked) return arrow::Status::CapacityError("Token windows blocked by materialization memory preflight: ",decision.reason);
    if (row_bound>static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
        return arrow::Status::CapacityError("Token window output exceeds Arrow list offset capacity; partition the input");

    std::vector<std::unique_ptr<arrow::ArrayBuilder>> provenance;
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (int col:retained) {
        ARROW_ASSIGN_OR_RAISE(auto builder,arrow::MakeBuilder(input->field(col)->type()));
        provenance.push_back(std::move(builder)); fields.push_back(input->field(col));
    }
    auto values=std::make_shared<arrow::Int64Builder>();
    // Use Parquet's canonical nested field name for exact schema round trips.
    arrow::ListBuilder ids(arrow::default_memory_pool(),values,arrow::list(arrow::field("element",arrow::int64())));
    arrow::Int64Builder indices,starts,ends,valid;
    int64_t emitted=0, total_ids=0;
    for (size_t r=0;r<texts.size();++r) {
        ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
        if (texts[r].empty()) continue;
        std::vector<int> tokens;
        try { tokens=tokenizer.Encode(texts[r]); }
        catch (const std::exception& e) { ARROW_RETURN_NOT_OK(CheckCancellation(GetName())); return arrow::Status::Invalid(e.what()); }
        std::vector<std::shared_ptr<arrow::Scalar>> source_values;
        for (int col:retained) {
            ARROW_ASSIGN_OR_RAISE(auto value,input->column(col)->GetScalar(static_cast<int64_t>(r)));
            source_values.push_back(std::move(value));
        }
        int64_t window=0;
        for (size_t start=0; start+1<tokens.size();start+=static_cast<size_t>(context),++window) {
            ARROW_RETURN_NOT_OK(CheckCancellation(GetName()));
            const size_t end=std::min(tokens.size(),start+static_cast<size_t>(context)+1);
            if (total_ids>std::numeric_limits<int32_t>::max()-static_cast<int64_t>(end-start))
                return arrow::Status::CapacityError("Token window list offsets exceed int32; partition the input");
            ARROW_RETURN_NOT_OK(ids.Append());
            for(size_t i=start;i<end;++i) {
                if(tokens[i]==tokenizer.GetVocabulary().PadIndex()) return arrow::Status::Invalid("Unpadded token windows cannot contain PAD control tokens");
                ARROW_RETURN_NOT_OK(values->Append(tokens[i]));
            }
            total_ids+=static_cast<int64_t>(end-start);
            for(size_t col=0;col<provenance.size();++col) ARROW_RETURN_NOT_OK(provenance[col]->AppendScalar(*source_values[col]));
            ARROW_RETURN_NOT_OK(indices.Append(window));
            ARROW_RETURN_NOT_OK(starts.Append(static_cast<int64_t>(start)));
            ARROW_RETURN_NOT_OK(ends.Append(static_cast<int64_t>(end)));
            ARROW_RETURN_NOT_OK(valid.Append(static_cast<int64_t>(end-start-1)));
            ++emitted;
        }
        if(progress_callback_ && (r%256==0 || r+1==texts.size())) {
            PipelineOperatorProgress progress;
            progress.stage="Building document token windows"; progress.status="running";
            progress.processed_items=r+1; progress.total_items=texts.size();
            progress.progress=0.4f+0.55f*static_cast<float>(r+1)/static_cast<float>(texts.size());
            progress.estimated_memory_bytes=estimate.estimated_peak_bytes; progress_callback_(progress);
        }
    }
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    for(auto& builder:provenance) { ARROW_ASSIGN_OR_RAISE(auto array,builder->Finish()); arrays.push_back(std::move(array)); }
    auto finish=[&](const char* name,arrow::ArrayBuilder& builder)->arrow::Status {
        ARROW_ASSIGN_OR_RAISE(auto array,builder.Finish());
        fields.push_back(arrow::field(name,array->type(),false)); arrays.push_back(std::move(array)); return arrow::Status::OK();
    };
    ARROW_RETURN_NOT_OK(finish("token_ids",ids));
    ARROW_RETURN_NOT_OK(finish("__window_index",indices));
    ARROW_RETURN_NOT_OK(finish("__token_start",starts));
    ARROW_RETURN_NOT_OK(finish("__token_end",ends));
    ARROW_RETURN_NOT_OK(finish("__valid_targets",valid));
    std::ostringstream artifact;
    if(!tokenizer.GetVocabulary().SaveToStream(artifact)) return arrow::Status::IOError("Cannot serialize token window vocabulary");
    auto metadata=input->schema()->metadata()?input->schema()->metadata()->Copy():std::make_shared<arrow::KeyValueMetadata>();
    ARROW_RETURN_NOT_OK(metadata->Set("cyxwiz.token_windows.version","1"));
    ARROW_RETURN_NOT_OK(metadata->Set("cyxwiz.token_windows.context",std::to_string(max_length_)));
    ARROW_RETURN_NOT_OK(metadata->Set("cyxwiz.token_windows.vocabulary",artifact.str()));
    ARROW_RETURN_NOT_OK(metadata->Set("cyxwiz.token_windows.tokenizer_type",std::to_string(tokenizer_type_)));
    ARROW_RETURN_NOT_OK(metadata->Set("cyxwiz.token_windows.lowercase",tokenizer.GetLowercase()?"true":"false"));
    last_vocab_size_=tokenizer.GetVocabulary().Size();
    return arrow::Table::Make(arrow::schema(fields,metadata),arrays,emitted);
}
} // namespace cyxwiz
