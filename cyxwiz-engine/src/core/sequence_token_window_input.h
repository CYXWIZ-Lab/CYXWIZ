#pragma once

#include "sequence_batcher.h"
#include <cyxwiz/tokenizer.h>
#include <arrow/api.h>
#include <arrow/util/key_value_metadata.h>
#include <charconv>
#include <sstream>

namespace cyxwiz {
// Typed host ingress only. Target shifting and mask construction remain in
// SequenceBatcher; integer IDs must never be fitted as string vocabulary items.
inline bool ReadTokenWindowSamples(const std::shared_ptr<arrow::Table>& table,
                                  int token_column, int configured_context,
                                  std::vector<SequenceSample>& samples,
                                  Vocabulary& vocabulary, int& context,
                                  std::string& error) {
    auto fail=[&](const std::string& message) { error="Token windows: "+message; return false; };
    auto metadata=table->schema()->metadata();
    if(!metadata) return fail("missing schema metadata; use Parquet/Arrow with the saved schema");
    auto get=[&](const char* key)->std::string { auto value=metadata->Get(key); return value.ok()?*value:std::string{}; };
    if(get("cyxwiz.token_windows.version")!="1") return fail("unsupported or missing window version");
    const auto length=get("cyxwiz.token_windows.context");
    auto parsed=std::from_chars(length.data(),length.data()+length.size(),context);
    if(parsed.ec!=std::errc{} || parsed.ptr!=length.data()+length.size() || context<1)
        return fail("invalid context metadata");
    if(configured_context>0 && configured_context!=context) return fail("configured context differs from exported windows; rebuild windows explicitly");
    std::istringstream artifact(get("cyxwiz.token_windows.vocabulary"));
    if(!vocabulary.LoadFromStream(artifact)) return fail("invalid or missing vocabulary artifact");
    const auto strategy=get("cyxwiz.token_windows.tokenizer_type");
    if(strategy!="0" && strategy!="1" && strategy!="2" && strategy!="3" && strategy!="4" && strategy!="5" && strategy!="6") return fail("invalid tokenizer strategy");
    if((strategy=="3")!=vocabulary.IsByteBPE()) return fail("tokenizer strategy/artifact mismatch");
    if(strategy=="4" && vocabulary.IsByteBPE()) return fail("tokenizer strategy/artifact mismatch");
    if(vocabulary.PadIndex()<0 || static_cast<size_t>(vocabulary.PadIndex())>=vocabulary.Size()) return fail("invalid PAD ID");
    auto column=table->column(token_column);
    auto type=std::static_pointer_cast<arrow::ListType>(column->type());
    if(type->value_type()->id()!=arrow::Type::INT64) return fail("token_ids must be list<int64>");
    const auto start_col=table->GetColumnByName("__token_start");
    const auto end_col=table->GetColumnByName("__token_end");
    const auto valid_col=table->GetColumnByName("__valid_targets");
    if(!start_col || !end_col || !valid_col || start_col->type()->id()!=arrow::Type::INT64 ||
       end_col->type()->id()!=arrow::Type::INT64 || valid_col->type()->id()!=arrow::Type::INT64)
        return fail("missing int64 window bounds/target counts");
    for(int64_t row=0;row<table->num_rows();++row) {
        const auto item=column->GetScalar(row);
        const auto start=start_col->GetScalar(row), end=end_col->GetScalar(row), valid=valid_col->GetScalar(row);
        if(!item.ok() || !start.ok() || !end.ok() || !valid.ok() || !(*item)->is_valid ||
            !(*start)->is_valid || !(*end)->is_valid || !(*valid)->is_valid) return fail("null or unreadable window");
        const auto list=std::static_pointer_cast<arrow::ListScalar>(*item);
        const auto ids=std::static_pointer_cast<arrow::Int64Array>(list->value);
        const int64_t begin=std::static_pointer_cast<arrow::Int64Scalar>(*start)->value;
        const int64_t finish=std::static_pointer_cast<arrow::Int64Scalar>(*end)->value;
        const int64_t targets=std::static_pointer_cast<arrow::Int64Scalar>(*valid)->value;
        if(ids->length()<2 || ids->length()>static_cast<int64_t>(context)+1 || begin<0 || finish<begin ||
            finish-begin!=ids->length() || targets!=ids->length()-1) return fail("invalid window length, offsets or target count");
        SequenceSample sample;
        sample.word_ids.reserve(static_cast<size_t>(ids->length()));
        for(int64_t i=0;i<ids->length();++i) {
            const int64_t id=ids->Value(i);
            if(ids->IsNull(i) || id<0 || static_cast<uint64_t>(id)>=vocabulary.Size() || id==vocabulary.PadIndex())
                return fail("invalid, null or padded token ID in an unpadded window");
            sample.word_ids.push_back(id);
        }
        samples.push_back(std::move(sample));
    }
    if(samples.empty()) return fail("no supervised windows");
    return true;
}
} // namespace cyxwiz
