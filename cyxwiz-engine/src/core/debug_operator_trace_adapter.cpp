#include "debug_operator_trace_adapter.h"

namespace cyxwiz {

namespace {

std::vector<size_t> TableShape(const std::shared_ptr<arrow::Table>& table) {
    if (!table) {
        return {};
    }
    return {
        static_cast<size_t>(table->num_rows()),
        static_cast<size_t>(table->num_columns())
    };
}

std::string TableSchemaText(const std::shared_ptr<arrow::Table>& table) {
    if (!table || !table->schema()) {
        return "";
    }
    return table->schema()->ToString();
}

} // namespace

DebugGraphTraceStep DebugOperatorTraceAdapter::BuildStep(
    int node_id,
    const std::string& node_name,
    const std::string& node_type,
    const std::shared_ptr<arrow::Table>& input,
    const std::shared_ptr<arrow::Table>& output,
    float duration_ms) const {
    DebugGraphTraceStep step;
    step.node_id = node_id;
    step.node_name = node_name;
    step.node_type = node_type;
    step.phase = "OperatorTransform";
    step.role = DebugTraceRole::PreprocessingOutput;
    step.input_shape = TableShape(input);
    step.output_shape = TableShape(output);
    step.dtype = "arrow::Table";
    step.backend = "CPU";
    step.status = output ? "ok" : "failed";
    step.duration_ms = duration_ms;
    step.payload["operator"] = node_type;
    step.payload["input_rows"] = input ? input->num_rows() : 0;
    step.payload["input_columns"] = input ? input->num_columns() : 0;
    step.payload["output_rows"] = output ? output->num_rows() : 0;
    step.payload["output_columns"] = output ? output->num_columns() : 0;
    step.payload["input_schema"] = TableSchemaText(input);
    step.payload["output_schema"] = TableSchemaText(output);
    return step;
}

} // namespace cyxwiz
