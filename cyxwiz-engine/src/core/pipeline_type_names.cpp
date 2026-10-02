#include "pipeline_type_names.h"

#include "pipeline_runtime_capabilities.h"

namespace cyxwiz {

std::string PipelineTypeName(gui::NodeType type) {
    using gui::NodeType;
    if (const char* runtime_name = ResolvePipelineRuntimeLegacyTypeName(type)) {
        return runtime_name;
    }
    switch (type) {
        // Smart I/O nodes (universal data input/output)
        case NodeType::DataInput: return "DataInput";
        case NodeType::DataOutput: return "DataOutput";
        case NodeType::DataConvert: return "DataConvert";
        case NodeType::CSVFile: return "FileInput";
        case NodeType::FilterRows: return "FilterRows";
        case NodeType::SelectColumns: return "SelectColumns";
        case NodeType::JoinTables: return "Join";
        case NodeType::GroupByAggregate: return "GroupBy";
        case NodeType::SortRows: return "SortRows";
        case NodeType::FillMissingValues: return "FillMissing";
        case NodeType::RemoveDuplicateRows: return "RemoveDuplicates";
        case NodeType::RenameColumns: return "RenameColumns";
        case NodeType::SampleRows: return "SampleRows";
        case NodeType::SQLQuery: return "SQLQuery";
        case NodeType::ParquetFile: return "ParquetInput";
        case NodeType::ExportCSV: return "ExportCSV";
        case NodeType::ExportParquet: return "ExportParquet";
        case NodeType::ExportJSON: return "ExportJSON";
        case NodeType::DescribeStats: return "DescribeStats";
        case NodeType::DecisionTreeClassifier: return "DecisionTreeClassifier";
        case NodeType::RandomForestClassifier: return "RandomForestClassifier";
        case NodeType::GradientBoostingClassifier: return "GradientBoostingClassifier";
        case NodeType::TreeModelPredictor: return "TreeModelPredictor";
        case NodeType::RegressionModelPredictor: return "RegressionModelPredictor";
        // KNIME-style table manipulation nodes
        case NodeType::ExcelFile: return "ExcelInput";
        case NodeType::ExportExcel: return "ExportExcel";
        case NodeType::RowToColumnNames: return "RowToColumnNames";
        case NodeType::TableSplitter: return "TableSplitter";
        case NodeType::CellExtractor: return "CellExtractor";
        case NodeType::CellUpdater: return "CellUpdater";
        case NodeType::TableCropper: return "TableCropper";
        case NodeType::ColumnAppender: return "ColumnAppender";
        case NodeType::RowAppender: return "RowAppender";
        case NodeType::Unpivot: return "Unpivot";
        case NodeType::StringManipulation: return "StringManipulation";
        case NodeType::MathFormula: return "MathFormula";
        case NodeType::RuleEngine: return "RuleEngine";
        default: return "Unknown";
    }
}

}  // namespace cyxwiz
