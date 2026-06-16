#include "../src/core/node_executors/pipeline_operator_factory.h"
#include "../src/core/node_metadata_registry.h"
#include "../src/core/pipeline_runtime_capabilities.h"
#include "../src/gui/data_studio/pipeline_canvas.h"
#include "../src/gui/data_studio/node_registry.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::string TypeId(gui::NodeType type) {
    return std::to_string(static_cast<int>(type));
}

bool HasInput(const cyxwiz::NodeMetadata* meta,
              const std::string& name,
              bool required) {
    if (!meta) return false;
    for (const auto& input : meta->inputs) {
        if (input.name == name && input.required == required) {
            return true;
        }
    }
    return false;
}

bool HasInputType(const cyxwiz::NodeMetadata* meta,
                  const std::string& name,
                  gui::PinType type) {
    if (!meta) return false;
    for (const auto& input : meta->inputs) {
        if (input.name == name && input.type == type) {
            return true;
        }
    }
    return false;
}

bool HasOutputType(const cyxwiz::NodeMetadata* meta,
                   const std::string& name,
                   gui::PinType type) {
    if (!meta) return false;
    for (const auto& output : meta->outputs) {
        if (output.name == name && output.type == type) {
            return true;
        }
    }
    return false;
}

bool HasParameter(const cyxwiz::NodeMetadata* meta,
                  const std::string& name) {
    if (!meta) return false;
    for (const auto& param : meta->parameters) {
        if (param.name == name) {
            return true;
        }
    }
    return false;
}

bool HasEnumValue(const cyxwiz::NodeMetadata* meta,
                  const std::string& param_name,
                  const std::string& enum_value) {
    if (!meta) return false;
    for (const auto& param : meta->parameters) {
        if (param.name != param_name) continue;
        for (const auto& value : param.enum_values) {
            if (value == enum_value) {
                return true;
            }
        }
    }
    return false;
}

bool ContainsString(const std::vector<std::string>& values,
                    const std::string& expected) {
    return std::find(values.begin(), values.end(), expected) != values.end();
}

const cyxwiz::ParameterDefinition* FindParameter(
    const cyxwiz::NodeMetadata* meta,
    const std::string& name) {
    if (!meta) return nullptr;
    for (const auto& param : meta->parameters) {
        if (param.name == name) {
            return &param;
        }
    }
    return nullptr;
}

const cyxwiz::SupportAxisDefinition* FindSupportAxis(
    const cyxwiz::NodeMetadata* meta,
    const std::string& name) {
    if (!meta) return nullptr;
    for (const auto& axis : meta->support_axes) {
        if (axis.name == name) {
            return &axis;
        }
    }
    return nullptr;
}

void CheckSupportAxis(const cyxwiz::NodeMetadata* meta,
                      const std::string& name,
                      const std::string& value,
                      bool supported,
                      const std::string& context) {
    const auto* axis = FindSupportAxis(meta, name);
    Check(axis != nullptr, "missing support axis " + name + ": " + context);
    Check(axis->value == value,
          "support axis " + name + " has wrong value for " + context +
              ": " + axis->value);
    Check(axis->supported == supported,
          "support axis " + name + " has wrong supported flag for " +
              context);
}

std::vector<std::string> ParseCatalogEnumValues(
    const std::string& parameter_type) {
    std::vector<std::string> values;
    if (parameter_type.rfind("enum:", 0) != 0) {
        return values;
    }

    const std::string raw_values = parameter_type.substr(5);
    std::size_t start = 0;
    while (start <= raw_values.size()) {
        const std::size_t end = raw_values.find(',', start);
        const std::string value = raw_values.substr(
            start,
            end == std::string::npos ? std::string::npos : end - start);
        if (!value.empty()) {
            values.push_back(value);
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return values;
}

} // namespace

int main() {
    auto& metadata = cyxwiz::NodeMetadataRegistry::Instance();
    metadata.Initialize();

    {
        const auto* data_input = metadata.GetMetadata(gui::NodeType::DataInput);
        Check(data_input != nullptr, "DataInput metadata should exist");
        Check(ContainsString(data_input->keywords, "csv"),
              "DataInput metadata should keep CSV keyword");
        Check(ContainsString(data_input->keywords, "tsv"),
              "DataInput metadata should keep TSV keyword");
        Check(ContainsString(data_input->keywords, "parquet"),
              "DataInput metadata should keep Parquet keyword");
        Check(ContainsString(data_input->keywords, "feather"),
              "DataInput metadata should keep Feather keyword");
        Check(ContainsString(data_input->keywords, "arrow"),
              "DataInput metadata should keep Arrow keyword");
        Check(!ContainsString(data_input->keywords, "json"),
              "DataInput metadata should not advertise JSON runtime input");
        Check(!ContainsString(data_input->keywords, "excel"),
              "DataInput metadata should not advertise Excel runtime input");
        Check(!ContainsString(data_input->keywords, "hdf5"),
              "DataInput metadata should not advertise HDF5 runtime input");

        const auto* file_path = FindParameter(data_input, "file_path");
        Check(file_path != nullptr,
              "DataInput metadata should keep file_path parameter");
        Check(file_path->validation.find("*.csv") != std::string::npos &&
                  file_path->validation.find("*.tsv") != std::string::npos &&
                  file_path->validation.find("*.parquet") != std::string::npos &&
                  file_path->validation.find("*.feather") != std::string::npos &&
                  file_path->validation.find("*.arrow") != std::string::npos &&
                  file_path->validation.find("*.ipc") != std::string::npos,
              "DataInput file filter should list runtime-supported tabular formats");
        Check(file_path->validation.find("*.json") == std::string::npos &&
                  file_path->validation.find("*.xlsx") == std::string::npos &&
                  file_path->validation.find("*.hdf5") == std::string::npos,
              "DataInput file filter should not advertise unsupported runtime formats");

        const auto* type = FindParameter(data_input, "type");
        Check(type != nullptr,
              "DataInput metadata should expose runtime file type selector");
        Check(type->default_value == "auto",
              "DataInput type should default to runtime auto detection");
        Check(ContainsString(type->enum_values, "csv") &&
                  ContainsString(type->enum_values, "tsv") &&
                  ContainsString(type->enum_values, "parquet") &&
                  ContainsString(type->enum_values, "feather") &&
                  ContainsString(type->enum_values, "arrow") &&
                  ContainsString(type->enum_values, "ipc"),
              "DataInput type enum should list runtime-supported formats");
        Check(!ContainsString(type->enum_values, "json") &&
                  !ContainsString(type->enum_values, "excel") &&
                  !ContainsString(type->enum_values, "hdf5"),
              "DataInput type enum should not advertise unsupported runtime formats");

        const auto source_type_axes =
            cyxwiz::ResolvePipelineAllowedParameterValues("DataInput");
        auto source_type_axis = std::find_if(
            source_type_axes.begin(),
            source_type_axes.end(),
            [](const cyxwiz::PipelineAllowedParameterValuesRuntimeCapability&
                   capability) {
                return std::string(capability.parameter_name) == "source_type";
            });
        Check(source_type_axis != source_type_axes.end(),
              "DataInput runtime support should expose source_type limits");
        const std::vector<std::string> source_type_values(
            source_type_axis->allowed_values.begin(),
            source_type_axis->allowed_values.end());
        Check(ContainsString(source_type_values, "file") &&
                  ContainsString(source_type_values, "folder"),
              "DataInput source_type should allow file and folder");
        Check(!ContainsString(source_type_values, "ml_dataset"),
              "DataInput source_type should not advertise ml_dataset without a runtime owner");
    }

    {
        const auto* data_output = metadata.GetMetadata(gui::NodeType::DataOutput);
        Check(data_output != nullptr, "DataOutput metadata should exist");
        const auto* file_type = FindParameter(data_output, "file_type");
        Check(file_type != nullptr,
              "DataOutput metadata should expose runtime output format selector");
        Check(file_type->default_value == "csv",
              "DataOutput file_type should default to CSV");
        Check(ContainsString(file_type->enum_values, "csv") &&
                  ContainsString(file_type->enum_values, "parquet"),
              "DataOutput file_type enum should list runtime-supported output formats");
        Check(!ContainsString(file_type->enum_values, "json") &&
                  !ContainsString(file_type->enum_values, "excel"),
              "DataOutput file_type enum should not advertise unsupported exporters");
    }

    auto& factory = cyxwiz::PipelineOperatorFactory::Instance();
    const auto supported = factory.GetSupportedTypes();
    Check(!supported.empty(), "PipelineOperatorFactory should register operators");

    {
        std::set<std::string> operator_names;
        for (const auto& capability : cyxwiz::GetPipelineOperatorRuntimeCapabilities()) {
            Check(operator_names.insert(capability.legacy_type_name).second,
                  std::string("duplicate operator runtime capability: ") +
                      capability.legacy_type_name);
        }

        std::set<std::string> fail_closed_names;
        for (const auto& capability : cyxwiz::GetPipelineFailClosedRuntimeCapabilities()) {
            Check(fail_closed_names.insert(capability.legacy_type_name).second,
                  std::string("duplicate fail-closed runtime capability: ") +
                      capability.legacy_type_name);
        }

        std::set<std::string> legacy_names;
        for (const auto& capability : cyxwiz::GetPipelineLegacyRuntimeCapabilities()) {
            Check(legacy_names.insert(capability.legacy_type_name).second,
                  std::string("duplicate legacy runtime capability: ") +
                      capability.legacy_type_name);
        }

        std::set<std::string> source_names;
        for (const auto& capability : cyxwiz::GetPipelineSourceRuntimeCapabilities()) {
            Check(source_names.insert(capability.legacy_type_name).second,
                  std::string("duplicate source runtime capability: ") +
                      capability.legacy_type_name);
        }

        std::set<std::string> input_arity_names;
        for (const auto& capability : cyxwiz::GetPipelineInputArityRuntimeCapabilities()) {
            Check(input_arity_names.insert(capability.legacy_type_name).second,
                  std::string("duplicate input-arity runtime capability: ") +
                      capability.legacy_type_name);
        }

        std::set<std::string> required_parameter_names;
        for (const auto& capability :
             cyxwiz::GetPipelineRequiredParameterRuntimeCapabilities()) {
            Check(required_parameter_names.insert(capability.legacy_type_name).second,
                  std::string("duplicate required-parameter runtime capability: ") +
                      capability.legacy_type_name);
        }

        std::set<std::string> allowed_parameter_names;
        for (const auto& capability :
             cyxwiz::GetPipelineAllowedParameterValuesRuntimeCapabilities()) {
            const std::string key =
                std::string(capability.legacy_type_name) + "." +
                capability.parameter_name;
            Check(allowed_parameter_names.insert(key).second,
                  "duplicate allowed-parameter runtime capability: " + key);
        }

        std::set<std::string> integer_parameter_names;
        for (const auto& capability :
             cyxwiz::GetPipelineIntegerParameterRuntimeCapabilities()) {
            const std::string key =
                std::string(capability.legacy_type_name) + "." +
                capability.parameter_name;
            Check(integer_parameter_names.insert(key).second,
                  "duplicate integer-parameter runtime capability: " + key);
        }

        std::set<std::string> float_parameter_names;
        for (const auto& capability :
             cyxwiz::GetPipelineFloatParameterRuntimeCapabilities()) {
            const std::string key =
                std::string(capability.legacy_type_name) + "." +
                capability.parameter_name;
            Check(float_parameter_names.insert(key).second,
                  "duplicate float-parameter runtime capability: " + key);
        }

        std::set<int> unsupported_training_layer_types;
        for (const auto& capability :
             cyxwiz::GetPipelineUnsupportedSequentialModelLayerCapabilities()) {
            const int key = static_cast<int>(capability.node_type);
            Check(unsupported_training_layer_types.insert(key).second,
                  "duplicate unsupported training layer capability: " +
                      TypeId(capability.node_type));
            Check(capability.reason != nullptr &&
                      std::string(capability.reason).size() > 16,
                  "unsupported training layer reason is too weak: " +
                      TypeId(capability.node_type));
            Check(cyxwiz::IsPipelineUnsupportedSequentialModelLayer(
                      capability.node_type),
                  "unsupported training layer capability does not resolve: " +
                      TypeId(capability.node_type));
            const auto support = cyxwiz::ResolvePipelineTrainingBackendSupport(
                capability.node_type);
            Check(support.mode ==
                      cyxwiz::PipelineTrainingBackendSupportMode::
                          UnsupportedSequentialModelLayer,
                  "unsupported training layer should resolve through unified support: " +
                      TypeId(capability.node_type));
            Check(!support.compile_supported && !support.training_supported,
                  "unsupported training layer should block compile/training: " +
                      TypeId(capability.node_type));
            Check(support.reason == capability.reason,
                  "unsupported training layer reason should be shared: " +
                      TypeId(capability.node_type));
        }

        std::set<int> unsupported_training_control_types;
        for (const auto& capability :
             cyxwiz::GetPipelineUnsupportedTrainingControlCapabilities()) {
            const int key = static_cast<int>(capability.node_type);
            Check(unsupported_training_control_types.insert(key).second,
                  "duplicate unsupported training control capability: " +
                      TypeId(capability.node_type));
            Check(capability.reason != nullptr &&
                      std::string(capability.reason).size() > 16,
                  "unsupported training control reason is too weak: " +
                      TypeId(capability.node_type));
            Check(cyxwiz::IsPipelineUnsupportedTrainingControlNode(
                      capability.node_type),
                  "unsupported training control capability does not resolve: " +
                      TypeId(capability.node_type));
            const auto support = cyxwiz::ResolvePipelineTrainingBackendSupport(
                capability.node_type);
            Check(support.mode ==
                      cyxwiz::PipelineTrainingBackendSupportMode::
                          UnsupportedTrainingControl,
                  "unsupported training control should resolve through unified support: " +
                      TypeId(capability.node_type));
            Check(!support.compile_supported && !support.training_supported,
                  "unsupported training control should block compile/training: " +
                      TypeId(capability.node_type));
            Check(support.reason == capability.reason,
                  "unsupported training control reason should be shared: " +
                      TypeId(capability.node_type));
        }

        std::set<int> supported_training_types;
        for (const auto& capability :
             cyxwiz::GetPipelineSupportedTrainingBackendCapabilities()) {
            const int key = static_cast<int>(capability.node_type);
            Check(supported_training_types.insert(key).second,
                  "duplicate supported training backend capability: " +
                      TypeId(capability.node_type));
            Check(capability.reason != nullptr &&
                      std::string(capability.reason).size() > 16,
                  "supported training backend reason is too weak: " +
                      TypeId(capability.node_type));
            Check(cyxwiz::IsPipelineSupportedTrainingBackendNode(
                      capability.node_type),
                  "supported training backend capability does not resolve: " +
                      TypeId(capability.node_type));
            Check(!cyxwiz::IsPipelineUnsupportedSequentialModelLayer(
                      capability.node_type) &&
                      !cyxwiz::IsPipelineUnsupportedTrainingControlNode(
                          capability.node_type),
                  "supported training backend should not overlap unsupported lists: " +
                      TypeId(capability.node_type));
            const auto support = cyxwiz::ResolvePipelineTrainingBackendSupport(
                capability.node_type);
            Check(support.mode ==
                      cyxwiz::PipelineTrainingBackendSupportMode::Allowed,
                  "supported training backend should resolve as allowed: " +
                      TypeId(capability.node_type));
            Check(support.compile_supported && support.training_supported,
                  "supported training backend should allow compile/training: " +
                      TypeId(capability.node_type));
            Check(support.reason == capability.reason,
                  "supported training backend reason should be shared: " +
                      TypeId(capability.node_type));
        }

        std::set<int> materializer_storage_backends;
        int materializer_supported_backends = 0;
        for (const auto& capability :
             cyxwiz::GetPipelineMaterializerStorageBackendCapabilities()) {
            const int key = static_cast<int>(capability.backend);
            Check(materializer_storage_backends.insert(key).second,
                  "duplicate materializer storage backend capability: " +
                      std::string(cyxwiz::PipelineStorageBackendName(
                          capability.backend)));
            const auto resolved =
                cyxwiz::ResolvePipelineMaterializerStorageBackendSupport(
                    capability.backend);
            Check(resolved.backend == capability.backend,
                  "materializer storage backend capability does not resolve: " +
                      std::string(cyxwiz::PipelineStorageBackendName(
                          capability.backend)));
            Check(resolved.materializer_supported ==
                      capability.materializer_supported,
                  "materializer storage backend support mismatch: " +
                      std::string(cyxwiz::PipelineStorageBackendName(
                          capability.backend)));
            Check(resolved.storage_support == capability.storage_support,
                  "materializer storage support scope mismatch: " +
                      std::string(cyxwiz::PipelineStorageBackendName(
                          capability.backend)));
            Check(capability.reason != nullptr &&
                      std::string(capability.reason).size() > 16,
                  "materializer storage backend reason is too weak: " +
                      std::string(cyxwiz::PipelineStorageBackendName(
                          capability.backend)));
            if (capability.materializer_supported) {
                ++materializer_supported_backends;
                Check(capability.backend == cyxwiz::PipelineStorageBackend::ArrowTable,
                      "only ArrowTable should be materializer-supported today");
                Check(capability.storage_support ==
                          cyxwiz::PipelineMaterializerStorageSupport::ArrowTableOnly,
                      "ArrowTable materializer support should be ArrowTableOnly");
            } else {
                Check(capability.storage_support ==
                          cyxwiz::PipelineMaterializerStorageSupport::None,
                      "unsupported materializer backend should advertise no storage scope");
            }
        }
        Check(materializer_supported_backends == 1,
              "exactly one materializer storage backend should be supported today");
    }

    for (auto type : supported) {
        auto op = factory.Create(type);
        Check(op != nullptr, "factory returned null for type " + TypeId(type));

        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr, "missing metadata for factory type " + TypeId(type));
        Check(meta->status == cyxwiz::NodeImplementationStatus::Implemented,
              "factory type " + TypeId(type) + " is not marked implemented");
        Check(meta->category != gui::NodeCategory::Unknown,
              "factory type " + TypeId(type) + " has unknown category");

        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(type);
        Check(support.mode == cyxwiz::PipelineRuntimeSupportMode::OperatorBacked,
              "factory type missing central operator-backed runtime support: " +
                  TypeId(type));
        Check(support.operator_type.has_value() && *support.operator_type == type,
              "factory type resolves to wrong operator runtime type: " +
                  TypeId(type));
        Check(support.implementation_owner ==
                  cyxwiz::PipelineRuntimeImplementationOwner::
                      PipelineOperatorFactory,
              "factory type should be owned by PipelineOperatorFactory: " +
                  TypeId(type));
        Check(support.materializer_arrow_table_supported,
              "factory type should advertise Arrow materializer support: " +
                  TypeId(type));
    }

    for (const auto& data_studio_node :
         cyxwiz::NodeRegistry::Instance().GetAllNodeTypes()) {
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            data_studio_node.type_id);
        Check(support.mode != cyxwiz::PipelineRuntimeSupportMode::Unknown,
              "Data Studio node registry advertises unknown runtime type: " +
                  data_studio_node.type_id);
        Check(support.pipeline_executor_supported,
              "Data Studio node registry advertises unsupported runtime type: " +
                  data_studio_node.type_id);
        Check(data_studio_node.type_id != "ArrowDataset" &&
                  data_studio_node.type_id != "Aggregate" &&
                  data_studio_node.type_id != "DetectOutliers",
              "Data Studio node registry should not advertise stale unsupported type: " +
                  data_studio_node.type_id);
        for (const auto& parameter : data_studio_node.parameters) {
            const auto catalog_values = ParseCatalogEnumValues(parameter.second);
            if (catalog_values.empty()) {
                continue;
            }

            const auto runtime_axis = std::find_if(
                support.allowed_parameter_values.begin(),
                support.allowed_parameter_values.end(),
                [&parameter](
                    const cyxwiz::PipelineAllowedParameterValuesRuntimeCapability&
                        axis) {
                    return std::string(axis.parameter_name) == parameter.first;
                });
            if (runtime_axis == support.allowed_parameter_values.end()) {
                continue;
            }

            for (const auto& catalog_value : catalog_values) {
                const bool runtime_accepts_value = std::find_if(
                    runtime_axis->allowed_values.begin(),
                    runtime_axis->allowed_values.end(),
                    [&catalog_value](const char* runtime_value) {
                        return runtime_value != nullptr &&
                               catalog_value == runtime_value;
                    }) != runtime_axis->allowed_values.end();
                Check(runtime_accepts_value,
                      "Data Studio node registry advertises unsupported enum value: " +
                          data_studio_node.type_id + "." + parameter.first +
                          "=" + catalog_value);
            }
        }
    }

    const std::set<std::string> intentional_quick_add_compatibility_nodes;
    const std::set<std::string> disallowed_quick_add_legacy_aliases = {
        "ArrowDataset",
        "FileInput",
        "SaveDataset",
        "RemoveDuplicates",
        "TextClean",
        "TextTokenize",
        "TextVectorize",
        "TSWindow",
        "TSFeatures",
        "TSLag",
        "TSDiff",
        "PCA",
        "PolynomialFeatures",
        "Binning",
    };
    std::vector<std::string> data_studio_quick_add_type_ids;
    for (const auto& item : cyxwiz::PipelineCanvas::GetQuickAddNodes()) {
        Check(item.label != nullptr && std::string(item.label).size() > 0,
              "Data Studio quick-add item should have a label");
        Check(item.type_id != nullptr && std::string(item.type_id).size() > 0,
              "Data Studio quick-add item should have a type id");
        const std::string type_id = item.type_id;
        data_studio_quick_add_type_ids.push_back(type_id);
        Check(disallowed_quick_add_legacy_aliases.count(type_id) == 0,
              "Data Studio quick-add should not promote legacy alias: " +
                  type_id);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(type_id);
        Check(support.mode != cyxwiz::PipelineRuntimeSupportMode::Unknown,
              "Data Studio quick-add node has unknown runtime support: " +
                  type_id);
        Check(support.pipeline_executor_supported,
              "Data Studio quick-add node is not PipelineExecutor-supported: " +
                  type_id);
        Check(support.fail_mode == cyxwiz::PipelineRuntimeFailMode::Real,
              "Data Studio quick-add node should have real fail mode: " +
                  type_id);
        if (intentional_quick_add_compatibility_nodes.count(type_id) == 0) {
            Check(support.node_type.has_value(),
                  "Data Studio quick-add canonical node should resolve typed metadata: " +
                      type_id);
            const auto* meta = metadata.GetMetadata(*support.node_type);
            Check(meta != nullptr,
                  "Data Studio quick-add canonical node metadata missing: " +
                      type_id);
            Check(meta->status == cyxwiz::NodeImplementationStatus::Implemented,
                  "Data Studio quick-add canonical node should be implemented metadata: " +
                      type_id);
        }
    }
    for (const auto& legacy_alias : disallowed_quick_add_legacy_aliases) {
        Check(std::find(data_studio_quick_add_type_ids.begin(),
                        data_studio_quick_add_type_ids.end(),
                        legacy_alias) == data_studio_quick_add_type_ids.end(),
              "Data Studio quick-add contract contains disallowed alias: " +
                  legacy_alias);
    }

    for (const auto& capability : cyxwiz::GetPipelineOperatorRuntimeCapabilities()) {
        Check(factory.HasOperator(capability.node_type),
              std::string("runtime capability has no factory operator: ") +
                  capability.legacy_type_name);
        auto resolved = cyxwiz::ResolvePipelineOperatorRuntimeType(
            capability.legacy_type_name);
        Check(resolved.has_value(),
              std::string("runtime capability does not resolve: ") +
                  capability.legacy_type_name);
        Check(*resolved == capability.node_type,
              std::string("runtime capability resolves to wrong type: ") +
                  capability.legacy_type_name);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            capability.legacy_type_name);
        Check(support.mode == cyxwiz::PipelineRuntimeSupportMode::OperatorBacked,
              std::string("runtime support mode is not operator-backed: ") +
                  capability.legacy_type_name);
        Check(support.fail_mode == cyxwiz::PipelineRuntimeFailMode::Real,
              std::string("operator-backed runtime should advertise real fail mode: ") +
                  capability.legacy_type_name);
        Check(support.operator_type.has_value() &&
                  *support.operator_type == capability.node_type,
              std::string("runtime support operator type mismatch: ") +
                  capability.legacy_type_name);
        Check(support.node_type.has_value() &&
                  *support.node_type == capability.node_type,
              std::string("runtime support node type mismatch: ") +
                  capability.legacy_type_name);
        const auto runtime_node_type =
            cyxwiz::ResolvePipelineRuntimeNodeType(capability.legacy_type_name);
        Check(runtime_node_type.has_value() &&
                  *runtime_node_type == capability.node_type,
              std::string("runtime name should resolve to operator node type: ") +
                  capability.legacy_type_name);
        const auto enum_support =
            cyxwiz::ResolvePipelineRuntimeSupport(capability.node_type);
        Check(enum_support.mode ==
                  cyxwiz::PipelineRuntimeSupportMode::OperatorBacked,
              std::string("operator node type should resolve support by enum: ") +
                  capability.legacy_type_name);
        Check(support.materializer_arrow_table_supported,
              std::string("operator-backed runtime should be Arrow-materializer capable: ") +
                  capability.legacy_type_name);
        Check(support.pipeline_executor_supported,
              std::string("operator-backed runtime should be pipeline-executor supported: ") +
                  capability.legacy_type_name);
        Check(support.implementation_owner ==
                  cyxwiz::PipelineRuntimeImplementationOwner::
                      PipelineOperatorFactory,
              std::string("operator-backed runtime should be owned by PipelineOperatorFactory: ") +
                  capability.legacy_type_name);
        Check(support.materializer_storage_support ==
                  cyxwiz::PipelineMaterializerStorageSupport::ArrowTableOnly,
              std::string("operator-backed runtime should advertise Arrow-only materializer scope: ") +
                  capability.legacy_type_name);
        const auto* meta = metadata.GetMetadata(capability.node_type);
        Check(meta != nullptr,
              std::string("operator-backed runtime metadata missing: ") +
                  capability.legacy_type_name);
        Check(meta->help_text.find(cyxwiz::PipelineRuntimeSupportModeName(
                  support.mode)) != std::string::npos,
              std::string("operator-backed metadata should expose runtime support mode: ") +
                  capability.legacy_type_name);
        Check(meta->help_text.find(cyxwiz::PipelineRuntimeFailModeName(
                  support.fail_mode)) != std::string::npos,
              std::string("operator-backed metadata should expose fail mode: ") +
                  capability.legacy_type_name);
        Check(meta->help_text.find("pipeline_executor=supported") !=
                  std::string::npos,
              std::string("operator-backed metadata should expose pipeline executor support: ") +
                  capability.legacy_type_name);
        Check(meta->help_text.find(cyxwiz::PipelineMaterializerStorageSupportName(
                  support.materializer_storage_support)) != std::string::npos,
              std::string("operator-backed metadata should expose materializer support scope: ") +
                  capability.legacy_type_name);
        Check(meta->help_text.find(cyxwiz::PipelineRuntimeImplementationOwnerName(
                  support.implementation_owner)) != std::string::npos,
              std::string("operator-backed metadata should expose implementation owner: ") +
                  capability.legacy_type_name);
        CheckSupportAxis(
            meta,
            "Runtime",
            cyxwiz::PipelineRuntimeSupportModeName(support.mode),
            true,
            capability.legacy_type_name);
        CheckSupportAxis(
            meta,
            "Fail Mode",
            cyxwiz::PipelineRuntimeFailModeName(support.fail_mode),
            true,
            capability.legacy_type_name);
        CheckSupportAxis(
            meta,
            "Pipeline Executor",
            "supported",
            true,
            capability.legacy_type_name);
        CheckSupportAxis(
            meta,
            "Materializer",
            cyxwiz::PipelineMaterializerStorageSupportName(
                support.materializer_storage_support),
            true,
            capability.legacy_type_name);
        CheckSupportAxis(
            meta,
            "Implementation Owner",
            cyxwiz::PipelineRuntimeImplementationOwnerName(
                support.implementation_owner),
            true,
            capability.legacy_type_name);
    }

    const std::set<std::string> allowed_untyped_fail_closed_names = {
        // Legacy canvas aliases with no one-to-one metadata node. The current
        // typed metadata entries are DataSplit and DataInput.
        "TrainTestSplit",
    };

    for (const auto& capability : cyxwiz::GetPipelineFailClosedRuntimeCapabilities()) {
        Check(!cyxwiz::IsPipelineOperatorRuntimeNode(capability.legacy_type_name),
              std::string("fail-closed runtime name is also operator-backed: ") +
                  capability.legacy_type_name);
        Check(!cyxwiz::IsPipelineLegacyRuntimeNode(capability.legacy_type_name),
              std::string("fail-closed runtime name is also legacy-dispatched: ") +
                  capability.legacy_type_name);
        Check(cyxwiz::ResolvePipelineFailClosedReason(capability.legacy_type_name) != nullptr,
              std::string("fail-closed runtime name does not resolve: ") +
                  capability.legacy_type_name);
        Check(capability.reason != nullptr && std::string(capability.reason).size() > 8,
              std::string("fail-closed runtime reason is too weak: ") +
                  capability.legacy_type_name);
        const std::string fail_closed_reason =
            capability.reason != nullptr ? capability.reason : "";
        Check(fail_closed_reason.find("is still a") == std::string::npos &&
                  fail_closed_reason.find("is still an") == std::string::npos,
              std::string("fail-closed runtime reason should describe current hard-fail state: ") +
                  capability.legacy_type_name);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            capability.legacy_type_name);
        Check(support.mode == cyxwiz::PipelineRuntimeSupportMode::FailClosed,
              std::string("runtime support mode is not fail-closed: ") +
                  capability.legacy_type_name);
        Check(support.fail_mode == cyxwiz::PipelineRuntimeFailMode::HardFail,
              std::string("fail-closed runtime should advertise hard-fail mode: ") +
                  capability.legacy_type_name);
        Check(support.fail_closed_reason != nullptr,
              std::string("runtime support fail-closed reason missing: ") +
                  capability.legacy_type_name);
        auto expected_metadata_type = capability.metadata_node_type;
        if (!expected_metadata_type.has_value()) {
            expected_metadata_type = capability.node_type;
        }
        if (!expected_metadata_type.has_value()) {
            Check(allowed_untyped_fail_closed_names.count(
                      capability.legacy_type_name) > 0,
                  std::string("fail-closed runtime should be typed unless it is a documented legacy alias: ") +
                      capability.legacy_type_name);
        }
        Check(support.metadata_node_type == expected_metadata_type,
              std::string("runtime support fail-closed metadata node mismatch: ") +
                  capability.legacy_type_name);
        Check(support.node_type == capability.node_type,
              std::string("runtime support fail-closed node type mismatch: ") +
                  capability.legacy_type_name);
        const auto runtime_node_type =
            cyxwiz::ResolvePipelineRuntimeNodeType(capability.legacy_type_name);
        Check(runtime_node_type == expected_metadata_type,
              std::string("fail-closed runtime name should resolve typed metadata: ") +
                  capability.legacy_type_name);
        Check(!support.materializer_arrow_table_supported,
              std::string("fail-closed runtime should not be Arrow-materializer capable: ") +
                  capability.legacy_type_name);
        Check(!support.pipeline_executor_supported,
              std::string("fail-closed runtime should not be pipeline-executor supported: ") +
                  capability.legacy_type_name);
        Check(support.implementation_owner ==
                  cyxwiz::PipelineRuntimeImplementationOwner::None,
              std::string("fail-closed runtime should advertise no implementation owner: ") +
                  capability.legacy_type_name);
        Check(support.materializer_storage_support ==
                  cyxwiz::PipelineMaterializerStorageSupport::None,
              std::string("fail-closed runtime should not advertise materializer scope: ") +
                  capability.legacy_type_name);
        if (expected_metadata_type.has_value()) {
            const auto* meta = metadata.GetMetadata(*expected_metadata_type);
            if (meta != nullptr) {
                const std::string reason =
                    capability.reason != nullptr ? capability.reason : "";
                CheckSupportAxis(
                    meta,
                    "Runtime",
                    cyxwiz::PipelineRuntimeSupportModeName(support.mode),
                    false,
                    capability.legacy_type_name);
                CheckSupportAxis(
                    meta,
                    "Fail Mode",
                    cyxwiz::PipelineRuntimeFailModeName(support.fail_mode),
                    false,
                    capability.legacy_type_name);
                CheckSupportAxis(
                    meta,
                    "Pipeline Executor",
                    "unsupported",
                    false,
                    capability.legacy_type_name);
                CheckSupportAxis(
                    meta,
                    "Materializer",
                    cyxwiz::PipelineMaterializerStorageSupportName(
                        support.materializer_storage_support),
                    false,
                    capability.legacy_type_name);
                CheckSupportAxis(
                    meta,
                    "Implementation Owner",
                    cyxwiz::PipelineRuntimeImplementationOwnerName(
                        support.implementation_owner),
                    false,
                    capability.legacy_type_name);
                Check(meta->help_text.find(reason) != std::string::npos,
                      std::string("fail-closed metadata should expose reason: ") +
                          capability.legacy_type_name);
            }
        }
    }

    const std::set<std::string> expected_string_only_legacy_names;
    std::set<std::string> observed_string_only_legacy_names;

    for (const auto& capability : cyxwiz::GetPipelineLegacyRuntimeCapabilities()) {
        Check(!cyxwiz::IsPipelineOperatorRuntimeNode(capability.legacy_type_name),
              std::string("legacy runtime name is also operator-backed: ") +
                  capability.legacy_type_name);
        Check(!cyxwiz::IsPipelineFailClosedRuntimeNode(capability.legacy_type_name),
              std::string("legacy runtime name is also fail-closed: ") +
                  capability.legacy_type_name);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            capability.legacy_type_name);
        Check(support.mode == cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
              std::string("runtime support mode is not legacy executor: ") +
                  capability.legacy_type_name);
        Check(support.fail_mode == cyxwiz::PipelineRuntimeFailMode::Real,
              std::string("legacy-dispatched runtime should advertise real fail mode: ") +
                  capability.legacy_type_name);
        Check(support.node_type == capability.node_type,
              std::string("legacy-dispatched runtime node type mismatch: ") +
                  capability.legacy_type_name);
        Check(support.legacy_dispatch_kind == capability.dispatch_kind,
              std::string("legacy-dispatched runtime dispatch kind mismatch: ") +
                  capability.legacy_type_name);
        if (capability.node_type.has_value()) {
            Check(capability.dispatch_kind ==
                      cyxwiz::PipelineLegacyDispatchKind::Unknown,
                  std::string("typed legacy runtime should not also carry "
                              "string-only dispatch kind: ") +
                      capability.legacy_type_name);
            Check(capability.compatibility_reason == nullptr,
                  std::string("typed legacy runtime should not carry string-only "
                              "compatibility reason: ") +
                      capability.legacy_type_name);
        } else {
            Check(capability.dispatch_kind !=
                      cyxwiz::PipelineLegacyDispatchKind::Unknown,
                  std::string("string-only legacy runtime missing dispatch kind: ") +
                      capability.legacy_type_name);
            Check(capability.compatibility_reason != nullptr &&
                      std::string(capability.compatibility_reason).size() > 0,
                  std::string("string-only legacy runtime missing compatibility "
                              "reason: ") +
                      capability.legacy_type_name);
            Check(expected_string_only_legacy_names.count(
                      capability.legacy_type_name) == 1,
                  std::string("unexpected string-only legacy runtime exception: ") +
                      capability.legacy_type_name);
            observed_string_only_legacy_names.insert(capability.legacy_type_name);
        }
        const auto runtime_node_type =
            cyxwiz::ResolvePipelineRuntimeNodeType(capability.legacy_type_name);
        Check(runtime_node_type == capability.node_type,
              std::string("legacy runtime name should resolve typed metadata: ") +
                  capability.legacy_type_name);
        Check(!support.materializer_arrow_table_supported,
              std::string("legacy-dispatched runtime should not claim Arrow materializer support: ") +
                  capability.legacy_type_name);
        Check(support.pipeline_executor_supported,
              std::string("legacy-dispatched runtime should be pipeline-executor supported: ") +
                  capability.legacy_type_name);
        Check(support.implementation_owner ==
                  cyxwiz::PipelineRuntimeImplementationOwner::PipelineExecutor,
              std::string("legacy-dispatched runtime should be owned by PipelineExecutor: ") +
                  capability.legacy_type_name);
        Check(support.materializer_storage_support ==
                  cyxwiz::PipelineMaterializerStorageSupport::None,
              std::string("legacy-dispatched runtime should not advertise materializer scope: ") +
                  capability.legacy_type_name);
        if (capability.node_type.has_value()) {
            const auto* meta = metadata.GetMetadata(*capability.node_type);
            if (meta != nullptr) {
                CheckSupportAxis(
                    meta,
                    "Runtime",
                    cyxwiz::PipelineRuntimeSupportModeName(support.mode),
                    true,
                    capability.legacy_type_name);
                CheckSupportAxis(
                    meta,
                    "Fail Mode",
                    cyxwiz::PipelineRuntimeFailModeName(support.fail_mode),
                    true,
                    capability.legacy_type_name);
                CheckSupportAxis(
                    meta,
                    "Pipeline Executor",
                    "supported",
                    true,
                    capability.legacy_type_name);
                CheckSupportAxis(
                    meta,
                    "Materializer",
                    cyxwiz::PipelineMaterializerStorageSupportName(
                        support.materializer_storage_support),
                    false,
                    capability.legacy_type_name);
                CheckSupportAxis(
                    meta,
                    "Implementation Owner",
                    cyxwiz::PipelineRuntimeImplementationOwnerName(
                        support.implementation_owner),
                    true,
                    capability.legacy_type_name);
            }
        }
    }

    for (const auto& expected_name : expected_string_only_legacy_names) {
        Check(observed_string_only_legacy_names.count(expected_name) == 1,
              "missing string-only legacy runtime exception: " + expected_name);
    }

    Check(std::string(cyxwiz::PipelineRuntimeSupportModeName(
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor)) ==
              "legacy_executor",
          "runtime support mode name for legacy executor is stable");
    Check(std::string(cyxwiz::PipelineRuntimeSupportModeName(
              cyxwiz::PipelineRuntimeSupportMode::OperatorBacked)) ==
              "operator_backed",
          "runtime support mode name for operator-backed is stable");
    Check(std::string(cyxwiz::PipelineRuntimeSupportModeName(
              cyxwiz::PipelineRuntimeSupportMode::FailClosed)) ==
              "fail_closed",
          "runtime support mode name for fail-closed is stable");

    Check(std::string(cyxwiz::PipelineRuntimeFailModeName(
              cyxwiz::PipelineRuntimeFailMode::Real)) == "real",
          "runtime fail-mode name for real is stable");
    Check(std::string(cyxwiz::PipelineRuntimeFailModeName(
              cyxwiz::PipelineRuntimeFailMode::HardFail)) == "hard_fail",
          "runtime fail-mode name for hard_fail is stable");
    Check(std::string(cyxwiz::PipelineRuntimeFailModeName(
              cyxwiz::PipelineRuntimeFailMode::Simulated)) == "simulated",
          "runtime fail-mode name for simulated is stable");
    Check(std::string(cyxwiz::PipelineRuntimeFailModeName(
              cyxwiz::PipelineRuntimeFailMode::Passthrough)) == "passthrough",
          "runtime fail-mode name for passthrough is stable");

    Check(std::string(cyxwiz::PipelineRuntimeImplementationOwnerName(
              cyxwiz::PipelineRuntimeImplementationOwner::None)) == "none",
          "runtime implementation owner name for none is stable");
    Check(std::string(cyxwiz::PipelineRuntimeImplementationOwnerName(
              cyxwiz::PipelineRuntimeImplementationOwner::PipelineExecutor)) ==
              "pipeline_executor",
          "runtime implementation owner name for pipeline executor is stable");
    Check(std::string(cyxwiz::PipelineRuntimeImplementationOwnerName(
              cyxwiz::PipelineRuntimeImplementationOwner::
                  PipelineOperatorFactory)) == "pipeline_operator_factory",
          "runtime implementation owner name for operator factory is stable");

    Check(std::string(cyxwiz::PipelineMaterializerStorageSupportName(
              cyxwiz::PipelineMaterializerStorageSupport::None)) == "none",
          "materializer storage support name for none is stable");
    Check(std::string(cyxwiz::PipelineMaterializerStorageSupportName(
              cyxwiz::PipelineMaterializerStorageSupport::ArrowTableOnly)) ==
              "arrow_table_only",
          "materializer storage support name for ArrowTableOnly is stable");

    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::FilterRows)) == "FilterRows",
          "legacy runtime enum lookup for FilterRows is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::FilterRows).mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "FilterRows enum support should resolve to legacy executor");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::RemoveDuplicateRows)) == "RemoveDuplicateRows",
          "runtime enum lookup for RemoveDuplicateRows should prefer canonical spelling");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(
              gui::NodeType::RemoveDuplicateRows).mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "RemoveDuplicateRows enum support should resolve to legacy executor");
    const auto canonical_dedup_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("RemoveDuplicateRows");
    Check(canonical_dedup_type.has_value() &&
              *canonical_dedup_type == gui::NodeType::RemoveDuplicateRows,
          "canonical RemoveDuplicateRows runtime name should resolve to typed metadata");
    const auto legacy_dedup_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("RemoveDuplicates");
    Check(legacy_dedup_type.has_value() &&
              *legacy_dedup_type == gui::NodeType::RemoveDuplicateRows,
          "legacy RemoveDuplicates runtime name should remain an executable alias");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::BinningNode)) == "BinningNode",
          "runtime enum lookup for BinningNode should prefer canonical spelling");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::BinningNode).mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "BinningNode enum support should resolve to legacy executor");
    const auto canonical_binning_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("BinningNode");
    Check(canonical_binning_type.has_value() &&
              *canonical_binning_type == gui::NodeType::BinningNode,
          "canonical BinningNode runtime name should resolve to typed metadata");
    const auto legacy_binning_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("Binning");
    Check(legacy_binning_type.has_value() &&
              *legacy_binning_type == gui::NodeType::BinningNode,
          "legacy Binning runtime name should remain an executable alias");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::PolynomialFeaturesNode)) == "PolynomialFeaturesNode",
          "runtime enum lookup for PolynomialFeaturesNode should prefer canonical spelling");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(
              gui::NodeType::PolynomialFeaturesNode).mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "PolynomialFeaturesNode enum support should resolve to legacy executor");
    const auto canonical_polynomial_features_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("PolynomialFeaturesNode");
    Check(canonical_polynomial_features_type.has_value() &&
              *canonical_polynomial_features_type ==
                  gui::NodeType::PolynomialFeaturesNode,
          "canonical PolynomialFeaturesNode runtime name should resolve to typed metadata");
    const auto legacy_polynomial_features_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("PolynomialFeatures");
    Check(legacy_polynomial_features_type.has_value() &&
              *legacy_polynomial_features_type ==
                  gui::NodeType::PolynomialFeaturesNode,
          "legacy PolynomialFeatures runtime name should remain an executable alias");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::TimeSeriesLag)) == "TimeSeriesLag",
          "runtime enum lookup for TimeSeriesLag should prefer canonical spelling");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::TimeSeriesLag).mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "TimeSeriesLag enum support should resolve to legacy executor");
    const auto canonical_ts_lag_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("TimeSeriesLag");
    Check(canonical_ts_lag_type.has_value() &&
              *canonical_ts_lag_type == gui::NodeType::TimeSeriesLag,
          "canonical TimeSeriesLag runtime name should resolve to typed metadata");
    const auto legacy_ts_lag_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("TSLag");
    Check(legacy_ts_lag_type.has_value() &&
              *legacy_ts_lag_type == gui::NodeType::TimeSeriesLag,
          "legacy TSLag runtime name should remain an executable alias");
    const auto legacy_ts_window_support =
        cyxwiz::ResolvePipelineRuntimeSupport("TSWindow");
    Check(legacy_ts_window_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "legacy TSWindow runtime name should remain legacy-executor routed");
    Check(legacy_ts_window_support.node_type.has_value() &&
              *legacy_ts_window_support.node_type ==
                  gui::NodeType::TimeSeriesWindow,
          "legacy TSWindow runtime name should resolve to TimeSeriesWindow metadata");
    const auto legacy_ts_window_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("TSWindow");
    Check(legacy_ts_window_type.has_value() &&
              *legacy_ts_window_type == gui::NodeType::TimeSeriesWindow,
          "legacy TSWindow runtime name should remain an executable alias");
    const auto legacy_ts_features_support =
        cyxwiz::ResolvePipelineRuntimeSupport("TSFeatures");
    Check(legacy_ts_features_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "legacy TSFeatures runtime name should remain legacy-executor routed");
    Check(legacy_ts_features_support.node_type.has_value() &&
              *legacy_ts_features_support.node_type ==
                  gui::NodeType::TimeSeriesFeatures,
          "legacy TSFeatures runtime name should resolve to TimeSeriesFeatures metadata");
    const auto legacy_ts_features_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("TSFeatures");
    Check(legacy_ts_features_type.has_value() &&
              *legacy_ts_features_type == gui::NodeType::TimeSeriesFeatures,
          "legacy TSFeatures runtime name should remain an executable alias");
    const auto legacy_ts_diff_support =
        cyxwiz::ResolvePipelineRuntimeSupport("TSDiff");
    Check(legacy_ts_diff_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "legacy TSDiff runtime name should remain legacy-executor routed");
    Check(legacy_ts_diff_support.node_type.has_value() &&
              *legacy_ts_diff_support.node_type == gui::NodeType::Differencing,
          "legacy TSDiff runtime name should resolve to Differencing metadata");
    const auto legacy_ts_diff_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("TSDiff");
    Check(legacy_ts_diff_type.has_value() &&
              *legacy_ts_diff_type == gui::NodeType::Differencing,
          "legacy TSDiff runtime name should remain an executable alias");
    const auto legacy_text_vectorize_support =
        cyxwiz::ResolvePipelineRuntimeSupport("TextVectorize");
    Check(legacy_text_vectorize_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "legacy TextVectorize runtime name should remain legacy-executor routed");
    Check(legacy_text_vectorize_support.node_type.has_value() &&
              *legacy_text_vectorize_support.node_type ==
                  gui::NodeType::CountVectorizer,
          "legacy TextVectorize runtime name should resolve to CountVectorizer metadata");
    const auto legacy_text_vectorize_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("TextVectorize");
    Check(legacy_text_vectorize_type.has_value() &&
              *legacy_text_vectorize_type == gui::NodeType::CountVectorizer,
          "legacy TextVectorize runtime name should remain an executable alias");
    const auto legacy_text_tokenize_support =
        cyxwiz::ResolvePipelineRuntimeSupport("TextTokenize");
    Check(legacy_text_tokenize_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "legacy TextTokenize runtime name should remain legacy-executor routed");
    Check(legacy_text_tokenize_support.node_type.has_value() &&
              *legacy_text_tokenize_support.node_type ==
                  gui::NodeType::TextTokenizer,
          "legacy TextTokenize runtime name should resolve to TextTokenizer metadata");
    const auto legacy_text_tokenize_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("TextTokenize");
    Check(legacy_text_tokenize_type.has_value() &&
              *legacy_text_tokenize_type == gui::NodeType::TextTokenizer,
          "legacy TextTokenize runtime name should remain an executable alias");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::TextCleanNode)) == "TextCleanNode",
          "runtime enum lookup for TextCleanNode should prefer canonical spelling");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::TextCleanNode).mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "TextCleanNode enum support should resolve to legacy executor");
    const auto canonical_text_clean_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("TextCleanNode");
    Check(canonical_text_clean_type.has_value() &&
              *canonical_text_clean_type == gui::NodeType::TextCleanNode,
          "canonical TextCleanNode runtime name should resolve to typed metadata");
    const auto legacy_text_clean_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("TextClean");
    Check(legacy_text_clean_type.has_value() &&
              *legacy_text_clean_type == gui::NodeType::TextCleanNode,
          "legacy TextClean runtime name should remain an executable alias");
    const auto legacy_save_dataset_support =
        cyxwiz::ResolvePipelineRuntimeSupport("SaveDataset");
    Check(legacy_save_dataset_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "legacy SaveDataset runtime name should remain legacy-executor routed");
    Check(legacy_save_dataset_support.node_type.has_value() &&
              *legacy_save_dataset_support.node_type == gui::NodeType::DataOutput,
          "legacy SaveDataset runtime name should resolve to DataOutput metadata");
    const auto legacy_save_dataset_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("SaveDataset");
    Check(legacy_save_dataset_type.has_value() &&
              *legacy_save_dataset_type == gui::NodeType::DataOutput,
          "legacy SaveDataset runtime name should remain an executable alias");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::DeployToNodeEditorNode)) == "DeployToNodeEditorNode",
          "runtime enum lookup for DeployToNodeEditorNode should prefer canonical spelling");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(
              gui::NodeType::DeployToNodeEditorNode).mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "DeployToNodeEditorNode enum support should resolve to legacy executor");
    const auto canonical_deploy_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("DeployToNodeEditorNode");
    Check(canonical_deploy_type.has_value() &&
              *canonical_deploy_type == gui::NodeType::DeployToNodeEditorNode,
          "canonical DeployToNodeEditorNode runtime name should resolve to typed metadata");
    const auto legacy_deploy_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("DeployToNodeEditor");
    Check(legacy_deploy_type.has_value() &&
              *legacy_deploy_type == gui::NodeType::DeployToNodeEditorNode,
          "legacy DeployToNodeEditor runtime name should remain an executable alias");
    Check(std::string(cyxwiz::ResolvePipelineOperatorRuntimeLegacyTypeName(
              gui::NodeType::PCANode)) == "PCANode",
          "operator runtime enum lookup for PCANode should prefer canonical spelling");
    Check(cyxwiz::ResolvePipelineRuntimeSupport("PCA").mode ==
              cyxwiz::PipelineRuntimeSupportMode::OperatorBacked,
          "legacy PCA runtime name should resolve to operator-backed support");
    const auto legacy_pca_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("PCA");
    Check(legacy_pca_type.has_value() &&
              *legacy_pca_type == gui::NodeType::PCANode,
          "legacy PCA runtime name should remain an executable alias");
    const auto legacy_parquet_input_support =
        cyxwiz::ResolvePipelineRuntimeSupport("ParquetInput");
    Check(legacy_parquet_input_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "legacy ParquetInput runtime name should resolve to legacy executor");
    Check(legacy_parquet_input_support.node_type.has_value() &&
              *legacy_parquet_input_support.node_type == gui::NodeType::DataInput,
          "legacy ParquetInput runtime name should resolve to DataInput metadata");
    const auto legacy_parquet_input_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("ParquetInput");
    Check(legacy_parquet_input_type.has_value() &&
              *legacy_parquet_input_type == gui::NodeType::DataInput,
          "legacy ParquetInput runtime name should remain an executable alias");
    const auto cell_extractor_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::CellExtractor);
    Check(cell_extractor_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "CellExtractor enum support should resolve to legacy executor");
    const auto cell_extractor_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("CellExtractor");
    Check(cell_extractor_type.has_value() &&
              *cell_extractor_type == gui::NodeType::CellExtractor,
          "CellExtractor runtime name should resolve typed metadata");
    const auto cell_updater_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::CellUpdater);
    Check(cell_updater_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "CellUpdater enum support should resolve to legacy executor");
    const auto cell_updater_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("CellUpdater");
    Check(cell_updater_type.has_value() &&
              *cell_updater_type == gui::NodeType::CellUpdater,
          "CellUpdater runtime name should resolve typed metadata");
    const auto row_appender_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::RowAppender);
    Check(row_appender_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "RowAppender enum support should resolve to legacy executor");
    const auto row_appender_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("RowAppender");
    Check(row_appender_type.has_value() &&
              *row_appender_type == gui::NodeType::RowAppender,
          "RowAppender runtime name should resolve typed metadata");
    const auto column_appender_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::ColumnAppender);
    Check(column_appender_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "ColumnAppender enum support should resolve to legacy executor");
    const auto column_appender_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("ColumnAppender");
    Check(column_appender_type.has_value() &&
              *column_appender_type == gui::NodeType::ColumnAppender,
          "ColumnAppender runtime name should resolve typed metadata");
    const auto unpivot_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::Unpivot);
    Check(unpivot_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "Unpivot enum support should resolve to legacy executor");
    const auto unpivot_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("Unpivot");
    Check(unpivot_type.has_value() && *unpivot_type == gui::NodeType::Unpivot,
          "Unpivot runtime name should resolve typed metadata");
    const auto export_json_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::ExportJSON);
    Check(export_json_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "ExportJSON enum support should resolve to legacy executor");
    const auto export_json_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("ExportJSON");
    Check(export_json_type.has_value() &&
              *export_json_type == gui::NodeType::ExportJSON,
          "ExportJSON runtime name should resolve typed metadata");
    const auto export_parquet_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::ExportParquet);
    Check(export_parquet_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "ExportParquet enum support should resolve to legacy executor");
    const auto export_parquet_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("ExportParquet");
    Check(export_parquet_type.has_value() &&
              *export_parquet_type == gui::NodeType::ExportParquet,
          "ExportParquet runtime name should resolve typed metadata");
    const auto rule_engine_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::RuleEngine);
    Check(rule_engine_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "RuleEngine enum support should resolve to legacy executor");
    const auto rule_engine_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("RuleEngine");
    Check(rule_engine_type.has_value() &&
              *rule_engine_type == gui::NodeType::RuleEngine,
          "RuleEngine runtime name should resolve typed metadata");
    const auto unit_converter_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::UnitConverter);
    Check(unit_converter_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "UnitConverter enum support should resolve to legacy executor");
    const auto unit_converter_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("UnitConverter");
    Check(unit_converter_type.has_value() &&
              *unit_converter_type == gui::NodeType::UnitConverter,
          "UnitConverter runtime name should resolve typed metadata");
    const auto calculator_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::CalculatorNode);
    Check(calculator_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "CalculatorNode enum support should resolve to legacy executor");
    const auto calculator_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("CalculatorNode");
    Check(calculator_type.has_value() &&
              *calculator_type == gui::NodeType::CalculatorNode,
          "CalculatorNode runtime name should resolve typed metadata");
    const auto json_path_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::JSONPathExtractor);
    Check(json_path_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "JSONPathExtractor enum support should resolve to legacy executor");
    const auto json_path_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("JSONPathExtractor");
    Check(json_path_type.has_value() &&
              *json_path_type == gui::NodeType::JSONPathExtractor,
          "JSONPathExtractor runtime name should resolve typed metadata");
    const auto regex_tester_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::RegexTester);
    Check(regex_tester_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "RegexTester enum support should resolve to legacy executor");
    const auto regex_tester_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("RegexTester");
    Check(regex_tester_type.has_value() &&
              *regex_tester_type == gui::NodeType::RegexTester,
          "RegexTester runtime name should resolve typed metadata");
    const auto data_profiler_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::DataProfiler);
    Check(data_profiler_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "DataProfiler enum support should resolve to legacy executor");
    const auto data_profiler_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("DataProfiler");
    Check(data_profiler_type.has_value() &&
              *data_profiler_type == gui::NodeType::DataProfiler,
          "DataProfiler runtime name should resolve typed metadata");
    const auto regression_metrics_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::RegressionMetricsNode);
    Check(regression_metrics_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "RegressionMetricsNode enum support should resolve to legacy executor");
    const auto regression_metrics_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("RegressionMetricsNode");
    Check(regression_metrics_type.has_value() &&
              *regression_metrics_type == gui::NodeType::RegressionMetricsNode,
          "RegressionMetricsNode runtime name should resolve typed metadata");
    const auto* regression_metrics_meta =
        metadata.GetMetadata(gui::NodeType::RegressionMetricsNode);
    Check(regression_metrics_meta != nullptr,
          "RegressionMetricsNode metadata should exist");
    Check(regression_metrics_meta->status ==
              cyxwiz::NodeImplementationStatus::Implemented,
          "RegressionMetricsNode metadata should be implemented");
    const auto confusion_matrix_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::ConfusionMatrixNode);
    Check(confusion_matrix_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "ConfusionMatrixNode enum support should resolve to legacy executor");
    const auto confusion_matrix_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("ConfusionMatrixNode");
    Check(confusion_matrix_type.has_value() &&
              *confusion_matrix_type == gui::NodeType::ConfusionMatrixNode,
          "ConfusionMatrixNode runtime name should resolve typed metadata");
    const auto* confusion_matrix_meta =
        metadata.GetMetadata(gui::NodeType::ConfusionMatrixNode);
    Check(confusion_matrix_meta != nullptr,
          "ConfusionMatrixNode metadata should exist");
    Check(confusion_matrix_meta->status ==
              cyxwiz::NodeImplementationStatus::Implemented,
          "ConfusionMatrixNode metadata should be implemented");
    const auto roc_curve_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::ROCCurveNode);
    Check(roc_curve_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "ROCCurveNode enum support should resolve to legacy executor");
    const auto roc_curve_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("ROCCurveNode");
    Check(roc_curve_type.has_value() &&
              *roc_curve_type == gui::NodeType::ROCCurveNode,
          "ROCCurveNode runtime name should resolve typed metadata");
    const auto* roc_curve_meta =
        metadata.GetMetadata(gui::NodeType::ROCCurveNode);
    Check(roc_curve_meta != nullptr,
          "ROCCurveNode metadata should exist");
    Check(roc_curve_meta->status ==
              cyxwiz::NodeImplementationStatus::Implemented,
          "ROCCurveNode metadata should be implemented");
    const auto pr_curve_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::PRCurveNode);
    Check(pr_curve_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "PRCurveNode enum support should resolve to legacy executor");
    const auto pr_curve_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("PRCurveNode");
    Check(pr_curve_type.has_value() &&
              *pr_curve_type == gui::NodeType::PRCurveNode,
          "PRCurveNode runtime name should resolve typed metadata");
    const auto* pr_curve_meta =
        metadata.GetMetadata(gui::NodeType::PRCurveNode);
    Check(pr_curve_meta != nullptr,
          "PRCurveNode metadata should exist");
    Check(pr_curve_meta->status ==
              cyxwiz::NodeImplementationStatus::Implemented,
          "PRCurveNode metadata should be implemented");
    const auto data_validator_support =
        cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::DataValidator);
    Check(data_validator_support.mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "DataValidator enum support should resolve to legacy executor");
    const auto data_validator_type =
        cyxwiz::ResolvePipelineRuntimeNodeType("DataValidator");
    Check(data_validator_type.has_value() &&
              *data_validator_type == gui::NodeType::DataValidator,
          "DataValidator runtime name should resolve typed metadata");
    const auto* data_validator_meta =
        metadata.GetMetadata(gui::NodeType::DataValidator);
    Check(data_validator_meta != nullptr,
          "DataValidator metadata should exist");
    Check(data_validator_meta->status ==
              cyxwiz::NodeImplementationStatus::Implemented,
          "DataValidator metadata should be implemented");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::CSVFile)) == "FileInput",
          "legacy runtime enum lookup for CSVFile is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::CSVFile).mode ==
              cyxwiz::PipelineRuntimeSupportMode::LegacyExecutor,
          "CSVFile enum support should resolve to legacy executor");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::ExcelFile)) == "ExcelInput",
          "fail-closed runtime enum lookup for ExcelFile is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::ExcelFile).mode ==
              cyxwiz::PipelineRuntimeSupportMode::FailClosed,
          "ExcelFile enum support should resolve to fail-closed");
    const auto* excel_file_meta = metadata.GetMetadata(gui::NodeType::ExcelFile);
    Check(excel_file_meta != nullptr,
          "ExcelFile metadata should exist");
    Check(excel_file_meta->status == cyxwiz::NodeImplementationStatus::Template,
          "ExcelFile metadata should be blocked until Excel loading is real");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::JSONFile)) == "JSONFile",
          "fail-closed runtime enum lookup for JSONFile is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::JSONFile).mode ==
              cyxwiz::PipelineRuntimeSupportMode::FailClosed,
          "JSONFile enum support should resolve to fail-closed");
    const auto* json_file_meta = metadata.GetMetadata(gui::NodeType::JSONFile);
    Check(json_file_meta != nullptr,
          "JSONFile metadata should exist");
    Check(json_file_meta->status == cyxwiz::NodeImplementationStatus::Template,
          "JSONFile metadata should be blocked until JSON loading is real");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::SQLQuery)) == "SQLQuery",
          "fail-closed runtime enum lookup for SQLQuery is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::SQLQuery).mode ==
              cyxwiz::PipelineRuntimeSupportMode::FailClosed,
          "SQLQuery enum support should resolve to fail-closed");
    const auto* sql_query_meta = metadata.GetMetadata(gui::NodeType::SQLQuery);
    Check(sql_query_meta != nullptr,
          "SQLQuery metadata should exist");
    Check(sql_query_meta->status == cyxwiz::NodeImplementationStatus::Template,
          "SQLQuery metadata should be blocked until SQL loading is real");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::HDF5Dataset)) == "HDF5Dataset",
          "fail-closed runtime enum lookup for HDF5Dataset is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::HDF5Dataset).mode ==
              cyxwiz::PipelineRuntimeSupportMode::FailClosed,
          "HDF5Dataset enum support should resolve to fail-closed");
    const auto* hdf5_meta = metadata.GetMetadata(gui::NodeType::HDF5Dataset);
    Check(hdf5_meta != nullptr,
          "HDF5Dataset metadata should exist");
    Check(hdf5_meta->status == cyxwiz::NodeImplementationStatus::Template,
          "HDF5Dataset metadata should be blocked until HDF5 loading is real");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::RESTAPISource)) == "RESTAPISource",
          "fail-closed runtime enum lookup for RESTAPISource is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::RESTAPISource).mode ==
              cyxwiz::PipelineRuntimeSupportMode::FailClosed,
          "RESTAPISource enum support should resolve to fail-closed");
    const auto* rest_api_meta = metadata.GetMetadata(gui::NodeType::RESTAPISource);
    Check(rest_api_meta != nullptr,
          "RESTAPISource metadata should exist");
    Check(rest_api_meta->status == cyxwiz::NodeImplementationStatus::Template,
          "RESTAPISource metadata should be blocked until REST loading is real");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::ExportSQL)) == "ExportSQL",
          "fail-closed runtime enum lookup for ExportSQL is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::ExportSQL).mode ==
              cyxwiz::PipelineRuntimeSupportMode::FailClosed,
          "ExportSQL enum support should resolve to fail-closed");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::StandardScaler)) == "StandardScaler",
          "operator runtime enum lookup for StandardScaler is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::StandardScaler).mode ==
              cyxwiz::PipelineRuntimeSupportMode::OperatorBacked,
          "StandardScaler enum support should resolve to operator-backed");
    Check(std::string(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(
              gui::NodeType::SVMClassifier)) == "SVMClassifier",
          "fail-closed runtime enum lookup for SVMClassifier is stable");
    Check(cyxwiz::ResolvePipelineRuntimeSupport(gui::NodeType::SVMClassifier).mode ==
              cyxwiz::PipelineRuntimeSupportMode::FailClosed,
          "SVMClassifier enum support should resolve to fail-closed");
    const gui::NodeType additional_fail_closed_enum_cases[] = {
        gui::NodeType::UMAPNode,
        gui::NodeType::SVMRegressor,
        gui::NodeType::LearningCurvesNode,
        gui::NodeType::FeatureImportanceNode,
        gui::NodeType::CrossValidationNode,
        gui::NodeType::QualityAnalyzer,
        gui::NodeType::TableSplitter,
        gui::NodeType::ExportExcel,
        gui::NodeType::IFFTNode,
        gui::NodeType::WaveletTransform,
        gui::NodeType::WordEmbeddings,
        gui::NodeType::NamedEntityRecognizer,
        gui::NodeType::ImagePreprocessor,
        gui::NodeType::ImageFolderDataset,
        gui::NodeType::AugmentationPreset,
    };
    for (const auto type : additional_fail_closed_enum_cases) {
        Check(cyxwiz::ResolvePipelineRuntimeLegacyTypeName(type) != nullptr,
              "typed fail-closed runtime enum should resolve legacy name: " +
                  TypeId(type));
        Check(cyxwiz::ResolvePipelineRuntimeSupport(type).mode ==
                  cyxwiz::PipelineRuntimeSupportMode::FailClosed,
              "typed fail-closed runtime enum should resolve fail-closed support: " +
                  TypeId(type));
    }

    const gui::NodeType blocked_metadata_cases[] = {
        gui::NodeType::LearningCurvesNode,
        gui::NodeType::FeatureImportanceNode,
        gui::NodeType::CrossValidationNode,
        gui::NodeType::ExportExcel,
        gui::NodeType::TableSplitter,
        gui::NodeType::IFFTNode,
        gui::NodeType::WaveletTransform,
        gui::NodeType::WordEmbeddings,
        gui::NodeType::NamedEntityRecognizer,
    };
    for (const auto type : blocked_metadata_cases) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr,
              "blocked fail-closed metadata should exist: " + TypeId(type));
        Check(meta->status == cyxwiz::NodeImplementationStatus::Template,
              "blocked fail-closed metadata should remain template: " +
                  TypeId(type));
    }

    Check(std::string(cyxwiz::PipelineTrainingBackendSupportModeName(
              cyxwiz::PipelineTrainingBackendSupportMode::Allowed)) ==
              "allowed",
          "training backend support mode name for allowed is stable");
    Check(std::string(cyxwiz::PipelineTrainingBackendSupportModeName(
              cyxwiz::PipelineTrainingBackendSupportMode::
                  UnsupportedSequentialModelLayer)) ==
              "unsupported_sequential_model_layer",
          "training backend support mode name for unsupported layer is stable");
    Check(std::string(cyxwiz::PipelineTrainingBackendSupportModeName(
              cyxwiz::PipelineTrainingBackendSupportMode::
                  UnsupportedTrainingControl)) ==
              "unsupported_training_control",
          "training backend support mode name for unsupported control is stable");

    for (const auto& capability : cyxwiz::GetPipelineSourceRuntimeCapabilities()) {
        Check(cyxwiz::IsPipelineSourceRuntimeNode(capability.legacy_type_name),
              std::string("source runtime name does not resolve: ") +
                  capability.legacy_type_name);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            capability.legacy_type_name);
        Check(support.mode != cyxwiz::PipelineRuntimeSupportMode::Unknown,
              std::string("source runtime name has unknown support: ") +
                  capability.legacy_type_name);
        Check(support.source_node,
              std::string("source runtime support should carry source-node axis: ") +
                  capability.legacy_type_name);
    }

    for (const auto& capability : cyxwiz::GetPipelineInputArityRuntimeCapabilities()) {
        const auto required_count = cyxwiz::ResolvePipelineRequiredInputCount(
            capability.legacy_type_name);
        Check(required_count.has_value(),
              std::string("input arity runtime name does not resolve: ") +
                  capability.legacy_type_name);
        Check(*required_count == capability.required_input_count,
              std::string("input arity runtime count mismatch: ") +
                  capability.legacy_type_name);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            capability.legacy_type_name);
        Check(support.mode != cyxwiz::PipelineRuntimeSupportMode::Unknown,
              std::string("input arity runtime name has unknown support: ") +
                  capability.legacy_type_name);
        Check(support.required_input_count.has_value() &&
                  *support.required_input_count ==
                      capability.required_input_count,
              std::string("runtime support should carry input-arity axis: ") +
                  capability.legacy_type_name);
        Check(capability.required_input_count > 1,
              std::string("input arity override should only list multi-input nodes: ") +
                  capability.legacy_type_name);
    }

    for (const auto& capability :
         cyxwiz::GetPipelineRequiredParameterRuntimeCapabilities()) {
        const auto required_parameters = cyxwiz::ResolvePipelineRequiredParameters(
            capability.legacy_type_name);
        Check(!required_parameters.empty(),
              std::string("required-parameter runtime name does not resolve: ") +
                  capability.legacy_type_name);
        Check(required_parameters.size() == capability.required_parameters.size(),
              std::string("required-parameter count mismatch: ") +
                  capability.legacy_type_name);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            capability.legacy_type_name);
        Check(support.mode != cyxwiz::PipelineRuntimeSupportMode::Unknown,
              std::string("required-parameter runtime name has unknown support: ") +
                  capability.legacy_type_name);
        Check(support.required_parameters.size() ==
                  capability.required_parameters.size(),
              std::string("runtime support should carry required-parameter axis: ") +
                  capability.legacy_type_name);
        for (const char* parameter : capability.required_parameters) {
            Check(parameter != nullptr && std::string(parameter).size() > 1,
                  std::string("required parameter name is too weak: ") +
                      capability.legacy_type_name);
        }
    }

    for (const auto& capability :
         cyxwiz::GetPipelineAllowedParameterValuesRuntimeCapabilities()) {
        const auto allowed_parameters =
            cyxwiz::ResolvePipelineAllowedParameterValues(
                capability.legacy_type_name);
        Check(!allowed_parameters.empty(),
              std::string("allowed-parameter runtime name does not resolve: ") +
                  capability.legacy_type_name);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            capability.legacy_type_name);
        Check(support.mode != cyxwiz::PipelineRuntimeSupportMode::Unknown,
              std::string("allowed-parameter runtime name has unknown support: ") +
                  capability.legacy_type_name);
        auto supported_axis = std::find_if(
            support.allowed_parameter_values.begin(),
            support.allowed_parameter_values.end(),
            [&capability](
                const cyxwiz::PipelineAllowedParameterValuesRuntimeCapability&
                    runtime_capability) {
                return std::string(runtime_capability.parameter_name) ==
                       capability.parameter_name;
            });
        Check(supported_axis != support.allowed_parameter_values.end(),
              std::string("runtime support should carry allowed-parameter axis: ") +
                  capability.legacy_type_name + "." + capability.parameter_name);
        Check(capability.parameter_name != nullptr &&
                  std::string(capability.parameter_name).size() > 1,
              std::string("allowed parameter name is too weak: ") +
                  capability.legacy_type_name);
        Check(capability.default_value != nullptr,
              std::string("allowed parameter default is missing: ") +
                  capability.legacy_type_name);
        Check(!capability.allowed_values.empty(),
              std::string("allowed parameter value list is empty: ") +
                  capability.legacy_type_name);
    }

    for (const auto& capability :
         cyxwiz::GetPipelineIntegerParameterRuntimeCapabilities()) {
        const auto integer_parameters =
            cyxwiz::ResolvePipelineIntegerParameters(capability.legacy_type_name);
        Check(!integer_parameters.empty(),
              std::string("integer-parameter runtime name does not resolve: ") +
                  capability.legacy_type_name);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            capability.legacy_type_name);
        Check(support.mode != cyxwiz::PipelineRuntimeSupportMode::Unknown,
              std::string("integer-parameter runtime name has unknown support: ") +
                  capability.legacy_type_name);
        auto supported_axis = std::find_if(
            support.integer_parameters.begin(),
            support.integer_parameters.end(),
            [&capability](
                const cyxwiz::PipelineIntegerParameterRuntimeCapability&
                    runtime_capability) {
                return std::string(runtime_capability.parameter_name) ==
                           capability.parameter_name &&
                       runtime_capability.minimum == capability.minimum &&
                       runtime_capability.comma_separated ==
                           capability.comma_separated;
            });
        Check(supported_axis != support.integer_parameters.end(),
              std::string("runtime support should carry integer-parameter axis: ") +
                  capability.legacy_type_name + "." + capability.parameter_name);
        Check(capability.parameter_name != nullptr &&
                  std::string(capability.parameter_name).size() > 1,
              std::string("integer parameter name is too weak: ") +
                  capability.legacy_type_name);
    }

    for (const auto& capability :
         cyxwiz::GetPipelineFloatParameterRuntimeCapabilities()) {
        const auto float_parameters =
            cyxwiz::ResolvePipelineFloatParameters(capability.legacy_type_name);
        Check(!float_parameters.empty(),
              std::string("float-parameter runtime name does not resolve: ") +
                  capability.legacy_type_name);
        const auto support = cyxwiz::ResolvePipelineRuntimeSupport(
            capability.legacy_type_name);
        Check(support.mode != cyxwiz::PipelineRuntimeSupportMode::Unknown,
              std::string("float-parameter runtime name has unknown support: ") +
                  capability.legacy_type_name);
        auto supported_axis = std::find_if(
            support.float_parameters.begin(),
            support.float_parameters.end(),
            [&capability](
                const cyxwiz::PipelineFloatParameterRuntimeCapability&
                    runtime_capability) {
                return std::string(runtime_capability.parameter_name) ==
                           capability.parameter_name &&
                       runtime_capability.minimum == capability.minimum &&
                       runtime_capability.maximum == capability.maximum &&
                       runtime_capability.minimum_inclusive ==
                           capability.minimum_inclusive &&
                       runtime_capability.maximum_inclusive ==
                           capability.maximum_inclusive;
            });
        Check(supported_axis != support.float_parameters.end(),
              std::string("runtime support should carry float-parameter axis: ") +
                  capability.legacy_type_name + "." + capability.parameter_name);
        Check(capability.parameter_name != nullptr &&
                  std::string(capability.parameter_name).size() > 1,
              std::string("float parameter name is too weak: ") +
                  capability.legacy_type_name);
        const bool range_has_bound =
            capability.minimum.has_value() || capability.maximum.has_value();
        const bool range_order_is_valid =
            !capability.minimum.has_value() ||
            !capability.maximum.has_value() ||
            *capability.maximum >= *capability.minimum;
        Check(range_has_bound && range_order_is_valid,
              std::string("float parameter range is invalid: ") +
                  capability.legacy_type_name + "." + capability.parameter_name);
    }

    const std::vector<gui::NodeType> supported_model_nodes = {
        gui::NodeType::Dense,
        gui::NodeType::Dropout,
        gui::NodeType::BatchNorm,
        gui::NodeType::LSTM,
        gui::NodeType::GRU,
    };
    for (auto type : supported_model_nodes) {
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr, "missing supported model metadata for type " + TypeId(type));
        Check(meta->status == cyxwiz::NodeImplementationStatus::Implemented,
              "supported model type " + TypeId(type) + " should be marked implemented");
        const auto support = cyxwiz::ResolvePipelineTrainingBackendSupport(type);
        Check(support.mode == cyxwiz::PipelineTrainingBackendSupportMode::Allowed,
              "supported model type should be allowed by training backend support: " +
                  TypeId(type));
        Check(support.compile_supported && support.training_supported,
              "supported model type should not be blocked by training backend support: " +
                  TypeId(type));
        Check(cyxwiz::IsPipelineSupportedTrainingBackendNode(type),
              "supported model type should be named in central training support: " +
                  TypeId(type));
        CheckSupportAxis(
            meta,
            "Training Backend",
            cyxwiz::PipelineTrainingBackendSupportModeName(
                cyxwiz::PipelineTrainingBackendSupportMode::Allowed),
            true,
            TypeId(type));
        CheckSupportAxis(meta, "Compile", "supported", true, TypeId(type));
        CheckSupportAxis(meta, "Training", "supported", true, TypeId(type));
    }

    for (const auto& capability :
         cyxwiz::GetPipelineFailClosedRuntimeCapabilities()) {
        auto expected_metadata_type = capability.metadata_node_type;
        if (!expected_metadata_type.has_value()) {
            expected_metadata_type = capability.node_type;
        }
        if (!expected_metadata_type.has_value()) {
            continue;
        }
        const auto* meta = metadata.GetMetadata(*expected_metadata_type);
        if (!meta) {
            Check(!capability.metadata_node_type.has_value(),
                  std::string("missing explicitly mapped fail-closed metadata for runtime name ") +
                      capability.legacy_type_name);
            continue;
        }
        if (capability.blocks_metadata_status) {
            Check(meta->status == cyxwiz::NodeImplementationStatus::Template,
                  std::string("fail-closed runtime metadata should not be marked implemented: ") +
                      capability.legacy_type_name);
            Check(meta->badge == "Blocked",
                  std::string("fail-closed runtime metadata should carry blocked badge: ") +
                      capability.legacy_type_name);
        } else {
            Check(meta->status == cyxwiz::NodeImplementationStatus::Implemented,
                  std::string("training-only fail-closed runtime metadata should keep implemented status: ") +
                      capability.legacy_type_name);
        }
        Check(capability.reason != nullptr &&
                  meta->help_text.find(capability.reason) != std::string::npos,
              std::string("fail-closed runtime metadata should expose central reason: ") +
                  capability.legacy_type_name);
        CheckSupportAxis(
            meta,
            "Runtime",
            "fail_closed",
            false,
            capability.legacy_type_name);
        const auto* runtime_axis = FindSupportAxis(meta, "Runtime");
        Check(runtime_axis != nullptr &&
                  capability.reason != nullptr &&
                  runtime_axis->reason.find(capability.reason) !=
                      std::string::npos,
              std::string("fail-closed runtime support axis should expose reason: ") +
                  capability.legacy_type_name);
        CheckSupportAxis(
            meta,
            "Pipeline Executor",
            "unsupported",
            false,
            capability.legacy_type_name);
    }

    for (const auto& capability :
         cyxwiz::GetPipelineUnsupportedSequentialModelLayerCapabilities()) {
        const auto type = capability.node_type;
        const auto* meta = metadata.GetMetadata(type);
        Check(meta != nullptr,
              "missing unsupported training metadata for type " + TypeId(type));
        Check(meta->status == cyxwiz::NodeImplementationStatus::Template,
              "unsupported training type " + TypeId(type) +
                  " should not be marked implemented");
        Check(meta->badge == "Blocked",
              "unsupported training type " + TypeId(type) +
                  " should carry blocked badge");
        Check(meta->help_text.find(cyxwiz::PipelineTrainingBackendSupportModeName(
                  cyxwiz::PipelineTrainingBackendSupportMode::
                      UnsupportedSequentialModelLayer)) != std::string::npos,
              "unsupported training type " + TypeId(type) +
                  " should expose training backend support mode");
        Check(capability.reason != nullptr &&
                  meta->help_text.find(capability.reason) != std::string::npos,
              "unsupported training type " + TypeId(type) +
                  " should expose central training backend reason");
        CheckSupportAxis(
            meta,
            "Training Backend",
            cyxwiz::PipelineTrainingBackendSupportModeName(
                cyxwiz::PipelineTrainingBackendSupportMode::
                    UnsupportedSequentialModelLayer),
            false,
            TypeId(type));
        CheckSupportAxis(meta, "Compile", "unsupported", false, TypeId(type));
        CheckSupportAxis(meta, "Training", "unsupported", false, TypeId(type));
        const auto* training_axis = FindSupportAxis(meta, "Training Backend");
        Check(training_axis != nullptr &&
                  capability.reason != nullptr &&
                  training_axis->reason.find(capability.reason) !=
                      std::string::npos,
              "unsupported training type " + TypeId(type) +
                  " should expose reason on structured support axis");
    }

    for (const auto& capability :
         cyxwiz::GetPipelineUnsupportedTrainingControlCapabilities()) {
        const auto* meta = metadata.GetMetadata(capability.node_type);
        Check(meta != nullptr,
              "missing unsupported training control metadata for type " +
                  TypeId(capability.node_type));
        Check(meta->status == cyxwiz::NodeImplementationStatus::Template,
              "unsupported training control " + TypeId(capability.node_type) +
                  " should not be marked implemented");
        Check(meta->badge == "Blocked",
              "unsupported training control " + TypeId(capability.node_type) +
                  " should carry blocked badge");
        Check(meta->help_text.find(cyxwiz::PipelineTrainingBackendSupportModeName(
                  cyxwiz::PipelineTrainingBackendSupportMode::
                      UnsupportedTrainingControl)) != std::string::npos,
              "unsupported training control " + TypeId(capability.node_type) +
                  " should expose training backend support mode");
        Check(capability.reason != nullptr &&
                  meta->help_text.find(capability.reason) != std::string::npos,
              "unsupported training control " + TypeId(capability.node_type) +
                  " should expose central training backend reason");
        CheckSupportAxis(
            meta,
            "Training Backend",
            cyxwiz::PipelineTrainingBackendSupportModeName(
                cyxwiz::PipelineTrainingBackendSupportMode::
                    UnsupportedTrainingControl),
            false,
            TypeId(capability.node_type));
        CheckSupportAxis(
            meta,
            "Compile",
            "unsupported",
            false,
            TypeId(capability.node_type));
        CheckSupportAxis(
            meta,
            "Training",
            "unsupported",
            false,
            TypeId(capability.node_type));
        const auto* training_axis = FindSupportAxis(meta, "Training Backend");
        Check(training_axis != nullptr &&
                  capability.reason != nullptr &&
                  training_axis->reason.find(capability.reason) !=
                      std::string::npos,
              "unsupported training control " + TypeId(capability.node_type) +
                  " should expose reason on structured support axis");
    }

    const auto* compare = metadata.GetMetadata(gui::NodeType::TensorCompare);
    Check(compare != nullptr, "missing TensorCompare metadata");
    Check(HasInput(compare, "A", true),
          "TensorCompare should expose required A input");
    Check(HasInput(compare, "B", false),
          "TensorCompare should expose optional B input");
    Check(HasEnumValue(compare, "op", "=="),
          "TensorCompare should expose tensor compare operators");

    const auto* logical = metadata.GetMetadata(gui::NodeType::TensorLogicalMask);
    Check(logical != nullptr, "missing TensorLogicalMask metadata");
    Check(HasInput(logical, "A", true),
          "TensorLogicalMask should expose required A input");
    Check(HasInput(logical, "B", false),
          "TensorLogicalMask should expose optional B input");
    Check(HasEnumValue(logical, "op", "not"),
          "TensorLogicalMask should keep unary not");
    Check(HasEnumValue(logical, "op", "and") &&
          HasEnumValue(logical, "op", "or"),
          "TensorLogicalMask should expose binary and/or");

    const auto* ts_window = metadata.GetMetadata(gui::NodeType::TimeSeriesWindow);
    Check(ts_window != nullptr, "missing TimeSeriesWindow metadata");
    Check(HasInputType(ts_window, "Data", gui::PinType::Dataset),
          "TimeSeriesWindow should expose Dataset input");
    Check(HasOutputType(ts_window, "Windowed", gui::PinType::Dataset),
          "TimeSeriesWindow should expose one windowed Dataset output");
    Check(HasParameter(ts_window, "value_col") &&
          HasParameter(ts_window, "feature_cols") &&
          HasParameter(ts_window, "time_col") &&
          HasParameter(ts_window, "input_width") &&
          HasParameter(ts_window, "label_width") &&
          HasParameter(ts_window, "shift"),
          "TimeSeriesWindow should expose canonical operator parameters");

    const auto* ts_features = metadata.GetMetadata(gui::NodeType::TimeSeriesFeatures);
    Check(ts_features != nullptr, "missing TimeSeriesFeatures metadata");
    Check(HasInputType(ts_features, "Data", gui::PinType::Dataset),
          "TimeSeriesFeatures should expose Dataset input");
    Check(HasOutputType(ts_features, "Enriched", gui::PinType::Dataset),
          "TimeSeriesFeatures should expose one enriched Dataset output");
    Check(HasParameter(ts_features, "value_col") &&
          HasParameter(ts_features, "lag_values") &&
          HasParameter(ts_features, "rolling_windows") &&
          HasParameter(ts_features, "rolling_aggregations"),
          "TimeSeriesFeatures should expose canonical operator parameters");

    const auto* ts_split = metadata.GetMetadata(gui::NodeType::TimeSeriesSplit);
    Check(ts_split != nullptr, "missing TimeSeriesSplit metadata");
    Check(HasInputType(ts_split, "Data", gui::PinType::Dataset),
          "TimeSeriesSplit should expose Dataset input");
    Check(HasOutputType(ts_split, "Partitioned", gui::PinType::Dataset),
          "TimeSeriesSplit should expose one partitioned Dataset output");
    Check(HasParameter(ts_split, "train_ratio") &&
          HasParameter(ts_split, "val_ratio") &&
          HasParameter(ts_split, "test_ratio"),
          "TimeSeriesSplit should expose train/val/test ratio parameters");

    std::cout << "Pipeline operator metadata drift guard passed\n";
    return 0;
}
