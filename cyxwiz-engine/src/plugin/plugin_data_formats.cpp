#include "plugin_data_formats.h"

#include "registries/plugin_data_loader_registry.h"
#include "../core/data_input_external_formats.h"
#include "../core/data_registry.h"

namespace cyxwiz::plugin {

namespace {

bool LoaderOf(const std::string& format, PluginDataLoaderRegistry::Loader& out) {
    for (auto& loader : PluginDataLoaderRegistry::Instance().Loaders()) {
        if (data_input::ExternalFormatName(loader.info.loader_id) == format) {
            out = std::move(loader);
            return true;
        }
    }
    return false;
}

}  // namespace

void InstallPluginDataFormats() {
    data_input::ExternalFormatSource source;
    source.list = [] {
        std::vector<data_input::ExternalFormat> formats;
        for (const auto& loader : PluginDataLoaderRegistry::Instance().Loaders()) {
            data_input::ExternalFormat format;
            format.name = data_input::ExternalFormatName(loader.info.loader_id);
            format.label = loader.info.format_name;
            format.extensions = loader.info.extensions;
            format.version = loader.plugin_id + "@" + loader.plugin_version;
            formats.push_back(std::move(format));
        }
        return formats;
    };
    source.detect = [](const std::string& path) {
        const auto loader_id = PluginDataLoaderRegistry::Instance().LoaderForFile(path);
        return loader_id.empty() ? std::string{} : data_input::ExternalFormatName(loader_id);
    };
    data_input::SetExternalFormatSource(std::move(source));

    DataRegistry::Instance().SetExternalReaderOpener(
        [](const std::string& format, const std::string& path,
           std::shared_ptr<arrow::RecordBatchReader>& reader, std::string& error) {
            PluginDataLoaderRegistry::Loader loader;
            if (!LoaderOf(format, loader)) {
                error = "The format '" + format + "' is not installed (its plugin is not loaded)";
                return false;
            }
            auto opened = PluginDataLoaderRegistry::Instance().Open(loader.info.loader_id, path);
            if (!opened.ok()) {
                error = opened.status().message();
                return false;
            }
            reader = *opened;
            return true;
        });
}

}  // namespace cyxwiz::plugin
