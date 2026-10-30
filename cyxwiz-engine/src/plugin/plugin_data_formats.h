#pragma once

// Connects the plugins' dataset formats (PluginDataLoaderRegistry) to the
// core's external-format hook (core/data_input_external_formats.h) and to
// DataRegistry's reader, so Data Input, pipelines and the asset browser see
// them without depending on the plugin system (TOFIX181). Called once at
// startup, before plugins load; the formats follow the registry live.

namespace cyxwiz::plugin {

void InstallPluginDataFormats();

}  // namespace cyxwiz::plugin
