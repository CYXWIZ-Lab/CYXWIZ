# Example Data Formats: a CyxWiz data plugin

A C++ plugin that adds a dataset file format and a dataset analytic:

- **ARFF (Weka)**: Data Input opens `.arff` files. Numeric attributes become
  number columns, the others text, `?` an empty value.
- **Coefficient of variation**: Data Studio's Explore > Plugins tab computes
  standard deviation / |mean| of the numeric columns you pick.

Both reach the Engine through the Arrow C stream interface
(`plugin/interfaces/arrow_c_stream.h` in the SDK), filled and read by hand:
no Arrow library is needed.

| File | What it is |
|---|---|
| `data_formats_plugin.cpp` | The plugin. Its first comment lists the order the Engine calls it in |
| `plugin.json` | The manifest: id, version, plugin API 1.3.0, the `DataRegistry` permission, the library of each platform |
| `CMakeLists.txt` | The build |
| `weather.arff` | A sample dataset: 14 rows, 5 attributes |
| `built/` | In the SDK only: this plugin already built by CyxWiz, ready to Install |

## Try it without building

In the Engine: Plugin Manager > Install..., choose
`<CyxWiz SDK folder>/examples/plugins/data-formats/built`. Then open
`weather.arff` in a Data Input, Apply, and run Explore > Plugins >
Coefficient of variation: temperature 0.0893, humidity 0.1260.

## Build it yourself

Copy this folder out of the installation, then:

```
cmake -S . -B build -DCMAKE_PREFIX_PATH=<CyxWiz SDK folder> -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
<CyxWiz installation>/cyxwiz-cli plugin-check build/data_formats
```

Use the compiler the Engine was built with (the SDK names it in
`lib/cmake/CyxWizSDK/CyxWizSDKConfig.cmake`, line `# Built with:`).
`build/data_formats/` is the plugin folder. If you installed `built/`
before, Unload it in the Plugin Manager first: both have the same id.

## Documentation

`docs/extending/data-plugin-guide.md` in the CyxWiz repository on GitHub,
<https://github.com/CYXWIZ-Lab/CYXWIZ>, walks through this plugin: building
and checking it, what the Engine does with each call, and how to write your
own format.

Licensed under the Apache License 2.0 (the SDK's `LICENSE` and `NOTICE`).
