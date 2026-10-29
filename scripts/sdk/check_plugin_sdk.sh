#!/usr/bin/env bash
# Checks the CyxWiz plugin SDK end to end (TOFIX125 P5 step 5.3):
#   1. installs the SDK from a CyxWiz build into <work>/sdk,
#   2. copies the template plugin and the example plugin out of the source
#      tree, and the data plugin example out of the installed SDK, and builds
#      them against that SDK only (the template also in Debug, which must
#      still load),
#   3. loads each result, and the data plugin example the SDK ships built,
#      with cyxwiz-plugin-check, which checks the build fingerprint, the
#      gradients of every node and that formats and analytics are added.
#
# Usage: scripts/sdk/check_plugin_sdk.sh <CyxWiz build folder> [<work folder>]
# The CyxWiz build must be a Release build that includes cyxwiz-plugin-check.
# ArrayFire's lib folder (and CUDA's bin folder for the CUDA backend) must be
# on PATH, as for the Engine's tests.
set -u

build=${1:?usage: check_plugin_sdk.sh <CyxWiz build folder> [<work folder>]}
build=$(cd "$build" && pwd)
source_root=$(cd "$(dirname "$0")/../.." && pwd)
work=${2:-${TMPDIR:-/tmp}/cyxwiz_plugin_sdk_check}
mkdir -p "$work"
work=$(cd "$work" && pwd)
rm -rf "$work/sdk" "$work/src" "$work/build"

failures=0
step() { echo; echo "== $*"; }
fail() { echo "FAIL: $*"; failures=$((failures + 1)); }

step "install the SDK into $work/sdk"
cmake --install "$build" --config Release --component plugin_sdk --prefix "$work/sdk" > "$work/install.log" 2>&1 \
    || { cat "$work/install.log"; echo "FAIL: install"; exit 1; }
echo "$(find "$work/sdk" -type f | wc -l) files"
for unexpected in $(cd "$work/sdk/include" && ls); do
    case $unexpected in cyxwiz|plugin|core) ;; *) fail "unexpected include folder $unexpected" ;; esac
done
core_headers=$(ls "$work/sdk/include/core" | tr '\n' ' ')
[ "$core_headers" = "extension_node_contract.h graph_model.h node_metadata.h " ] \
    || fail "core/ should hold only the three contract headers, has: $core_headers"
[ -f "$work/sdk/lib/cmake/CyxWizSDK/CyxWizSDKConfig.cmake" ] || fail "no CyxWizSDKConfig.cmake"
backend_found=no
for library in "$work/sdk/lib/cyxwiz-backend.lib" "$work/sdk/lib/libcyxwiz-backend.so" "$work/sdk/lib/libcyxwiz-backend.dylib"; do
    if [ -f "$library" ] && [ ! -L "$library" ]; then backend_found=yes; fi
done
[ "$backend_found" = yes ] || fail "no backend library (a real file, not a symlink) in sdk/lib"
examples="$work/sdk/examples/python-nodes/builtin-layers"
for file in builtin_layers.py test_builtin_layers.py blobs_python_layers.cyxgraph blobs.csv             pycyxwiz_test_env.py nodes/builtin-layers.cyxnode/node.json nodes/builtin-layers.cyxnode/node.py; do
    [ -f "$examples/$file" ] || fail "the Python examples lack $file"
done
# The Python examples' own test, from the installed SDK, when a Python with
# pycyxwiz is given: CYXWIZ_SDK_CHECK_PYTHON=<python> (pycyxwiz from the build).
if [ -n "${CYXWIZ_SDK_CHECK_PYTHON:-}" ]; then
    step "Python example nodes: their test, run from the installed SDK"
    (cd "$examples" && "$CYXWIZ_SDK_CHECK_PYTHON" test_builtin_layers.py "$build/lib/Release" "$build/bin/Release"         > "$work/python_examples.log" 2>&1) || fail "test_builtin_layers.py (see $work/python_examples.log)"
    tail -1 "$work/python_examples.log"
fi

data_example="$work/sdk/examples/plugins/data-formats"
for file in CMakeLists.txt plugin.json data_formats_plugin.cpp README.md weather.arff built/plugin.json; do
    [ -f "$data_example/$file" ] || fail "the data plugin example lacks $file"
done
ls "$data_example/built/bin" 2>/dev/null | grep -qE "^(lib)?data_formats\.(dll|so|dylib)$" \
    || fail "the data plugin example's built/bin has no library"

# name source-folder [config]
build_plugin() {
    local name=$1 source=$2 config=${3:-Release}
    local copy="$work/src/$name"
    step "build $name ($config) out of tree"
    mkdir -p "$copy"
    cp -r "$source"/. "$copy"/
    rm -rf "$copy/bin" "$copy/built"
    if ! cmake -S "$copy" -B "$work/build/$name" -DCMAKE_PREFIX_PATH="$work/sdk" \
            -DCMAKE_BUILD_TYPE="$config" > "$work/$name.configure.log" 2>&1; then
        cat "$work/$name.configure.log"; fail "$name configure"; return
    fi
    if ! cmake --build "$work/build/$name" --config "$config" > "$work/$name.build.log" 2>&1; then
        grep -E "error|Error" "$work/$name.build.log" | head -20; fail "$name build"; return
    fi
    grep -E "warning" "$work/$name.build.log" | grep -v MSB8029 | head -5  # MSB8029: work folder under TEMP
    echo "built: $(ls "$work/build/$name"/*/bin 2>/dev/null | tr '\n' ' ')"
}

build_plugin my_nodes "$source_root/plugins/template"
build_plugin my_nodes_debug "$source_root/plugins/template" Debug
build_plugin activation_nodes "$source_root/plugins/examples/activation_nodes"
build_plugin data_formats "$data_example"

step "load and check with cyxwiz-plugin-check"
check="$build/bin/Release/cyxwiz-plugin-check"
export PATH="$build/bin/Release:$PATH"
folders=()
for folder in "$work/build/my_nodes/my_nodes" "$work/build/my_nodes_debug/my_nodes" \
              "$work/build/activation_nodes/activation_nodes" "$work/build/data_formats/data_formats"               "$data_example/built"; do
    if [ -f "$folder/plugin.json" ]; then folders+=("$folder"); else fail "no plugin folder $folder"; fi
done
if [ ${#folders[@]} -gt 0 ]; then
    "$check" "${folders[@]}" > "$work/check.log" 2>&1
    rc=$?
    grep -v "\] \[info\]" "$work/check.log"
    [ $rc -eq 0 ] || fail "cyxwiz-plugin-check"
fi

echo
if [ $failures -eq 0 ]; then
    echo "check_plugin_sdk: all passed"
else
    echo "check_plugin_sdk: $failures failure(s); logs in $work"
fi
[ $failures -eq 0 ]
