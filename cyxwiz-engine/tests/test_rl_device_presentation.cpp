// What the RL surfaces say about a PPO run's device (TOFIX206): the
// dashboard chip and note, the toolbar word, Compile's small-network advice.
#include "../src/core/rl_device_presentation.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

using namespace cyxwiz::rl_device;

RunDevice Cpu() {
    RunDevice d;
    d.valid = true;
    d.requested_backend = d.effective_backend = "arrayfire_cpu";
    d.device_name = "Intel";
    d.policy = "strict";
    return d;
}

RunDevice Cuda() {
    RunDevice d;
    d.valid = true;
    d.requested_backend = d.effective_backend = "arrayfire_cuda";
    d.device_name = "GeForce GTX 1050 Ti";
    d.policy = "compatibility";
    return d;
}

void CheckChipAndToolbar() {
    const Chip cpu = DeviceChip(Cpu());
    Check(cpu.text == "CPU \xC2\xB7 Intel" && !cpu.fallback, "CPU chip: " + cpu.text);
    Check(cpu.tooltip == "arrayfire_cpu device 0 \xC2\xB7 Intel \xC2\xB7 policy Strict", "CPU route: " + cpu.tooltip);
    Check(DeviceChip(Cuda()).text == "CUDA \xC2\xB7 GeForce GTX 1050 Ti", "CUDA chip");
    Check(ToolbarSuffix(Cuda()) == " \xC2\xB7 CUDA", "toolbar word");

    RunDevice fb = Cpu();
    fb.requested_backend = "arrayfire_cuda";
    fb.policy = "compatibility";
    fb.fallback = true;
    fb.error = "not in the current inventory";
    const Chip chip = DeviceChip(fb);
    Check(chip.fallback && chip.text == "CPU \xC2\xB7 fallback", "fallback chip: " + chip.text);
    Check(chip.tooltip.find("The chosen CUDA device 0 failed its check (not in the current inventory)") == 0,
          "the fallback's reason: " + chip.tooltip);
    Check(ToolbarSuffix(fb) == " \xC2\xB7 CPU (fallback)", "toolbar fallback word");
    Check(DashboardNote(fb, {64, 64}, {64, 64}, "").find("Strict would have refused") != std::string::npos,
          "the dashboard explains the fallback (at once)");

    Check(DeviceChip(RunDevice{}).text.empty() && ToolbarSuffix(RunDevice{}).empty(), "nothing before preflight");
    Check(BackendWord("arrayfire_opencl") == "OpenCL" && BackendWord("arrayfire_oneapi") == "oneAPI", "backend words");
}

void CheckAdvice() {
    Check(SmallNetworks({64, 64}, {64, 64}) && SmallNetworks({256, 256}, {256}) && !SmallNetworks({512}, {64}),
          "small = every hidden layer at most 256 units");
    Check(Widths({64, 64}, {64, 64}) == "64-64" && Widths({64, 64}, {128}) == "64-64 and 128", "widths");
    const std::string advice = CompileAdvice("CUDA", {64, 64}, {64, 64}, false);
    Check(advice == "Small networks (64-64) usually train faster on the CPU, and the many small GPU steps of PPO strain "
                    "some graphics drivers. Training uses CUDA, as chosen in Preferences > Devices.",
          "Compile advice: " + advice);
    Check(CompileAdvice("CPU", {64, 64}, {64, 64}, false).empty(), "no advice on the CPU");
    Check(CompileAdvice("CUDA", {512, 512}, {512, 512}, false).empty(), "no advice for large networks");
    Check(CompileAdvice("CUDA", {64, 64}, {64, 64}, true).empty(),
          "no advice when the device's provider will run the networks as fused kernels");
    const std::string layer_path = "ArrayFire operations (no neural provider runs this network on opencl)";
    Check(DashboardNote(Cuda(), {64, 64}, {64, 64}, layer_path).find("Preferences > Devices chooses where training runs") !=
              std::string::npos,
          "the dashboard repeats the advice on a GPU");
    Check(DashboardNote(Cpu(), {64, 64}, {64, 64}, layer_path).empty(), "no note on the CPU");
    Check(DashboardNote(Cuda(), {64, 64}, {64, 64}, "fused small-network kernels (cyxwiz.nvidia-cublas-cell)").empty(),
          "no note when the run uses the fused kernels");
    Check(DashboardNote(Cuda(), {64, 64}, {64, 64}, "").empty(), "no advice before the run says how it computes");
}

}  // namespace

int main() {
    CheckChipAndToolbar();
    CheckAdvice();
    if (failures == 0) std::printf("test_rl_device_presentation: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
