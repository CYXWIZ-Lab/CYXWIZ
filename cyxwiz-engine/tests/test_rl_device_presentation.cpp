// What the RL surfaces say about a PPO run's device (TOFIX206): the
// device route and dashboard note, the toolbar word, and Compile's line on how
// the networks will compute: the actual path and its reason, with what to do
// when a provider is missing, and the small-network advice only where small
// networks are the cause.
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

void CheckRouteAndToolbar() {
    const std::string cpu = DeviceRoute(Cpu());
    Check(cpu == "arrayfire_cpu device 0 \xC2\xB7 Intel \xC2\xB7 policy Strict", "CPU route: " + cpu);
    Check(DeviceRoute(Cuda()) == "arrayfire_cuda device 0 \xC2\xB7 GeForce GTX 1050 Ti \xC2\xB7 policy Compatibility",
          "CUDA route");
    Check(ToolbarSuffix(Cuda()) == " \xC2\xB7 CUDA", "toolbar word");

    RunDevice fb = Cpu();
    fb.requested_backend = "arrayfire_cuda";
    fb.policy = "compatibility";
    fb.fallback = true;
    fb.error = "not in the current inventory";
    const std::string route = DeviceRoute(fb);
    Check(route.find("The chosen CUDA device 0 failed its check (not in the current inventory)") == 0,
          "the fallback's reason: " + route);
    Check(ToolbarSuffix(fb) == " \xC2\xB7 CPU (fallback)", "toolbar fallback word");
    Check(DashboardNote(fb, {64, 64}, {64, 64}, "", "").find("Strict would have refused") != std::string::npos,
          "the dashboard explains the fallback (at once)");

    Check(DeviceRoute(RunDevice{}).empty() && ToolbarSuffix(RunDevice{}).empty(), "nothing before preflight");
    Check(BackendWord("arrayfire_opencl") == "OpenCL" && BackendWord("arrayfire_oneapi") == "oneAPI", "backend words");
}

void CheckAdvice() {
    Check(SmallNetworks({64, 64}, {64, 64}) && SmallNetworks({256, 256}, {256}) && !SmallNetworks({512}, {64}),
          "small = every hidden layer at most 256 units");
    Check(Widths({64, 64}, {64, 64}) == "64-64" && Widths({64, 64}, {128}) == "64-64 and 128", "widths");
    const std::string small = "Small networks (64-64) usually train faster on the CPU, and the many small GPU steps of RL "
                              "training strain some graphics drivers.";
    const std::string missing = "NVIDIA provider not installed. Install the CUDA GPU pack.";

    // A provider that is missing: the path, the reason and what to do. The
    // small-network advice is not given: it would name the wrong cause.
    NetworkForecast no_provider;
    no_provider.missing_provider = missing;
    const std::string absent = CompileAdvice("CUDA", {64, 64}, {64, 64}, no_provider);
    Check(absent == "The networks run as ArrayFire operations on CUDA, not as fused kernels: NVIDIA provider not "
                    "installed. Install the CUDA GPU pack. Training uses CUDA, as chosen in Preferences > Devices.",
          "Compile, missing provider: " + absent);
    Check(absent.find("Small networks") == std::string::npos, "a missing provider is not blamed on small networks");

    // A provider serves the device and declines these networks, or none is
    // offered for them there: the path and that reason, and the small-network
    // advice, which is the real cause of a slow run then
    NetworkForecast not_offered;
    not_offered.why = "No provider is offered for this on OpenCL";
    const std::string advice = CompileAdvice("OpenCL", {64, 64}, {64, 64}, not_offered);
    Check(advice == "The networks run as ArrayFire operations on OpenCL: No provider is offered for this on OpenCL. " +
                        small + " Training uses OpenCL, as chosen in Preferences > Devices.",
          "Compile, nothing offered: " + advice);
    NetworkForecast declined;
    declined.why = "cyxwiz.nvidia-cublas-cell declined this network: layer width 768 exceeds 512";
    const std::string wide = CompileAdvice("CUDA", {768, 768}, {768, 768}, declined);
    Check(wide == "The networks run as ArrayFire operations on CUDA: cyxwiz.nvidia-cublas-cell declined this network: "
                  "layer width 768 exceeds 512. Training uses CUDA, as chosen in Preferences > Devices.",
          "Compile, a declined network that is not small: the reason, no small-network advice: " + wide);

    // A provider runs them: the path, and nothing to do
    NetworkForecast fused;
    fused.fused = true;
    fused.provider_id = "cyxwiz.nvidia-cublas-cell";
    const std::string on_provider = CompileAdvice("CUDA", {64, 64}, {64, 64}, fused);
    Check(on_provider == "The networks run as fused small-network kernels on cyxwiz.nvidia-cublas-cell. Training uses "
                         "CUDA, as chosen in Preferences > Devices.",
          "Compile, fused: " + on_provider);
    Check(CompileAdvice("CPU", {64, 64}, {64, 64}, not_offered).empty() &&
              CompileAdvice("CPU", {64, 64}, {64, 64}, no_provider).empty(),
          "nothing on the CPU: ArrayFire is the designed path there");
    Check(CompileAdvice("", {64, 64}, {64, 64}, no_provider).empty(), "nothing before the route is known");

    // The dashboard: the same causes, told apart the same way
    const std::string layer_path = "ArrayFire operations (No provider is offered for this on OpenCL)";
    const std::string advice_note = DashboardNote(Cuda(), {64, 64}, {64, 64}, layer_path, "");
    Check(advice_note == small + " Preferences > Devices chooses where training runs.",
          "the dashboard gives the small-network advice where it is the cause: " + advice_note);
    const std::string absent_path = "ArrayFire operations (" + missing + ")";
    const std::string absent_note = DashboardNote(Cuda(), {64, 64}, {64, 64}, absent_path, missing);
    Check(absent_note == "The networks run as ArrayFire operations on CUDA, not as fused kernels: NVIDIA provider not "
                         "installed. Install the CUDA GPU pack.",
          "the dashboard names a missing provider and what to do: " + absent_note);
    Check(DashboardNote(Cuda(), {768, 768}, {768, 768}, absent_path, missing) == absent_note,
          "and does so whatever the networks' size");
    Check(DashboardNote(Cuda(), {768, 768}, {768, 768}, layer_path, "").empty(),
          "no note for large networks a provider declines: the compute path row says why");
    Check(DashboardNote(Cpu(), {64, 64}, {64, 64}, layer_path, "").empty() &&
              DashboardNote(Cpu(), {64, 64}, {64, 64}, absent_path, missing).empty(),
          "no note on the CPU");
    Check(DashboardNote(Cuda(), {64, 64}, {64, 64}, "fused small-network kernels (cyxwiz.nvidia-cublas-cell)", "").empty(),
          "no note when the run uses the fused kernels");
    Check(DashboardNote(Cuda(), {64, 64}, {64, 64}, "", "").empty() &&
              DashboardNote(Cuda(), {64, 64}, {64, 64}, "", missing).empty(),
          "no note before the run says how it computes");
}

}  // namespace

int main() {
    CheckRouteAndToolbar();
    CheckAdvice();
    if (failures == 0) std::printf("test_rl_device_presentation: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
