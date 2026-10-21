// ui_watchdog.cpp - names what blocks the UI thread (TOFIX139 hang).
#include "gui/ui_watchdog.h"

#include <spdlog/spdlog.h>

#include <atomic>
#include <string>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#endif

namespace cyxwiz::servernode::gui {

namespace {

std::atomic<long long> g_last_frame_ms{0};
std::atomic<bool> g_running{false};
std::thread g_thread;
#ifdef _WIN32
HANDLE g_ui_thread = nullptr;
#endif

long long NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

#ifdef _WIN32
// Walks the (suspended) UI thread's stack. Symbols come from the PDB beside
// the executable when there is one; otherwise the module and offset still
// say which code it is.
std::string CaptureUiStack() {
    std::string out;
    const HANDLE process = GetCurrentProcess();
    static bool symbols_ready = false;
    if (!symbols_ready) {
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
        symbols_ready = SymInitialize(process, nullptr, TRUE) != FALSE;
    }
    if (SuspendThread(g_ui_thread) == static_cast<DWORD>(-1)) return "could not suspend the UI thread";
    CONTEXT context = {};
    context.ContextFlags = CONTEXT_FULL;
    if (!GetThreadContext(g_ui_thread, &context)) {
        ResumeThread(g_ui_thread);
        return "could not read the UI thread context";
    }
    STACKFRAME64 frame = {};
    frame.AddrPC.Mode = frame.AddrFrame.Mode = frame.AddrStack.Mode = AddrModeFlat;
#ifdef _M_X64
    const DWORD machine = IMAGE_FILE_MACHINE_AMD64;
    frame.AddrPC.Offset = context.Rip;
    frame.AddrFrame.Offset = context.Rbp;
    frame.AddrStack.Offset = context.Rsp;
#else
    const DWORD machine = IMAGE_FILE_MACHINE_I386;
    frame.AddrPC.Offset = context.Eip;
    frame.AddrFrame.Offset = context.Ebp;
    frame.AddrStack.Offset = context.Esp;
#endif
    for (int depth = 0; depth < 40; ++depth) {
        if (!StackWalk64(machine, process, g_ui_thread, &frame, &context, nullptr, SymFunctionTableAccess64,
                         SymGetModuleBase64, nullptr) ||
            frame.AddrPC.Offset == 0) {
            break;
        }
        char module_name[MAX_PATH] = "?";
        HMODULE module = nullptr;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCSTR>(frame.AddrPC.Offset), &module)) {
            GetModuleFileNameA(module, module_name, MAX_PATH);
        }
        const char* base_name = strrchr(module_name, '\\');
        base_name = base_name ? base_name + 1 : module_name;
        char symbol_buffer[sizeof(SYMBOL_INFO) + 256] = {};
        auto* symbol = reinterpret_cast<SYMBOL_INFO*>(symbol_buffer);
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        symbol->MaxNameLen = 255;
        DWORD64 displacement = 0;
        std::string name;
        if (symbols_ready && SymFromAddr(process, frame.AddrPC.Offset, &displacement, symbol)) {
            name = std::string(symbol->Name) + "+0x" + std::to_string(displacement);
            IMAGEHLP_LINE64 line = {};
            line.SizeOfStruct = sizeof(line);
            DWORD line_displacement = 0;
            if (SymGetLineFromAddr64(process, frame.AddrPC.Offset, &line_displacement, &line)) {
                const char* file = strrchr(line.FileName, '\\');
                name += " (" + std::string(file ? file + 1 : line.FileName) + ":" + std::to_string(line.LineNumber) + ")";
            }
        } else {
            char offset[32];
            snprintf(offset, sizeof(offset), "+0x%llx", static_cast<unsigned long long>(frame.AddrPC.Offset - reinterpret_cast<DWORD64>(module)));
            name = offset;
        }
        out += "\n    #" + std::to_string(depth) + " " + base_name + "!" + name;
    }
    ResumeThread(g_ui_thread);
    return out;
}
#endif

void WatchLoop(std::chrono::milliseconds stall_after) {
    long long reported_for = 0;
    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        const long long last = g_last_frame_ms.load();
        if (last == 0) continue;
        const long long stalled_ms = NowMs() - last;
        if (stalled_ms < stall_after.count()) {
            reported_for = 0;
            continue;
        }
        if (reported_for == last) continue;  // one report per stall
        reported_for = last;
#ifdef _WIN32
        spdlog::error("UI thread has not drawn a frame for {} ms; its stack:{}", stalled_ms, CaptureUiStack());
#else
        spdlog::error("UI thread has not drawn a frame for {} ms", stalled_ms);
#endif
    }
}

}  // namespace

void UiWatchdog::Start(std::chrono::milliseconds stall_after) {
    if (g_running.exchange(true)) return;
#ifdef _WIN32
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &g_ui_thread, 0, FALSE,
                    DUPLICATE_SAME_ACCESS);
#endif
    g_last_frame_ms.store(NowMs());
    g_thread = std::thread(WatchLoop, stall_after);
}

void UiWatchdog::Frame() { g_last_frame_ms.store(NowMs()); }

void UiWatchdog::Stop() {
    if (!g_running.exchange(false)) return;
    if (g_thread.joinable()) g_thread.join();
#ifdef _WIN32
    if (g_ui_thread) {
        CloseHandle(g_ui_thread);
        g_ui_thread = nullptr;
    }
#endif
}

UiWatchdog::~UiWatchdog() { Stop(); }

}  // namespace cyxwiz::servernode::gui
