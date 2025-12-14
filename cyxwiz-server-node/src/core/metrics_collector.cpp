// metrics_collector.cpp - System metrics collection implementation
#include "core/metrics_collector.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#include <dxgi.h>
#pragma comment(lib, "pdh.lib")
#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "dxgi.lib")
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/statvfs.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <net/if.h>
#include <net/if_dl.h>
#include <ifaddrs.h>
#include <fstream>
#include <sstream>
#include <cstring>
#else
// Linux
#include <sys/sysinfo.h>
#include <sys/statvfs.h>
#include <fstream>
#include <sstream>
#endif

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz::servernode::core {

// ========== NVML Types and Function Pointers ==========
#ifdef _WIN32

// NVML return type
typedef int nvmlReturn_t;
#define NVML_SUCCESS 0

// NVML device handle
typedef void* nvmlDevice_t;

// NVML memory info
typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

// NVML utilization
typedef struct {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;

// NVML function pointer types
typedef nvmlReturn_t (*nvmlInit_t)(void);
typedef nvmlReturn_t (*nvmlShutdown_t)(void);
typedef nvmlReturn_t (*nvmlDeviceGetCount_t)(unsigned int*);
typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndex_t)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t (*nvmlDeviceGetName_t)(nvmlDevice_t, char*, unsigned int);
typedef nvmlReturn_t (*nvmlDeviceGetMemoryInfo_t)(nvmlDevice_t, nvmlMemory_t*);
typedef nvmlReturn_t (*nvmlDeviceGetUtilizationRates_t)(nvmlDevice_t, nvmlUtilization_t*);
typedef nvmlReturn_t (*nvmlDeviceGetTemperature_t)(nvmlDevice_t, int, unsigned int*);
typedef nvmlReturn_t (*nvmlDeviceGetPowerUsage_t)(nvmlDevice_t, unsigned int*);
typedef nvmlReturn_t (*nvmlDeviceGetEncoderUtilization_t)(nvmlDevice_t, unsigned int*, unsigned int*);
typedef nvmlReturn_t (*nvmlDeviceGetDecoderUtilization_t)(nvmlDevice_t, unsigned int*, unsigned int*);

// NVML function pointers (dynamically loaded)
static HMODULE nvml_lib = nullptr;
static nvmlInit_t pNvmlInit = nullptr;
static nvmlShutdown_t pNvmlShutdown = nullptr;
static nvmlDeviceGetCount_t pNvmlDeviceGetCount = nullptr;
static nvmlDeviceGetHandleByIndex_t pNvmlDeviceGetHandleByIndex = nullptr;
static nvmlDeviceGetName_t pNvmlDeviceGetName = nullptr;
static nvmlDeviceGetMemoryInfo_t pNvmlDeviceGetMemoryInfo = nullptr;
static nvmlDeviceGetUtilizationRates_t pNvmlDeviceGetUtilizationRates = nullptr;
static nvmlDeviceGetTemperature_t pNvmlDeviceGetTemperature = nullptr;
static nvmlDeviceGetPowerUsage_t pNvmlDeviceGetPowerUsage = nullptr;
static nvmlDeviceGetEncoderUtilization_t pNvmlDeviceGetEncoderUtilization = nullptr;
static nvmlDeviceGetDecoderUtilization_t pNvmlDeviceGetDecoderUtilization = nullptr;

// Temperature sensor type
#define NVML_TEMPERATURE_GPU 0

static bool LoadNVML() {
    if (nvml_lib) return true;  // Already loaded

    nvml_lib = LoadLibraryA("nvml.dll");
    if (!nvml_lib) {
        spdlog::debug("NVML not found (nvml.dll) - NVIDIA metrics unavailable");
        return false;
    }

    pNvmlInit = (nvmlInit_t)GetProcAddress(nvml_lib, "nvmlInit_v2");
    pNvmlShutdown = (nvmlShutdown_t)GetProcAddress(nvml_lib, "nvmlShutdown");
    pNvmlDeviceGetCount = (nvmlDeviceGetCount_t)GetProcAddress(nvml_lib, "nvmlDeviceGetCount_v2");
    pNvmlDeviceGetHandleByIndex = (nvmlDeviceGetHandleByIndex_t)GetProcAddress(nvml_lib, "nvmlDeviceGetHandleByIndex_v2");
    pNvmlDeviceGetName = (nvmlDeviceGetName_t)GetProcAddress(nvml_lib, "nvmlDeviceGetName");
    pNvmlDeviceGetMemoryInfo = (nvmlDeviceGetMemoryInfo_t)GetProcAddress(nvml_lib, "nvmlDeviceGetMemoryInfo");
    pNvmlDeviceGetUtilizationRates = (nvmlDeviceGetUtilizationRates_t)GetProcAddress(nvml_lib, "nvmlDeviceGetUtilizationRates");
    pNvmlDeviceGetTemperature = (nvmlDeviceGetTemperature_t)GetProcAddress(nvml_lib, "nvmlDeviceGetTemperature");
    pNvmlDeviceGetPowerUsage = (nvmlDeviceGetPowerUsage_t)GetProcAddress(nvml_lib, "nvmlDeviceGetPowerUsage");
    pNvmlDeviceGetEncoderUtilization = (nvmlDeviceGetEncoderUtilization_t)GetProcAddress(nvml_lib, "nvmlDeviceGetEncoderUtilization");
    pNvmlDeviceGetDecoderUtilization = (nvmlDeviceGetDecoderUtilization_t)GetProcAddress(nvml_lib, "nvmlDeviceGetDecoderUtilization");

    if (!pNvmlInit || !pNvmlDeviceGetCount || !pNvmlDeviceGetHandleByIndex) {
        spdlog::warn("NVML loaded but essential functions not found");
        FreeLibrary(nvml_lib);
        nvml_lib = nullptr;
        return false;
    }

    spdlog::debug("NVML loaded successfully");
    return true;
}

static void UnloadNVML() {
    if (nvml_lib) {
        if (pNvmlShutdown) {
            pNvmlShutdown();
        }
        FreeLibrary(nvml_lib);
        nvml_lib = nullptr;
    }
}

// ========== AMD ADL Types and Function Pointers ==========

// ADL return codes
#define ADL_OK 0
#define ADL_ERR -1
#define ADL_ERR_NOT_INIT -2
#define ADL_ERR_INVALID_PARAM -3
#define ADL_ERR_NOT_SUPPORTED -8

// Maximum number of GL-Sync connectors per display
#define ADL_MAX_PATH 256

// ADL adapter info structure
typedef struct ADLAdapterInfo {
    int iSize;                          // Size of the structure
    int iAdapterIndex;                  // The ADL index handle
    char strUDID[ADL_MAX_PATH];         // UDID string
    int iBusNumber;                     // Bus number
    int iDeviceNumber;                  // Device number
    int iFunctionNumber;                // Function number
    int iVendorID;                      // Vendor ID
    char strAdapterName[ADL_MAX_PATH];  // Adapter name
    char strDisplayName[ADL_MAX_PATH];  // Display name
    int iPresent;                       // Present or not
    int iExist;                         // Exist or not
    char strDriverPath[ADL_MAX_PATH];   // Driver path
    char strDriverPathExt[ADL_MAX_PATH];// Driver path extension
    char strPNPString[ADL_MAX_PATH];    // PNP string
    int iOSDisplayIndex;                // OS display index
} ADLAdapterInfo, *LPADLAdapterInfo;

// ADL temperature structure
typedef struct ADLTemperature {
    int iSize;                  // Size of the structure
    int iTemperature;           // Temperature in millidegrees Celsius
} ADLTemperature;

// ADL PM activity structure (Overdrive5)
typedef struct ADLPMActivity {
    int iSize;                  // Size of the structure
    int iEngineClock;           // Current engine clock in 10kHz
    int iMemoryClock;           // Current memory clock in 10kHz
    int iVddc;                  // Current VDDC in mV
    int iActivityPercent;       // GPU activity percentage
    int iCurrentPerformanceLevel;// Current performance level
    int iCurrentBusSpeed;       // Current bus speed
    int iCurrentBusLanes;       // Current bus lanes
    int iMaximumBusLanes;       // Maximum bus lanes
    int iReserved;              // Reserved
} ADLPMActivity;

// ADL memory info (Overdrive6)
typedef struct ADLMemoryInfo {
    long long iMemorySize;      // Memory size in bytes
    char strMemoryType[ADL_MAX_PATH]; // Memory type string
    long long iMemoryBandwidth; // Memory bandwidth in bytes/s
} ADLMemoryInfo;

// Memory allocation callback for ADL
typedef void* (__stdcall *ADL_MAIN_MALLOC_CALLBACK)(int);

// ADL function pointer types
typedef int (*ADL_Main_Control_Create_t)(ADL_MAIN_MALLOC_CALLBACK, int);
typedef int (*ADL_Main_Control_Destroy_t)(void);
typedef int (*ADL_Adapter_NumberOfAdapters_Get_t)(int*);
typedef int (*ADL_Adapter_AdapterInfo_Get_t)(LPADLAdapterInfo, int);
typedef int (*ADL_Adapter_Active_Get_t)(int, int*);
typedef int (*ADL_Overdrive5_Temperature_Get_t)(int, int, ADLTemperature*);
typedef int (*ADL_Overdrive5_CurrentActivity_Get_t)(int, ADLPMActivity*);
typedef int (*ADL_Adapter_MemoryInfo_Get_t)(int, ADLMemoryInfo*);

// ADL function pointers (dynamically loaded)
static HMODULE adl_lib = nullptr;
static ADL_Main_Control_Create_t pADL_Main_Control_Create = nullptr;
static ADL_Main_Control_Destroy_t pADL_Main_Control_Destroy = nullptr;
static ADL_Adapter_NumberOfAdapters_Get_t pADL_Adapter_NumberOfAdapters_Get = nullptr;
static ADL_Adapter_AdapterInfo_Get_t pADL_Adapter_AdapterInfo_Get = nullptr;
static ADL_Adapter_Active_Get_t pADL_Adapter_Active_Get = nullptr;
static ADL_Overdrive5_Temperature_Get_t pADL_Overdrive5_Temperature_Get = nullptr;
static ADL_Overdrive5_CurrentActivity_Get_t pADL_Overdrive5_CurrentActivity_Get = nullptr;
static ADL_Adapter_MemoryInfo_Get_t pADL_Adapter_MemoryInfo_Get = nullptr;

// ADL global state
static std::vector<ADLAdapterInfo> adl_adapter_infos;
static std::vector<int> adl_adapter_indices;  // Active AMD adapter indices

// Memory allocation callback for ADL
static void* __stdcall ADL_Main_Memory_Alloc(int iSize) {
    void* lpBuffer = malloc(iSize);
    return lpBuffer;
}

static bool LoadADL() {
    if (adl_lib) return true;  // Already loaded

    // Try to load ADL (atiadlxx.dll for 64-bit, atiadlxy.dll for 32-bit)
    adl_lib = LoadLibraryA("atiadlxx.dll");
    if (!adl_lib) {
        adl_lib = LoadLibraryA("atiadlxy.dll");
    }

    if (!adl_lib) {
        spdlog::debug("ADL not found (atiadlxx.dll) - AMD metrics unavailable");
        return false;
    }

    // Load function pointers
    pADL_Main_Control_Create = (ADL_Main_Control_Create_t)GetProcAddress(adl_lib, "ADL_Main_Control_Create");
    pADL_Main_Control_Destroy = (ADL_Main_Control_Destroy_t)GetProcAddress(adl_lib, "ADL_Main_Control_Destroy");
    pADL_Adapter_NumberOfAdapters_Get = (ADL_Adapter_NumberOfAdapters_Get_t)GetProcAddress(adl_lib, "ADL_Adapter_NumberOfAdapters_Get");
    pADL_Adapter_AdapterInfo_Get = (ADL_Adapter_AdapterInfo_Get_t)GetProcAddress(adl_lib, "ADL_Adapter_AdapterInfo_Get");
    pADL_Adapter_Active_Get = (ADL_Adapter_Active_Get_t)GetProcAddress(adl_lib, "ADL_Adapter_Active_Get");
    pADL_Overdrive5_Temperature_Get = (ADL_Overdrive5_Temperature_Get_t)GetProcAddress(adl_lib, "ADL_Overdrive5_Temperature_Get");
    pADL_Overdrive5_CurrentActivity_Get = (ADL_Overdrive5_CurrentActivity_Get_t)GetProcAddress(adl_lib, "ADL_Overdrive5_CurrentActivity_Get");
    pADL_Adapter_MemoryInfo_Get = (ADL_Adapter_MemoryInfo_Get_t)GetProcAddress(adl_lib, "ADL_Adapter_MemoryInfo_Get");

    if (!pADL_Main_Control_Create || !pADL_Adapter_NumberOfAdapters_Get) {
        spdlog::warn("ADL loaded but essential functions not found");
        FreeLibrary(adl_lib);
        adl_lib = nullptr;
        return false;
    }

    spdlog::debug("ADL loaded successfully");
    return true;
}

static void UnloadADL() {
    if (adl_lib) {
        if (pADL_Main_Control_Destroy) {
            pADL_Main_Control_Destroy();
        }
        FreeLibrary(adl_lib);
        adl_lib = nullptr;
        adl_adapter_infos.clear();
        adl_adapter_indices.clear();
    }
}

// ========== D3DKMT Types for Intel/Generic GPU Monitoring ==========
// D3DKMT is the Windows Display Driver Model kernel thunk - same API Task Manager uses

typedef LONG NTSTATUS;
#define STATUS_SUCCESS ((NTSTATUS)0x00000000L)

typedef UINT D3DKMT_HANDLE;

typedef struct _D3DKMT_OPENADAPTERFROMLUID {
    LUID AdapterLuid;
    D3DKMT_HANDLE hAdapter;
} D3DKMT_OPENADAPTERFROMLUID;

typedef struct _D3DKMT_CLOSEADAPTER {
    D3DKMT_HANDLE hAdapter;
} D3DKMT_CLOSEADAPTER;

// Query statistics types
typedef enum _D3DKMT_QUERYSTATISTICS_TYPE {
    D3DKMT_QUERYSTATISTICS_ADAPTER = 0,
    D3DKMT_QUERYSTATISTICS_PROCESS = 1,
    D3DKMT_QUERYSTATISTICS_PROCESS_ADAPTER = 2,
    D3DKMT_QUERYSTATISTICS_SEGMENT = 3,
    D3DKMT_QUERYSTATISTICS_PROCESS_SEGMENT = 4,
    D3DKMT_QUERYSTATISTICS_NODE = 5,
    D3DKMT_QUERYSTATISTICS_PROCESS_NODE = 6,
    D3DKMT_QUERYSTATISTICS_VIDPNSOURCE = 7,
    D3DKMT_QUERYSTATISTICS_PROCESS_VIDPNSOURCE = 8
} D3DKMT_QUERYSTATISTICS_TYPE;

typedef struct _D3DKMT_QUERYSTATISTICS_COUNTER {
    ULONGLONG Count;
    ULONGLONG Bytes;
} D3DKMT_QUERYSTATISTICS_COUNTER;

typedef struct _D3DKMT_QUERYSTATISTICS_DMA_PACKET_TYPE_INFORMATION {
    ULONG PacketSubmitted;
    ULONG PacketCompleted;
    ULONG PacketPreempted;
    ULONG PacketFaulted;
} D3DKMT_QUERYSTATISTICS_DMA_PACKET_TYPE_INFORMATION;

typedef struct _D3DKMT_QUERYSTATISTICS_QUEUE_PACKET_TYPE_INFORMATION {
    ULONG PacketSubmitted;
    ULONG PacketCompleted;
} D3DKMT_QUERYSTATISTICS_QUEUE_PACKET_TYPE_INFORMATION;

typedef struct _D3DKMT_QUERYSTATISTICS_PACKET_INFORMATION {
    D3DKMT_QUERYSTATISTICS_QUEUE_PACKET_TYPE_INFORMATION QueuePacket[8];
    D3DKMT_QUERYSTATISTICS_DMA_PACKET_TYPE_INFORMATION DmaPacket[4];
} D3DKMT_QUERYSTATISTICS_PACKET_INFORMATION;

typedef struct _D3DKMT_QUERYSTATISTICS_NODE_INFORMATION {
    D3DKMT_QUERYSTATISTICS_COUNTER GlobalInformation;
    D3DKMT_QUERYSTATISTICS_COUNTER SystemInformation;
    BYTE NodeInformation[64];  // Simplified - actual structure is more complex
} D3DKMT_QUERYSTATISTICS_NODE_INFORMATION;

typedef struct _D3DKMT_QUERYSTATISTICS_PROCESS_NODE_INFORMATION {
    LARGE_INTEGER RunningTime;
    ULONG ContextSwitch;
    D3DKMT_QUERYSTATISTICS_COUNTER Preempted;
    D3DKMT_QUERYSTATISTICS_PACKET_INFORMATION PacketInformation;
    BYTE Reserved[64];
} D3DKMT_QUERYSTATISTICS_PROCESS_NODE_INFORMATION;

typedef struct _D3DKMT_QUERYSTATISTICS_ADAPTER_INFORMATION {
    ULONG NbSegments;
    ULONG NodeCount;
    ULONG VidPnSourceCount;
    ULONG VSyncEnabled;
    ULONG TdrDetectedCount;
    LONGLONG ZeroLengthDmaBuffers;
    ULONGLONG RestartedPeriod;
    D3DKMT_QUERYSTATISTICS_COUNTER ReferenceDma;
    D3DKMT_QUERYSTATISTICS_COUNTER RenamedDma;
    D3DKMT_QUERYSTATISTICS_COUNTER PresentHistoryToken;
    BYTE Reserved[256];
} D3DKMT_QUERYSTATISTICS_ADAPTER_INFORMATION;

typedef struct _D3DKMT_QUERYSTATISTICS {
    D3DKMT_QUERYSTATISTICS_TYPE Type;
    LUID AdapterLuid;
    HANDLE hProcess;
    union {
        D3DKMT_QUERYSTATISTICS_ADAPTER_INFORMATION AdapterInformation;
        D3DKMT_QUERYSTATISTICS_NODE_INFORMATION NodeInformation;
        D3DKMT_QUERYSTATISTICS_PROCESS_NODE_INFORMATION ProcessNodeInformation;
        BYTE Reserved[512];
    } QueryResult;
    union {
        ULONG NodeId;
        ULONG SegmentId;
        ULONG VidPnSourceId;
    } QueryParam;
} D3DKMT_QUERYSTATISTICS;

// Function pointer types
typedef NTSTATUS (WINAPI *D3DKMTQueryStatistics_t)(D3DKMT_QUERYSTATISTICS*);
typedef NTSTATUS (WINAPI *D3DKMTOpenAdapterFromLuid_t)(D3DKMT_OPENADAPTERFROMLUID*);
typedef NTSTATUS (WINAPI *D3DKMTCloseAdapter_t)(D3DKMT_CLOSEADAPTER*);

// Function pointers (dynamically loaded from gdi32.dll)
static HMODULE d3dkmt_lib = nullptr;
static D3DKMTQueryStatistics_t pD3DKMTQueryStatistics = nullptr;
static D3DKMTOpenAdapterFromLuid_t pD3DKMTOpenAdapterFromLuid = nullptr;
static D3DKMTCloseAdapter_t pD3DKMTCloseAdapter = nullptr;

// D3DKMT state - per-adapter GPU engine running time for usage calculation
struct D3DKMTAdapterState {
    LUID adapter_luid = {0, 0};
    ULONG node_count = 0;
    std::vector<LONGLONG> last_running_times;  // Per-node running time in 100-ns units
    std::chrono::steady_clock::time_point last_sample_time;
    float last_usage = 0.0f;
    bool initialized = false;
};
static std::unordered_map<uint64_t, D3DKMTAdapterState> d3dkmt_adapter_states;

// PDH-based GPU Engine counters (Windows 10+)
struct GPUEngineCounterState {
    PDH_HQUERY query = nullptr;
    PDH_HCOUNTER counter = nullptr;
    uint64_t adapter_luid_key = 0;
    bool initialized = false;
    bool counter_found = false;  // True if we successfully found and added a counter
    float last_value = 0.0f;
    int sample_count = 0;  // Need at least 2 samples for valid data
};
static std::unordered_map<uint64_t, GPUEngineCounterState> gpu_engine_counters;

static bool LoadD3DKMT() {
    if (d3dkmt_lib) return true;  // Already loaded

    d3dkmt_lib = GetModuleHandleA("gdi32.dll");
    if (!d3dkmt_lib) {
        d3dkmt_lib = LoadLibraryA("gdi32.dll");
    }

    if (!d3dkmt_lib) {
        spdlog::debug("D3DKMT: gdi32.dll not found");
        return false;
    }

    pD3DKMTQueryStatistics = (D3DKMTQueryStatistics_t)GetProcAddress(d3dkmt_lib, "D3DKMTQueryStatistics");
    pD3DKMTOpenAdapterFromLuid = (D3DKMTOpenAdapterFromLuid_t)GetProcAddress(d3dkmt_lib, "D3DKMTOpenAdapterFromLuid");
    pD3DKMTCloseAdapter = (D3DKMTCloseAdapter_t)GetProcAddress(d3dkmt_lib, "D3DKMTCloseAdapter");

    if (!pD3DKMTQueryStatistics) {
        spdlog::debug("D3DKMT: D3DKMTQueryStatistics not found");
        return false;
    }

    spdlog::debug("D3DKMT loaded successfully");
    return true;
}

// Get GPU usage via PDH GPU Engine counters (Windows 10+)
// This is the same method Windows Task Manager uses
static float GetPDHGpuUsage(const LUID& adapter_luid, const std::string& adapter_name) {
    // Create unique key from LUID
    uint64_t luid_key = ((uint64_t)adapter_luid.HighPart << 32) | (uint64_t)adapter_luid.LowPart;

    // Get or create counter state
    auto& state = gpu_engine_counters[luid_key];

    if (!state.initialized) {
        // Create PDH query for GPU Engine counters
        // Counter path format: \GPU Engine(pid_*_luid_0xHHHHHHHH_0xHHHHHHHH_*)\Utilization Percentage
        // We need to find counters matching our adapter LUID

        PDH_STATUS status = PdhOpenQuery(NULL, 0, &state.query);
        if (status != ERROR_SUCCESS) {
            spdlog::debug("PDH GPU: Failed to open query for {}", adapter_name);
            return 0.0f;
        }

        // Format the LUID for matching
        char luid_pattern[64];
        snprintf(luid_pattern, sizeof(luid_pattern), "luid_0x%08X_0x%08X",
                 adapter_luid.HighPart, adapter_luid.LowPart);

        // Try to add a wildcard counter for all engines of this GPU
        // The counter pattern matches all processes and engine types for this adapter
        char counter_path[512];
        snprintf(counter_path, sizeof(counter_path),
                 "\\GPU Engine(pid_*_%s_*_engtype_3D)\\Utilization Percentage",
                 luid_pattern);

        status = PdhAddEnglishCounterA(state.query, counter_path, 0, &state.counter);
        if (status == ERROR_SUCCESS) {
            spdlog::info("PDH GPU: Added wildcard 3D counter for {}", adapter_name);
            state.counter_found = true;
        } else {
            // Enumerate counters to find ones matching our LUID
            DWORD buffer_size = 0;
            DWORD instance_list_size = 0;
            PdhEnumObjectItemsA(NULL, NULL, "GPU Engine", NULL, &buffer_size,
                               NULL, &instance_list_size, PERF_DETAIL_WIZARD, 0);

            spdlog::debug("PDH GPU: Enumeration buffer sizes - counters: {}, instances: {}",
                         buffer_size, instance_list_size);

            if (instance_list_size > 0) {
                std::vector<char> counter_list(buffer_size + 1, 0);
                std::vector<char> instance_list(instance_list_size + 1, 0);

                status = PdhEnumObjectItemsA(NULL, NULL, "GPU Engine",
                                            counter_list.data(), &buffer_size,
                                            instance_list.data(), &instance_list_size,
                                            PERF_DETAIL_WIZARD, 0);

                if (status == ERROR_SUCCESS) {
                    // Parse instance list to find matching LUID
                    char* instance = instance_list.data();
                    bool found_3d = false;
                    int matching_luid_count = 0;
                    std::string first_matching_instance;

                    // First pass: count and log matching instances
                    while (*instance) {
                        std::string inst_name(instance);

                        // Check if this instance matches our LUID
                        if (inst_name.find(luid_pattern) != std::string::npos) {
                            matching_luid_count++;
                            if (first_matching_instance.empty()) {
                                first_matching_instance = inst_name;
                            }
                            // Check for 3D engine specifically
                            if (inst_name.find("engtype_3D") != std::string::npos && !found_3d) {
                                // Add this specific counter
                                snprintf(counter_path, sizeof(counter_path),
                                         "\\GPU Engine(%s)\\Utilization Percentage", instance);

                                status = PdhAddEnglishCounterA(state.query, counter_path, 0, &state.counter);
                                if (status == ERROR_SUCCESS) {
                                    spdlog::info("PDH GPU: Added 3D counter for {}: {}", adapter_name, inst_name);
                                    found_3d = true;
                                    state.counter_found = true;
                                }
                            }
                        }

                        instance += strlen(instance) + 1;
                    }

                    spdlog::debug("PDH GPU: Found {} instances matching LUID {} for {}",
                                 matching_luid_count, luid_pattern, adapter_name);

                    // If no 3D engine, try to use any engine type (Copy, VideoDecode, etc.)
                    if (!found_3d && matching_luid_count > 0) {
                        spdlog::debug("PDH GPU: No 3D engine, trying first available: {}", first_matching_instance);
                        snprintf(counter_path, sizeof(counter_path),
                                 "\\GPU Engine(%s)\\Utilization Percentage", first_matching_instance.c_str());

                        status = PdhAddEnglishCounterA(state.query, counter_path, 0, &state.counter);
                        if (status == ERROR_SUCCESS) {
                            spdlog::info("PDH GPU: Added fallback counter for {}: {}", adapter_name, first_matching_instance);
                            state.counter_found = true;
                        }
                    }

                    if (!state.counter_found) {
                        spdlog::info("PDH GPU: No counter found for {} (LUID: {}), will use D3DKMT fallback",
                                     adapter_name, luid_pattern);
                        PdhCloseQuery(state.query);
                        state.query = nullptr;
                        state.initialized = true;  // Mark as initialized but failed
                        return 0.0f;
                    }
                }
            } else {
                spdlog::debug("PDH GPU: GPU Engine object not available (Windows 10+ required)");
                PdhCloseQuery(state.query);
                state.query = nullptr;
                state.initialized = true;
                return 0.0f;
            }
        }

        // Collect initial data (need two samples to get rate)
        PdhCollectQueryData(state.query);
        state.adapter_luid_key = luid_key;
        state.initialized = true;
        state.sample_count = 1;
        state.last_value = 0.0f;

        spdlog::info("PDH GPU: Initialized counter for {} (LUID: {})", adapter_name, luid_pattern);
        return 0.0f;  // First sample always returns 0
    }

    if (!state.query || !state.counter) {
        return state.last_value;  // Return last known value if query failed
    }

    // Collect and get counter value
    PDH_STATUS status = PdhCollectQueryData(state.query);
    if (status != ERROR_SUCCESS) {
        return state.last_value;
    }

    state.sample_count++;

    PDH_FMT_COUNTERVALUE value;
    status = PdhGetFormattedCounterValue(state.counter, PDH_FMT_DOUBLE, NULL, &value);
    // PDH_CSTATUS_VALID_DATA = 0x00000000, PDH_CSTATUS_NEW_DATA = 0x00000001
    if (status == ERROR_SUCCESS && value.CStatus <= 1) {
        // Need at least 2 samples to get a valid rate-based value
        if (state.sample_count >= 2) {
            state.last_value = static_cast<float>(value.doubleValue) / 100.0f;  // Convert to 0-1 range
        }
    }

    return state.last_value;
}

// Check if PDH was successfully initialized for a GPU
static bool IsPDHInitialized(const LUID& adapter_luid) {
    uint64_t luid_key = ((uint64_t)adapter_luid.HighPart << 32) | (uint64_t)adapter_luid.LowPart;
    auto it = gpu_engine_counters.find(luid_key);
    return it != gpu_engine_counters.end() && it->second.initialized && it->second.counter_found;
}

// Fallback: Get GPU usage via D3DKMT running time (works for Intel and other non-NVIDIA/AMD GPUs)
static float GetD3DKMTGpuUsage(const LUID& adapter_luid, const std::string& adapter_name) {
    // First try PDH (Windows 10+, more accurate)
    float pdh_usage = GetPDHGpuUsage(adapter_luid, adapter_name);

    // If PDH was successfully initialized (found a counter), use its value even if 0
    // Only fall back to D3DKMT if PDH failed to find a counter
    if (IsPDHInitialized(adapter_luid)) {
        return pdh_usage;
    }

    // Fall back to D3DKMT if PDH doesn't work
    if (!pD3DKMTQueryStatistics) {
        return 0.0f;
    }

    // Create unique key from LUID
    uint64_t luid_key = ((uint64_t)adapter_luid.HighPart << 32) | (uint64_t)adapter_luid.LowPart;

    // Get or create adapter state
    auto& state = d3dkmt_adapter_states[luid_key];
    auto now = std::chrono::steady_clock::now();

    // First, get adapter info to find node count
    if (!state.initialized) {
        D3DKMT_QUERYSTATISTICS query = {};
        query.Type = D3DKMT_QUERYSTATISTICS_ADAPTER;
        query.AdapterLuid = adapter_luid;

        if (pD3DKMTQueryStatistics(&query) == STATUS_SUCCESS) {
            state.node_count = query.QueryResult.AdapterInformation.NodeCount;
            state.adapter_luid = adapter_luid;
            state.last_running_times.resize(state.node_count, 0);
            state.initialized = true;
            spdlog::debug("D3DKMT: {} has {} GPU nodes", adapter_name, state.node_count);
        } else {
            spdlog::debug("D3DKMT: Failed to query adapter info for {}", adapter_name);
            return 0.0f;
        }
    }

    if (state.node_count == 0) {
        return 0.0f;
    }

    // Query running time for each GPU node
    LONGLONG total_running_time = 0;
    for (ULONG node = 0; node < state.node_count; node++) {
        D3DKMT_QUERYSTATISTICS query = {};
        query.Type = D3DKMT_QUERYSTATISTICS_NODE;
        query.AdapterLuid = adapter_luid;
        query.QueryParam.NodeId = node;

        if (pD3DKMTQueryStatistics(&query) == STATUS_SUCCESS) {
            // GlobalInformation.Count represents the running time in 100-ns units
            total_running_time += query.QueryResult.NodeInformation.GlobalInformation.Count;
        }
    }

    // Calculate usage from running time rate
    float usage = 0.0f;
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - state.last_sample_time).count();

    if (elapsed > 0 && state.last_sample_time.time_since_epoch().count() > 0) {
        LONGLONG last_total = 0;
        for (auto rt : state.last_running_times) {
            last_total += rt;
        }

        if (total_running_time >= last_total) {
            LONGLONG delta_running_time = total_running_time - last_total;
            // Convert 100-ns units to microseconds (divide by 10)
            // Then calculate percentage of elapsed time
            double running_us = delta_running_time / 10.0;
            usage = static_cast<float>(running_us / static_cast<double>(elapsed));
            usage = std::min(1.0f, std::max(0.0f, usage));
        }
    }

    // Update state
    state.last_running_times.clear();
    state.last_running_times.push_back(total_running_time);
    state.last_sample_time = now;
    state.last_usage = usage;

    return usage;
}

#endif // _WIN32

// ========== MetricsCollector Implementation ==========

MetricsCollector::MetricsCollector() {
    spdlog::debug("MetricsCollector created");

#ifdef _WIN32
    // Initialize PDH for CPU monitoring
    PDH_STATUS status = PdhOpenQuery(NULL, 0, reinterpret_cast<PDH_HQUERY*>(&cpu_query_));
    if (status == ERROR_SUCCESS) {
        PdhAddEnglishCounter(
            reinterpret_cast<PDH_HQUERY>(cpu_query_),
            "\\Processor(_Total)\\% Processor Time",
            0,
            reinterpret_cast<PDH_HCOUNTER*>(&cpu_counter_)
        );
        PdhCollectQueryData(reinterpret_cast<PDH_HQUERY>(cpu_query_));
    }

    // Initialize NVML for NVIDIA GPU monitoring
    if (LoadNVML()) {
        if (pNvmlInit && pNvmlInit() == NVML_SUCCESS) {
            nvml_initialized_ = true;

            // Enumerate NVIDIA GPUs
            unsigned int device_count = 0;
            if (pNvmlDeviceGetCount && pNvmlDeviceGetCount(&device_count) == NVML_SUCCESS) {
                nvidia_gpu_count_ = device_count;
                nvml_device_count_ = device_count;

                spdlog::info("NVML: Found {} NVIDIA GPU(s)", device_count);

                for (unsigned int i = 0; i < device_count; i++) {
                    nvmlDevice_t device;
                    if (pNvmlDeviceGetHandleByIndex(i, &device) == NVML_SUCCESS) {
                        nvml_devices_.push_back(device);

                        char name[256] = {0};
                        if (pNvmlDeviceGetName) {
                            pNvmlDeviceGetName(device, name, sizeof(name));
                        }
                        nvml_device_names_.push_back(name);
                        spdlog::info("  GPU {}: {}", i, name);
                    }
                }
            }
        } else {
            spdlog::warn("NVML initialization failed");
        }
    }

    // Initialize ADL for AMD GPU monitoring
    if (LoadADL()) {
        if (pADL_Main_Control_Create &&
            pADL_Main_Control_Create(ADL_Main_Memory_Alloc, 1) == ADL_OK) {
            adl_initialized_ = true;

            // Enumerate AMD GPUs
            int adapter_count = 0;
            if (pADL_Adapter_NumberOfAdapters_Get &&
                pADL_Adapter_NumberOfAdapters_Get(&adapter_count) == ADL_OK &&
                adapter_count > 0) {

                adl_adapter_infos.resize(adapter_count);
                if (pADL_Adapter_AdapterInfo_Get &&
                    pADL_Adapter_AdapterInfo_Get(adl_adapter_infos.data(),
                        sizeof(ADLAdapterInfo) * adapter_count) == ADL_OK) {

                    // Find active AMD adapters
                    for (int i = 0; i < adapter_count; i++) {
                        int active = 0;
                        if (pADL_Adapter_Active_Get &&
                            pADL_Adapter_Active_Get(adl_adapter_infos[i].iAdapterIndex, &active) == ADL_OK &&
                            active) {
                            // Check for duplicate adapter (same bus number)
                            bool duplicate = false;
                            for (int idx : adl_adapter_indices) {
                                if (adl_adapter_infos[idx].iBusNumber == adl_adapter_infos[i].iBusNumber) {
                                    duplicate = true;
                                    break;
                                }
                            }
                            if (!duplicate) {
                                adl_adapter_indices.push_back(i);
                                spdlog::info("ADL: Found AMD GPU: {} (adapter {})",
                                    adl_adapter_infos[i].strAdapterName,
                                    adl_adapter_infos[i].iAdapterIndex);
                            }
                        }
                    }

                    amd_gpu_count_ = static_cast<int>(adl_adapter_indices.size());
                    spdlog::info("ADL: Found {} active AMD GPU(s)", amd_gpu_count_);
                }
            }
        } else {
            spdlog::warn("ADL initialization failed");
        }
    }

    // Initialize D3DKMT for Intel/generic GPU monitoring
    if (LoadD3DKMT()) {
        spdlog::info("D3DKMT: Initialized for Intel/generic GPU monitoring");
    }
#endif

    last_sample_time_ = std::chrono::steady_clock::now();
}

MetricsCollector::~MetricsCollector() {
    StopCollection();

#ifdef _WIN32
    if (cpu_query_) {
        PdhCloseQuery(reinterpret_cast<PDH_HQUERY>(cpu_query_));
    }

    if (nvml_initialized_) {
        UnloadNVML();
        nvml_initialized_ = false;
    }

    if (adl_initialized_) {
        UnloadADL();
        adl_initialized_ = false;
    }
#endif
}

void MetricsCollector::StartCollection(int interval_ms) {
    if (running_.load()) {
        return;
    }

    interval_ms_.store(interval_ms);
    running_.store(true);
    collection_thread_ = std::thread(&MetricsCollector::CollectionLoop, this);
    spdlog::info("MetricsCollector started with {}ms interval", interval_ms);
}

void MetricsCollector::StopCollection() {
    if (!running_.load()) {
        return;
    }

    running_.store(false);
    if (collection_thread_.joinable()) {
        collection_thread_.join();
    }
    spdlog::info("MetricsCollector stopped");
}

void MetricsCollector::SetInterval(int interval_ms) {
    interval_ms_.store(interval_ms);
}

SystemMetrics MetricsCollector::GetCurrentMetrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_metrics_;
}

std::vector<GPUMetrics> MetricsCollector::GetPerGPUMetrics() {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_metrics_.gpus;
}

std::vector<float> MetricsCollector::GetHistory(MetricType type, int samples) const {
    std::lock_guard<std::mutex> lock(mutex_);

    const std::deque<float>* source = nullptr;
    switch (type) {
        case MetricType::CPU: source = &cpu_history_; break;
        case MetricType::GPU: source = &gpu_history_; break;
        case MetricType::RAM: source = &ram_history_; break;
        case MetricType::VRAM: source = &vram_history_; break;
        case MetricType::NetworkIn: source = &net_in_history_; break;
        case MetricType::NetworkOut: source = &net_out_history_; break;
        default: return {};
    }

    if (!source) return {};

    int count = std::min(samples, static_cast<int>(source->size()));
    std::vector<float> result(count);
    auto start = source->end() - count;
    std::copy(start, source->end(), result.begin());
    return result;
}

void MetricsCollector::CollectionLoop() {
    while (running_.load()) {
        auto start = std::chrono::steady_clock::now();

        // Collect all metrics
        SystemMetrics metrics;
        metrics.cpu_usage = CollectCPUUsage();
        metrics.ram_used_bytes = CollectRAMUsed();
        metrics.ram_total_bytes = CollectRAMTotal();
        metrics.network_in_mbps = CollectNetworkIn();
        metrics.network_out_mbps = CollectNetworkOut();

        // Collect all GPU metrics (DXGI + NVML)
        CollectAllGPUMetrics(metrics);

        // Calculate RAM percentage
        if (metrics.ram_total_bytes > 0) {
            metrics.ram_usage = static_cast<float>(metrics.ram_used_bytes) / metrics.ram_total_bytes;
        }

        // Set primary GPU metrics for backward compatibility
        if (!metrics.gpus.empty()) {
            // Use NVIDIA GPU as primary if available, otherwise first GPU
            int primary_idx = 0;
            for (size_t i = 0; i < metrics.gpus.size(); i++) {
                if (metrics.gpus[i].is_nvidia) {
                    primary_idx = static_cast<int>(i);
                    break;
                }
            }

            const auto& primary = metrics.gpus[primary_idx];
            metrics.gpu_usage = primary.usage_3d;
            metrics.vram_used_bytes = primary.vram_used_bytes;
            metrics.vram_total_bytes = primary.vram_total_bytes;
            metrics.vram_usage = primary.memory_usage;
            metrics.temperature_celsius = primary.temperature_celsius;
            metrics.power_watts = primary.power_watts;
        }

        // Update state and history
        {
            std::lock_guard<std::mutex> lock(mutex_);
            current_metrics_ = metrics;

            // Update history buffers
            auto addToHistory = [](std::deque<float>& history, float value) {
                history.push_back(value);
                if (history.size() > MAX_HISTORY_SIZE) {
                    history.pop_front();
                }
            };

            addToHistory(cpu_history_, metrics.cpu_usage);
            addToHistory(gpu_history_, metrics.gpu_usage);
            addToHistory(ram_history_, metrics.ram_usage);
            addToHistory(vram_history_, metrics.vram_usage);
            addToHistory(net_in_history_, metrics.network_in_mbps);
            addToHistory(net_out_history_, metrics.network_out_mbps);
        }

        // Sleep for remaining interval
        auto elapsed = std::chrono::steady_clock::now() - start;
        auto sleep_time = std::chrono::milliseconds(interval_ms_.load()) - elapsed;
        if (sleep_time > std::chrono::milliseconds(0)) {
            std::this_thread::sleep_for(sleep_time);
        }
    }
}

// ========== Collect All GPU Metrics (DXGI + NVML) ==========

void MetricsCollector::CollectAllGPUMetrics(SystemMetrics& metrics) {
#ifdef _WIN32
    metrics.gpus.clear();

    // Create DXGI factory to enumerate all GPUs
    IDXGIFactory1* factory = nullptr;
    HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory);
    if (FAILED(hr)) {
        spdlog::debug("Failed to create DXGI factory");
        return;
    }

    // Map NVIDIA device names to NVML indices for matching
    std::unordered_map<std::string, int> nvml_name_to_idx;
    for (size_t i = 0; i < nvml_device_names_.size(); i++) {
        nvml_name_to_idx[nvml_device_names_[i]] = static_cast<int>(i);
    }

    // Enumerate all display adapters via DXGI
    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        if (SUCCEEDED(adapter->GetDesc1(&desc))) {
            // Skip software/remote adapters
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
                adapter->Release();
                continue;
            }

            // Convert wide string name to UTF-8
            char name[256] = {0};
            WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name), nullptr, nullptr);

            GPUMetrics gpu;
            gpu.device_id = static_cast<int>(metrics.gpus.size());
            gpu.name = name;

            // Determine vendor from Vendor ID
            // 0x10DE = NVIDIA, 0x1002 = AMD, 0x8086 = Intel
            if (desc.VendorId == 0x10DE) {
                gpu.vendor = "NVIDIA";
                gpu.is_nvidia = true;
            } else if (desc.VendorId == 0x1002) {
                gpu.vendor = "AMD";
                gpu.is_nvidia = false;
            } else if (desc.VendorId == 0x8086) {
                gpu.vendor = "Intel";
                gpu.is_nvidia = false;
            } else {
                gpu.vendor = "Unknown";
                gpu.is_nvidia = false;
            }

            // Get VRAM from DXGI (works for all GPUs)
            gpu.vram_total_bytes = desc.DedicatedVideoMemory;

            // For NVIDIA GPUs, get detailed metrics from NVML
            if (gpu.is_nvidia && nvml_initialized_) {
                // Find matching NVML device by name
                int nvml_idx = -1;
                auto it = nvml_name_to_idx.find(name);
                if (it != nvml_name_to_idx.end()) {
                    nvml_idx = it->second;
                } else {
                    // Try partial match (DXGI and NVML names may differ slightly)
                    for (const auto& pair : nvml_name_to_idx) {
                        if (pair.first.find("GeForce") != std::string::npos &&
                            std::string(name).find("GeForce") != std::string::npos) {
                            // Match GTX/RTX model numbers
                            if (pair.first.find("1050") != std::string::npos &&
                                std::string(name).find("1050") != std::string::npos) {
                                nvml_idx = pair.second;
                                break;
                            }
                        }
                    }
                }

                if (nvml_idx >= 0 && nvml_idx < static_cast<int>(nvml_devices_.size())) {
                    nvmlDevice_t device = static_cast<nvmlDevice_t>(nvml_devices_[nvml_idx]);

                    // GPU utilization
                    if (pNvmlDeviceGetUtilizationRates) {
                        nvmlUtilization_t util;
                        if (pNvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS) {
                            gpu.usage_3d = util.gpu / 100.0f;
                            gpu.memory_usage = util.memory / 100.0f;
                        }
                    }

                    // Memory info
                    if (pNvmlDeviceGetMemoryInfo) {
                        nvmlMemory_t mem;
                        if (pNvmlDeviceGetMemoryInfo(device, &mem) == NVML_SUCCESS) {
                            gpu.vram_used_bytes = mem.used;
                            gpu.vram_total_bytes = mem.total;
                            if (mem.total > 0) {
                                gpu.memory_usage = static_cast<float>(mem.used) / mem.total;
                            }
                        }
                    }

                    // Temperature
                    if (pNvmlDeviceGetTemperature) {
                        unsigned int temp = 0;
                        if (pNvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
                            gpu.temperature_celsius = static_cast<float>(temp);
                        }
                    }

                    // Power usage
                    if (pNvmlDeviceGetPowerUsage) {
                        unsigned int power_mw = 0;
                        if (pNvmlDeviceGetPowerUsage(device, &power_mw) == NVML_SUCCESS) {
                            gpu.power_watts = power_mw / 1000.0f;
                        }
                    }

                    // Video encoder utilization
                    if (pNvmlDeviceGetEncoderUtilization) {
                        unsigned int util = 0, period = 0;
                        if (pNvmlDeviceGetEncoderUtilization(device, &util, &period) == NVML_SUCCESS) {
                            gpu.usage_video_encode = util / 100.0f;
                        }
                    }

                    // Video decoder utilization
                    if (pNvmlDeviceGetDecoderUtilization) {
                        unsigned int util = 0, period = 0;
                        if (pNvmlDeviceGetDecoderUtilization(device, &util, &period) == NVML_SUCCESS) {
                            gpu.usage_video_decode = util / 100.0f;
                        }
                    }
                }
            } else if (gpu.vendor == "AMD" && adl_initialized_) {
                // For AMD GPUs, get detailed metrics from ADL
                // Try to find matching ADL adapter by name
                int adl_adapter_idx = -1;
                for (int idx : adl_adapter_indices) {
                    std::string adl_name = adl_adapter_infos[idx].strAdapterName;
                    // Match by partial name (DXGI and ADL names may differ)
                    if (adl_name.find("Radeon") != std::string::npos &&
                        std::string(name).find("Radeon") != std::string::npos) {
                        adl_adapter_idx = adl_adapter_infos[idx].iAdapterIndex;
                        break;
                    }
                    // Also try direct match
                    if (adl_name == name) {
                        adl_adapter_idx = adl_adapter_infos[idx].iAdapterIndex;
                        break;
                    }
                }

                if (adl_adapter_idx >= 0) {
                    // GPU utilization (Overdrive5)
                    if (pADL_Overdrive5_CurrentActivity_Get) {
                        ADLPMActivity activity;
                        activity.iSize = sizeof(ADLPMActivity);
                        if (pADL_Overdrive5_CurrentActivity_Get(adl_adapter_idx, &activity) == ADL_OK) {
                            gpu.usage_3d = activity.iActivityPercent / 100.0f;
                        }
                    }

                    // Temperature (Overdrive5)
                    if (pADL_Overdrive5_Temperature_Get) {
                        ADLTemperature temp;
                        temp.iSize = sizeof(ADLTemperature);
                        // Thermal controller index 0 is typically the GPU
                        if (pADL_Overdrive5_Temperature_Get(adl_adapter_idx, 0, &temp) == ADL_OK) {
                            // Temperature is in millidegrees Celsius
                            gpu.temperature_celsius = temp.iTemperature / 1000.0f;
                        }
                    }

                    // Memory info
                    if (pADL_Adapter_MemoryInfo_Get) {
                        ADLMemoryInfo memInfo;
                        if (pADL_Adapter_MemoryInfo_Get(adl_adapter_idx, &memInfo) == ADL_OK) {
                            gpu.vram_total_bytes = static_cast<size_t>(memInfo.iMemorySize);
                        }
                    }
                } else {
                    // ADL adapter not found - use basic DXGI info
                    gpu.vram_used_bytes = 0;
                    gpu.usage_3d = 0.0f;
                }
            } else {
                // Non-NVIDIA/AMD GPU (Intel) - use D3DKMT for usage metrics
                // D3DKMT is the Windows Display Driver Model kernel thunk (same as Task Manager)
                gpu.usage_3d = GetD3DKMTGpuUsage(desc.AdapterLuid, name);
                gpu.vram_used_bytes = 0;  // D3DKMT doesn't easily expose memory usage
                gpu.usage_copy = 0.0f;    // Would need per-engine query (TODO)
                gpu.usage_video_encode = 0.0f;
                gpu.usage_video_decode = 0.0f;

                // Intel integrated GPUs use shared memory
                if (desc.SharedSystemMemory > 0) {
                    gpu.vram_total_bytes = desc.SharedSystemMemory;
                }
            }

            metrics.gpus.push_back(gpu);
        }
        adapter->Release();
    }

    factory->Release();
    metrics.gpu_count = static_cast<int>(metrics.gpus.size());

#else
    // Linux: Would need to parse /sys/class/drm/ or use vendor-specific libraries
    metrics.gpu_count = 0;
#endif
}

// ========== Platform-specific implementations ==========

#ifdef _WIN32

float MetricsCollector::CollectCPUUsage() {
    if (!cpu_query_ || !cpu_counter_) return 0.0f;

    PDH_FMT_COUNTERVALUE value;
    PdhCollectQueryData(reinterpret_cast<PDH_HQUERY>(cpu_query_));
    PdhGetFormattedCounterValue(
        reinterpret_cast<PDH_HCOUNTER>(cpu_counter_),
        PDH_FMT_DOUBLE,
        NULL,
        &value
    );
    return static_cast<float>(value.doubleValue) / 100.0f;
}

size_t MetricsCollector::CollectRAMUsed() {
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    return memInfo.ullTotalPhys - memInfo.ullAvailPhys;
}

size_t MetricsCollector::CollectRAMTotal() {
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    return memInfo.ullTotalPhys;
}

float MetricsCollector::CollectNetworkIn() {
    // Simplified: would need to track actual network bytes
    // For now, return placeholder
    return 0.0f;
}

float MetricsCollector::CollectNetworkOut() {
    return 0.0f;
}

#elif defined(__APPLE__)  // macOS

float MetricsCollector::CollectCPUUsage() {
    // macOS: Use host_processor_info or read from sysctl
    // Simplified: use host_statistics for CPU load
    host_cpu_load_info_data_t cpuinfo;
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;

    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                       (host_info_t)&cpuinfo, &count) != KERN_SUCCESS) {
        return 0.0f;
    }

    unsigned long long total = 0;
    for (int i = 0; i < CPU_STATE_MAX; i++) {
        total += cpuinfo.cpu_ticks[i];
    }

    unsigned long long idle = cpuinfo.cpu_ticks[CPU_STATE_IDLE];

    float usage = 0.0f;
    if (last_cpu_total_ > 0) {
        unsigned long long total_diff = total - last_cpu_total_;
        unsigned long long idle_diff = idle - last_cpu_idle_;
        if (total_diff > 0) {
            usage = 1.0f - (static_cast<float>(idle_diff) / total_diff);
        }
    }

    last_cpu_total_ = total;
    last_cpu_idle_ = idle;

    return usage;
}

size_t MetricsCollector::CollectRAMUsed() {
    // macOS: Use vm_statistics to get memory usage
    vm_size_t page_size;
    mach_port_t mach_port = mach_host_self();
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = sizeof(vm_stats) / sizeof(natural_t);

    host_page_size(mach_port, &page_size);

    if (host_statistics64(mach_port, HOST_VM_INFO64,
                         (host_info64_t)&vm_stats, &count) != KERN_SUCCESS) {
        return 0;
    }

    // Used memory = total - free - inactive - speculative - purgeable
    // For a simpler calculation, use: wired + active + compressed
    size_t used = (vm_stats.wire_count + vm_stats.active_count +
                   vm_stats.compressor_page_count) * page_size;
    return used;
}

size_t MetricsCollector::CollectRAMTotal() {
    // macOS: Use sysctl to get total physical memory
    int mib[2] = { CTL_HW, HW_MEMSIZE };
    int64_t memsize = 0;
    size_t len = sizeof(memsize);

    if (sysctl(mib, 2, &memsize, &len, NULL, 0) == 0) {
        return static_cast<size_t>(memsize);
    }
    return 0;
}

float MetricsCollector::CollectNetworkIn() {
    // macOS: Use getifaddrs to get network statistics
    // Note: This is a simplified implementation - for full stats would need IOKit
    struct ifaddrs *ifaddr, *ifa;
    uint64_t total_bytes = 0;

    if (getifaddrs(&ifaddr) == -1) {
        return 0.0f;
    }

    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) continue;
        if (ifa->ifa_addr->sa_family != AF_LINK) continue;
        // Skip loopback
        if (strcmp(ifa->ifa_name, "lo0") == 0) continue;

        struct if_data *if_data = (struct if_data *)ifa->ifa_data;
        if (if_data) {
            total_bytes += if_data->ifi_ibytes;
        }
    }

    freeifaddrs(ifaddr);

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_sample_time_).count();

    float mbps = 0.0f;
    if (elapsed > 0 && last_net_bytes_in_ > 0) {
        uint64_t bytes_diff = total_bytes - last_net_bytes_in_;
        mbps = (bytes_diff * 8.0f) / (elapsed * 1000.0f);
    }

    last_net_bytes_in_ = total_bytes;
    return mbps;
}

float MetricsCollector::CollectNetworkOut() {
    struct ifaddrs *ifaddr, *ifa;
    uint64_t total_bytes = 0;

    if (getifaddrs(&ifaddr) == -1) {
        return 0.0f;
    }

    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) continue;
        if (ifa->ifa_addr->sa_family != AF_LINK) continue;
        if (strcmp(ifa->ifa_name, "lo0") == 0) continue;

        struct if_data *if_data = (struct if_data *)ifa->ifa_data;
        if (if_data) {
            total_bytes += if_data->ifi_obytes;
        }
    }

    freeifaddrs(ifaddr);

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_sample_time_).count();

    float mbps = 0.0f;
    if (elapsed > 0 && last_net_bytes_out_ > 0) {
        uint64_t bytes_diff = total_bytes - last_net_bytes_out_;
        mbps = (bytes_diff * 8.0f) / (elapsed * 1000.0f);
    }

    last_net_bytes_out_ = total_bytes;
    last_sample_time_ = now;
    return mbps;
}

#else  // Linux

float MetricsCollector::CollectCPUUsage() {
    std::ifstream stat("/proc/stat");
    if (!stat.is_open()) return 0.0f;

    std::string cpu;
    long user, nice, system, idle, iowait, irq, softirq;
    stat >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq;

    long total = user + nice + system + idle + iowait + irq + softirq;
    long idle_time = idle + iowait;

    float usage = 0.0f;
    if (last_cpu_total_ > 0) {
        long total_diff = total - last_cpu_total_;
        long idle_diff = idle_time - last_cpu_idle_;
        if (total_diff > 0) {
            usage = 1.0f - (static_cast<float>(idle_diff) / total_diff);
        }
    }

    last_cpu_total_ = total;
    last_cpu_idle_ = idle_time;

    return usage;
}

size_t MetricsCollector::CollectRAMUsed() {
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return (info.totalram - info.freeram) * info.mem_unit;
    }
    return 0;
}

size_t MetricsCollector::CollectRAMTotal() {
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.totalram * info.mem_unit;
    }
    return 0;
}

float MetricsCollector::CollectNetworkIn() {
    std::ifstream net("/proc/net/dev");
    if (!net.is_open()) return 0.0f;

    std::string line;
    uint64_t total_bytes = 0;

    while (std::getline(net, line)) {
        if (line.find(':') == std::string::npos) continue;
        if (line.find("lo:") != std::string::npos) continue;  // Skip loopback

        std::istringstream iss(line.substr(line.find(':') + 1));
        uint64_t bytes_in;
        iss >> bytes_in;
        total_bytes += bytes_in;
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_sample_time_).count();

    float mbps = 0.0f;
    if (elapsed > 0 && last_net_bytes_in_ > 0) {
        uint64_t bytes_diff = total_bytes - last_net_bytes_in_;
        mbps = (bytes_diff * 8.0f) / (elapsed * 1000.0f);  // Convert to Mbps
    }

    last_net_bytes_in_ = total_bytes;
    return mbps;
}

float MetricsCollector::CollectNetworkOut() {
    std::ifstream net("/proc/net/dev");
    if (!net.is_open()) return 0.0f;

    std::string line;
    uint64_t total_bytes = 0;

    while (std::getline(net, line)) {
        if (line.find(':') == std::string::npos) continue;
        if (line.find("lo:") != std::string::npos) continue;

        std::istringstream iss(line.substr(line.find(':') + 1));
        uint64_t bytes_in, packets_in, errin, dropin, fifoin, framein, compressedin, multicastin;
        uint64_t bytes_out;
        iss >> bytes_in >> packets_in >> errin >> dropin >> fifoin >> framein >> compressedin >> multicastin >> bytes_out;
        total_bytes += bytes_out;
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_sample_time_).count();

    float mbps = 0.0f;
    if (elapsed > 0 && last_net_bytes_out_ > 0) {
        uint64_t bytes_diff = total_bytes - last_net_bytes_out_;
        mbps = (bytes_diff * 8.0f) / (elapsed * 1000.0f);
    }

    last_net_bytes_out_ = total_bytes;
    last_sample_time_ = now;
    return mbps;
}

#endif  // Platform-specific

// ========== GPU metrics (legacy single-GPU functions for backward compatibility) ==========

float MetricsCollector::CollectGPUUsage() {
    // Now handled by CollectAllGPUMetrics
    return 0.0f;
}

size_t MetricsCollector::CollectVRAMUsed() {
    // Now handled by CollectAllGPUMetrics
    return 0;
}

size_t MetricsCollector::CollectVRAMTotal() {
    // Now handled by CollectAllGPUMetrics
    return 0;
}

float MetricsCollector::CollectTemperature() {
    // Now handled by CollectAllGPUMetrics
    return 0.0f;
}

float MetricsCollector::CollectPowerUsage() {
    // Now handled by CollectAllGPUMetrics
    return 0.0f;
}

} // namespace cyxwiz::servernode::core
