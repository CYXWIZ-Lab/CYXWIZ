#pragma once

// MjcfParser — Extract actuator, sensor, and camera metadata from MJCF models
//
// Uses MuJoCo's own mj_loadXML to parse the model, then reads structural
// info (actuator names, joint ranges, sensor types, etc.) without keeping
// the model loaded. This is used for dynamic pin creation in the node editor.

#include <mujoco/mujoco.h>

#include <string>
#include <vector>

namespace cyxwiz::plugin::mujoco {

struct ActuatorInfo {
    std::string name;
    std::string joint;       // Associated joint name
    std::string type;        // "motor", "position", "velocity", etc.
    int index = 0;           // Actuator index in ctrl array
    float range_low = 0.0f;  // Control range lower bound
    float range_high = 0.0f; // Control range upper bound
    bool has_range = false;  // Whether ctrlrange is defined
};

struct SensorInfo {
    std::string name;
    std::string type;    // "jointpos", "jointvel", "accelerometer", etc.
    int dim = 1;         // Number of scalar values this sensor outputs
    int adr = 0;         // Start address in sensordata array
};

struct CameraInfo {
    std::string name;
    int index = 0;
};

struct BodyInfo {
    std::string name;
    int index = 0;
};

struct MjcfModelInfo {
    std::string model_name;
    bool valid = false;
    std::string error;

    // Model structure
    std::vector<ActuatorInfo> actuators;
    std::vector<SensorInfo> sensors;
    std::vector<CameraInfo> cameras;
    std::vector<BodyInfo> bodies;

    // Dimensions
    int nq = 0;  // Generalized positions
    int nv = 0;  // Generalized velocities
    int nu = 0;  // Actuators (controls)
    int nsensor = 0;
    int nbody = 0;

    // Timestep
    double timestep = 0.002;
};

// Parse an MJCF XML file and extract model metadata.
// Loads the model temporarily, reads structure, then frees it.
// Returns MjcfModelInfo with valid=false on error.
MjcfModelInfo ParseMjcfFile(const std::string& mjcf_path);

// Parse from an already-loaded model (does not free it).
MjcfModelInfo ParseMjcfModel(const mjModel* model);

} // namespace cyxwiz::plugin::mujoco
