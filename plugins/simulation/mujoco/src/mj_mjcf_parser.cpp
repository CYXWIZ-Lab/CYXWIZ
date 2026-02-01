#include "mj_mjcf_parser.h"
#include <spdlog/spdlog.h>

namespace cyxwiz::plugin::mujoco {

// Map MuJoCo actuator type enum to string
static const char* ActuatorTypeName(int type) {
    switch (type) {
        case mjTRN_JOINT:      return "motor";
        case mjTRN_TENDON:     return "tendon";
        case mjTRN_SITE:       return "site";
        case mjTRN_BODY:       return "body";
        default:               return "unknown";
    }
}

// Map MuJoCo sensor type enum to string
static const char* SensorTypeName(int type) {
    switch (type) {
        case mjSENS_JOINTPOS:       return "jointpos";
        case mjSENS_JOINTVEL:       return "jointvel";
        case mjSENS_TENDONPOS:      return "tendonpos";
        case mjSENS_TENDONVEL:      return "tendonvel";
        case mjSENS_ACTUATORFRC:    return "actuatorfrc";
        case mjSENS_TOUCH:          return "touch";
        case mjSENS_ACCELEROMETER:  return "accelerometer";
        case mjSENS_GYRO:           return "gyro";
        case mjSENS_FORCE:          return "force";
        case mjSENS_TORQUE:         return "torque";
        case mjSENS_MAGNETOMETER:   return "magnetometer";
        case mjSENS_RANGEFINDER:    return "rangefinder";
        case mjSENS_FRAMEPOS:       return "framepos";
        case mjSENS_FRAMEQUAT:      return "framequat";
        case mjSENS_FRAMEXAXIS:     return "framexaxis";
        case mjSENS_FRAMELINVEL:    return "framelinvel";
        case mjSENS_FRAMEANGVEL:    return "frameangvel";
        case mjSENS_FRAMELINACC:    return "framelinacc";
        case mjSENS_FRAMEANGACC:    return "frameangacc";
        default:                    return "other";
    }
}

MjcfModelInfo ParseMjcfModel(const mjModel* m) {
    MjcfModelInfo info;
    if (!m) {
        info.error = "Null model pointer";
        return info;
    }

    info.valid = true;

    // Model name
    const char* name = mj_id2name(m, mjOBJ_BODY, 0);  // worldbody name or model name
    if (m->names && m->names[0] != '\0') {
        // First entry in names buffer is often the model name
        info.model_name = m->names;
    }
    if (info.model_name.empty() && name) {
        info.model_name = name;
    }

    // Dimensions
    info.nq = m->nq;
    info.nv = m->nv;
    info.nu = m->nu;
    info.nsensor = m->nsensor;
    info.nbody = m->nbody;
    info.timestep = m->opt.timestep;

    // Actuators
    info.actuators.reserve(m->nu);
    for (int i = 0; i < m->nu; i++) {
        ActuatorInfo act;
        act.index = i;

        const char* aname = mj_id2name(m, mjOBJ_ACTUATOR, i);
        act.name = aname ? aname : ("actuator_" + std::to_string(i));

        // Transmission type
        act.type = ActuatorTypeName(m->actuator_trntype[i]);

        // Associated joint (if motor type)
        if (m->actuator_trntype[i] == mjTRN_JOINT && m->actuator_trnid[2 * i] >= 0) {
            const char* jname = mj_id2name(m, mjOBJ_JOINT, m->actuator_trnid[2 * i]);
            if (jname) act.joint = jname;
        }

        // Control range
        act.has_range = (m->actuator_ctrllimited[i] != 0);
        if (act.has_range) {
            act.range_low = static_cast<float>(m->actuator_ctrlrange[2 * i]);
            act.range_high = static_cast<float>(m->actuator_ctrlrange[2 * i + 1]);
        }

        info.actuators.push_back(std::move(act));
    }

    // Sensors
    info.sensors.reserve(m->nsensor);
    for (int i = 0; i < m->nsensor; i++) {
        SensorInfo sens;

        const char* sname = mj_id2name(m, mjOBJ_SENSOR, i);
        sens.name = sname ? sname : ("sensor_" + std::to_string(i));
        sens.type = SensorTypeName(m->sensor_type[i]);
        sens.dim = m->sensor_dim[i];
        sens.adr = m->sensor_adr[i];

        info.sensors.push_back(std::move(sens));
    }

    // Cameras (skip default camera at index -1)
    for (int i = 0; i < m->ncam; i++) {
        CameraInfo cam;
        cam.index = i;

        const char* cname = mj_id2name(m, mjOBJ_CAMERA, i);
        cam.name = cname ? cname : ("camera_" + std::to_string(i));

        info.cameras.push_back(std::move(cam));
    }

    // Bodies (skip worldbody at index 0)
    for (int i = 1; i < m->nbody; i++) {
        BodyInfo body;
        body.index = i;

        const char* bname = mj_id2name(m, mjOBJ_BODY, i);
        body.name = bname ? bname : ("body_" + std::to_string(i));

        info.bodies.push_back(std::move(body));
    }

    return info;
}

MjcfModelInfo ParseMjcfFile(const std::string& mjcf_path) {
    MjcfModelInfo info;

    char error_buf[1000] = {0};
    mjModel* m = mj_loadXML(mjcf_path.c_str(), nullptr, error_buf, sizeof(error_buf));

    if (!m) {
        info.error = error_buf[0] ? error_buf : "Unknown error loading " + mjcf_path;
        spdlog::error("[MjcfParser] Failed to load '{}': {}", mjcf_path, info.error);
        return info;
    }

    info = ParseMjcfModel(m);

    spdlog::info("[MjcfParser] Parsed '{}' — {} actuators, {} sensors, {} cameras, nq={}, nv={}",
                 mjcf_path, info.nu, info.nsensor, static_cast<int>(info.cameras.size()),
                 info.nq, info.nv);

    mj_deleteModel(m);
    return info;
}

} // namespace cyxwiz::plugin::mujoco
