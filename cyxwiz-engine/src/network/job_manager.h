#pragma once
#include <string>

namespace network {

class GRPCClient;

class JobManager {
public:
    explicit JobManager(GRPCClient* client);
    ~JobManager();

    void Update();

    bool SubmitJob(const std::string& job_config);
    void CancelJob(const std::string& job_id);

private:
    GRPCClient* client_;
};

} // namespace network
