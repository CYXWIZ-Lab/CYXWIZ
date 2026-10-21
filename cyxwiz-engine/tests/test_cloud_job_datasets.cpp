// The Engine's side of a CyxCloud dataset in a remote job (TOFIX164 slice 4a).
#include "network/cloud_job_datasets.h"
#include "network/datastream_client.h"

#include <datastream.grpc.pb.h>
#include <grpcpp/grpcpp.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

bool Contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

// The gateway's DataStreamService, as far as tokens go.
class FakeDataStream final : public cyxcloud::datastream::DataStreamService::Service {
public:
    std::vector<cyxcloud::datastream::CreateAccessTokenRequest> requests;
    bool refuse = false;

    grpc::Status CreateAccessToken(grpc::ServerContext*, const cyxcloud::datastream::CreateAccessTokenRequest* request,
                                   cyxcloud::datastream::AccessTokenResponse* response) override {
        requests.push_back(*request);
        if (refuse) return grpc::Status(grpc::StatusCode::PERMISSION_DENIED, "No access to dataset");
        response->set_token("tok-" + request->dataset_id());
        response->set_token_id("id-" + request->dataset_id());
        response->set_expires_at(1'900'000'000);
        for (const auto& scope : request->scopes()) response->add_scopes(scope);
        return grpc::Status::OK;
    }
};

cyxwiz::CloudJobDataset Entry(const std::string& name, const std::string& dataset, const std::string& file,
                              const std::string& file_name, uint64_t size) {
    cyxwiz::CloudJobDataset entry;
    entry.dataset_name = name;
    entry.dataset_id = dataset;
    entry.file_id = file;
    entry.file_name = file_name;
    entry.size_bytes = size;
    return entry;
}

}  // namespace

int main() {
    FakeDataStream service;
    int port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    if (!server || port == 0) {
        std::cerr << "could not start the fake gateway\n";
        return EXIT_FAILURE;
    }
    const std::string gateway = "127.0.0.1:" + std::to_string(port);

    network::DataStreamClient client;
    Check(client.Connect(gateway), "client connects to the fake gateway");
    client.SetAuthToken("engine-jwt");

    cyxwiz::GraphJobDatasetPlan plan;
    plan.cloud.push_back(Entry("tokens", "dataset-a", "file-1", "tokens.parquet", 4096));
    plan.cloud.push_back(Entry("heldout", "dataset-a", "file-2", "heldout.parquet", 512));
    plan.cloud.push_back(Entry("vocab", "dataset-b", "file-3", "vocab.arrow", 0));

    std::cout << "one token per dataset, into the job\n";
    {
        cyxwiz::protocol::JobConfig config;
        config.set_job_id("job-1");
        std::string error;
        Check(network::FillCloudJobDatasets(plan, &client, "gw.local:50060", 3600, config, error), "filled: " + error);
        Check(config.cloud_datasets_size() == 3, "three sources");
        Check(service.requests.size() == 2, "two tokens minted for two datasets");
        if (config.cloud_datasets_size() == 3) {
            const auto& a = config.cloud_datasets(0);
            const auto& b = config.cloud_datasets(1);
            const auto& c = config.cloud_datasets(2);
            Check(a.dataset_name() == "tokens" && a.file_id() == "file-1" && a.file_name() == "tokens.parquet" &&
                      a.size_bytes() == 4096 && a.gateway_address() == "gw.local:50060",
                  "first source carries name, file, size and gateway");
            Check(a.access_token() == "tok-dataset-a" && b.access_token() == "tok-dataset-a",
                  "files of one dataset share its token");
            Check(c.access_token() == "tok-dataset-b", "the other dataset has its own token");
        }
        if (!service.requests.empty()) {
            const auto& request = service.requests.front();
            bool stream_scope = false;
            for (const auto& scope : request.scopes()) stream_scope = stream_scope || scope == "stream";
            Check(stream_scope, "token asks for the stream scope");
            Check(request.ttl_seconds() == 3600, "token lives as long as asked");
        }
    }

    std::cout << "a refusal names the dataset and the gateway's words\n";
    {
        service.refuse = true;
        cyxwiz::protocol::JobConfig config;
        auto* stale = config.add_cloud_datasets();
        stale->set_dataset_name("stale");
        std::string error;
        Check(!network::FillCloudJobDatasets(plan, &client, "gw.local:50060", 3600, config, error), "refused");
        Check(Contains(error, "tokens") && Contains(error, "No access to dataset"), "reason: " + error);
        Check(config.cloud_datasets_size() == 1 && config.cloud_datasets(0).dataset_name() == "stale",
              "config left as it was");
        service.refuse = false;
    }

    std::cout << "no connection, no tokens\n";
    {
        cyxwiz::protocol::JobConfig config;
        std::string error;
        const bool no_client = !network::FillCloudJobDatasets(plan, nullptr, "gw.local:50060", 3600, config, error);
        Check(no_client && Contains(error, "no CyxCloud connection"), "no client: " + error);
        network::DataStreamClient idle;
        Check(!network::FillCloudJobDatasets(plan, &idle, "gw.local:50060", 3600, config, error), "idle client");
        const bool no_address = !network::FillCloudJobDatasets(plan, &client, "", 3600, config, error);
        Check(no_address && Contains(error, "gateway address"), "no gateway address: " + error);
    }

    std::cout << "a plan without cloud inputs clears the list\n";
    {
        cyxwiz::protocol::JobConfig config;
        config.add_cloud_datasets()->set_dataset_name("old");
        std::string error;
        Check(network::FillCloudJobDatasets({}, nullptr, "", 0, config, error), "nothing to do");
        Check(config.cloud_datasets_size() == 0, "cleared");
    }

    std::cout << "token life follows the reservation\n";
    {
        Check(network::CloudAccessTtlSeconds(1'000'000 + 7200, 1'000'000) == 7200 + 300, "end + margin");
        Check(network::CloudAccessTtlSeconds(1'000'000 + 60, 1'000'000) == 15 * 60, "at least fifteen minutes");
        Check(network::CloudAccessTtlSeconds(0, 1'000'000) == 24 * 3600, "unknown end: a day");
    }

    server->Shutdown();
    if (g_failures == 0) std::cout << "test_cloud_job_datasets: all checks passed\n";
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
