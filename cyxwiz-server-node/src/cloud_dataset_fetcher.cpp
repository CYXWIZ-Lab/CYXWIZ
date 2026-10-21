#include "cloud_dataset_fetcher.h"

#include "core/sha256_digest.h"

#include <cyxcloud_data.grpc.pb.h>
#include <grpcpp/grpcpp.h>
#include <spdlog/spdlog.h>

#include <cctype>
#include <chrono>
#include <fstream>
#include <thread>

namespace fs = std::filesystem;

namespace cyxwiz {
namespace server_node {

namespace {

const char* GatewayStatusName(grpc::StatusCode code) {
    switch (code) {
        case grpc::StatusCode::UNAUTHENTICATED: return "refused the token (UNAUTHENTICATED)";
        case grpc::StatusCode::PERMISSION_DENIED: return "refused the token for this file (PERMISSION_DENIED)";
        case grpc::StatusCode::NOT_FOUND: return "does not have the file (NOT_FOUND)";
        case grpc::StatusCode::DATA_LOSS: return "could not serve the whole file (DATA_LOSS)";
        case grpc::StatusCode::UNAVAILABLE: return "is unreachable (UNAVAILABLE)";
        case grpc::StatusCode::DEADLINE_EXCEEDED: return "took too long (DEADLINE_EXCEEDED)";
        case grpc::StatusCode::CANCELLED: return "stream cancelled (CANCELLED)";
        default: return "failed";
    }
}

}  // namespace

std::string SafeCloudFileName(const std::string& dataset_name) {
    std::string safe;
    for (const char c : dataset_name) {
        safe.push_back((std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') ? c : '_');
    }
    return safe.empty() ? std::string("dataset") : safe;
}

std::string CloudFileExtension(const std::string& file_name) {
    std::string extension = fs::path(file_name).extension().string();
    for (auto& c : extension) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return extension == ".parquet" ? ".parquet" : ".arrow";
}

CloudDatasetFetcher::CloudDatasetFetcher(fs::path directory, int chunk_timeout_ms)
    : directory_(std::move(directory)), chunk_timeout_ms_(chunk_timeout_ms) {}

std::shared_ptr<grpc::Channel> CloudDatasetFetcher::ChannelFor(const std::string& gateway_address) {
    // Plain channel on the LAN for now; TLS between machines is TOFIX163 D4.
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(64 * 1024 * 1024);
    return grpc::CreateCustomChannel(gateway_address, grpc::InsecureChannelCredentials(), args);
}

void CloudDatasetFetcher::Cancel() { cancelled_.store(true); }

bool CloudDatasetFetcher::Fetch(const cyxwiz::protocol::CloudDatasetSource& source, fs::path& file,
                                std::string& error) {
    bytes_received_ = 0;
    const std::string where = "dataset '" + source.dataset_name() + "' (file " + source.file_id() + ") from " +
                              source.gateway_address();
    if (source.gateway_address().empty() || source.file_id().empty()) {
        error = where + ": the job names no gateway or no file for it";
        return false;
    }
    if (source.access_token().empty()) {
        error = where + ": the job carries no access token for it";
        return false;
    }

    std::error_code ec;
    fs::create_directories(directory_, ec);
    const fs::path final_path =
        directory_ / (SafeCloudFileName(source.dataset_name()) + CloudFileExtension(source.file_name()));
    const fs::path part_path = final_path.string() + ".part";
    std::ofstream out(part_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        error = where + ": cannot write " + part_path.string();
        return false;
    }

    auto stub = cyxcloud::data::DataService::NewStub(ChannelFor(source.gateway_address()));
    cyxcloud::data::StreamDataRequest request;
    request.set_dataset_id(source.file_id());
    request.set_access_token(source.access_token());
    request.set_batch_size(1024);  // KB per message
    request.set_shuffle(false);
    request.set_prefetch(4);

    grpc::ClientContext context;
    auto reader = stub->StreamData(&context, request);

    // No message for chunk_timeout_ms, or a cancel, ends the call from the
    // side: Read() blocks and gRPC's deadline covers the whole call only.
    std::atomic<long long> last_message_ms{0};
    std::atomic<bool> finished{false};
    std::atomic<bool> timed_out{false};
    const auto now_ms = [] {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };
    last_message_ms.store(now_ms());
    std::thread watchdog([&] {
        while (!finished.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (cancelled_.load()) {
                context.TryCancel();
                return;
            }
            if (now_ms() - last_message_ms.load() > chunk_timeout_ms_) {
                timed_out.store(true);
                context.TryCancel();
                return;
            }
        }
    });

    Sha256Hasher hasher;
    cyxcloud::data::DataChunk chunk;
    bool saw_last = false;
    std::string trailer_sha256;
    uint64_t trailer_bytes = 0;
    while (reader->Read(&chunk)) {
        last_message_ms.store(now_ms());
        if (!chunk.data().empty()) {
            out.write(chunk.data().data(), static_cast<std::streamsize>(chunk.data().size()));
            if (!out) {
                error = where + ": write failed at byte " + std::to_string(bytes_received_);
                break;
            }
            if (!hasher.Update(chunk.data(), error)) break;
            bytes_received_ += chunk.data().size();
        }
        if (chunk.is_last()) {
            saw_last = true;
            trailer_sha256 = chunk.stream_sha256();
            trailer_bytes = chunk.total_bytes();
            break;
        }
    }
    finished.store(true);
    out.close();
    const grpc::Status status = reader->Finish();
    watchdog.join();

    const auto fail = [&](const std::string& why) {
        error = where + ": " + why;
        std::error_code remove_ec;
        fs::remove(part_path, remove_ec);
        return false;
    };
    if (!error.empty()) return fail(error.substr(where.size() + 2));
    if (cancelled_.load()) return fail("fetch cancelled");
    if (timed_out.load()) {
        return fail("no data for " + std::to_string(chunk_timeout_ms_ / 1000) + " s (" +
                    std::to_string(bytes_received_) + " bytes received)");
    }
    if (!status.ok()) {
        // The gateway's own words: DATA_LOSS names the chunk, UNAUTHENTICATED
        // and PERMISSION_DENIED say what the token lacks.
        return fail(std::string("gateway ") + GatewayStatusName(status.error_code()) + ": " + status.error_message());
    }
    if (!saw_last) return fail("the stream ended without a last chunk");
    std::string digest;
    if (!hasher.Finish(digest, error)) return fail(error);
    if (trailer_sha256.empty() || digest != trailer_sha256) {
        return fail("SHA-256 of the received bytes does not match the gateway's (" +
                    std::to_string(bytes_received_) + " bytes received)");
    }
    if (trailer_bytes != bytes_received_) {
        return fail("received " + std::to_string(bytes_received_) + " bytes, the gateway sent " +
                    std::to_string(trailer_bytes));
    }
    fs::rename(part_path, final_path, ec);
    if (ec) return fail("could not move the file into place: " + ec.message());
    file = final_path;
    spdlog::info("Fetched {} ({} bytes, SHA-256 verified)", where, bytes_received_);
    return true;
}

}  // namespace server_node
}  // namespace cyxwiz
