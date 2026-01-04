#include "datastream_client.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <chrono>

namespace network {

// Helper: format bytes as human-readable
static std::string FormatBytes(int64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = static_cast<double>(bytes);
    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        unit++;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << size << " " << units[unit];
    return oss.str();
}

// Helper: format timestamp
static std::string FormatTimestamp(int64_t timestamp) {
    if (timestamp == 0) return "Unknown";
    std::time_t time = static_cast<std::time_t>(timestamp);
    std::tm* tm = std::localtime(&time);
    if (!tm) return "Invalid";
    std::ostringstream oss;
    oss << std::put_time(tm, "%Y-%m-%d %H:%M");
    return oss.str();
}

std::string CloudDatasetInfo::GetSizeString() const {
    return FormatBytes(total_size_bytes);
}

std::string CloudDatasetInfo::GetTrustLevelString() const {
    switch (trust_level) {
        case TrustLevel::Self: return "Self";
        case TrustLevel::Signed: return "Signed";
        case TrustLevel::Verified: return "Verified";
        case TrustLevel::Attested: return "Attested";
        case TrustLevel::Untrusted: return "Untrusted";
        default: return "Unknown";
    }
}

std::string CloudDatasetInfo::GetCreatedAtString() const {
    return FormatTimestamp(created_at);
}

DataStreamClient::DataStreamClient() = default;

DataStreamClient::~DataStreamClient() {
    StopStreaming();
    Disconnect();
}

bool DataStreamClient::Connect(const std::string& gateway_address) {
    if (connected_ && gateway_address_ == gateway_address) {
        return true;
    }

    Disconnect();

    try {
        grpc::ChannelArguments args;
        args.SetMaxReceiveMessageSize(100 * 1024 * 1024);  // 100 MB for large batches
        args.SetMaxSendMessageSize(100 * 1024 * 1024);

        channel_ = grpc::CreateCustomChannel(
            gateway_address,
            grpc::InsecureChannelCredentials(),
            args
        );

        if (!channel_) {
            last_error_ = "Failed to create channel";
            return false;
        }

        stub_ = cyxwiz::protocol::DataStreamService::NewStub(channel_);
        if (!stub_) {
            last_error_ = "Failed to create stub";
            return false;
        }

        gateway_address_ = gateway_address;
        connected_ = true;
        return true;
    } catch (const std::exception& e) {
        last_error_ = std::string("Connection error: ") + e.what();
        return false;
    }
}

void DataStreamClient::Disconnect() {
    StopStreaming();
    stub_.reset();
    channel_.reset();
    connected_ = false;
    gateway_address_.clear();
}

void DataStreamClient::AddAuthMetadata(grpc::ClientContext& context) {
    if (!auth_token_.empty()) {
        context.AddMetadata("authorization", "Bearer " + auth_token_);
    }
}

TrustLevel DataStreamClient::ConvertTrustLevel(cyxwiz::protocol::TrustLevel proto) {
    switch (proto) {
        case cyxwiz::protocol::TRUST_SELF: return TrustLevel::Self;
        case cyxwiz::protocol::TRUST_SIGNED: return TrustLevel::Signed;
        case cyxwiz::protocol::TRUST_VERIFIED: return TrustLevel::Verified;
        case cyxwiz::protocol::TRUST_ATTESTED: return TrustLevel::Attested;
        case cyxwiz::protocol::TRUST_UNTRUSTED: return TrustLevel::Untrusted;
        default: return TrustLevel::Untrusted;
    }
}

CloudDatasetInfo DataStreamClient::ConvertDatasetInfo(const cyxwiz::protocol::DatasetInfo& proto) {
    CloudDatasetInfo info;
    info.id = proto.id();
    info.name = proto.name();
    info.owner_id = proto.owner_id();
    info.description = proto.description();
    info.total_size_bytes = proto.total_size_bytes();
    info.file_count = proto.file_count();
    info.trust_level = ConvertTrustLevel(proto.trust_level());
    info.version = proto.version();
    info.created_at = proto.created_at();
    info.updated_at = proto.updated_at();
    info.schema_json = proto.schema_json();

    if (!proto.content_hash().empty()) {
        info.content_hash.assign(proto.content_hash().begin(), proto.content_hash().end());
    }

    return info;
}

CloudFileInfo DataStreamClient::ConvertFileInfo(const cyxwiz::protocol::DatasetFileInfo& proto) {
    CloudFileInfo info;
    info.id = proto.id();
    info.path = proto.path_in_dataset();
    info.size_bytes = proto.size_bytes();
    info.file_index = proto.file_index();

    if (!proto.content_hash().empty()) {
        info.content_hash.assign(proto.content_hash().begin(), proto.content_hash().end());
    }

    return info;
}

PublicDatasetInfo DataStreamClient::ConvertPublicDatasetInfo(const cyxwiz::protocol::PublicDatasetInfo& proto) {
    PublicDatasetInfo info;
    info.id = proto.id();
    info.name = proto.name();
    info.version = proto.version();
    info.official_url = proto.official_url();
    info.paper_url = proto.paper_url();
    info.license = proto.license();
    info.cached = proto.cached();
    info.cached_dataset_id = proto.cached_dataset_id();

    for (const auto& v : proto.verified_by()) {
        info.verified_by.push_back(v);
    }

    return info;
}

// ============================================================================
// Dataset Browsing
// ============================================================================

bool DataStreamClient::ListDatasets(std::vector<CloudDatasetInfo>& out_datasets,
                                     int limit,
                                     int offset,
                                     bool include_shared,
                                     TrustLevel max_trust_level) {
    if (!connected_ || !stub_) {
        last_error_ = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    AddAuthMetadata(context);

    cyxwiz::protocol::ListDatasetsRequest request;
    request.set_limit(limit);
    request.set_offset(offset);
    request.set_include_shared(include_shared);
    request.set_max_trust_level(static_cast<int32_t>(max_trust_level));

    cyxwiz::protocol::ListDatasetsResponse response;
    grpc::Status status = stub_->ListDatasets(&context, request, &response);

    if (!status.ok()) {
        last_error_ = "ListDatasets failed: " + status.error_message();
        return false;
    }

    out_datasets.clear();
    out_datasets.reserve(response.datasets_size());

    for (const auto& ds : response.datasets()) {
        out_datasets.push_back(ConvertDatasetInfo(ds));
    }

    return true;
}

bool DataStreamClient::GetDatasetInfo(const std::string& dataset_id,
                                       CloudDatasetInfo& out_info,
                                       std::vector<CloudFileInfo>& out_files) {
    if (!connected_ || !stub_) {
        last_error_ = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    AddAuthMetadata(context);

    cyxwiz::protocol::GetDatasetInfoRequest request;
    request.set_dataset_id(dataset_id);

    cyxwiz::protocol::DatasetInfoResponse response;
    grpc::Status status = stub_->GetDatasetInfo(&context, request, &response);

    if (!status.ok()) {
        last_error_ = "GetDatasetInfo failed: " + status.error_message();
        return false;
    }

    out_info = ConvertDatasetInfo(response.dataset());

    out_files.clear();
    out_files.reserve(response.files_size());
    for (const auto& f : response.files()) {
        out_files.push_back(ConvertFileInfo(f));
    }

    return true;
}

bool DataStreamClient::ListPublicDatasets(std::vector<PublicDatasetInfo>& out_datasets,
                                           const std::string& name_filter) {
    if (!connected_ || !stub_) {
        last_error_ = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    AddAuthMetadata(context);

    cyxwiz::protocol::ListPublicDatasetsRequest request;
    request.set_name_filter(name_filter);

    cyxwiz::protocol::ListPublicDatasetsResponse response;
    grpc::Status status = stub_->ListPublicDatasets(&context, request, &response);

    if (!status.ok()) {
        last_error_ = "ListPublicDatasets failed: " + status.error_message();
        return false;
    }

    out_datasets.clear();
    out_datasets.reserve(response.datasets_size());

    for (const auto& ds : response.datasets()) {
        out_datasets.push_back(ConvertPublicDatasetInfo(ds));
    }

    return true;
}

// ============================================================================
// Dataset Management
// ============================================================================

bool DataStreamClient::CreateDataset(const std::string& name,
                                      const std::string& description,
                                      const std::vector<std::string>& file_ids,
                                      const std::string& schema_json,
                                      CloudDatasetInfo& out_dataset) {
    if (!connected_ || !stub_) {
        last_error_ = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    AddAuthMetadata(context);

    cyxwiz::protocol::CreateDatasetRequest request;
    request.set_name(name);
    request.set_description(description);
    for (const auto& id : file_ids) {
        request.add_file_ids(id);
    }
    request.set_schema_json(schema_json);

    cyxwiz::protocol::CreateDatasetResponse response;
    grpc::Status status = stub_->CreateDataset(&context, request, &response);

    if (!status.ok()) {
        last_error_ = "CreateDataset failed: " + status.error_message();
        return false;
    }

    out_dataset = ConvertDatasetInfo(response.dataset());
    return true;
}

bool DataStreamClient::DeleteDataset(const std::string& dataset_id) {
    if (!connected_ || !stub_) {
        last_error_ = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    AddAuthMetadata(context);

    cyxwiz::protocol::DeleteDatasetRequest request;
    request.set_dataset_id(dataset_id);

    cyxwiz::protocol::DeleteDatasetResponse response;
    grpc::Status status = stub_->DeleteDataset(&context, request, &response);

    if (!status.ok()) {
        last_error_ = "DeleteDataset failed: " + status.error_message();
        return false;
    }

    if (!response.success()) {
        last_error_ = response.error();
        return false;
    }

    return true;
}

bool DataStreamClient::ShareDataset(const std::string& dataset_id,
                                     const std::string& share_with_user_id,
                                     const std::vector<std::string>& permissions,
                                     int64_t expires_at) {
    if (!connected_ || !stub_) {
        last_error_ = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    AddAuthMetadata(context);

    cyxwiz::protocol::ShareDatasetRequest request;
    request.set_dataset_id(dataset_id);
    request.set_share_with_user_id(share_with_user_id);
    for (const auto& p : permissions) {
        request.add_permissions(p);
    }
    request.set_expires_at(expires_at);

    cyxwiz::protocol::ShareDatasetResponse response;
    grpc::Status status = stub_->ShareDataset(&context, request, &response);

    if (!status.ok()) {
        last_error_ = "ShareDataset failed: " + status.error_message();
        return false;
    }

    return response.success();
}

// ============================================================================
// File Upload
// ============================================================================

bool DataStreamClient::UploadFile(const std::string& filepath,
                                   std::string& out_file_id,
                                   UploadProgressCallback progress_callback) {
    // Read file
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) {
        last_error_ = "Failed to open file: " + filepath;
        return false;
    }

    int64_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(size);
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        last_error_ = "Failed to read file: " + filepath;
        return false;
    }

    // Extract filename from path
    std::string filename = filepath;
    size_t pos = filepath.find_last_of("/\\");
    if (pos != std::string::npos) {
        filename = filepath.substr(pos + 1);
    }

    // Determine content type (simple heuristic)
    std::string content_type = "application/octet-stream";
    size_t ext_pos = filename.rfind('.');
    if (ext_pos != std::string::npos) {
        std::string ext = filename.substr(ext_pos);
        if (ext == ".jpg" || ext == ".jpeg") content_type = "image/jpeg";
        else if (ext == ".png") content_type = "image/png";
        else if (ext == ".csv") content_type = "text/csv";
        else if (ext == ".json") content_type = "application/json";
        else if (ext == ".gz") content_type = "application/gzip";
        else if (ext == ".zip") content_type = "application/zip";
    }

    return UploadFileData(filename, data, content_type, out_file_id, progress_callback);
}

bool DataStreamClient::UploadFileData(const std::string& filename,
                                       const std::vector<uint8_t>& data,
                                       const std::string& content_type,
                                       std::string& out_file_id,
                                       UploadProgressCallback progress_callback) {
    if (!connected_ || !stub_) {
        last_error_ = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    AddAuthMetadata(context);

    cyxwiz::protocol::UploadFileResponse response;
    auto writer = stub_->UploadFile(&context, &response);
    if (!writer) {
        last_error_ = "Failed to create upload stream";
        return false;
    }

    // Send metadata first
    cyxwiz::protocol::UploadFileRequest meta_request;
    auto* metadata = meta_request.mutable_metadata();
    metadata->set_filename(filename);
    metadata->set_size_bytes(static_cast<int64_t>(data.size()));
    metadata->set_content_type(content_type);
    // TODO: Compute Blake3 hash

    if (!writer->Write(meta_request)) {
        last_error_ = "Failed to send metadata";
        return false;
    }

    // Send data chunks
    const size_t chunk_size = 64 * 1024;  // 64 KB chunks
    int64_t bytes_sent = 0;

    for (size_t offset = 0; offset < data.size(); offset += chunk_size) {
        size_t remaining = data.size() - offset;
        size_t this_chunk = std::min(chunk_size, remaining);

        cyxwiz::protocol::UploadFileRequest chunk_request;
        chunk_request.set_chunk(data.data() + offset, this_chunk);

        if (!writer->Write(chunk_request)) {
            last_error_ = "Failed to send chunk";
            return false;
        }

        bytes_sent += static_cast<int64_t>(this_chunk);

        if (progress_callback) {
            progress_callback(bytes_sent, static_cast<int64_t>(data.size()));
        }
    }

    writer->WritesDone();
    grpc::Status status = writer->Finish();

    if (!status.ok()) {
        last_error_ = "Upload failed: " + status.error_message();
        return false;
    }

    if (!response.success()) {
        last_error_ = response.error();
        return false;
    }

    out_file_id = response.file_id();
    return true;
}

// ============================================================================
// Verification
// ============================================================================

bool DataStreamClient::VerifyDataset(const std::string& dataset_id,
                                      DatasetVerificationResult& out_result,
                                      bool check_public_registry,
                                      bool full_verification) {
    if (!connected_ || !stub_) {
        last_error_ = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    AddAuthMetadata(context);

    cyxwiz::protocol::VerifyDatasetRequest request;
    request.set_dataset_id(dataset_id);
    request.set_check_public_registry(check_public_registry);
    request.set_full_verification(full_verification);

    cyxwiz::protocol::VerificationResult response;
    grpc::Status status = stub_->VerifyDataset(&context, request, &response);

    if (!status.ok()) {
        last_error_ = "VerifyDataset failed: " + status.error_message();
        return false;
    }

    out_result.dataset_id = response.dataset_id();
    out_result.manifest_valid = response.manifest_valid();
    out_result.all_files_valid = response.all_files_valid();
    out_result.computed_trust_level = ConvertTrustLevel(response.computed_trust_level());
    out_result.message = response.message();

    if (response.has_public_match()) {
        out_result.public_match_name = response.public_match().name();
        out_result.public_match_confidence = response.public_match().confidence();
    }

    return true;
}

// ============================================================================
// Access Tokens
// ============================================================================

bool DataStreamClient::CreateAccessToken(const std::string& dataset_id,
                                          DataStreamAccessToken& out_token,
                                          const std::string& node_id,
                                          int64_t ttl_seconds) {
    if (!connected_ || !stub_) {
        last_error_ = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    AddAuthMetadata(context);

    cyxwiz::protocol::CreateAccessTokenRequest request;
    request.set_dataset_id(dataset_id);
    if (!node_id.empty()) {
        request.set_node_id(node_id);
    }
    request.add_scopes("read");
    request.add_scopes("stream");
    request.set_ttl_seconds(ttl_seconds);

    cyxwiz::protocol::AccessTokenResponse response;
    grpc::Status status = stub_->CreateAccessToken(&context, request, &response);

    if (!status.ok()) {
        last_error_ = "CreateAccessToken failed: " + status.error_message();
        return false;
    }

    out_token.token = response.token();
    out_token.token_id = response.token_id();
    out_token.expires_at = response.expires_at();
    out_token.scopes.clear();
    for (const auto& s : response.scopes()) {
        out_token.scopes.push_back(s);
    }

    return true;
}

bool DataStreamClient::RevokeAccessToken(const std::string& token_id) {
    if (!connected_ || !stub_) {
        last_error_ = "Not connected";
        return false;
    }

    grpc::ClientContext context;
    AddAuthMetadata(context);

    cyxwiz::protocol::RevokeAccessTokenRequest request;
    request.set_token_id(token_id);

    cyxwiz::protocol::RevokeAccessTokenResponse response;
    grpc::Status status = stub_->RevokeAccessToken(&context, request, &response);

    if (!status.ok()) {
        last_error_ = "RevokeAccessToken failed: " + status.error_message();
        return false;
    }

    return response.success();
}

// ============================================================================
// Batch Streaming
// ============================================================================

bool DataStreamClient::StartStreaming(const std::string& dataset_id,
                                       int32_t batch_size,
                                       BatchCallback on_batch,
                                       StreamErrorCallback on_error,
                                       StreamCompleteCallback on_complete,
                                       int64_t start_index,
                                       int32_t prefetch_batches,
                                       TrustLevel max_trust_level,
                                       bool shuffle,
                                       int64_t seed,
                                       const std::string& access_token) {
    if (!connected_ || !stub_) {
        last_error_ = "Not connected";
        return false;
    }

    if (streaming_.load()) {
        StopStreaming();
    }

    streaming_.store(true);
    should_stop_.store(false);
    current_batch_.store(0);
    total_batches_.store(0);

    // Start streaming thread
    streaming_thread_ = std::make_unique<std::thread>(
        &DataStreamClient::StreamingThread, this,
        dataset_id, batch_size, on_batch, on_error, on_complete,
        start_index, prefetch_batches, max_trust_level, shuffle, seed, access_token
    );

    return true;
}

void DataStreamClient::StopStreaming() {
    should_stop_.store(true);

    if (stream_context_) {
        stream_context_->TryCancel();
    }

    if (streaming_thread_ && streaming_thread_->joinable()) {
        streaming_thread_->join();
    }

    streaming_thread_.reset();
    stream_context_.reset();
    streaming_.store(false);
}

float DataStreamClient::GetStreamingProgress() const {
    uint64_t total = total_batches_.load();
    if (total == 0) return 0.0f;
    return static_cast<float>(current_batch_.load()) / static_cast<float>(total);
}

void DataStreamClient::StreamingThread(const std::string& dataset_id,
                                         int32_t batch_size,
                                         BatchCallback on_batch,
                                         StreamErrorCallback on_error,
                                         StreamCompleteCallback on_complete,
                                         int64_t start_index,
                                         int32_t prefetch_batches,
                                         TrustLevel max_trust_level,
                                         bool shuffle,
                                         int64_t seed,
                                         const std::string& access_token) {
    stream_context_ = std::make_unique<grpc::ClientContext>();
    AddAuthMetadata(*stream_context_);

    cyxwiz::protocol::StreamBatchesRequest request;
    request.set_dataset_id(dataset_id);
    if (!access_token.empty()) {
        request.set_access_token(access_token);
    }
    request.set_batch_size(batch_size);
    request.set_start_index(start_index);
    request.set_prefetch_batches(prefetch_batches);
    request.set_max_trust_level(static_cast<int32_t>(max_trust_level));
    request.set_shuffle(shuffle);
    request.set_seed(seed);

    auto reader = stub_->StreamBatches(stream_context_.get(), request);
    if (!reader) {
        if (on_error) {
            on_error("Failed to create stream reader");
        }
        streaming_.store(false);
        return;
    }

    cyxwiz::protocol::BatchResponse proto_batch;
    while (!should_stop_.load() && reader->Read(&proto_batch)) {
        // Update progress
        current_batch_.store(proto_batch.batch_index() + 1);
        total_batches_.store(proto_batch.total_batches());

        // Convert batch
        StreamingBatch batch;
        batch.batch_index = proto_batch.batch_index();
        batch.total_batches = proto_batch.total_batches();
        batch.is_last = proto_batch.is_last();

        batch.items.reserve(proto_batch.items_size());
        for (const auto& item : proto_batch.items()) {
            batch.items.emplace_back(item.begin(), item.end());
        }

        batch.item_hashes.reserve(proto_batch.item_hashes_size());
        for (const auto& hash : proto_batch.item_hashes()) {
            batch.item_hashes.emplace_back(hash.begin(), hash.end());
        }

        if (!proto_batch.batch_hash().empty()) {
            batch.batch_hash.assign(proto_batch.batch_hash().begin(),
                                    proto_batch.batch_hash().end());
        }

        // Invoke callback
        if (on_batch) {
            on_batch(batch);
        }

        if (batch.is_last) {
            break;
        }
    }

    grpc::Status status = reader->Finish();
    streaming_.store(false);

    if (!status.ok() && !should_stop_.load()) {
        if (on_error) {
            on_error(status.error_message());
        }
    } else if (on_complete) {
        on_complete(total_batches_.load());
    }
}

} // namespace network
