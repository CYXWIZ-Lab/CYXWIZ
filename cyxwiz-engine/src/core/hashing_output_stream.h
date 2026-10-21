#pragma once

// An output stream that hashes what it writes (TOFIX157): an export's
// SHA-256 comes with writing it, instead of reading the file again for the
// lineage.

#include "sha256_digest.h"

#include <arrow/buffer.h>
#include <arrow/io/interfaces.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include <memory>
#include <string>
#include <string_view>

namespace cyxwiz {

class HashingOutputStream : public arrow::io::OutputStream {
public:
    explicit HashingOutputStream(std::shared_ptr<arrow::io::OutputStream> output) : output_(std::move(output)) {}

    arrow::Status Close() override { return output_->Close(); }
    bool closed() const override { return output_->closed(); }
    arrow::Result<int64_t> Tell() const override { return output_->Tell(); }
    arrow::Status Flush() override { return output_->Flush(); }
    arrow::Status Write(const void* data, int64_t nbytes) override {
        Hash(data, nbytes);
        return output_->Write(data, nbytes);
    }
    arrow::Status Write(const std::shared_ptr<arrow::Buffer>& data) override {
        Hash(data->data(), data->size());
        return output_->Write(data);
    }

    uint64_t bytes() const { return bytes_; }
    // The digest of everything written; "" when hashing failed.
    std::string Finish() {
        std::string digest, error;
        if (!ok_ || !hasher_.Finish(digest, error)) return {};
        return digest;
    }

private:
    void Hash(const void* data, int64_t nbytes) {
        if (nbytes <= 0) return;
        std::string error;
        if (ok_) ok_ = hasher_.Update(std::string_view(static_cast<const char*>(data), static_cast<size_t>(nbytes)), error);
        bytes_ += static_cast<uint64_t>(nbytes);
    }

    std::shared_ptr<arrow::io::OutputStream> output_;
    Sha256Hasher hasher_;
    bool ok_ = true;
    uint64_t bytes_ = 0;
};

}  // namespace cyxwiz
