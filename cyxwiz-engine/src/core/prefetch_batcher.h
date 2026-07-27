#pragma once

#include "dataset_batcher.h"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <spdlog/spdlog.h>

namespace cyxwiz {

class PrefetchBatcher final : public IBatcher {
public:
    PrefetchBatcher(IBatcher& source, size_t queue_depth, std::string name)
        : source_(&source)
        , queue_depth_(std::max<size_t>(1, queue_depth))
        , name_(std::move(name)) {}

    PrefetchBatcher(std::shared_ptr<IBatcher> source,
                    size_t queue_depth,
                    std::string name)
        : owned_source_(std::move(source))
        , source_(owned_source_.get())
        , queue_depth_(std::max<size_t>(1, queue_depth))
        , name_(std::move(name)) {}

    ~PrefetchBatcher() override {
        StopWorker();
    }

    Batch GetNextBatch() override {
        StartWorker();

        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_cv_.wait(lock, [this]() {
            return stop_requested_ || !queue_.empty() || source_complete_;
        });

        if (queue_.empty()) {
            return {};
        }

        Batch batch = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        not_full_cv_.notify_one();
        return batch;
    }

    void Reset() override {
        StopWorker();
        source_->Reset();

        std::lock_guard<std::mutex> lock(mutex_);
        source_complete_ = false;
        started_ = false;
        ClearQueue();
    }

    bool IsEpochComplete() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!started_) {
            return source_->IsEpochComplete();
        }
        return source_complete_ && queue_.empty();
    }

    size_t GetNumBatches() const override { return source_->GetNumBatches(); }
    size_t GetNumSamples() const override { return source_->GetNumSamples(); }

    void AdoptSourceOwnership(std::shared_ptr<IBatcher> source) {
        StopWorker();
        if (!source || source.get() != source_) {
            throw std::invalid_argument(
                "PrefetchBatcher ownership source must match its wrapped source");
        }
        owned_source_ = std::move(source);
        source_ = owned_source_.get();
    }

    void SetNormalization(float mean, float std_dev) override {
        StopWorker();
        source_->SetNormalization(mean, std_dev);
    }

    void SetOneHotEncoding(size_t num_classes) override {
        StopWorker();
        source_->SetOneHotEncoding(num_classes);
    }

    void SetScalarLabelMode(bool enable) override {
        StopWorker();
        source_->SetScalarLabelMode(enable);
    }

    void SetFlatten(bool flatten) override {
        StopWorker();
        source_->SetFlatten(flatten);
    }

    void SetPhase(BatcherPhase phase) override {
        StopWorker();
        source_->SetPhase(phase);
    }

private:
    void StartWorker() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (started_) {
            return;
        }

        stop_requested_ = false;
        source_complete_ = false;
        started_ = true;
        worker_ = std::thread([this]() { WorkerLoop(); });
        spdlog::info("PrefetchBatcher '{}': started async queue with depth={}",
                     name_, queue_depth_);
    }

    void StopWorker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_requested_ = true;
        }
        not_empty_cv_.notify_all();
        not_full_cv_.notify_all();

        if (worker_.joinable()) {
            worker_.join();
        }

        std::lock_guard<std::mutex> lock(mutex_);
        stop_requested_ = false;
        started_ = false;
        source_complete_ = false;
        ClearQueue();
    }

    void WorkerLoop() {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                not_full_cv_.wait(lock, [this]() {
                    return stop_requested_ || queue_.size() < queue_depth_;
                });
                if (stop_requested_) {
                    return;
                }
            }

            Batch batch = source_->GetNextBatch();

            std::unique_lock<std::mutex> lock(mutex_);
            if (stop_requested_) {
                return;
            }
            if (!batch.IsValid()) {
                source_complete_ = true;
                lock.unlock();
                not_empty_cv_.notify_all();
                return;
            }

            queue_.push(std::move(batch));
            lock.unlock();
            not_empty_cv_.notify_one();
        }
    }

    void ClearQueue() {
        std::queue<Batch> empty;
        queue_.swap(empty);
    }

    std::shared_ptr<IBatcher> owned_source_;
    IBatcher* source_ = nullptr;
    size_t queue_depth_;
    std::string name_;

    mutable std::mutex mutex_;
    std::condition_variable not_empty_cv_;
    std::condition_variable not_full_cv_;
    std::queue<Batch> queue_;
    std::thread worker_;
    bool started_ = false;
    bool stop_requested_ = false;
    bool source_complete_ = false;
};

} // namespace cyxwiz
