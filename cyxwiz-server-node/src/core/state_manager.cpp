// state_manager.cpp - Observable state implementation
#include "core/state_manager.h"
#include <algorithm>
#include <spdlog/spdlog.h>

namespace cyxwiz::servernode::core {

StateManager::StateManager() {
    spdlog::debug("StateManager created");
}

void StateManager::AddObserver(StateObserver* observer) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (std::find(observers_.begin(), observers_.end(), observer) == observers_.end()) {
        observers_.push_back(observer);
    }
}

void StateManager::RemoveObserver(StateObserver* observer) {
    std::lock_guard<std::mutex> lock(mutex_);
    observers_.erase(
        std::remove(observers_.begin(), observers_.end(), observer),
        observers_.end()
    );
}

// ========== Getters ==========

std::vector<JobState> StateManager::GetActiveJobs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return jobs_;
}

std::vector<DeploymentState> StateManager::GetDeployments() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return deployments_;
}

SystemMetrics StateManager::GetMetrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return metrics_;
}

EarningsInfo StateManager::GetEarningsToday() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return earnings_today_;
}

EarningsInfo StateManager::GetEarningsThisWeek() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return earnings_week_;
}

EarningsInfo StateManager::GetEarningsThisMonth() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return earnings_month_;
}

ConnectionStatus StateManager::GetConnectionStatus() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return connection_status_;
}

std::string StateManager::GetWalletAddress() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return wallet_address_;
}

double StateManager::GetWalletBalance() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return wallet_balance_;
}

// ========== Setters ==========

void StateManager::UpdateJobs(const std::vector<JobState>& jobs) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        jobs_ = jobs;
    }
    NotifyJobsChanged();
}

void StateManager::UpdateJob(const JobState& job) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = std::find_if(jobs_.begin(), jobs_.end(),
            [&job](const JobState& j) { return j.id == job.id; });

        if (it != jobs_.end()) {
            *it = job;
        } else {
            jobs_.push_back(job);
        }
    }
    NotifyJobsChanged();
}

void StateManager::RemoveJob(const std::string& job_id) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        jobs_.erase(
            std::remove_if(jobs_.begin(), jobs_.end(),
                [&job_id](const JobState& j) { return j.id == job_id; }),
            jobs_.end()
        );
    }
    NotifyJobsChanged();
}

void StateManager::UpdateDeployments(const std::vector<DeploymentState>& deployments) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        deployments_ = deployments;
    }
    NotifyDeploymentsChanged();
}

void StateManager::UpdateDeployment(const DeploymentState& deployment) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = std::find_if(deployments_.begin(), deployments_.end(),
            [&deployment](const DeploymentState& d) { return d.id == deployment.id; });

        if (it != deployments_.end()) {
            *it = deployment;
        } else {
            deployments_.push_back(deployment);
        }
    }
    NotifyDeploymentsChanged();
}

void StateManager::RemoveDeployment(const std::string& deployment_id) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        deployments_.erase(
            std::remove_if(deployments_.begin(), deployments_.end(),
                [&deployment_id](const DeploymentState& d) { return d.id == deployment_id; }),
            deployments_.end()
        );
    }
    NotifyDeploymentsChanged();
}

void StateManager::UpdateMetrics(const SystemMetrics& metrics) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_ = metrics;
    }
    NotifyMetricsUpdated();
}

void StateManager::UpdateConnectionStatus(ConnectionStatus status) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (connection_status_ == status) return;
        connection_status_ = status;
    }
    NotifyConnectionStatusChanged();
}

void StateManager::UpdateEarnings(const EarningsInfo& today, const EarningsInfo& week, const EarningsInfo& month) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        earnings_today_ = today;
        earnings_week_ = week;
        earnings_month_ = month;
    }
    NotifyEarningsChanged();
}

void StateManager::UpdateWallet(const std::string& address, double balance) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        wallet_address_ = address;
        wallet_balance_ = balance;
    }
    NotifyWalletChanged();
}

// ========== Notifications ==========

void StateManager::NotifyJobsChanged() {
    std::vector<StateObserver*> observers_copy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        observers_copy = observers_;
    }
    for (auto* observer : observers_copy) {
        if (observer) {
            observer->OnJobsChanged();
        }
    }
}

void StateManager::NotifyDeploymentsChanged() {
    std::vector<StateObserver*> observers_copy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        observers_copy = observers_;
    }
    for (auto* observer : observers_copy) {
        if (observer) {
            observer->OnDeploymentsChanged();
        }
    }
}

void StateManager::NotifyMetricsUpdated() {
    std::vector<StateObserver*> observers_copy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        observers_copy = observers_;
    }
    for (auto* observer : observers_copy) {
        if (observer) {
            observer->OnMetricsUpdated();
        }
    }
}

void StateManager::NotifyConnectionStatusChanged() {
    std::vector<StateObserver*> observers_copy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        observers_copy = observers_;
    }
    for (auto* observer : observers_copy) {
        if (observer) {
            observer->OnConnectionStatusChanged();
        }
    }
}

void StateManager::NotifyEarningsChanged() {
    std::vector<StateObserver*> observers_copy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        observers_copy = observers_;
    }
    for (auto* observer : observers_copy) {
        if (observer) {
            observer->OnEarningsChanged();
        }
    }
}

void StateManager::NotifyWalletChanged() {
    std::vector<StateObserver*> observers_copy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        observers_copy = observers_;
    }
    for (auto* observer : observers_copy) {
        if (observer) {
            observer->OnWalletChanged();
        }
    }
}

} // namespace cyxwiz::servernode::core
