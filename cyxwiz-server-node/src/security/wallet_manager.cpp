// wallet_manager.cpp - Solana wallet integration implementation
#include "security/wallet_manager.h"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <httplib.h>
#include <fstream>
#include <regex>
#include <chrono>

namespace cyxwiz::servernode::security {

// CYXWIZ token mint address (placeholder - would be real address in production)
static const std::string CYXWIZ_MINT_ADDRESS = "CYXWiz1111111111111111111111111111111111111";

const char* GetNetworkName(SolanaNetwork network) {
    switch (network) {
        case SolanaNetwork::Mainnet: return "mainnet-beta";
        case SolanaNetwork::Devnet: return "devnet";
        case SolanaNetwork::Testnet: return "testnet";
        default: return "unknown";
    }
}

std::string GetDefaultRPCUrl(SolanaNetwork network) {
    switch (network) {
        case SolanaNetwork::Mainnet:
            return "https://api.mainnet-beta.solana.com";
        case SolanaNetwork::Devnet:
            return "https://api.devnet.solana.com";
        case SolanaNetwork::Testnet:
            return "https://api.testnet.solana.com";
        default:
            return "https://api.devnet.solana.com";
    }
}

WalletManager::WalletManager(SolanaNetwork network)
    : network_(network) {
    rpc_url_ = GetDefaultRPCUrl(network);
}

WalletManager::WalletManager(const std::string& rpc_url, SolanaNetwork network)
    : rpc_url_(rpc_url)
    , network_(network) {
}

bool WalletManager::IsValidSolanaAddress(const std::string& address) {
    // Solana addresses are Base58 encoded and 32-44 characters
    if (address.length() < 32 || address.length() > 44) {
        return false;
    }

    // Check for valid Base58 characters
    static const std::regex base58_regex("^[1-9A-HJ-NP-Za-km-z]+$");
    return std::regex_match(address, base58_regex);
}

bool WalletManager::ConnectExternalWallet(const std::string& public_address) {
    if (!IsValidSolanaAddress(public_address)) {
        spdlog::error("WalletManager: Invalid Solana address: {}", public_address);
        return false;
    }

    public_address_ = public_address;
    connected_ = true;

    // Fetch initial balance
    RefreshBalance();

    spdlog::info("WalletManager: Connected to external wallet {}", public_address_);

    if (event_callback_) {
        event_callback_("WALLET_CONNECTED", public_address_);
    }

    return true;
}

void WalletManager::Disconnect() {
    if (connected_) {
        spdlog::info("WalletManager: Disconnected wallet {}", public_address_);

        if (event_callback_) {
            event_callback_("WALLET_DISCONNECTED", public_address_);
        }
    }

    public_address_.clear();
    connected_ = false;
    cached_balance_ = WalletBalance{};
}

std::string WalletManager::BuildRPCRequest(const std::string& method, const std::string& params) {
    nlohmann::json request = {
        {"jsonrpc", "2.0"},
        {"id", 1},
        {"method", method},
        {"params", nlohmann::json::parse(params)}
    };
    return request.dump();
}

WalletManager::RPCResponse WalletManager::SolanaRPC(const std::string& method,
                                                     const std::string& params) {
    RPCResponse response;

    try {
        // Parse URL
        std::string host;
        std::string path = "/";
        int port = 443;
        bool use_ssl = true;

        size_t proto_end = rpc_url_.find("://");
        if (proto_end != std::string::npos) {
            std::string proto = rpc_url_.substr(0, proto_end);
            use_ssl = (proto == "https");
            host = rpc_url_.substr(proto_end + 3);
        } else {
            host = rpc_url_;
        }

        size_t path_start = host.find('/');
        if (path_start != std::string::npos) {
            path = host.substr(path_start);
            host = host.substr(0, path_start);
        }

        size_t port_start = host.find(':');
        if (port_start != std::string::npos) {
            port = std::stoi(host.substr(port_start + 1));
            host = host.substr(0, port_start);
        } else {
            port = use_ssl ? 443 : 80;
        }

        // Make request
        std::string request_body = BuildRPCRequest(method, params);
        spdlog::debug("WalletManager: RPC {} to {}", method, host);

        std::unique_ptr<httplib::Client> client;
        if (use_ssl) {
            client = std::make_unique<httplib::Client>(("https://" + host).c_str());
        } else {
            client = std::make_unique<httplib::Client>(host, port);
        }

        client->set_connection_timeout(10);
        client->set_read_timeout(30);

        auto res = client->Post(path, request_body, "application/json");

        if (!res) {
            response.error = "HTTP request failed";
            spdlog::error("WalletManager: RPC request failed: {}", response.error);
            return response;
        }

        if (res->status != 200) {
            response.error = "HTTP " + std::to_string(res->status);
            spdlog::error("WalletManager: RPC error: {}", response.error);
            return response;
        }

        // Parse response
        nlohmann::json json_response = nlohmann::json::parse(res->body);

        if (json_response.contains("error")) {
            response.error = json_response["error"]["message"].get<std::string>();
            response.error_code = json_response["error"]["code"].get<int>();
            spdlog::error("WalletManager: RPC error: {}", response.error);
            return response;
        }

        response.success = true;
        response.result = json_response["result"].dump();

    } catch (const std::exception& e) {
        response.error = e.what();
        spdlog::error("WalletManager: RPC exception: {}", e.what());
    }

    return response;
}

double WalletManager::ParseSOLBalance(const std::string& response) {
    try {
        nlohmann::json j = nlohmann::json::parse(response);
        int64_t lamports = j["value"].get<int64_t>();
        return lamports / 1e9;  // Lamports to SOL
    } catch (...) {
        return 0.0;
    }
}

double WalletManager::ParseTokenBalance(const std::string& response) {
    try {
        nlohmann::json j = nlohmann::json::parse(response);
        auto accounts = j["value"];
        for (const auto& account : accounts) {
            auto token_amount = account["account"]["data"]["parsed"]["info"]["tokenAmount"];
            return std::stod(token_amount["uiAmountString"].get<std::string>());
        }
    } catch (...) {
    }
    return 0.0;
}

bool WalletManager::RefreshBalance() {
    if (!connected_) {
        return false;
    }

    // Get SOL balance
    std::string params = "[\"" + public_address_ + "\"]";
    auto sol_response = SolanaRPC("getBalance", params);
    if (sol_response.success) {
        cached_balance_.sol_balance = ParseSOLBalance(sol_response.result);
    }

    // Get CYXWIZ token balance
    std::string token_params = R"([
        ")" + public_address_ + R"(",
        {"mint": ")" + CYXWIZ_MINT_ADDRESS + R"("},
        {"encoding": "jsonParsed"}
    ])";
    auto token_response = SolanaRPC("getTokenAccountsByOwner", token_params);
    if (token_response.success) {
        cached_balance_.cyxwiz_balance = ParseTokenBalance(token_response.result);
    }

    cached_balance_.last_updated = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    spdlog::debug("WalletManager: Balance refreshed - SOL: {}, CYXWIZ: {}",
                 cached_balance_.sol_balance, cached_balance_.cyxwiz_balance);

    return true;
}

std::optional<WalletBalance> WalletManager::GetBalance() {
    if (!connected_) {
        return std::nullopt;
    }

    // Refresh if stale (older than 30 seconds)
    int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    if (now - cached_balance_.last_updated > 30) {
        RefreshBalance();
    }

    return cached_balance_;
}

double WalletManager::GetSOLBalance() {
    auto balance = GetBalance();
    return balance ? balance->sol_balance : 0.0;
}

double WalletManager::GetCYXWIZBalance() {
    auto balance = GetBalance();
    return balance ? balance->cyxwiz_balance : 0.0;
}

UnsignedTransaction WalletManager::BuildWithdrawTransaction(
    const std::string& destination,
    double amount_cyxwiz) {

    UnsignedTransaction tx;

    if (!connected_) {
        spdlog::error("WalletManager: Not connected");
        return tx;
    }

    if (!IsValidSolanaAddress(destination)) {
        spdlog::error("WalletManager: Invalid destination address");
        return tx;
    }

    if (amount_cyxwiz <= 0) {
        spdlog::error("WalletManager: Invalid amount");
        return tx;
    }

    // Get recent blockhash
    auto blockhash_response = SolanaRPC("getLatestBlockhash", "[]");
    if (!blockhash_response.success) {
        spdlog::error("WalletManager: Failed to get blockhash");
        return tx;
    }

    try {
        nlohmann::json j = nlohmann::json::parse(blockhash_response.result);
        tx.blockhash = j["value"]["blockhash"].get<std::string>();
    } catch (...) {
        spdlog::error("WalletManager: Failed to parse blockhash");
        return tx;
    }

    // Build transaction message (simplified - real implementation would use Solana SDK)
    // The actual transaction would be built by the frontend wallet

    tx.destination = destination;
    tx.amount = amount_cyxwiz;
    tx.created_at = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    tx.valid = true;

    // For now, store metadata that frontend can use to build actual transaction
    nlohmann::json metadata = {
        {"type", "token_transfer"},
        {"from", public_address_},
        {"to", destination},
        {"mint", CYXWIZ_MINT_ADDRESS},
        {"amount", amount_cyxwiz},
        {"blockhash", tx.blockhash}
    };
    tx.serialized_message = metadata.dump();

    spdlog::info("WalletManager: Built withdraw transaction for {} CYXWIZ to {}",
                 amount_cyxwiz, destination);

    return tx;
}

UnsignedTransaction WalletManager::BuildSOLTransferTransaction(
    const std::string& destination,
    double amount_sol) {

    UnsignedTransaction tx;

    if (!connected_ || !IsValidSolanaAddress(destination) || amount_sol <= 0) {
        return tx;
    }

    // Get recent blockhash
    auto blockhash_response = SolanaRPC("getLatestBlockhash", "[]");
    if (!blockhash_response.success) {
        return tx;
    }

    try {
        nlohmann::json j = nlohmann::json::parse(blockhash_response.result);
        tx.blockhash = j["value"]["blockhash"].get<std::string>();
    } catch (...) {
        return tx;
    }

    tx.destination = destination;
    tx.amount = amount_sol;
    tx.created_at = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    tx.valid = true;

    nlohmann::json metadata = {
        {"type", "sol_transfer"},
        {"from", public_address_},
        {"to", destination},
        {"amount", amount_sol},
        {"blockhash", tx.blockhash}
    };
    tx.serialized_message = metadata.dump();

    return tx;
}

std::string WalletManager::SubmitSignedTransaction(const std::string& signed_tx) {
    std::string params = R"([")" + signed_tx + R"(", {"encoding": "base64"}])";
    auto response = SolanaRPC("sendTransaction", params);

    if (!response.success) {
        spdlog::error("WalletManager: Failed to submit transaction: {}", response.error);
        return "";
    }

    try {
        nlohmann::json j = nlohmann::json::parse(response.result);
        std::string signature = j.get<std::string>();
        spdlog::info("WalletManager: Transaction submitted: {}", signature);

        if (event_callback_) {
            event_callback_("TRANSACTION_SUBMITTED", signature);
        }

        return signature;
    } catch (...) {
        return "";
    }
}

std::vector<Transaction> WalletManager::GetTransactionHistory(int limit) {
    std::vector<Transaction> transactions;

    if (!connected_) {
        return transactions;
    }

    std::string params = R"([
        ")" + public_address_ + R"(",
        {"limit": )" + std::to_string(limit) + R"(}
    ])";

    auto response = SolanaRPC("getSignaturesForAddress", params);
    if (!response.success) {
        return transactions;
    }

    try {
        nlohmann::json sigs = nlohmann::json::parse(response.result);
        for (const auto& sig_info : sigs) {
            Transaction tx;
            tx.signature = sig_info["signature"].get<std::string>();
            tx.timestamp = sig_info.value("blockTime", 0);
            tx.status = sig_info.value("confirmationStatus", "confirmed");

            if (sig_info.contains("err") && !sig_info["err"].is_null()) {
                tx.status = "failed";
            }

            transactions.push_back(tx);
        }
    } catch (...) {
    }

    return transactions;
}

std::optional<Transaction> WalletManager::GetTransaction(const std::string& signature) {
    std::string params = R"([")" + signature + R"(", {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}])";
    auto response = SolanaRPC("getTransaction", params);

    if (!response.success) {
        return std::nullopt;
    }

    try {
        nlohmann::json j = nlohmann::json::parse(response.result);
        Transaction tx;
        tx.signature = signature;
        tx.timestamp = j.value("blockTime", 0);
        tx.status = "confirmed";

        // Parse transaction details (simplified)
        if (j.contains("meta") && j["meta"].contains("err") && !j["meta"]["err"].is_null()) {
            tx.status = "failed";
        }

        return tx;
    } catch (...) {
        return std::nullopt;
    }
}

void WalletManager::RecordEarning(double amount, const std::string& job_id) {
    pending_earnings_ += amount;
    total_earnings_ += amount;

    spdlog::info("WalletManager: Recorded earning of {} CYXWIZ for job {}",
                 amount, job_id);

    if (event_callback_) {
        nlohmann::json details = {{"amount", amount}, {"job_id", job_id}};
        event_callback_("EARNING_RECORDED", details.dump());
    }
}

std::string WalletManager::GetCYXWIZMintAddress() {
    return CYXWIZ_MINT_ADDRESS;
}

void WalletManager::SetEventCallback(WalletEventCallback callback) {
    event_callback_ = std::move(callback);
}

bool WalletManager::SaveState(const std::string& path) {
    try {
        nlohmann::json state = {
            {"public_address", public_address_},
            {"network", static_cast<int>(network_)},
            {"total_earnings", total_earnings_},
            {"pending_earnings", pending_earnings_}
        };

        std::ofstream file(path);
        file << state.dump(2);
        return true;
    } catch (...) {
        return false;
    }
}

bool WalletManager::LoadState(const std::string& path) {
    try {
        std::ifstream file(path);
        if (!file) return false;

        nlohmann::json state = nlohmann::json::parse(file);

        if (state.contains("public_address") && !state["public_address"].get<std::string>().empty()) {
            ConnectExternalWallet(state["public_address"].get<std::string>());
        }

        total_earnings_ = state.value("total_earnings", 0.0);
        pending_earnings_ = state.value("pending_earnings", 0.0);

        return true;
    } catch (...) {
        return false;
    }
}

// Singleton
std::unique_ptr<WalletManager> WalletManagerSingleton::instance_;

void WalletManagerSingleton::Initialize(SolanaNetwork network) {
    instance_ = std::make_unique<WalletManager>(network);
}

WalletManager& WalletManagerSingleton::Instance() {
    if (!instance_) {
        instance_ = std::make_unique<WalletManager>(SolanaNetwork::Devnet);
    }
    return *instance_;
}

} // namespace cyxwiz::servernode::security
