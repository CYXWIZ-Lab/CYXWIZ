// wallet_manager.h - Solana wallet integration (external wallet model)
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <functional>
#include <memory>

namespace cyxwiz::servernode::security {

// Solana network types
enum class SolanaNetwork {
    Mainnet,
    Devnet,
    Testnet
};

const char* GetNetworkName(SolanaNetwork network);
std::string GetDefaultRPCUrl(SolanaNetwork network);

// Transaction types
enum class TransactionType {
    EARN,           // Mining/compute rewards
    WITHDRAW,       // Withdrawal to external address
    DEPOSIT,        // Deposit from external address
    STAKE,          // Staking
    UNSTAKE,        // Unstaking
    UNKNOWN
};

// Transaction record
struct Transaction {
    std::string signature;     // Transaction signature
    int64_t timestamp = 0;     // Unix timestamp
    TransactionType type = TransactionType::UNKNOWN;
    double amount = 0.0;       // Amount in CYXWIZ or SOL
    std::string token;         // "SOL" or "CYXWIZ"
    std::string status;        // "confirmed", "pending", "failed"
    std::string from_address;
    std::string to_address;
    std::string error;         // Error message if failed
};

// Unsigned transaction for external signing
struct UnsignedTransaction {
    std::string serialized_message;  // Base58 encoded message
    std::string blockhash;           // Recent blockhash
    int64_t created_at = 0;         // Unix timestamp
    double amount = 0.0;
    std::string destination;
    bool valid = false;
};

// Balance information
struct WalletBalance {
    double sol_balance = 0.0;       // SOL balance
    double cyxwiz_balance = 0.0;    // CYXWIZ token balance
    int64_t last_updated = 0;       // Unix timestamp
};

// Callback for wallet events
using WalletEventCallback = std::function<void(const std::string& event, const std::string& details)>;

class WalletManager {
public:
    explicit WalletManager(SolanaNetwork network = SolanaNetwork::Mainnet);
    WalletManager(const std::string& rpc_url, SolanaNetwork network = SolanaNetwork::Mainnet);
    ~WalletManager() = default;

    // Connection (external wallet - no private key storage)
    bool ConnectExternalWallet(const std::string& public_address);
    void Disconnect();
    bool IsConnected() const { return connected_; }
    std::string GetAddress() const { return public_address_; }

    // Network
    SolanaNetwork GetNetwork() const { return network_; }
    std::string GetRPCUrl() const { return rpc_url_; }
    void SetRPCUrl(const std::string& url) { rpc_url_ = url; }

    // Balance queries (via RPC)
    std::optional<WalletBalance> GetBalance();
    double GetSOLBalance();
    double GetCYXWIZBalance();
    bool RefreshBalance();

    // Transaction building (for external signing)
    // Returns unsigned transaction that can be signed by external wallet
    UnsignedTransaction BuildWithdrawTransaction(
        const std::string& destination,
        double amount_cyxwiz);

    UnsignedTransaction BuildSOLTransferTransaction(
        const std::string& destination,
        double amount_sol);

    // Submit pre-signed transaction
    std::string SubmitSignedTransaction(const std::string& signed_tx);

    // Transaction history
    std::vector<Transaction> GetTransactionHistory(int limit = 50);
    std::optional<Transaction> GetTransaction(const std::string& signature);

    // Earnings tracking (local accumulator)
    void RecordEarning(double amount, const std::string& job_id);
    double GetTotalEarnings() const { return total_earnings_; }
    double GetPendingEarnings() const { return pending_earnings_; }

    // Address validation
    static bool IsValidSolanaAddress(const std::string& address);

    // Token info
    static std::string GetCYXWIZMintAddress();

    // Event callback
    void SetEventCallback(WalletEventCallback callback);

    // Persistence
    bool SaveState(const std::string& path);
    bool LoadState(const std::string& path);

private:
    std::string public_address_;
    std::string rpc_url_;
    SolanaNetwork network_;
    bool connected_ = false;

    WalletBalance cached_balance_;
    double total_earnings_ = 0.0;
    double pending_earnings_ = 0.0;
    WalletEventCallback event_callback_;

    // Solana JSON-RPC helpers
    struct RPCResponse {
        bool success = false;
        std::string result;
        std::string error;
        int error_code = 0;
    };

    RPCResponse SolanaRPC(const std::string& method, const std::string& params);
    std::string BuildRPCRequest(const std::string& method, const std::string& params);

    // Balance parsing
    double ParseSOLBalance(const std::string& response);
    double ParseTokenBalance(const std::string& response);

    // Transaction parsing
    std::vector<Transaction> ParseTransactionHistory(const std::string& response);
};

// Global wallet manager singleton
class WalletManagerSingleton {
public:
    static WalletManager& Instance();
    static void Initialize(SolanaNetwork network = SolanaNetwork::Mainnet);

private:
    WalletManagerSingleton() = default;
    static std::unique_ptr<WalletManager> instance_;
};

} // namespace cyxwiz::servernode::security
