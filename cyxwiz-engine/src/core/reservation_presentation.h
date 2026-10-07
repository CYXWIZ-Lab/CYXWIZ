#pragma once

// Server Connection: the node reservation card (TOFIX118 gaps 5-6, mockup
// approved 2026-09-30). Pure data in and out, so the renderer only draws: the
// reserve quote, the active card (time left, Extend, End), the End summary, the
// receipt after a reservation ends, and the reconnect rows. Money is shown in
// US dollars when the node lists a dollar price; credits and balances come with
// TOFIX127, until then the balance rows say so.

#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

using ReservationDetails = std::vector<std::pair<std::string, std::string>>;

// The node as the listing described it.
struct ReservationNodeFacts {
    std::string name;
    std::string device_type;         // "CUDA", "OpenCL", "CPU"
    long long vram_bytes = 0;
    double reputation = 0.0;         // 0..1
    long long jobs_completed = 0;
    std::string region;
    bool online = false;
    double price_usd_per_hour = 0.0; // <= 0: the node lists no dollar price
    bool free_tier = false;
};

// ---- Reserve ---------------------------------------------------------------

struct ReserveQuoteInputs {
    ReservationNodeFacts node;
    int duration_minutes = 60;
    bool has_account = false;        // signed in (the account is the identity)
    bool reserving = false;
};

struct ReserveQuote {
    std::string title;               // node name
    std::string subtitle;            // "OpenCL · 7.8 GB · reputation 92% · 14 jobs"
    std::string duration;            // "2 h 00 min"
    std::string price;               // "$0.25 / hour (fee included)"
    std::string hold;                // "$0.50"
    std::string balance_after;
    std::string pay_rule;
    std::string button;              // "Reserve for 2 h 00 min"
    bool enabled = false;
    std::string disabled_reason;     // shown next to the button and on hover
    ReservationDetails details;
};

ReserveQuote BuildReserveQuote(const ReserveQuoteInputs& in);

// ---- Active reservation ----------------------------------------------------

enum class ReservationUrgency { Normal, Soon, Urgent, Ended };  // >=10 min, <10, <5, 0

struct ActiveReservationInputs {
    ReservationNodeFacts node;
    bool node_known = false;         // false after a reconnect with no listing
    std::string endpoint;
    long long now = 0;               // Unix seconds
    long long start_time = 0;        // 0: unknown (reconnected)
    long long end_time = 0;
    long long server_seconds_left = -1;   // heartbeat; -1: no reply yet
    long long server_checked_at = 0;      // Unix seconds of that reply
    bool p2p_connected = false;
    bool training_running = false;
    int epoch = 0;                   // training position, 0: unknown
    int total_epochs = 0;
    std::string reservation_id;
    std::string node_id;
    std::string current_job_id;
    long long access_until = 0;      // P2P token expiry
    int epochs_setting = 0;
    int batch_setting = 0;
};

struct ActiveReservationCard {
    std::string title;
    std::string subtitle;
    std::string connection;          // "Connected to node" / "Not connected"
    bool connected = false;
    long long seconds_left = 0;
    std::string time_left;           // "1:12:40", "4:12"
    ReservationUrgency urgency = ReservationUrgency::Normal;
    bool has_bar = false;            // start known
    double fill = 0.0;               // share of the reservation used
    std::string started;             // "Started 13:05"
    std::string ends;                // "Ends 15:05"
    std::string source;              // where time left comes from
    bool stale = false;              // the central server stopped answering
    std::string price;               // "$0.50 (2 h 00 min)": the reserved time
    bool warn = false;
    std::string warning;             // ending soon: what happens at the end
    ReservationDetails details;
};

ActiveReservationCard BuildActiveReservationCard(const ActiveReservationInputs& in);

struct ExtendOption {
    int minutes = 0;
    std::string label;               // "+30 min"
    std::string cost;                // "$0.13" or "-"
};

std::vector<ExtendOption> BuildExtendOptions(double price_usd_per_hour);

// ---- End and after ---------------------------------------------------------

// Leaving does not end the reservation: the reserved time stays the user's
// until it runs out (owner rule 2026-10-07); they come back from the
// reconnect prompt.
struct LeaveSummary {
    std::string title;               // "Leave dell-pc? The clock keeps running"
    std::string ends;                // "Your reservation ends at 15:39 (51 min left) whether ..."
    std::string body;                // what happens to training; how to come back
    std::string button;              // "Leave, keep the clock running"
};

LeaveSummary BuildLeaveSummary(const ActiveReservationInputs& in);

// TimeRanOut: the reserved time is over. Lost: the central server no longer
// knows the reservation (it was restarted, or ended it on its side).
enum class ReservationEndReason { TimeRanOut, Lost };

struct ReservationEndFacts {
    std::string node_name;
    long long ended_at = 0;          // Unix seconds
    ReservationEndReason reason = ReservationEndReason::TimeRanOut;
    long long seconds_used = -1;     // -1: not known
    int jobs_started = 0;
    std::string error;               // Lost: the Central Server's words
    std::string reservation_id;
};

struct ReservationReceipt {
    std::string title;               // "The reservation on dell-pc ended at 15:05"
    std::string why;                 // "time ran out", "the central server no longer knows it"
    bool failed = false;             // Lost: the note is a warning
    std::string time_used;
    std::string paid;
    std::string jobs;
    std::string note;
    ReservationDetails details;
};

ReservationReceipt BuildReservationReceipt(const ReservationEndFacts& facts);

// ---- Reconnect -------------------------------------------------------------

struct ActiveReservationListing {
    std::string reservation_id;
    std::string node_id;
    std::string node_name;           // empty: not in the node list
    long long ends_at = 0;           // Unix seconds
    bool engine_connected = false;
    int jobs_completed = 0;
};

struct ReconnectRow {
    std::string reservation_id;
    std::string node;                // name, else the node id
    std::string time_left;           // "ends at 15:39 (51 min left)"
    std::string note;                // "connected from another Engine" etc.
};

// Longest time left first; reservations that ended by now are left out.
std::vector<ReconnectRow> BuildReconnectRows(std::vector<ActiveReservationListing> listings, long long now);

// ---- Shared words ----------------------------------------------------------

std::string FormatCountdown(long long seconds);        // "1:12:40", "4:12", "0:05"
std::string FormatReservationLength(int minutes);      // "2 h 00 min", "45 min"
std::string FormatDollars(double dollars);             // "$0.50", "$12.40", "$0.004"
std::string FormatClockTime(long long unix_seconds);   // local "15:05"

}  // namespace cyxwiz
