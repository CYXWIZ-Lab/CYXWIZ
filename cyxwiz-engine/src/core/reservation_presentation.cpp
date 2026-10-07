#include "reservation_presentation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>

namespace cyxwiz {

namespace {

std::string Gigabytes(long long bytes) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.1f GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return text;
}

std::string NodeSubtitle(const ReservationNodeFacts& node) {
    std::string text = node.device_type.empty() ? "Device not listed" : node.device_type;
    if (node.vram_bytes > 0) text += " · " + Gigabytes(node.vram_bytes);
    text += " · reputation " + std::to_string(static_cast<int>(std::lround(node.reputation * 100.0))) + "%";
    text += " · " + std::to_string(node.jobs_completed) + (node.jobs_completed == 1 ? " job" : " jobs");
    return text;
}

long long MinutesRoundedUp(long long seconds) {
    return seconds <= 0 ? 0 : (seconds + 59) / 60;
}

std::string Minutes(long long minutes) {
    return FormatReservationLength(static_cast<int>(minutes));
}

}  // namespace

std::string FormatCountdown(long long seconds) {
    if (seconds < 0) seconds = 0;
    const long long h = seconds / 3600;
    const long long m = (seconds % 3600) / 60;
    const long long s = seconds % 60;
    char text[32];
    if (h > 0) {
        std::snprintf(text, sizeof(text), "%lld:%02lld:%02lld", h, m, s);
    } else {
        std::snprintf(text, sizeof(text), "%lld:%02lld", m, s);
    }
    return text;
}

std::string FormatReservationLength(int minutes) {
    if (minutes < 60) return std::to_string(minutes) + " min";
    char text[32];
    std::snprintf(text, sizeof(text), "%d h %02d min", minutes / 60, minutes % 60);
    return text;
}

std::string FormatDollars(double dollars) {
    char text[32];
    if (dollars > 0.0 && dollars < 0.01) {
        std::snprintf(text, sizeof(text), "$%.3f", dollars);
    } else {
        std::snprintf(text, sizeof(text), "$%.2f", dollars);
    }
    return text;
}

std::string FormatClockTime(long long unix_seconds) {
    const std::time_t value = static_cast<std::time_t>(unix_seconds);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &value);
#else
    localtime_r(&value, &local);
#endif
    char text[16];
    std::strftime(text, sizeof(text), "%H:%M", &local);
    return text;
}

ReserveQuote BuildReserveQuote(const ReserveQuoteInputs& in) {
    ReserveQuote quote;
    quote.title = in.node.name.empty() ? "Selected node" : in.node.name;
    quote.subtitle = NodeSubtitle(in.node);
    quote.duration = FormatReservationLength(in.duration_minutes);
    const double hours = in.duration_minutes / 60.0;
    if (in.node.free_tier) {
        quote.price = "Free (the node is rebuilding its reputation)";
        quote.hold = FormatDollars(0.0);
    } else if (in.node.price_usd_per_hour > 0.0) {
        quote.price = FormatDollars(in.node.price_usd_per_hour) + " / hour (fee included)";
        quote.hold = FormatDollars(in.node.price_usd_per_hour * hours);
    } else {
        quote.price = "The node lists no price";
        quote.hold = "-";
    }
    quote.balance_after = "Not tracked yet: balances start with credits";
    quote.pay_rule =
        "You pay for the reserved time; it is yours until it runs out, whether you use it or not. Leaving early "
        "returns nothing.";
    quote.button = in.reserving ? "Reserving..." : "Reserve for " + quote.duration;
    if (in.reserving) {
        quote.disabled_reason = "Waiting for the central server.";
    } else if (!in.node.online) {
        quote.disabled_reason = "This node is offline.";
    } else if (!in.has_account) {
        quote.disabled_reason = "Sign in to reserve a node.";
    }
    quote.enabled = quote.disabled_reason.empty();

    quote.details.emplace_back("Region", in.node.region.empty() ? "-" : in.node.region);
    quote.details.emplace_back("Billing", "per hour, for the time reserved");
    if (in.node.vram_bytes > 0) quote.details.emplace_back("Device memory", Gigabytes(in.node.vram_bytes));
    return quote;
}

ActiveReservationCard BuildActiveReservationCard(const ActiveReservationInputs& in) {
    ActiveReservationCard card;
    card.title = in.node_known && !in.node.name.empty() ? in.node.name : "Reserved node";
    card.subtitle = in.node_known ? NodeSubtitle(in.node) : in.endpoint;
    card.connected = in.p2p_connected;
    card.connection = in.p2p_connected ? "Connected to node" : "Not connected";

    // Time left: the Central Server's heartbeat when there is one (it counts
    // extensions and its own clock), else the end time and this clock.
    if (in.server_seconds_left >= 0) {
        card.seconds_left = std::max(0LL, in.server_seconds_left - std::max(0LL, in.now - in.server_checked_at));
        const long long ago = std::max(0LL, in.now - in.server_checked_at);
        // The heartbeat runs every 30 s; after two missed ones say so.
        card.stale = ago > 75;
        card.source = card.stale ? "The central server has not answered for " + std::to_string(ago) +
                                       " s; counting down from its last reply"
                                 : "From the central server, checked " + std::to_string(ago) + " s ago";
    } else {
        card.seconds_left = std::max(0LL, in.end_time - in.now);
        card.source = "From this computer's clock (waiting for the central server)";
    }
    card.time_left = FormatCountdown(card.seconds_left);
    if (card.seconds_left == 0) {
        card.urgency = ReservationUrgency::Ended;
    } else if (card.seconds_left < 300) {
        card.urgency = ReservationUrgency::Urgent;
    } else if (card.seconds_left < 600) {
        card.urgency = ReservationUrgency::Soon;
    }

    const long long end = in.now + card.seconds_left;
    card.ends = "Ends " + FormatClockTime(end);
    if (in.start_time > 0 && end > in.start_time) {
        card.has_bar = true;
        card.started = "Started " + FormatClockTime(in.start_time);
        card.fill = std::clamp(static_cast<double>(in.now - in.start_time) / static_cast<double>(end - in.start_time),
                               0.0, 1.0);
        const long long reserved = end - in.start_time;
        card.price = in.node.price_usd_per_hour > 0.0
                         ? FormatDollars(in.node.price_usd_per_hour * static_cast<double>(reserved) / 3600.0) + " (" +
                               FormatReservationLength(static_cast<int>(MinutesRoundedUp(reserved))) + ")"
                         : FormatReservationLength(static_cast<int>(MinutesRoundedUp(reserved)));
    } else {
        card.price = "-";
    }

    if (card.urgency == ReservationUrgency::Soon || card.urgency == ReservationUrgency::Urgent) {
        card.warn = true;
        const long long minutes = MinutesRoundedUp(card.seconds_left);
        card.warning = "Ends in " + std::to_string(minutes) + (minutes == 1 ? " minute. " : " minutes. ");
        if (in.training_running) {
            if (in.epoch > 0 && in.total_epochs > 0) {
                card.warning += "Training still running (epoch " + std::to_string(in.epoch) + " of " +
                                std::to_string(in.total_epochs) + "). ";
            } else {
                card.warning += "Training still running. ";
            }
            card.warning +=
                "At the end the node saves a checkpoint and stops the job; you can resume it in a new "
                "reservation. Extend to keep going.";
        } else {
            card.warning += "Extend to keep the node.";
        }
    }

    const auto add = [&card](const char* key, const std::string& value) {
        if (!value.empty()) card.details.emplace_back(key, value);
    };
    add("Reservation", in.reservation_id);
    add("Node id", in.node_id);
    add("Endpoint", in.endpoint);
    add("Current job", in.current_job_id);
    if (in.epochs_setting > 0) {
        add("Epochs / batch", std::to_string(in.epochs_setting) + " / " + std::to_string(in.batch_setting));
    }
    if (in.access_until > 0) add("Access valid until", FormatClockTime(in.access_until) + " (end + 5 min)");
    return card;
}

std::vector<ExtendOption> BuildExtendOptions(double price_usd_per_hour) {
    std::vector<ExtendOption> options;
    for (const auto& [minutes, label] :
         {std::pair{15, "+15 min"}, std::pair{30, "+30 min"}, std::pair{60, "+1 hour"}, std::pair{120, "+2 hours"}}) {
        ExtendOption option;
        option.minutes = minutes;
        option.label = label;
        option.cost = price_usd_per_hour > 0.0 ? FormatDollars(price_usd_per_hour * minutes / 60.0) : "-";
        options.push_back(option);
    }
    return options;
}

LeaveSummary BuildLeaveSummary(const ActiveReservationInputs& in) {
    LeaveSummary summary;
    const std::string name = in.node_known && !in.node.name.empty() ? in.node.name : "the node";
    summary.title = "Leave " + name + "?";
    if (in.training_running) {
        summary.body = "Training is running";
        if (in.epoch > 0 && in.total_epochs > 0) {
            summary.body += " (epoch " + std::to_string(in.epoch) + " of " + std::to_string(in.total_epochs) + ")";
        }
        summary.body += ". Leaving stops it with a checkpoint; the node keeps it. ";
    }
    summary.body +=
        "The reserved time stays yours: come back from this screen to use it. Nothing is returned for time "
        "left when it runs out.";
    const long long left =
        in.server_seconds_left >= 0
            ? std::max(0LL, in.server_seconds_left - std::max(0LL, in.now - in.server_checked_at))
            : std::max(0LL, in.end_time - in.now);
    summary.ends = "The reservation ends at " + FormatClockTime(in.now + left) + " (" +
                   FormatReservationLength(static_cast<int>(MinutesRoundedUp(left))) + " left)";
    return summary;
}

ReservationReceipt BuildReservationReceipt(const ReservationEndFacts& facts) {
    ReservationReceipt receipt;
    const std::string name = facts.node_name.empty() ? "the node" : facts.node_name;
    if (facts.reason == ReservationEndReason::Lost) {
        receipt.failed = true;
        receipt.title = "The reservation on " + name + " is over";
        receipt.why = "the central server no longer knows it";
        receipt.note = (facts.error.empty() ? std::string("No reason given.") : facts.error) +
                       " The central server was restarted or ended it on its side; the node keeps any checkpoints. "
                       "Reserve again to continue.";
    } else {
        receipt.title = "The reservation on " + name + " ended at " + FormatClockTime(facts.ended_at);
        receipt.why = "time ran out";
        receipt.note =
            "A job still training was stopped with a checkpoint; resume it from the P2P Training panel after "
            "reserving a node again.";
    }
    receipt.time_used = facts.seconds_used >= 0 ? Minutes(MinutesRoundedUp(facts.seconds_used)) : "not known";
    receipt.paid = "Nothing charged: payments are not live yet";
    receipt.jobs = std::to_string(facts.jobs_started) + (facts.jobs_started == 1 ? " job started" : " jobs started");
    if (!facts.reservation_id.empty()) receipt.details.emplace_back("Reservation", facts.reservation_id);
    return receipt;
}

std::vector<ReconnectRow> BuildReconnectRows(std::vector<ActiveReservationListing> listings, long long now) {
    std::sort(listings.begin(), listings.end(), [](const auto& a, const auto& b) { return a.ends_at > b.ends_at; });
    std::vector<ReconnectRow> rows;
    for (const auto& listing : listings) {
        if (listing.ends_at <= now) continue;
        ReconnectRow row;
        row.reservation_id = listing.reservation_id;
        row.node = listing.node_name.empty() ? "Node " + listing.node_id.substr(0, 8) : listing.node_name;
        row.time_left = FormatCountdown(listing.ends_at - now) + " left";
        if (listing.engine_connected) row.note = "open in another Engine";
        if (listing.jobs_completed > 0) {
            if (!row.note.empty()) row.note += " · ";
            row.note += std::to_string(listing.jobs_completed) + (listing.jobs_completed == 1 ? " job done" : " jobs done");
        }
        rows.push_back(row);
    }
    return rows;
}

}  // namespace cyxwiz
