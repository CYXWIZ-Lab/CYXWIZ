// Reservation card presentation (TOFIX118 gaps 5-6, mockup approved
// 2026-09-30). Scenario: the owner's Dell (Intel Iris Xe over OpenCL) reserved
// for two hours at a listed $0.25 / hour.
#include "core/reservation_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++failures;
}

std::string Detail(const cyxwiz::ReservationDetails& details, const std::string& key) {
    for (const auto& [k, v] : details) {
        if (k == key) return v;
    }
    return "";
}

cyxwiz::ReservationNodeFacts Dell() {
    cyxwiz::ReservationNodeFacts node;
    node.name = "dell-pc";
    node.device_type = "OpenCL";
    node.vram_bytes = 8375186227LL;  // 7.8 GB shared
    node.reputation = 0.92;
    node.jobs_completed = 14;
    node.region = "local";
    node.online = true;
    node.price_usd_per_hour = 0.25;
    return node;
}

}  // namespace

int main() {
    using namespace cyxwiz;

    std::cout << "shared words\n";
    Check(FormatCountdown(4352) == "1:12:32", "countdown with hours: " + FormatCountdown(4352));
    Check(FormatCountdown(252) == "4:12" && FormatCountdown(-5) == "0:00", "countdown under an hour, never negative");
    Check(FormatReservationLength(120) == "2 h 00 min" && FormatReservationLength(45) == "45 min", "lengths");
    Check(FormatDollars(0.5) == "$0.50" && FormatDollars(12.4) == "$12.40" && FormatDollars(0.004) == "$0.004",
          "dollars keep small amounts visible");
    Check(FormatClockTime(0).size() == 5, "clock time is HH:MM");

    std::cout << "reserve quote\n";
    ReserveQuoteInputs reserve;
    reserve.node = Dell();
    reserve.duration_minutes = 120;
    reserve.has_account = true;
    const auto quote = BuildReserveQuote(reserve);
    Check(quote.title == "dell-pc", "the node's name is the title");
    Check(quote.subtitle == "OpenCL · 7.8 GB · reputation 92% · 14 jobs", "subtitle: " + quote.subtitle);
    Check(quote.price == "$0.25 / hour (fee included)", "the listed price includes the fee: " + quote.price);
    Check(quote.hold == "$0.50", "two hours are held: " + quote.hold);
    Check(quote.button == "Reserve for 2 h 00 min" && quote.enabled, "button names the length");
    Check(quote.pay_rule.find("reserved time") != std::string::npos, "pay rule: the reserved time is yours");
    Check(Detail(quote.details, "Region") == "local", "details keep the region");

    auto offline = reserve;
    offline.node.online = false;
    const auto offline_quote = BuildReserveQuote(offline);
    Check(!offline_quote.enabled && offline_quote.disabled_reason == "This node is offline.", "offline has a reason");
    auto no_account = reserve;
    no_account.has_account = false;
    Check(!BuildReserveQuote(no_account).enabled &&
              BuildReserveQuote(no_account).disabled_reason == "Sign in to reserve a node.",
          "not signed in has a reason");
    auto busy = reserve;
    busy.reserving = true;
    Check(BuildReserveQuote(busy).button == "Reserving..." && !BuildReserveQuote(busy).enabled, "reserving is shown");
    auto free = reserve;
    free.node.free_tier = true;
    Check(BuildReserveQuote(free).hold == "$0.00", "a free node holds nothing");

    std::cout << "active card\n";
    const long long start = 1790745000;  // reserved for two hours
    ActiveReservationInputs active;
    active.node = Dell();
    active.node_known = true;
    active.endpoint = "192.168.1.222:50052";
    active.start_time = start;
    active.end_time = start + 7200;
    active.now = start + 47 * 60;
    active.p2p_connected = true;
    active.training_running = true;
    active.epoch = 6;
    active.total_epochs = 10;
    active.reservation_id = "4f1c0000-a92e";
    active.node_id = "8b27-03d1";
    active.epochs_setting = 10;
    active.batch_setting = 32;
    active.access_until = start + 7200 + 300;

    const auto local_card = BuildActiveReservationCard(active);
    Check(local_card.time_left == "1:13:00" && local_card.urgency == ReservationUrgency::Normal,
          "time left from the end time before a heartbeat: " + local_card.time_left);
    Check(local_card.source.find("this computer") != std::string::npos, "says where the time comes from");
    Check(local_card.has_bar && local_card.fill > 0.39 && local_card.fill < 0.40, "bar: 47 of 120 minutes used");
    Check(local_card.price == "$0.50 (2 h 00 min)", "price of the reserved time: " + local_card.price);
    Check(!local_card.stale, "a local clock is not stale");
    Check(local_card.title == "dell-pc" && local_card.connection == "Connected to node", "name, not the endpoint");
    Check(!local_card.warn, "no warning with over an hour left");
    Check(Detail(local_card.details, "Endpoint") == "192.168.1.222:50052" &&
              Detail(local_card.details, "Epochs / batch") == "10 / 32",
          "details keep the endpoint and settings");

    auto heartbeat = active;
    heartbeat.server_seconds_left = 4372;  // the server knows a little more time (its clock)
    heartbeat.server_checked_at = active.now - 12;
    const auto server_card = BuildActiveReservationCard(heartbeat);
    Check(server_card.seconds_left == 4360 && server_card.time_left == "1:12:40",
          "the heartbeat wins, aged by the time since it came: " + server_card.time_left);
    Check(server_card.source == "From the central server, checked 12 s ago", server_card.source);

    auto soon = active;
    soon.now = active.end_time - 252;
    const auto soon_card = BuildActiveReservationCard(soon);
    Check(soon_card.urgency == ReservationUrgency::Urgent && soon_card.time_left == "4:12", "under 5 minutes is urgent");
    Check(soon_card.warn && soon_card.warning.rfind("Ends in 5 minutes. Training still running (epoch 6 of 10).", 0) == 0,
          "warning says what happens: " + soon_card.warning);
    auto idle = soon;
    idle.training_running = false;
    Check(BuildActiveReservationCard(idle).warning == "Ends in 5 minutes. Extend to keep the node.", "idle warning");
    auto nine = active;
    nine.now = active.end_time - 540;
    Check(BuildActiveReservationCard(nine).urgency == ReservationUrgency::Soon, "under 10 minutes is soon");
    auto over = active;
    over.now = active.end_time + 3;
    Check(BuildActiveReservationCard(over).urgency == ReservationUrgency::Ended &&
              BuildActiveReservationCard(over).time_left == "0:00",
          "past the end is ended");

    auto reconnected = active;
    reconnected.node_known = false;
    reconnected.start_time = 0;  // the reconnect token gives time left only
    const auto re_card = BuildActiveReservationCard(reconnected);
    Check(!re_card.has_bar && re_card.price == "-", "no invented start after a reconnect");
    Check(re_card.subtitle == "192.168.1.222:50052", "unknown node shows its endpoint");

    std::cout << "extend and leave\n";
    const auto options = BuildExtendOptions(0.25);
    Check(options.size() == 4 && options[1].label == "+30 min" && options[0].cost == "$0.06" &&
              options[3].minutes == 120 && options[3].cost == "$0.50",
          "extend options with their cost");
    Check(BuildExtendOptions(0.0)[0].cost == "-", "no price, no cost");
    const auto leave = BuildLeaveSummary(active);
    Check(leave.title == "Leave dell-pc? The clock keeps running", leave.title);
    Check(leave.ends.rfind("Your reservation ends at ", 0) == 0 &&
              leave.ends.find("(1 h 13 min left) whether you are connected or not. Leaving does not pause it.") !=
                  std::string::npos,
          "the end time comes first and says the clock runs: " + leave.ends);
    Check(leave.body.rfind("Training is running (epoch 6 of 10). Leaving stops it with a checkpoint;", 0) == 0,
          "leave body: " + leave.body);
    Check(leave.body.find("Reconnect") != std::string::npos, "says how to come back");
    Check(leave.button == "Leave, keep the clock running", leave.button);
    Check(BuildLeaveSummary(reconnected).title == "Leave the node? The clock keeps running",
          "unknown node after a reconnect");

    std::cout << "stale heartbeat\n";
    auto stale = active;
    stale.server_seconds_left = 3000;
    stale.server_checked_at = stale.now - 400;
    const auto stale_card = BuildActiveReservationCard(stale);
    Check(stale_card.stale && stale_card.source.find("has not answered for 400 s") != std::string::npos,
          "says the central server stopped answering: " + stale_card.source);
    Check(stale_card.time_left == "43:20", "still counts down from the last reply: " + stale_card.time_left);

    std::cout << "receipt\n";
    ReservationEndFacts ended;
    ended.node_name = "dell-pc";
    ended.ended_at = start + 7200;
    ended.reason = ReservationEndReason::TimeRanOut;
    ended.seconds_used = 7200;
    ended.jobs_started = 2;
    ended.reservation_id = "4f1c0000-a92e";
    const auto receipt = BuildReservationReceipt(ended);
    Check(receipt.title.rfind("The reservation on dell-pc ended at ", 0) == 0 && receipt.why == "time ran out",
          receipt.title);
    Check(receipt.time_used == "2 h 00 min" && receipt.jobs == "2 jobs started", "receipt facts");
    Check(receipt.paid.find("not live") != std::string::npos, "no invented charge");
    Check(receipt.note.find("checkpoint") != std::string::npos, "points to the resume");
    auto lost = ended;
    lost.reason = ReservationEndReason::Lost;
    lost.error = "Reservation not found";
    const auto lost_receipt = BuildReservationReceipt(lost);
    Check(lost_receipt.failed && lost_receipt.title == "The reservation on dell-pc is over" &&
              lost_receipt.why == "the central server no longer knows it" &&
              lost_receipt.note.rfind("Reservation not found", 0) == 0,
          "a lost reservation says why");

    std::cout << "reconnect\n";
    ActiveReservationListing dell{"r1", "8b27aaaa-03d1", "dell-pc", start + 2285, false, 0};
    ActiveReservationListing mac{"r2", "5c01bbbb-7700", "", start + 6690, true, 1};
    ActiveReservationListing gone{"r3", "aaaa0000-0000", "old-pc", start - 5, false, 0};
    const auto rows = BuildReconnectRows({dell, mac, gone}, start);
    Check(rows.size() == 2 && rows[0].reservation_id == "r2", "longest time left first; the ended one is left out");
    Check(rows[0].node == "Node 5c01bbbb" && rows[0].time_left.rfind("ends at ", 0) == 0 &&
              rows[0].time_left.find("(1 h 52 min left)") != std::string::npos,
          "unknown node by id; row says when it ends: " + rows[0].time_left);
    Check(rows[0].note == "open in another Engine · 1 job done", rows[0].note);
    Check(rows[1].node == "dell-pc" && rows[1].note.empty(), "known node by name");

    if (failures) {
        std::cout << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Reservation presentation passed\n";
    return EXIT_SUCCESS;
}
