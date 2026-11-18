// The Training Dashboard's words (TOFIX129 B4, approved mockup 2026-11-18),
// checked on a real run: the Handwritten Digits starter, 10 epochs on the CPU
// (run log 2026-11-18). The cards show the finished epoch, not the last batch
// window (the dashboard read 0.2322 while epoch 10 ended at 0.2223).
#include "core/training_dashboard_presentation.h"
#include "core/training_progress_estimate.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

namespace td = cyxwiz::training_dashboard;

const double kTrainLoss[] = {2.2147, 1.8785, 1.4826, 1.0660, 0.7337, 0.5238, 0.3965, 0.3107, 0.2599, 0.2223};
const double kValLoss[] = {2.0328, 1.6772, 1.2632, 0.8795, 0.6247, 0.4709, 0.3864, 0.3223, 0.2886, 0.2554};
const double kTrainAcc[] = {22.33, 59.39, 72.37, 83.88, 87.16, 90.51, 92.18, 93.44, 94.42, 95.25};
const double kValAcc[] = {46.02, 63.64, 78.98, 88.07, 89.77, 90.34, 88.64, 90.91, 90.91, 92.61};
// Batch windows 1 and 10 of each epoch (of 12), as the dashboard records them.
const double kBatchLoss[10][2] = {{2.3896, 2.2329}, {2.0534, 1.9009}, {1.6215, 1.5112}, {1.2777, 1.0888},
                                  {0.9463, 0.7439}, {0.5229, 0.5287}, {0.4029, 0.4040}, {0.3667, 0.3085},
                                  {0.2560, 0.2597}, {0.2103, 0.2211}};
const double kBatchAcc[10][2] = {{7.03, 19.61}, {48.44, 58.28}, {71.09, 71.17}, {78.12, 83.28}, {83.59, 87.19},
                                 {92.19, 90.70}, {91.41, 91.88}, {91.41, 93.75}, {94.53, 94.53}, {97.66, 95.23}};

// The run after `done` finished epochs; `batch_into_next` > 0: that far into the next epoch.
td::RunFacts Run(int done, int batch_into_next) {
    td::RunFacts f;
    td::Series loss_points, acc_points;
    for (int e = 0; e < 10; ++e) {
        const int epoch = e + 1;
        for (int k = 0; k < 2; ++k) {
            const int batch = k == 0 ? 1 : 10;
            if (epoch > done && !(epoch == done + 1 && batch <= batch_into_next)) continue;
            const double x = e + batch / 12.0;
            loss_points.x.push_back(x);
            loss_points.y.push_back(kBatchLoss[e][k]);
            acc_points.x.push_back(x);
            acc_points.y.push_back(kBatchAcc[e][k]);
        }
        if (epoch > done) continue;
        // The epoch's end is a point too, then the whole-epoch values.
        loss_points.x.push_back(epoch);
        loss_points.y.push_back(kTrainLoss[e]);
        acc_points.x.push_back(epoch);
        acc_points.y.push_back(kTrainAcc[e]);
        f.train_loss_epochs.x.push_back(epoch);
        f.train_loss_epochs.y.push_back(kTrainLoss[e]);
        f.train_accuracy_epochs.x.push_back(epoch);
        f.train_accuracy_epochs.y.push_back(kTrainAcc[e]);
        f.val_loss.x.push_back(epoch);
        f.val_loss.y.push_back(kValLoss[e]);
        f.val_accuracy.x.push_back(epoch);
        f.val_accuracy.y.push_back(kValAcc[e]);
    }
    if (!loss_points.Empty()) f.latest_loss_point = td::Point{loss_points.x.back(), loss_points.y.back()};
    if (!acc_points.Empty()) f.latest_accuracy_point = td::Point{acc_points.x.back(), acc_points.y.back()};
    f.total_epochs = 10;
    f.total_batches = 12;
    return f;
}

const td::Card* Find(const std::vector<td::Card>& cards, const std::string& id) {
    for (const auto& c : cards)
        if (c.id == id) return &c;
    return nullptr;
}

}  // namespace

int main() {
    // Completed: the epoch-10 values, not the last batch window.
    td::RunFacts done = Run(10, 0);
    // The run's last batch window (0.2322 on the old card) after the epoch end.
    done.latest_loss_point = td::Point{10.0, 0.2322};
    done.training = false;
    done.last_executed_epoch = 10;
    done.total_seconds = 2.17;
    done.avg_epoch_seconds = 0.217;
    auto cards = td::BuildCards(done);
    const td::Card* loss = Find(cards, "##kpi_loss");
    Check(loss && loss->value == "0.2223", "the Loss card shows epoch 10 (0.2223), not the batch window");
    Check(loss && loss->label == "Loss \xC2\xB7 epoch 10", "the card names its epoch");
    Check(loss && loss->change == "0.0376 since epoch 9" && loss->direction < 0 && loss->change_tone == td::Tone::Good,
          "loss fell 0.0376 since epoch 9");
    Check(loss && loss->line2 == "best 0.2223 at epoch 10", "best training loss");
    Check(loss && loss->line3.empty(), "no 'so far' line after the run");
    const td::Card* acc = Find(cards, "##kpi_acc");
    Check(acc && acc->value == "95.25%" && acc->change == "0.83 points since epoch 9", "accuracy in points");
    const td::Card* val = Find(cards, "##kpi_val_loss");
    Check(val && val->value == "0.2554" && val->change == "0.0332 since epoch 9", "validation loss and its change");
    Check(val && val->line2 == "0.0331 above the training loss: a small gap" && val->line2_tone == td::Tone::Good,
          "the gap against epoch 10's training loss (the old card said +0.0232): " + (val ? val->line2 : ""));
    const td::Card* vacc = Find(cards, "##kpi_val_acc");
    Check(vacc && vacc->value == "92.61%" && vacc->change == "1.70 points since epoch 9" &&
              vacc->line2 == "best 92.61% at epoch 10",
          "validation accuracy");
    const td::Card* time = Find(cards, "##kpi_time");
    Check(time && time->label == "Total time" && time->value == "2.2s" && time->line2 == "10 of 10 epochs, 0.22s each",
          "total time keeps a tenth: " + (time ? time->value + " / " + time->line2 : ""));
    Check(cards.size() == 5, "five cards");

    // Training: epoch 6, batch 10 of 12.
    td::RunFacts running = Run(5, 10);
    running.training = true;
    running.current_epoch = 6;
    running.current_batch = 10;
    running.samples_per_second = 23145.0;
    running.remaining_seconds = 0.9;
    cards = td::BuildCards(running);
    loss = Find(cards, "##kpi_loss");
    Check(loss && loss->value == "0.7337" && loss->label == "Loss \xC2\xB7 epoch 5", "running: epoch 5's loss");
    Check(loss && loss->line3 == "latest batches 0.5287", "running: the latest batches as a second line");
    acc = Find(cards, "##kpi_acc");
    Check(acc && acc->line3 == "latest batches 90.70%" && acc->change == "3.28 points since epoch 4", "running accuracy");
    val = Find(cards, "##kpi_val_loss");
    Check(val && val->line2 == "0.1090 below the training loss", "running: validation below training");
    time = Find(cards, "##kpi_time");
    Check(time && time->label == "Time left" && time->value == "0.9s" && time->line2 == "4 epochs and 2 batches to go" &&
              time->line3 == "23,145 samples/s",
          "time left: " + (time ? time->value + " / " + time->line2 + " / " + time->line3 : ""));

    // The first epoch, before it ends.
    td::RunFacts first = Run(0, 10);
    first.training = true;
    cards = td::BuildCards(first);
    loss = Find(cards, "##kpi_loss");
    Check(loss && loss->label == "Loss \xC2\xB7 epoch 1, latest batches" && loss->value == "2.2329", "the first epoch");
    first.training = false;  // stopped before the first epoch ended
    cards = td::BuildCards(first);
    loss = Find(cards, "##kpi_loss");
    Check(loss && loss->label == "Loss \xC2\xB7 stopped in epoch 1" && loss->change == "no epoch finished",
          "a run stopped in epoch 1 does not say it is running");
    Check(!Find(cards, "##kpi_val_loss"), "no validation card before the first epoch ends");

    // How the run is going.
    td::Insights ins = td::BuildInsights(done);
    Check(ins.suggestion == "Suggested: train longer", "finished, still improving: train longer");
    Check(ins.rows.size() == 4 && ins.rows[0].value == "fell every epoch (2.21 to 0.222)", "training loss in words");
    Check(ins.rows.size() == 4 && ins.rows[1].value.find("0.0331 above the training loss: a small gap") != std::string::npos,
          "validation loss with the gap");
    Check(ins.rows.size() == 4 && ins.rows[2].value == "epoch 10: loss 0.2554, accuracy 92.61%", "best validation");
    ins = td::BuildInsights(running);
    Check(ins.suggestion == "Suggested: keep going" && ins.rows[0].value == "falling every epoch (2.21 to 0.734)",
          "running: keep going");
    // A rising validation loss after its best epoch.
    td::RunFacts over = Run(10, 0);
    over.val_loss.y = {1.0, 0.8, 0.6, 0.5, 0.45, 0.50, 0.58, 0.70, 0.85, 1.0};
    ins = td::BuildInsights(over);
    bool rising = false;
    for (const auto& r : ins.rows) rising |= r.label == "Rising" && r.value == "validation loss has risen since epoch 6";
    Check(rising, "a validation rise is named with its epoch");
    Check(ins.suggestion_tone == td::Tone::Bad, "and the suggestion warns");

    Check(td::Thousands(23145) == "23,145" && td::Thousands(999) == "999" && td::Thousands(1234567) == "1,234,567",
          "thousands");
    Check(td::PerEpoch(0.217) == "0.22s" && td::PerEpoch(3.4) == "3.4s" && td::PerEpoch(75) == "1m 15s", "per epoch");

    if (g_failures) {
        std::printf("%d training dashboard checks failed\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("Training dashboard presentation passed\n");
    return EXIT_SUCCESS;
}
