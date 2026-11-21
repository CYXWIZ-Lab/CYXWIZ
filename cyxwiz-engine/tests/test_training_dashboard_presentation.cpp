// The Training Dashboard's words (TOFIX129 B4, approved mockup 2026-11-18),
// checked on a real run: the Handwritten Digits starter, 10 epochs on the CPU
// (run log 2026-11-18). The cards show the finished epoch, not the last batch
// window (the dashboard read 0.2322 while epoch 10 ended at 0.2223).
#include "core/training_dashboard_presentation.h"
#include "core/training_progress_estimate.h"

#include <cmath>
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

    // The header (B4-4): the finished run in plain words, the codes under Run details.
    td::HeaderFacts h;
    h.terminal_status = "completed";
    h.terminal_reason = "completed_all_epochs";
    h.route = td::Route{"arrayfire_cpu", "arrayfire_cpu", 0, "", "compatibility", false};
    h.dataset = "digits_8x8";
    h.train_rows = 1433;
    h.validation_rows = 176;
    h.test_rows = 188;
    h.samples_per_batch = 128;
    h.metric_interval = 10;
    h.active_model_provenance = "restored_best_checkpoint";
    h.checkpoint_path = "D:\\tmp\\gui129\\nbproj\\checkpoints\\train-1794985565683\\best";
    h.checkpoint_epoch = 10;
    h.checkpoint_step = 12;
    h.has_checkpoint_validation = true;
    h.checkpoint_val_loss = 0.2554;
    h.checkpoint_val_accuracy = 0.9261;
    h.has_test = true;
    h.test_loss = 0.2220;
    h.test_accuracy = 0.9415;
    td::Header head = td::BuildHeader(done, h);
    Check(head.state == td::RunState::Completed && head.pill == "Completed", "completed");
    Check(head.status == "All 10 epochs in 2.2s", "status: " + head.status);
    Check(head.show_progress && head.progress == 1.0, "a full bar");
    Check(head.where == "Ran on the CPU (ArrayFire CPU) \xC2\xB7 compatibility policy", "where: " + head.where);
    Check(head.data == "digits_8x8 \xC2\xB7 1,433 training, 176 validation, 188 test rows \xC2\xB7 batches of 128",
          "data: " + head.data);
    Check(head.outcome == "Finished: every epoch ran." && head.outcome_tone == td::Tone::Good, "outcome");
    Check(head.active_model == "Active model: the best validation checkpoint, epoch 10" &&
              head.active_model_detail == "(validation loss 0.2554, accuracy 92.61%)",
          "active model: " + head.active_model + " " + head.active_model_detail);
    Check(head.test == "Held-out test: accuracy 94.15%, loss 0.2220" &&
              head.test_detail == "(188 rows the model never saw)",
          "held-out test: " + head.test);
    Check(head.now.empty(), "no latest batches after the run");
    const auto row = [](const td::Header& hd, const std::string& key) {
        for (const auto& r : hd.run_rows)
            if (r.key == key) return r.value;
        return std::string();
    };
    Check(row(head, "Stop code") == "completed_all_epochs", "the stop code under Run details");
    Check(row(head, "Checkpoint") == "epoch 10, step 12", "checkpoint epoch and step");
    Check(row(head, "Path") == h.checkpoint_path, "checkpoint path");
    Check(row(head, "Batches") == "128 samples each \xC2\xB7 metrics every 10 batches", "batches: " + row(head, "Batches"));
    Check(row(head, "Last batch loss").empty(), "no last batch loss after the run");

    // Training: epoch 6, batch 10 of 12.
    td::HeaderFacts live = h;
    live.terminal_status.clear();
    live.terminal_reason.clear();
    live.active_model_provenance.clear();
    live.checkpoint_path.clear();
    live.checkpoint_epoch = 0;
    live.checkpoint_step = 0;
    live.has_test = false;
    live.last_batch_loss = 0.5198f;
    live.tasks = {{"Training Model (Arrow)", 0.55f, ""}, {"Reading digits_8x8", 1.0f, "done"},
                  {"Cache", 0.1f, "writing"}, {"Fourth", 0.0f, ""}};
    running.batches_per_second = 204.3;
    head = td::BuildHeader(running, live);
    Check(head.pill == "Training" && head.status == "Epoch 6 of 10 \xC2\xB7 batch 10 of 12", "training status: " + head.status);
    Check(head.where == "Runs on the CPU (ArrayFire CPU) \xC2\xB7 compatibility policy", "runs on: " + head.where);
    Check(head.now == "Latest batches: loss 0.5287 \xC2\xB7 accuracy 90.70%" &&
              head.now_detail == "(204 batches/s)",
          "latest batches: " + head.now + " " + head.now_detail);
    Check(std::abs(head.progress - (5.0 + 10.0 / 12.0) / 10.0) < 1e-9, "progress counts the running epoch's batches");
    Check(head.outcome.empty() && head.active_model.empty() && head.test.empty(), "no outcome while training");
    Check(row(head, "Last batch loss") == "0.5198", "last batch loss while training");
    int task_rows = 0;
    for (const auto& r : head.run_rows) task_rows += r.key == "Engine task";
    Check(task_rows == 3 && row(head, "Engine tasks") == "and 1 more" &&
              row(head, "Engine task") == "Training Model (Arrow) \xC2\xB7 55% \xC2\xB7 running",
          "three engine tasks and the rest counted");
    live.paused = true;
    Check(td::BuildHeader(running, live).pill == "Paused", "paused");
    live.paused = false;
    live.waiting_for_gpu = "Another run holds the GPU (Cats vs dogs)";
    head = td::BuildHeader(running, live);
    Check(head.state == td::RunState::WaitingForGpu && head.status == "Starts when the other run ends" &&
              head.where.empty(),
          "waiting for the GPU: no route yet");

    // Early stopped: 7 of 10 epochs, the eighth running when it stopped.
    td::RunFacts early = Run(7, 0);
    early.total_seconds = 1.52;
    early.last_executed_epoch = 7;
    early.current_epoch = 8;
    td::HeaderFacts e = h;
    e.terminal_status = "early_stopped";
    e.terminal_reason = "validation_loss_plateau_patience_3";
    head = td::BuildHeader(early, e);
    Check(head.pill == "Early stopped" && head.status == "7 of 10 epochs in 1.5s, stopped during epoch 8",
          "early stop status: " + head.status);
    Check(head.outcome == "Stopped early: the validation loss did not improve for 3 epochs." &&
              head.outcome_tone == td::Tone::Bad,
          "early stop outcome: " + head.outcome);
    Check(std::abs(head.progress - 0.7) < 1e-9, "the bar shows the finished epochs");
    e.terminal_status = "cancelled";
    e.terminal_reason = "user_cancelled";
    head = td::BuildHeader(early, e);
    Check(head.pill == "Stopped" && head.outcome == "Stopped: you stopped the run before its last epoch.", "stopped by the user");
    e.terminal_status = "failed";
    e.terminal_reason = "Out of memory while building batch 3";
    head = td::BuildHeader(early, e);
    Check(head.pill == "Failed" && head.outcome == "Failed: Out of memory while building batch 3" &&
              head.outcome_tone == td::Tone::Error && row(head, "Stop code") == "failed",
          "a failure's sentence on the surface, its status as the code");
    Check(!head.missing_values, "no fix card for another failure");
    // Refused for missing values: a plain sentence, the code missing_values
    // and the fix card (TOFIX223 0.3); NaN found while reading a file says so.
    e.terminal_reason = "[CW-T-0501] Training failed. Detail: ArrowDatasetBatcher: the data has missing values ...";
    cyxwiz::MissingColumn x1;
    x1.name = "x1";
    x1.rows = 2;
    e.missing_columns = {x1};
    head = td::BuildHeader(early, e);
    Check(head.outcome == "Failed: the data has missing values (empty cells or NaN), which training does not use." &&
              head.missing_values && row(head, "Stop code") == "missing_values",
          "missing values: " + head.outcome);
    e.missing_columns.front().at_least = true;
    head = td::BuildHeader(early, e);
    Check(head.outcome == "Failed: the data has missing values the check before training could not see (NaN in a "
                          "file on disk).",
          "NaN read from disk: " + head.outcome);
    e.missing_columns.clear();

    // A regression run: no accuracy anywhere.
    td::HeaderFacts reg = h;
    reg.classification = false;
    reg.test_mae = 0.3125;
    head = td::BuildHeader(done, reg);
    Check(head.test == "Held-out test: loss 0.2220, mean absolute error 0.3125" &&
              head.active_model_detail == "(validation loss 0.2554)",
          "regression: " + head.test + " " + head.active_model_detail);
    // The final state of the run, not a checkpoint.
    reg.active_model_provenance = "run_final_state";
    reg.checkpoint_path.clear();
    reg.checkpoint_epoch = 0;
    head = td::BuildHeader(done, reg);
    Check(head.active_model == "Active model: the final state of the run, epoch 10" && head.active_model_detail.empty(),
          "final state: " + head.active_model);

    // A GPU that failed its check: the run fell back to the CPU.
    td::HeaderFacts fb = h;
    fb.route = td::Route{"arrayfire_cuda", "arrayfire_cpu", 0, "", "compatibility", true};
    head = td::BuildHeader(done, fb);
    Check(head.where == "Ran on the CPU (ArrayFire CPU) (the chosen CUDA device failed its check) \xC2\xB7 compatibility policy",
          "fallback: " + head.where);
    fb.route = td::Route{"arrayfire_cuda", "arrayfire_cuda", 0, "GeForce GTX 1050 Ti", "strict", false};
    Check(td::BuildHeader(done, fb).where == "Ran on CUDA (GeForce GTX 1050 Ti) \xC2\xB7 strict policy", "a GPU run");

    // Nothing yet; a checkpoint loaded for testing.
    td::RunFacts none;
    td::HeaderFacts idle;
    head = td::BuildHeader(none, idle);
    Check(head.pill == "Idle" && head.status == "No training run yet" && !head.show_progress, "idle");
    idle.checkpoint_loaded = true;
    idle.checkpoint_epoch = 4;
    idle.checkpoint_path = "best";
    idle.has_checkpoint_validation = true;
    idle.checkpoint_val_loss = 0.5;
    idle.checkpoint_val_accuracy = 0.8;
    head = td::BuildHeader(none, idle);
    Check(head.pill == "Model loaded" && head.active_model == "Active model: a checkpoint loaded for testing, epoch 4" &&
              head.active_model_detail == "(validation loss 0.5000, accuracy 80.00%)",
          "model loaded: " + head.active_model);
    // A checkpoint loaded after a finished run: its model, not the run's ending or test.
    td::HeaderFacts loaded = h;
    loaded.checkpoint_loaded = true;
    loaded.active_model_provenance = "loaded_checkpoint_for_testing";
    loaded.checkpoint_epoch = 4;
    loaded.checkpoint_step = 0;
    head = td::BuildHeader(done, loaded);
    Check(head.pill == "Model loaded" && head.outcome.empty() && head.test.empty() &&
              head.active_model == "Active model: a checkpoint loaded for testing, epoch 4" &&
              row(head, "Stop code").empty() && row(head, "Checkpoint") == "epoch 4",
          "a checkpoint loaded after a run: " + head.pill + " / " + head.test);
    td::HeaderFacts prep;
    prep.preparing = true;
    prep.preparation_progress = 0.4;
    head = td::BuildHeader(none, prep);
    Check(head.pill == "Preparing" && head.show_progress && head.progress == 0.4 &&
              head.progress_text == "Preparing training...",
          "preparing");
    prep.preparing = false;
    prep.preparation_failed = true;
    prep.preparation_error = "The label column 'target' is missing.";
    head = td::BuildHeader(none, prep);
    Check(head.pill == "Preparation failed" && head.outcome == prep.preparation_error && head.outcome_tone == td::Tone::Error,
          "preparation failed");

    // The Runs card: the digits run, then a stopped GPU run with no test rows.
    cyxwiz::TrainingRunComparisonRecord digits;
    digits.run_id = "training-task-3";
    digits.run_status = "completed";
    digits.dataset_name = "digits_8x8";
    digits.finished_at = "11:06";
    digits.epochs = 10;
    digits.epochs_run = 10;
    digits.elapsed_seconds = 2.2f;
    digits.has_validation_metrics = true;
    digits.best_val_loss = 0.2554f;
    digits.best_val_loss_epoch = 10;
    digits.best_val_accuracy = 0.9261f;
    digits.best_val_accuracy_epoch = 10;
    digits.has_test_metrics = true;
    digits.final_test_accuracy = 0.9415f;
    digits.final_test_loss = 0.2220f;
    digits.test_sample_count = 188;
    digits.train_sample_count = 1433;
    digits.val_sample_count = 176;
    digits.train_ratio = 0.8f;
    digits.val_ratio = 0.1f;
    digits.test_ratio = 0.1f;
    digits.learning_rate = 0.001f;
    digits.partition_manifest_fingerprint = "a1b2c3d4e5f6";
    digits.device_backend = "arrayfire_cpu";
    cyxwiz::TrainingRunComparisonRecord gpu = digits;
    gpu.run_id = "training-task-4";
    gpu.run_status = "cancelled";
    gpu.finished_at = "11:20";
    gpu.epochs_run = 7;
    gpu.has_test_metrics = false;
    gpu.partition_manifest_fingerprint = "ffff0000";
    gpu.device_backend = "arrayfire_cuda";
    gpu.device_name = "GeForce GTX 1050 Ti";
    const td::RunsTable runs = td::BuildRunsTable({digits, gpu}, false);
    const auto cell = [&runs](size_t row, const std::string& column) -> const td::RunCell& {
        for (size_t i = 0; i < runs.columns.size(); ++i)
            if (runs.columns[i] == column) return runs.rows[row][i];
        static const td::RunCell none{"(no column)", {}, td::Tone::Plain};
        return none;
    };
    Check(runs.columns.size() == 28 && runs.rows.size() == 2 && runs.rows[0].size() == 28 &&
              runs.columns[td::RunsTable::kKeyColumns - 1] == "Device",
          "runs table shape");
    Check(cell(0, "Run").text == "11:06 \xC2\xB7 digits_8x8" && cell(0, "Result").text == "Completed" &&
              cell(0, "Result").tone == td::Tone::Good && cell(0, "Best val loss").text == "0.2554" &&
              cell(0, "Best val acc").text == "92.61%" && cell(0, "Test acc").text == "94.15%" &&
              cell(0, "Test acc").tooltip == "Loss 0.2220 on 188 test rows" && cell(0, "Epochs").text == "10" &&
              cell(0, "Time").text == "2.2s" && cell(0, "Device").text == "CPU",
          "runs table digits row: " + cell(0, "Run").text + " / " + cell(0, "Test acc").tooltip);
    Check(cell(1, "Result").text == "Stopped" && cell(1, "Result").tone == td::Tone::Bad &&
              cell(1, "Test acc").tone == td::Tone::Faint && cell(1, "Epochs").text == "7 of 10" &&
              cell(1, "Device").text == "CUDA" &&
              cell(1, "Device").tooltip == "GeForce GTX 1050 Ti\nBackend: arrayfire_cuda" &&
              cell(1, "Partition match").text == "different" && cell(0, "Partition match").text == "same",
          "runs table stopped GPU row: " + cell(1, "Device").tooltip);
    Check(cell(0, "Rows").text == "1,433 / 176 / 188" && cell(0, "Split").text == "80/10/10%" &&
              cell(0, "Learning rate").text == "0.001" && cell(0, "Partition").text == "a1b2c3d4",
          "runs table other columns: " + cell(0, "Split").text);
    Check(runs.key_note.rfind("8 of 28 columns", 0) == 0 && runs.loaded_note.empty(), "runs table note");
    cyxwiz::TrainingRunComparisonRecord failed = digits;
    failed.run_status = "failed";
    failed.device_backend.clear();
    const td::RunsTable more = td::BuildRunsTable({failed}, true);
    Check(more.rows[0][7].text == "\xE2\x80\x94" && more.rows[0][7].tone == td::Tone::Faint &&
              more.rows[0][1].tone == td::Tone::Error && !more.loaded_note.empty(),
          "runs table: no device recorded, failed, loaded checkpoint");
    cyxwiz::TrainingRunComparisonRecord fallback = digits;
    fallback.device_fallback = true;
    Check(td::BuildRunsTable({fallback}, false).rows[0][7].text == "CPU (fallback)", "runs table fallback device");
    Check(td::BuildRunsTable({}, false).rows.empty(), "runs table empty");
    // Stopped during epoch 1: none of its epochs ran; early stopped; no
    // validation rows; no dataset name.
    cyxwiz::TrainingRunComparisonRecord stopped_first = digits;
    stopped_first.run_status = "cancelled";
    stopped_first.epochs_run = 0;
    stopped_first.has_validation_metrics = false;
    stopped_first.dataset_name.clear();
    cyxwiz::TrainingRunComparisonRecord early_run = digits;
    early_run.run_status = "early_stopped";
    early_run.epochs_run = 6;
    const td::RunsTable edge = td::BuildRunsTable({stopped_first, early_run}, false);
    Check(edge.rows[0][5].text == "0 of 10" && edge.rows[0][2].tone == td::Tone::Faint &&
              edge.rows[0][3].text == "\xE2\x80\x94" && edge.rows[0][0].text == "11:06 \xC2\xB7 training-task-3",
          "runs table: stopped in epoch 1, no validation, no dataset: " + edge.rows[0][5].text + " / " +
              edge.rows[0][0].text);
    Check(edge.rows[1][1].text == "Early stopped" && edge.rows[1][1].tone == td::Tone::Bad &&
              edge.rows[1][5].text == "6 of 10",
          "runs table: early stopped");
    cyxwiz::TrainingRunComparisonRecord unreported = digits;
    unreported.epochs_run = 0;  // completed, epochs not reported: all ran
    Check(td::BuildRunsTable({unreported}, false).rows[0][5].text == "10", "runs table: completed without epochs");

    // The metrics CSV: batch points and epoch ends of the digits run's first
    // two epochs; validation only at the epoch ends, each value on its own row.
    td::CsvColumn train_loss_column{"TrainLoss", {{0.0833333, 0.833333, 1.0, 1.833333, 2.0}, {2.3896, 2.2154, 2.0632, 1.9009, 1.6912}}};
    td::CsvColumn val_loss_column{"ValLoss", {{1.0, 2.0}, {2.0328, 1.6772}}};
    td::CsvColumn epoch_loss{"TrainLossEpoch", {{1.0, 2.0}, {2.21, 1.79}}};
    td::CsvColumn named{"F1, macro", {{2.0}, {0.5}}};
    const std::string csv = td::MetricsCsv({train_loss_column, val_loss_column, epoch_loss, named});
    Check(csv == "Epoch,TrainLoss,ValLoss,TrainLossEpoch,\"F1, macro\"\n"
                 "0.0833333,2.3896,,,\n"
                 "0.833333,2.2154,,,\n"
                 "1,2.0632,2.0328,2.21,\n"
                 "1.833333,1.9009,,,\n"
                 "2,1.6912,1.6772,1.79,0.5\n",
          "metrics CSV rows:\n" + csv);

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
