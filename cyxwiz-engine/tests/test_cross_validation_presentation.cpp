// Results > Cross-validation and Learning curve wording (TOFIX129 D4), on the
// approved mockup's penguins figures (5 stratified folds of a small classifier,
// species from bill length and body mass): the headline with its spread, the
// range and the single test beside it, the fold that stands out, the learning
// curve's plateau and gap, a regression's MAE, progress and the shares field.

#include "../src/core/cross_validation_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

using namespace cyxwiz;
using namespace cyxwiz::folds_view;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

bool Has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

CrossValidationOutcome PenguinsFolds() {
    CrossValidationOutcome o;
    o.ok = true;
    o.folds = 5;
    o.rows = 333;
    o.seconds = 161.0;
    const double accuracy[] = {0.9403, 0.9701, 0.9403, 0.9545, 0.9242};
    const double f1[] = {0.9369, 0.9698, 0.9442, 0.9488, 0.9244};
    const double loss[] = {0.1515, 0.1419, 0.1136, 0.1522, 0.2452};
    for (int k = 0; k < 5; ++k) {
        CrossValidationRun r;
        r.fold = k + 1;
        r.train_rows = k < 3 ? 266 : 267;
        r.val_rows = k < 3 ? 67 : 66;
        r.val.accuracy = accuracy[k];
        r.val.macro_f1 = f1[k];
        r.val.loss = loss[k];
        o.runs.push_back(r);
    }
    return o;
}

void CheckCrossValidation() {
    const CrossValidationView v = BuildCrossValidationView(PenguinsFolds(), 0.985);
    Check(v.headline == "Accuracy 94.6% \xC2\xB1 1.7 over 5 folds", "headline: " + v.headline);
    Check(v.pills.size() == 4 && v.pills[0] == "macro F1 0.945 \xC2\xB1 0.017" && v.pills[2] == "range 92.4% .. 97.0%" &&
              v.pills[3] == "2 min 41 s",
          "pills: " + (v.pills.empty() ? std::string() : v.pills[0] + " | " + v.pills.back()));
    Check(Has(v.sentence, "Each fold scores 92.4% to 97.0%, so expect about 95% on new rows.") &&
              Has(v.sentence, "The single test said 98.5%, better than every fold: that split was a lucky one"),
          "sentence: " + v.sentence);
    Check(v.rows.size() == 5 && v.rows[4].flagged && !v.rows[0].flagged && Has(v.note, "Fold 5's loss is high (0.245)"),
          "fold 5 stands out: " + v.note);
    Check(v.rows[0].scores[0] == "94.0%" && v.rows[0].train_rows == "266" && v.mean.scores[0] == "94.6%", "fold rows and mean");
    Check(Has(v.note, "and its accuracy low"), "fold 5's accuracy is low too: " + v.note);
    CrossValidationOutcome confident = PenguinsFolds();
    confident.runs[4].val.accuracy = 0.973;  // the highest accuracy, the highest loss (seen in the Engine)
    Check(Has(BuildCrossValidationView(confident).note, "though its accuracy is not low: a few mistakes made with great confidence"),
          "high loss, high accuracy: " + BuildCrossValidationView(confident).note);
    Check(v.bars.size() == 5 && v.percent && v.bars_label == "accuracy", "bars");
    Check(Has(BuildCrossValidationView(PenguinsFolds(), 0.95).sentence, "The single test's 95.0% is inside that range."), "a test inside the range");
    Check(!Has(BuildCrossValidationView(PenguinsFolds()).sentence, "single test"), "no test: not mentioned");
    const std::string csv = OutcomeCsv(PenguinsFolds());
    Check(csv.rfind("fold,share,train_rows,val_rows,val_accuracy,val_macro_f1,val_loss,epochs,seconds\n1,1,266,67,0.9403,", 0) == 0 &&
              Has(csv, "\n5,1,267,66,0.9242,0.9244,0.2452,") && Has(csv, "# rows used,333"),
          "csv: " + csv.substr(0, 140));
}

void CheckLearningCurve() {
    CrossValidationOutcome o;
    o.ok = true;
    const double shares[] = {0.1, 0.25, 0.5, 0.75, 1.0};
    const int rows[] = {27, 67, 133, 200, 266};
    const double train[] = {0.993, 0.943, 0.952, 0.955, 0.953};
    const double val[] = {0.922, 0.934, 0.940, 0.940, 0.943};
    for (int s = 0; s < 5; ++s)
        for (int k = 0; k < 5; ++k) {
            CrossValidationRun r;
            r.fold = k + 1;
            r.share = shares[s];
            r.train_rows = rows[s];
            r.train.accuracy = train[s];
            r.val.accuracy = val[s];
            o.runs.push_back(r);
        }
    const LearningCurveView v = BuildLearningCurveView(o, CurveScore::Accuracy);
    Check(v.points.size() == 5 && v.percent && v.higher_is_better, "five points of accuracy");
    Check(v.table[4][0] == "266" && v.table[4][1] == "95.3%" && v.table[4][2] == "94.3%" && v.table[4][3] == "1.0", "table at full size");
    Check(v.sentence == "Validation accuracy levels off at about 94.3% from 67 rows; more of the same rows will not help much.",
          "sentence: " + v.sentence);
    Check(Has(v.advice, "The gap is 1.0 points at full size, so the model is not memorising"), "advice: " + v.advice);

    // A dip at full size inside the folds' spread is not a fall (seen in the
    // Engine: 88.3%, 87.7% then 84.1% with a spread of about 5 points).
    CrossValidationOutcome dip = o;
    const double dip_val[] = {0.353, 0.702, 0.883, 0.877, 0.841};
    for (auto& r : dip.runs) {
        const int s = r.share == 0.1 ? 0 : r.share == 0.25 ? 1 : r.share == 0.5 ? 2 : r.share == 0.75 ? 3 : 4;
        r.val.accuracy = dip_val[s] + (r.fold - 3) * 0.025;  // sd about 4 points
    }
    const LearningCurveView d = BuildLearningCurveView(dip, CurveScore::Accuracy);
    Check(Has(d.sentence, "levels off at about 84.1% from 200 rows"), "dip within the spread: " + d.sentence);

    // Still rising, and memorising.
    for (auto& r : o.runs) {
        if (r.share == 1.0) r.val.accuracy = 0.80;
        if (r.share == 0.75) r.val.accuracy = 0.75;
        if (r.share == 1.0) r.train.accuracy = 0.99;
    }
    const LearningCurveView w = BuildLearningCurveView(o, CurveScore::Accuracy);
    Check(Has(w.sentence, "is still improving at 266 rows (75.0% to 80.0%): more rows would likely help"), "rising: " + w.sentence);
    Check(Has(w.advice, "Training scores 19.0 points better than validation at full size: the model memorises"), "memorising: " + w.advice);

    // Loss: lower is better.
    const LearningCurveView loss = BuildLearningCurveView(o, CurveScore::Loss);
    Check(!loss.higher_is_better && !loss.percent && loss.score_name == "loss", "loss is lower-better and not a percent");
}

void CheckRegression() {
    CrossValidationOutcome o;
    o.ok = true;
    o.regression = true;
    const double mae[] = {318.2, 300.5, 340.0, 290.4, 600.0};
    for (int k = 0; k < 5; ++k) {
        CrossValidationRun r;
        r.fold = k + 1;
        r.val.mae = mae[k];
        r.val.rmse = mae[k] * 1.2;
        r.val.r2 = 0.7;
        o.runs.push_back(r);
    }
    const CrossValidationView v = BuildCrossValidationView(o, 318.2);
    Check(v.headline.rfind("MAE 369.8 \xC2\xB1 ", 0) == 0 && v.columns[0] == "MAE" && !v.percent, "regression headline: " + v.headline);
    Check(Has(v.sentence, "expect an error of about 369.8 on new rows.") && Has(v.sentence, "inside that range"), "regression sentence: " + v.sentence);
    Check(v.rows[4].flagged && Has(v.note, "Fold 5's MAE (600.0) stands apart"), "a hard fold: " + v.note);
}

void CheckProgressAndShares() {
    CrossValidationProgress p;
    p.run = 3;
    p.runs = 5;
    p.fold = 3;
    p.folds = 5;
    p.epoch = 22;
    p.epochs = 40;
    const std::string line = ProgressLine(p, 62.0);
    Check(line.rfind("Fold 3 of 5 \xC2\xB7 epoch 22 of 40 \xC2\xB7 about ", 0) == 0 && Has(line, " left"), "progress: " + line);
    std::vector<double> shares;
    std::string error;
    Check(ParseShares("10%, 25%, 50%, 75%, 100%", shares, error) && shares.size() == 5 && shares[0] == 0.1 && shares[4] == 1.0, "percent list");
    Check(ParseShares("0.5 0.25 1", shares, error) && shares.size() == 3 && shares[0] == 0.25, "fractions, sorted");
    Check(SharesText({0.1, 0.25, 1.0}) == "10%, 25%, 100%", "shares text");
    Check(!ParseShares("10%, 150%", shares, error) && Has(error, "at most 100%"), "over 100% refused: " + error);
    Check(!ParseShares("half", shares, error) && Has(error, "not a number"), "words refused: " + error);
    Check(!ParseShares("50%", shares, error) && Has(error, "two shares"), "one share refused: " + error);
}

}  // namespace

int main() {
    CheckCrossValidation();
    CheckLearningCurve();
    CheckRegression();
    CheckProgressAndShares();
    std::cout << "cross-validation presentation: headline, range and single test, standing-out fold, learning curve, regression, "
                 "progress, shares. OK\n";
    return 0;
}
