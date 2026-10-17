# Testing a trained model

After training, test the model on the graph's test split:

- **Train > Test > Run Test** (F8) runs the whole test split.
- **Train > Test > Run Quick Test** runs the first 10 batches only. Its
  results say *quick test* and are never taken for a full run.

The test runs in the background (Task View) and opens **Test Results** when it
ends. **Train > Test > View Test Results** opens the panel again later.

## Test Results

The header shows the headline figure (accuracy for classification; 1-NN
accuracy for metric learning; MAE for regression; token accuracy for
sequence models), how many samples were tested and how long it took, and the
other figures as chips (loss, F1, samples per second).

If a test fails, the panel says *The test failed* with the reason, above the
last successful results (a failed run is never compared against).

Tabs (the tab bar scrolls when the panel is narrow):

| Tab | What it shows |
| --- | --- |
| Overview | Precision, recall, macro F1 and weighted F1; the best class and the worst classes ("never predicted" when the model never chose a class). |
| Confusion matrix | Rows are the true class, columns the predicted class, as **Counts** or **Share of each true class**. The largest confusion is written under it ("true Depression predicted as Suicidal (641)"). Hover a cell for its numbers. |
| Classes | One row per class: precision, recall, F1, found (TP), false alarms (FP), missed (FN), support. Click a header to sort, type to filter by name. |
| Curves | ROC and precision-recall curves (classification). |
| Predictions | Every test sample's true class, predicted class and confidence; **All** or **Only mistakes** (the most confident mistakes first). A regression lists every row's true and predicted value and error; **All** or **Largest errors first** (with several targets the first is shown, and the order uses each row's mean error over all of them). **Dataset row** is the row of the tested table (counted from 1, as the Table Viewer counts) when the test recorded it. |
| By column value | Where the model goes wrong: pick a column of the tested table; each value gets its test rows and the share wrong (a classifier) or the mean \|error\| (regression), highest first; a value above the overall error by a quarter or more is amber. A number column with more than 20 values is cut into five ranges of about equal rows; a text column keeps its 20 most common values and puts the rest under *(other values)*. A table on disk is read one row group at a time (a task in Task View). It needs the tested table loaded; one re-loaded with fewer rows is refused as changed since the test. |
| Comparison | The latest run beside the previous one, with the change of each figure. **Train > Test > Compare Test Results...** opens this tab. |

The Overview of a regression also has its **Bias** (the mean of predicted -
true: positive predicts high). Data Studio > Results > Test run repeats the
headline of the latest test and opens this panel.

Class names come from the data: a Text Tokenizer's label keeps the class
names of its label column, so the sentiment demo shows *Anxiety*, *Normal*,
*Depression* and so on instead of class numbers.

## Footer

- **Export CSV...** and **Export JSON...** save what the panel shows (the
  metrics, every class with its name, the confusion matrix, each prediction
  with its dataset row, a regression's rows, the tested table's name; JSON also
  has the curves and the quick-test flag). The save dialog starts
  in the project's `exports` folder; a note says where the file went.
  **Train > Test > Export Test Report...** is the same JSON export.
- **Clear** empties the panel.
- **Run test again** tests the current graph's trained model again.

See [export_import.md](export_import.md) to save the trained model itself.
