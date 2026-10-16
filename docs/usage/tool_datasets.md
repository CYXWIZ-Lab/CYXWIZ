# Where the tools get their data

Data enters the Engine in one place: a **Data Input** node in CyxWiz Studio.
Everything else reads what Data Input nodes loaded.

- **Data > Import Dataset** and **Data > Create Custom Dataset** add a new Data
  Input node and open its dialog.
- **Data > Dataset Statistics** opens Data Studio's Profile tab (row counts,
  columns, value summaries) for the dataset picked there.

## The Dataset dropdown

Every analysis tool has the same Dataset dropdown as Data Studio:
- Data > Transform
- Tools > Clustering and Projection
- Tools > Statistics
- Tools > Evaluation
- Data > Explore

It lists:
- datasets loaded by Data Input nodes, shown with the node's name
  ("Customer CSV · Data Input · 248 rows · 6 cols");
- tables made by graph nodes from them when a pipeline runs, shown as
  "node result" with the node's name (for example "PCA (2 components)").

Picking one reads it in the background, whether it is in memory or on disk.
The tool shows "Reading the dataset..." while it reads, or the reason when it
cannot read it.

Nothing appears until a Data Input node has loaded data. To fill the list, open
its dialog and apply it, or run the pipeline (Execute Pipeline, or double-click
a Plot node).

The Data Explorer, which opened files directly, has been removed. Data Studio
covers querying, profiling and visualizing datasets.
