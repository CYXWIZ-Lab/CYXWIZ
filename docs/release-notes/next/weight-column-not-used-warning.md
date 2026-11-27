kind: changed
area: training
page: usage/data_studio.md

Compile now warns when a Data Input column has the Weight role: "The Weight column '...' is not used in training yet" (*CW-C-0208*). Training keeps the column out of the features, as before, but weighs no rows by it: every row counts the same in the loss and the metrics. Before, the compile report only listed it among the columns kept out of the features.
