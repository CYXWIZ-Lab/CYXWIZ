kind: fixed
area: training
page: usage/training_dashboard.md

A few run figures now tell the truth.
- **Samples per second:** the Dashboard and the log time the training pass alone; the validation pass used to be counted in its time, so the rate read low.
- **Range test:** a learning rate range test that has its curve now ends as completed, not as cancelled by the user. The Tasks panel says so.
- **Remote jobs:** a remote training job reports the samples it trained (each batch's real size); a short last batch used to count as full.
- **Large test sets:** test results count past 2,147,483,647 rows (samples, correct predictions, the confusion matrix and per-class counts).
