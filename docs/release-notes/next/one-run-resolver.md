kind: changed
area: training
page: usage/server-connection.md

A training job on a server node (remote training) now picks how to read its data by the same rules as Train in the Engine, in the same order: token-window graphs first, then Validation and Test inputs wired to the Data Split, then a table in memory, then a Parquet file kept on disk. Two things change for a job. Its data-loader worker count is limited to what the node's processor offers, as in the Engine. And a table's input width is read from its columns when the run is built, as in the Engine. A dataset the host has not loaded is refused with the dataset's name; that includes image, audio and raw text datasets on a node, which loads none of them. The messages for these refusals are now the same in the Engine and on a node.
