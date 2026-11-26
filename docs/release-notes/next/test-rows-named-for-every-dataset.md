kind: fixed
area: training
page: usage/data_studio.md

A test of a dataset registered through the older dataset interface now names each tested row: `test_predictions` gets its `row_id` column and the Worst rows table its `dataset_row`, as tests of tables, images and audio already did. Before, both were missing for that kind of dataset.
