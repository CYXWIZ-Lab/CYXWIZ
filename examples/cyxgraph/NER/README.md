# NER Sequence Tagging CyxGraph

This folder contains a CyxGraph design for training a named entity
recognition model on:

```text
D:\Dev\DataSet_List\NER\NER dataset.csv
```

The source dataset is CoNLL-style CSV:

```text
Sentence #,Word,POS,Tag
Sentence: 1,Thousands,NNS,O
,of,IN,O
,demonstrators,NNS,O
...
```

Important: this is not a normal text-classification dataset. Each token
has its own BIO tag, so the model must predict one label per token.

## Files

```text
prepare_ner_demo.py
ner_inference.py
ner_bilstm_sequence_tagger.cyxgraph
README.md
```

## Current Data Shape

Raw CSV rows are token rows:

```text
Word -> one token
POS  -> part-of-speech tag
Tag  -> BIO named-entity tag
```

Sentence boundaries are represented by `Sentence #`. Only the first row
of a sentence has a value such as `Sentence: 1`; following rows are blank
until the next sentence starts.

The prep script converts this into sentence-level rows:

```text
sentence_id,tokens,pos_tags,ner_tags
1,"Thousands of demonstrators ...","NNS IN NNS ...","O O O ..."
```

This is the format the graph expects.

## Prepare The Dataset

Run:

```powershell
python "D:\Dev\CyxWiz_Claude\examples\cyxgraph\NER\prepare_ner_demo.py"
```

Default input:

```text
D:\Dev\DataSet_List\NER\NER dataset.csv
```

Default output directory:

```text
D:\Dev\DataSet_List\NER\cyxwiz_ner
```

Generated files:

```text
ner_sentences.csv
ner_word_vocab.txt
ner_pos_vocab.txt
ner_tag_vocab.txt
ner_metadata.json
```

Validation run against the provided dataset produced:

```text
sentences: 47,959
tokens: 1,048,575
max source sentence length: 104
POS vocab size: 44
NER tag vocab size: 19
```

The generated tag order is:

```text
[PAD], [UNK], O,
B-geo, B-gpe, B-per, I-geo, B-org, I-org, B-tim,
B-art, I-art, I-per, I-gpe, I-tim, B-nat, B-eve, I-eve, I-nat
```

## Intended Model

The graph is designed for a BiLSTM token classifier:

```text
NER CSV
  -> NERSequenceBuilder
  -> TokenVocabulary
  -> POSVocabulary
  -> NERTagVocabulary
  -> SequencePadding
  -> DataSplit
  -> DataLoader
  -> Word Embedding
  -> optional POS Embedding
  -> FeatureConcat
  -> BiLSTM return_sequences=true
  -> Dropout
  -> TimeDistributed Dense
  -> Token CrossEntropy
  -> Adam
  -> SequenceTagOutput
```

The model output shape should be:

```text
[batch_size, max_sequence_length, num_tags]
```

The label tensor shape should be:

```text
[batch_size, max_sequence_length]
```

Padding tokens should be ignored by the loss.

## Engine Gap

The current CyxWiz text examples support whole-sequence classification:

```text
TextTokenizer -> TextVocabulary -> TextPadding
  -> Embedding -> GRU/LSTM -> Dense -> CrossEntropy
```

NER needs sequence tagging support:

```text
Embedding -> BiLSTM(return_sequences=true)
  -> TimeDistributedDense -> TokenCrossEntropy(ignore_pad)
```

Needed backend/graph nodes:

- `NERSequenceBuilder`
- `TokenVocabulary`
- `NERTagVocabulary`
- `SequencePadding` for both token IDs and tag IDs
- `FeatureConcat` for optional word + POS embeddings
- `TimeDistributedDense`
- `TokenCrossEntropyLoss`
- `SequenceTagOutput`
- token-level metrics:
  - token accuracy
  - entity precision
  - entity recall
  - entity F1

The `.cyxgraph` file in this folder is therefore the target design for
the NER engine path. It should be used to guide implementation and then
become directly trainable once the missing sequence-tagging nodes are
added.

## Why Not Use The Sentiment Graph Directly?

Sentiment analysis predicts one label for the whole text.

NER predicts one label for every token:

```text
London -> B-geo
Iraq   -> B-geo
British -> B-gpe
```

Using the sentiment graph directly would collapse a full sentence into
one class and lose the token-level labels. That would not train a real
NER model.

## Proof Use Case

After the sequence-tagging path is implemented, a proof run should:

1. Generate `ner_sentences.csv` and vocab files.
2. Open `ner_bilstm_sequence_tagger.cyxgraph`.
3. Confirm the `DataInput` path points to `ner_sentences.csv`.
4. Confirm the vocabulary paths point to generated vocab files.
5. Train for a small number of epochs.
6. Track:
   - token-level loss
   - token accuracy ignoring padding
   - entity-level precision/recall/F1
7. Save a `.cyxmodel` containing:
   - graph
   - weights
   - word vocabulary
   - POS vocabulary
   - tag vocabulary
   - metadata
   - max sequence length

## Inference Helper

After preparing the dataset, inspect the encoded inference payload:

```powershell
python "D:\Dev\CyxWiz_Claude\examples\cyxgraph\NER\ner_inference.py" `
  --dry-run `
  --sentence "British troops marched through London ."
```

With a deployed sequence-tagging model loaded in the embedded server:

```powershell
python "D:\Dev\CyxWiz_Claude\examples\cyxgraph\NER\ner_inference.py" `
  --sentence "British troops marched through London ."
```

Batch mode against prepared sentence rows:

```powershell
python "D:\Dev\CyxWiz_Claude\examples\cyxgraph\NER\ner_inference.py" `
  --csv-file "D:\Dev\DataSet_List\NER\cyxwiz_ner\ner_sentences.csv" `
  --num-samples 10
```

The helper expects sequence-tagging output shaped as either:

```text
[max_length, num_tags]
```

or a flat vector that can be reshaped to:

```text
max_length * num_tags
```

It then decodes one BIO tag per visible token.
