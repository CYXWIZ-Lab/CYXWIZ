# To Fix 11 - Text Vocabulary Dialog and Sentiment Vocab Workflow

This note tracks the next text-vocabulary pass for CyxWiz Data Studio.

## Goal

Make `TextVocabulary` a first-class dialog-backed node, like `DataInput`, so users can manage vocabularies without being forced through inline properties controls.

The dialog should also let the user choose the vocabulary/tokenization method rather than forcing a single fixed mode.

For third-party tokenization, we can add a plugin-backed `TextVocabulary_3prt` path using SentencePiece:
https://github.com/google/sentencepiece

## Current Gap

The backend already has vocabulary support through `cyxwiz::Vocabulary` and `cyxwiz::Tokenizer`, including:
- `Vocabulary::BuildFromDocuments()`
- `Vocabulary::WordToIndex()`
- `Vocabulary::IndexToWord()`
- `Vocabulary::LoadFromFile()` / `SaveToFile()`

The UI still exposes only the tokenizer dialog path, so the vocabulary node is not handled as a proper editor workflow.

## Sentiment Demo Follow-Up

For the sentiment analysis demo under `D:/demo/mrcj/datasets/sentiment analysis/`, we should mirror the CodeFeedback flow:
- build a vocabulary from the training CSV
- write the vocab to disk
- point the CyxGraph `TextVocabulary` node at that file
- let the engine read the vocabulary through its existing API

That keeps the example reproducible and avoids building the vocab ad hoc inside the inference helper.

## Suggested Next Steps

1. Add a dedicated `VocabularyDialog` for `TextVocabulary`.
2. Expose multiple vocabulary methods in the dialog, such as word-level, character-level, subword, and frequency-based selection rules.
3. Keep the in-house `TextVocabulary` path for native word-level workflows.
4. Add a plugin-backed `TextVocabulary_3prt` node for SentencePiece when the user wants third-party subword vocabularies.
5. Add a sentiment vocab prep script that generates `sentiment_analysis_vocab.txt`.
6. Wire the sentiment graph to that vocab file.
7. Update the inference helper to read metadata and reuse the generated vocab.
