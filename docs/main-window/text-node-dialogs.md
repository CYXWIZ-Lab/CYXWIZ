---
title: Text Tokenizer and Embedding dialogs
description: The dialogs of the Tokenizer, Embedding and the older Vocabulary and Padding nodes, tab by tab.
type: reference
area: graph-editor
audience: user
status: stable
since: 1.0.7
owner: graph-editor
last_reviewed: 2026-11-25
related:
  - node-dialogs.md
  - data-input-formats.md
  - ../usage/word_embeddings.md
redirect_from: []
---

The text nodes that turn words into numbers are set up in dialogs of their
own. The buttons every dialog shares are in [Node dialogs](node-dialogs.md);
why a text input needs one of these nodes is in
[Data Input formats](data-input-formats.md#text-needs-a-node-that-turns-it-into-numbers).

## Text Tokenizer

The **Tokenizer** node opens **Text Tokenizer Configuration**. The tabs are
named by what you decide:

| Tab | What it sets |
| --- | --- |
| **Words** | The text and label columns, **Split into** (Spaces, Words, Characters, WordPiece, Byte BPE; Byte BPE adds its starting units and space handling) and **Lowercase first**. A vocabulary card shows the file the node uses (*sentiment_analysis_vocab.txt, 5000 tokens, min frequency 5*), warns when its entries are broken bytes left by an older Engine (rebuild it), and has **Build and save vocabulary** and **Inspect file**. |
| **Vocabulary** | Source CSV, text column, minimum frequency, maximum size, the file, and "build and save it when the file is missing" |
| **Length** | The output (padded token columns, causal document windows, decode token IDs, round-trip check), maximum length, pad value, pad and cut |
| **Try it** | Type a sample: its tokens show as you type ([UNK] in orange, hover for the id) with the count against the maximum length, and the ids and round trip under a disclosure |

Help for each field is the hover text on its label. A new Text Tokenizer
takes its text and label columns from the Data Input before it.

## Embedding

The **Embedding** node opens **Embedding Configuration**, with the tabs
**Shape** (vocabulary size and vector size), **Weights** (start from
pretrained word vectors) and **Advanced**. Apply stays off until you change
something ("Nothing changed"). See [Word embeddings](../usage/word_embeddings.md).

## Older graphs: Vocabulary and Padding

Older graphs may hold **Vocabulary** and **Padding** nodes. They open
**Text Vocabulary Configuration** and **Text Padding Configuration**. In
new graphs the Tokenizer's **Vocabulary** and **Length** tabs set both.

## Related

- [Node dialogs](node-dialogs.md)
- [Data Input formats](data-input-formats.md)
- [Word embeddings](../usage/word_embeddings.md)
- [Text in Data Studio](../user-guide/data-studio/text.md)
