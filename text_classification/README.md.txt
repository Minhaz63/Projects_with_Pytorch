# Sarcasm Detection using BERT and PyTorch

This project builds a text classification model to detect sarcasm in news headlines using a fine-tuned **BERT (bert-base-uncased)** model with **PyTorch**.

---

## Overview
The model takes a news headline and predicts whether it is sarcastic (1) or not sarcastic (0).  
It learns contextual cues, tone, and subtle irony from text data.

Example:
- "Scientists discover water on Mars." → Not sarcastic  
- "Scientists discover water on Mars, finally good news for people stuck there." → Sarcastic  

---

## Model Used
- Base Model: `bert-base-uncased` (Hugging Face Transformers)
- Architecture:
  - BERT Encoder (outputs 768-dim vector)
  - Linear(768 → 384)
  - Dropout(0.25)
  - Linear(384 → 1)
  - Sigmoid activation for binary classification
- Loss Function: `BCEWithLogitsLoss`
- Optimizer: `Adam`
- Batch Size: `32`
- Epochs: `3`

---

## How the Algorithm Works

1. **Tokenization**  
   - Each headline is tokenized using BERT's tokenizer → converted to WordPiece tokens and integer IDs.

2. **Encoding**  
   - Generates `input_ids` and `attention_mask` tensors (padded/truncated to length 100).

3. **Contextual Embedding**  
   - BERT encodes each token with self-attention, capturing meaning and context.

4. **[CLS] Representation**  
   - The `[CLS]` token embedding summarizes the entire headline.

5. **Classification Head**  
   - The `[CLS]` vector is passed through two fully connected layers → output is a probability (0–1).

6. **Training and Validation**  
   - Model trains with backpropagation, tracks accuracy and loss for each epoch using `tqdm`.

7. **Testing**  
   - Final evaluation is done on unseen test data to check generalization.

---

## Dataset
**Dataset:** Sarcasm Headlines Dataset (JSON format)

Each record contains:
- `headline`: news headline text
- `is_sarcastic`: 1 if sarcastic, 0 otherwise

| Split | Percentage |
|--------|-------------|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

Example record:
```json
{"headline": "thirtysomething scientists unveil doomsday clock of hair loss", "is_sarcastic": 1}
