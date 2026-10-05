# Turkish Criminal Law Q&A – LLM Fine-Tuning

Fine-tuning a small LLM to answer questions on Turkish substantive criminal law (TCK general provisions) with short, article-referenced answers. Built in Google Colab with [Unsloth](https://github.com/unslothai/unsloth).

> Educational project. The model's answers are not legal advice.

## Setup
| | |
|---|---|
| Base model | `unsloth/Llama-3.1-8B` (4-bit) |
| Method | LoRA fine-tuning (Unsloth + TRL `SFTTrainer`) |
| Data | [X] hand-written Q&A pairs in Alpaca format (`MCHD.json`), checked against the TCK text |
| Hardware | Colab T4 GPU |

The dataset also includes examples where the model declines to judge a personal case and refers the user to a lawyer.

## Experiment Log

**Attempt 1 – no learning.** The model answered "Kast nedir?" with an unrelated definition, even though the question was in the dataset. Cause: the `trainer.train()` cell was never run, so inference used the untouched base model. The original settings would also have given only ~12 training steps for this dataset size.

**Attempt 2 – fixed.** Ran training with `num_train_epochs = 10` and `gradient_accumulation_steps = 1` (~[Y] steps).

| Question | In dataset? | Result |
|---|---|---|
| Kast nedir? | Yes | Correct, reproduced the training answer (TCK m.21) |
| Taksir nedir? | [Yes/No] | Correct definition with article reference (TCK m.22) |
| Komşum bana hakaret etti, ne ceza alır? | [Yes/No] | Declined to give a case-specific answer, cited TCK m.125 |

## Findings
- Fine-tuning mainly taught the **answer style**: citing the article number, keeping answers short and refusing case-specific advice.
- [Veri setinde yoksa:] Correct answers to unseen questions suggest the base model already knew the TCK text from pre-training; fine-tuning helped it surface that knowledge in the right format.
- With 10 epochs on a small dataset the model memorizes training answers word for word.

## Limitations & Next Steps
- [✓] Build a separate test set of unseen questions to measure accuracy properly
- [✓] Add retrieval (RAG) over the official TCK text so answers are grounded in the source
- [✓] Try a base model with stronger Turkish support

## Run It
Open the notebook in Colab, select **Runtime → T4 GPU**, upload `MCHD.json` and run all cells in order.
