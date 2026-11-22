Status: Planned — implementation not started yet.

# telugu-lora-eval

Fine-tune a small open-weight language model on Telugu instruction data with LoRA/QLoRA, then measure honestly whether it got better than the base model on held-out Telugu benchmarks and a small human-rated set.

Nothing in this repository has been built, trained or measured yet. There are no results below on purpose. Every number the project eventually reports will come from a logged run that is linked from this README.

- Plan, week by week: [PLAN.md](PLAN.md)
- Plan page: [`site/index.html`](site/index.html)
- Author: Monish Chaparla (B.Tech CSE, Andhra University; native Telugu speaker)

## Problem and why it matters

Most "AI projects" call a hosted API. This one shows the other half of the job: adapting a model, controlling the data, and proving the change with an evaluation that can say "no, it did not help".

Telugu is spoken by tens of millions of people, but small open models handle it unevenly. Two things make it hard:

1. Tokenisation. Byte-level BPE vocabularies built mostly on English, Chinese and European text can split Telugu script into many pieces per word. That costs context length, speed and often quality. (To be measured in week 2, not assumed.)
2. Thin evaluation. Many Indic benchmarks are machine-translated from English, so a score may reward translationese rather than natural Telugu.

The project asks one narrow question: **does parameter-efficient fine-tuning on open Telugu instruction data measurably improve a 1–3B model on held-out Telugu tasks, and what does it cost in English ability?**

A negative or mixed answer is a valid outcome and will be reported as such.

## Approach

1. **Pick the model.** Primary: `Qwen/Qwen3-1.7B-Base` (Apache-2.0). Reasons below.
2. **Build the data.** Collect open Telugu instruction data, filter it, remove anything that overlaps the evaluation sets, and freeze the split with a hash manifest.
3. **Fine-tune.** QLoRA (4-bit NF4 base, LoRA adapters) with `peft` + `transformers` on a free T4. Small ablation over LoRA rank or data mix.
4. **Evaluate.** Same prompts, same decoding, same scorer for every system:
   - base model (few-shot),
   - fine-tuned model (adapter),
   - the official `Qwen/Qwen3-1.7B` post-trained model as a strong reference.
5. **Report.** Confidence intervals, per-task tables, error analysis, a model card, and a list of what did not work.

### Model choice (checked at planning time)

| Candidate | Licence (Hugging Face metadata) | Size | Telugu coverage | Verdict |
|---|---|---|---|---|
| `Qwen/Qwen3-1.7B-Base` | Apache-2.0 | 1.72B params | Telugu is in the Qwen3 blog's list of supported languages and dialects (Dravidian: Tamil, Telugu, Kannada, Malayalam) | **Primary.** Permissive licence, ungated, fits QLoRA on a T4. |
| `Qwen/Qwen2.5-1.5B` | Apache-2.0 | 1.54B | Not in Qwen2.5's headline language list | Optional ablation only. |
| `google/gemma-3-1b-pt` | Gemma terms of use (not an OSI licence), gated | 1.0B | Google says 140+ languages; no Telugu-specific figure found | Fallback only; extra terms and gating. |
| `sarvamai/sarvam-1` | No licence field in Hugging Face metadata | 2.5B | Lists Telugu and 9 other Indic languages | Stretch comparison **only after** its licence text is read. |
| `meta-llama/Llama-3.2-1B` | Llama 3.2 community licence, gated | 1.24B | Telugu not among its listed languages | Rejected. |

"Telugu is listed as supported" does not mean "good at Telugu". The first evaluation task is to measure the base model, including tokens per word and Telugu-script output rate, before any training.

## Data sources and licences

Licence fields below were read from Hugging Face metadata or cards at planning time. A licence on a card does not automatically cover upstream sources, so each data component gets its own check and an entry in `DATA_LICENCES.md` before training.

### Training candidates

| Source | What it is | Licence on card | Notes |
|---|---|---|---|
| [`CohereLabs/aya_dataset`](https://huggingface.co/datasets/CohereLabs/aya_dataset) | Human-written prompt/completion pairs, 65 languages including Telugu (`tel`) | Apache-2.0 | About 8.4k Telugu training rows per the Hugging Face datasets server at planning time. Highest-quality source. |
| [`CohereLabs/aya_collection`](https://huggingface.co/datasets/CohereLabs/aya_collection) (Telugu subsets) | Templated, translated and synthetic instruction data | Apache-2.0 | Some subsets are templated from public NLP datasets. **Must be screened for overlap with the eval sets.** |
| [`ai4bharat/indic-align`](https://huggingface.co/datasets/ai4bharat/indic-align) (IndicAlign-Instruct, Telugu) | Translated and curated instruction data for 14 Indic languages | CC-BY-4.0 | Contains machine-translated components (for example Dolly-T, OpenAssistant-T). Check each component's upstream licence. |

### Evaluation sets (never trained on)

| Task | Dataset | Licence on card | Metric |
|---|---|---|---|
| Natural language inference | [`Divyanshu/indicxnli`](https://huggingface.co/datasets/Divyanshu/indicxnli) (`te`) | CC0-1.0 | Accuracy, macro-F1 |
| Sentiment | [`ai4bharat/IndicSentiment`](https://huggingface.co/datasets/ai4bharat/IndicSentiment) (`te`) | No licence field on the card; AI4Bharat's IndicXTREME page states CC0 for its packaging | Accuracy, macro-F1 |
| Reading comprehension | [`facebook/belebele`](https://huggingface.co/datasets/facebook/belebele) (`tel_Telu`) | CC-BY-SA-4.0 | Accuracy |
| Translation en→te and te→en | [`google/IndicGenBench_flores_in`](https://huggingface.co/datasets/google/IndicGenBench_flores_in) (FLORES-derived) | CC-BY-SA-4.0 | chrF++ (sacreBLEU) |
| Open-ended instruction following | Human-rated set (below) | Own work, to be released CC-BY-4.0 | Rubric scores, pairwise preference |

Optional: `ai4bharat/IndicQA` (CC-BY-4.0) for extractive QA; `ai4bharat/IN22-Gen` (CC-BY-4.0, auto-gated) for translation.

### Checked and excluded

- `ai4bharat/indic-instruct-data-v0.1`: its card lists only English and Hindi, so it has no Telugu data.
- `ai4bharat/IndicParaphrase`: CC-BY-NC-4.0, which is non-commercial.
- Telugu translations of Alpaca (for example `Telugu-LLM-Labs/*alpaca*`): the card says CC-BY-4.0, but the upstream Alpaca data is non-commercial and generated with OpenAI model outputs, so the chain is not clean. Excluded to keep the final adapter's licence simple.
- `ai4bharat/indic_glue`: licence listed as `other`; use the individual datasets above instead.

Planning-time caveats to resolve in week 1: the IndicXNLI card does not state the licence of the English XNLI it was translated from, and the IndicSentiment licence is not on its card. Both are used for **evaluation only**, and the exact terms get recorded before any number is published.

## Evaluation design

### Principles

- Test data is frozen before training starts. No tuning decisions use test sets; a separate validation split (from the training pool) drives early stopping and hyperparameters.
- Contamination check: exact-match and 8-gram overlap between every training example and every test item. Overlapping training rows are removed and the count is reported.
- One harness for all systems: same prompt templates, same decoding settings (greedy, fixed max tokens), same scorer.
- Every metric has a 95% bootstrap confidence interval. Differences between systems use a paired bootstrap test.
- Seeds: at least 3 training seeds for the main configuration if compute allows; otherwise one seed and the report says so.

### Automatic benchmarks

| Task | Scoring method | Metric |
|---|---|---|
| IndicXNLI-te, IndicSentiment-te, Belebele-tel | Few-shot prompt; choose the option with the highest log-likelihood (no free-text parsing) | Accuracy, macro-F1 |
| FLORES-derived en↔te | Greedy generation with a fixed instruction template | chrF++ (sacreBLEU signature recorded) |
| Output hygiene | Share of generations mainly in Telugu script when Telugu was requested; repetition rate; empty-output rate | Percent |
| English regression check | Small fixed English subset (for example ARC-Easy or a 500-item MMLU slice) | Accuracy, to detect forgetting |
| Tokenisation | Tokens per word on a Telugu sample (the tokenizer is not modified, so this is a property of the base model) | Ratio |

Classification and MCQ scores for a 1.7B base model can sit close to chance, and a fine-tune may or may not move them. That is exactly why the confidence intervals and the chance-level baseline are part of every table.

### Human-rated set

- 50 prompts written by the author across: factual questions, short writing, summarisation, translation, everyday reasoning, and culturally local topics (festivals, food, geography of Andhra Pradesh and Telangana). Prompts are written fresh and are not in any training source.
- Outputs from the 3 systems are shuffled and anonymised; raters do not know which system produced which output.
- Rubric (1–5): Telugu fluency, correctness, instruction following. Plus a pairwise "which is better" between base-with-few-shot and fine-tuned.
- Two raters: the author and one other Telugu speaker. Agreement reported as Cohen's κ / weighted κ.
- Limits stated up front: n=50, the author is both prompt writer and a rater, so the set supports a directional claim only.

### Success criteria (decided before running)

The project counts as "worked" if the fine-tuned model beats the base model on at least two of the three classification/MCQ tasks or on translation chrF++, with confidence intervals reported, **and** loses no more than a small pre-declared margin on the English check. If not, the write-up documents the negative result and the likely causes. The margin is fixed in the week-1 evaluation protocol file, before training.

## Tech stack

- Python 3.11, PyTorch
- Hugging Face `transformers`, `peft`, `datasets`, `bitsandbytes` (QLoRA), `trl` (SFT trainer), `accelerate`, `evaluate`
- `sacrebleu` for chrF++; `scikit-learn` for F1 and κ; `numpy` for bootstrap
- Experiment tracking: Weights & Biases (free tier) or MLflow local. Decide in week 1 and use one.
- Notebooks for Colab/Kaggle runs, thin scripts for the harness so runs are reproducible from a config file
- Hugging Face Hub for the adapter and model card

## Compute plan and cost

| Item | Plan |
|---|---|
| Training GPU | Kaggle notebooks (T4 ×2 or P100) and Google Colab free (T4, 16 GB). Free-tier quotas change and are not guaranteed; re-check the published limits in week 1. Kaggle has historically offered about 30 GPU hours per week with a per-session cap. |
| Why it should fit | 1.7B base in 4-bit NF4 plus LoRA adapters, gradient checkpointing, batch via gradient accumulation, max sequence about 1024. T4 has no bf16, so compute in fp16. Peak memory is to be measured in the week-3 smoke run, not assumed. |
| Run length | Unknown until the smoke run. The plan is short, checkpointed runs (resume from checkpoint on every session start) so a disconnect costs minutes, not the run. |
| Evaluation | Inference only; 4-bit or fp16 on the same GPUs. |
| Storage | Adapters are small (tens of MB expected); the base model is pulled from the Hub each session. |
| Cost | Target ₹0 / $0 on free tiers. If free quota runs out, a small paid-compute top-up with a cap decided and recorded before spending. No cloud spend without a note in the repo. |

## Milestones and done-criteria

| # | Milestone | Done when |
|---|---|---|
| M0 | Protocol and licence audit | `EVAL_PROTOCOL.md` and `DATA_LICENCES.md` committed: tasks, metrics, success margin, per-source licence and upstream check. Test sets frozen with hashes. |
| M1 | Eval harness + base-model baseline | One command runs all benchmarks for any model/adapter. Base model scores (with CIs) logged and linked. Scorers have unit tests. |
| M2 | Data pipeline | Deterministic build script produces train/val splits from the chosen sources; contamination report committed; dataset card draft written. |
| M3 | Smoke run | A tiny QLoRA run completes end-to-end on a free GPU, resumes from a checkpoint, and logs memory and throughput. |
| M4 | Main fine-tune | Main configuration trained (plus extra seeds/ablations if quota allows); adapters pushed to the Hub; every run in the tracker. |
| M5 | Full evaluation + human study | All systems scored with the same harness; human ratings collected from two raters; agreement computed. |
| M6 | Write-up | Model card, results tables, error analysis, limitations, reproduction instructions. The README status line is updated only at this point, to describe what actually exists. |

Week-by-week tasks are in [PLAN.md](PLAN.md).

## Risks and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Free GPU quota or session limits | Training could stall | Checkpoint every N steps, resume logic tested in M3, short runs, small data mix first |
| Telugu tokenisation is expensive | Long sequences, fewer examples fit | Measure tokens per word first; cap length; report as a limitation; vocabulary extension is out of scope |
| Test contamination | Inflated scores | N-gram overlap filter, frozen test sets, report removed counts |
| Licence chain issues | A public adapter could carry restricted terms | Per-source audit, exclude NC and unclear sources, keep `DATA_LICENCES.md` |
| Machine-translated data teaches translationese | Fine-tune may sound unnatural | Prefer human-written Aya data; compare data mixes; human evaluation of fluency |
| Catastrophic forgetting | English or reasoning may drop | English regression check; low-rank ablation; report the trade-off |
| Benchmarks near chance for a 1.7B model | Differences are not significant | Bootstrap CIs, paired tests, chance baselines, say "no significant difference" when true |
| Small, author-written human set | Bias, low power | Blind and randomise, second rater, κ, call it directional |
| Unfair base vs fine-tuned comparison | Inflated gain | Give the base the same few-shot prompt; include the official instruct model as a reference |
| Scope creep | Never finishes | Fixed 8-week plan, stretch goals listed separately |

## Model card outline (to be filled after training)

1. Model details: base model and revision, adapter type, LoRA rank/alpha/targets, quantisation, licence of the adapter (Apache-2.0 expected if all data allows), contact.
2. Intended use and out-of-scope use (research on Telugu adaptation; not for medical, legal or safety-critical use).
3. Training data: each source, version/revision, licence, filters, counts, contamination removals.
4. Training procedure: hyperparameters, seeds, hardware, tracker links, total GPU time.
5. Evaluation: benchmark list, prompts, scoring, results with confidence intervals, human-study protocol and agreement.
6. Limitations and biases: dataset translation artefacts, dialect coverage (standard written Telugu vs regional speech), small human set, English regression.
7. Ethical considerations: hallucinations, sensitive topics, data provenance, who rated.
8. How to run: loading the adapter, reproducing the evaluation.
9. Citation and acknowledgements for each dataset and model.

## Interview talking points

- Why parameter-efficient fine-tuning: memory maths for full fine-tuning vs QLoRA on a 16 GB T4; what rank and alpha change; why 4-bit NF4 with double quantisation works and what it costs.
- Why the evaluation comes first: the harness and base-model baseline are built before any training, success criteria are fixed in advance, and negative results are publishable here.
- Contamination and licence hygiene: why overlap filtering matters and why a card's licence field is not the end of the audit.
- Statistics: bootstrap confidence intervals, paired tests, chance baselines, and what "not significant" means for a 1.7B model.
- Telugu specifics: script and tokenisation cost, machine-translated vs natural data, why chrF++ is preferred over BLEU for morphologically rich languages.
- Trade-offs: forgetting, rank vs data size, base+few-shot vs instruct reference, and what would be done with more compute (vocabulary extension, continued pre-training, a larger model).
- Engineering: reproducible configs, checkpoint/resume on pre-emptible free GPUs, experiment tracking.

## Out of scope for now

Vocabulary extension or continued pre-training, RLHF/DPO, serving or a chat UI, models above 3B, multilingual claims beyond Telugu and the English regression check.

## Licence

Code in this repository will be released under MIT when it exists. Data and model licences are listed per source in `DATA_LICENCES.md` (to be created in M0).
