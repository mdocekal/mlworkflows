# mlworkflows

Config-driven machine learning workflows built on top of
[🤗 Transformers](https://github.com/huggingface/transformers),
[🤗 Datasets](https://github.com/huggingface/datasets) and
[🤗 Accelerate](https://github.com/huggingface/accelerate).

Every experiment (training, prediction, evaluation, preprocessing) is described by a single **YAML
configuration file**. The Python code defines *what can be configured*; the YAML file says *how it is
configured for this particular run*. No experiment-specific Python scripts are needed.

## Contents

- [Available workflows](#available-workflows)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Design philosophy](#design-philosophy)
- [Command line interface](#command-line-interface)
- [Preparing an experiment step by step](#preparing-an-experiment-step-by-step)
- [Examples](#examples)
  - [1. Sequence labeling (NER-like) with ModernBERT](#1-sequence-labeling-ner-like-with-modernbert)
  - [2. Sequence classification on local JSONL files](#2-sequence-classification-on-local-jsonl-files)
  - [3. Pseudo-perplexity evaluation of a masked LM](#3-pseudo-perplexity-evaluation-of-a-masked-lm)
  - [4. Pre-tokenizing a corpus and masked language modeling](#4-pre-tokenizing-a-corpus-and-masked-language-modeling)
- [Running tests](#running-tests)

## Available workflows

| Workflow module                                                                                            | Purpose                                                     | Example                                                                                        |
|------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| [train_sequence_labeling.py](mlworkflows/workflows/train_sequence_labeling.py)                             | Fine-tune a token classification model (NER, span tagging). | [Example 1](#1-sequence-labeling-ner-like-with-modernbert)                                     |
| [predict_sequence_labeling.py](mlworkflows/workflows/predict_sequence_labeling.py)                         | Predict with a trained token classification model.          | uses the model saved in [Example 1](#1-sequence-labeling-ner-like-with-modernbert)             |
| [train_sequence_classification.py](mlworkflows/workflows/train_sequence_classification.py)                 | Fine-tune a sequence (document/sentence) classifier.        | [Example 2](#2-sequence-classification-on-local-jsonl-files)                                   |
| [evaluations/pseudo_perplexity.py](mlworkflows/workflows/evaluations/pseudo_perplexity.py)                 | Compute pseudo-perplexity of a masked language model.       | [Example 3](#3-pseudo-perplexity-evaluation-of-a-masked-lm)                                    |
| [preprocessing/tokenize_dataset.py](mlworkflows/workflows/preprocessing/tokenize_dataset.py)               | Render inputs with a Jinja2 template, tokenize and save.    | [Example 4](#4-pre-tokenizing-a-corpus-and-masked-language-modeling)                           |
| [train_masked_language_modeling.py](mlworkflows/workflows/train_masked_language_modeling.py)               | (Continued) pre-training with masked language modeling.     | [Example 4](#4-pre-tokenizing-a-corpus-and-masked-language-modeling)                           |

All modules are in [mlworkflows/workflows/](mlworkflows/workflows/).

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/mdocekal/mlworkflows.git
cd mlworkflows

# install PyTorch first (pick the build matching your CUDA version, see https://pytorch.org)
pip install torch

# then the package itself (editable install is recommended, as you will point to workflow files inside it)
pip install -e .
```

Notes:

- `torch-scatter` is built against the installed PyTorch, so PyTorch must be installed before the package.
- Optional: install [`flash-attn`](https://github.com/Dao-AILab/flash-attention) if you want to use
  `attn_implementation: flash_attention_2` in the model configuration.
- Multi-GPU runs are launched via `accelerate`. Create an accelerate config once with `accelerate config`
  (or copy the `accelerate.yaml` from the example below).

The installation provides the `mlworkflows` command. Equivalently, you can call `run.py` from the repository
root (`python run.py ...`), which is handy with `accelerate launch`.

## Quick start

Fine-tune a token classifier on a Hugging Face Hub dataset in four steps.

```bash
# 1. create a directory for the experiment
mkdir my_first_run && cd my_first_run

# 2. generate a documented configuration template with all options and their defaults
mlworkflows make_config ~/mlworkflows/mlworkflows/workflows/train_sequence_labeling.py > config.yaml
```

3. Edit `config.yaml`; you need to set the dataset and the model, everything else has a usable default:

```yaml
dataset:
  cls: DatasetFactory
  config:
    path: SoFairOA/sofair_softcite_somesci
    name: sofair_softcite_somesci_documents
model:
  cls: TokenClassificationModelFactory
  config:
    model_path: answerdotai/ModernBERT-base
training:
  output_dir: ./training
  per_device_train_batch_size: 2
  num_train_epochs: 3
save_model: best
```

```bash
# 4. run it
mlworkflows run ~/mlworkflows/mlworkflows/workflows/train_sequence_labeling.py -c config.yaml
```

Checkpoints end up in `my_first_run/training/` and the final model in `my_first_run/best/`. See
[Examples](#examples) for the full setups of every workflow, including multi-GPU runs and preparing your
own data.

## Design philosophy

### A workflow is a configurable class, an experiment is a YAML file

A workflow is a Python module that defines a class and exposes it as `RUN`:

```python
class SequenceLabelingTrainingWorkflow(BaseTrainingWorkflow):
    text_field: str = ConfigurableValue(
        desc="Field in the dataset that contains the text. ...",
        user_default="text")
    ...

RUN = SequenceLabelingTrainingWorkflow
```

All parameters are declared as configurable attributes (via the
[classconfig](https://pypi.org/project/classconfig/) library), each with a description, a default value,
and optionally a validator and a transformation. Because of this, the workflow can **generate its own
configuration template**, documented line by line:

```bash
mlworkflows make_config mlworkflows/workflows/train_sequence_labeling.py > train_sequence_labeling.yaml
```

The resulting YAML file contains every option with its default value and a comment explaining it. You
edit the values you care about, and the file then fully describes the experiment.

### Components are pluggable through `cls` + `config`

Larger components (dataset, model, tokenizer, metrics, optimizer, early stopping, ...) are factories. In
the YAML they always appear in the same form:

```yaml
model:
  cls: TokenClassificationModelFactory   # which class (a subclass of the expected base class)
  config:                                # its own parameters
    model_path: answerdotai/ModernBERT-base
    ...
```

Swapping a component means changing `cls` (e.g. `optimizer_factory.cls: LionFactory` → `AdamWFactory`) and
its `config`. Use `make_config` to see the defaults; the available optimizers are in
[mlworkflows/training_utils/optimizer_factory.py](mlworkflows/training_utils/optimizer_factory.py)
(`SGDFactory`, `AdamFactory`, `AdamWFactory`, `LionFactory`).

### The whole Hugging Face stack is reachable from the config

- `dataset` → `datasets.load_dataset` / `load_from_disk` (Hub name, local directory, `.json`/`.jsonl`,
  `.csv` or `.txt` file),
- `model` → `transformers.Auto*.from_pretrained` (including quantization, `torch_dtype`,
  `attn_implementation`, config overrides, random initialization),
- `training` → all `transformers.TrainingArguments`,
- `compute_metrics` → any `evaluate` metric.

### One experiment = one directory

The recommended way of working is to create a directory per experiment that contains everything needed
to reproduce it, and in which all outputs are written:

```
experiments/<project>/<model>/<setting>/
├── accelerate.yaml               # hardware setup (number of GPUs, mixed precision, ...)
├── train_sequence_labeling.yaml  # the experiment itself
├── train.sh                      # one-line launcher
├── log.txt                       # captured stdout (created by the run)
├── logs/                         # TensorBoard logs (created by the run)
└── training/                     # checkpoints (created by the run)
```

Relative paths in the configuration (`output_dir`, `logging_dir`, `save_model`, local dataset paths, ...)
are resolved relative to the configuration file, so the outputs stay inside the experiment directory. The
experiment directory can be copied, versioned and re-run as is; changing a hyper-parameter means copying
the directory and editing the YAML.

### Voluntary parameters

Parameters marked as *voluntary* may be omitted from the YAML file and their default value is used.
The generated config is therefore a complete reference, but your configs may be much shorter.

## Command line interface

```text
mlworkflows make_config <path/to/workflow.py>               # print configuration template to stdout
mlworkflows run <path/to/workflow.py> -c <path/to/config>   # run the workflow with given configuration
```

Single process (single GPU / CPU):

```bash
mlworkflows run mlworkflows/workflows/train_sequence_labeling.py -c train_sequence_labeling.yaml
```

Multiple GPUs with `accelerate`:

```bash
accelerate launch --config_file accelerate.yaml ~/mlworkflows/run.py \
    run ~/mlworkflows/mlworkflows/workflows/train_sequence_labeling.py -c train_sequence_labeling.yaml
```

## Preparing an experiment step by step

1. **Choose the workflow** from the [table above](#available-workflows).
2. **Prepare the data** in the format the workflow expects (see the examples). It can be a dataset on the
   Hugging Face Hub, a local dataset directory, or a single `.json`/`.jsonl`/`.csv`/`.txt` file. Training
   workflows use the `train` split for training and the `validation` split for evaluation (names are
   configurable via `train_split`/`validation_split`).
3. **Create the experiment directory** and generate a configuration template into it:
   ```bash
   mkdir -p experiments/my_project/my_run && cd experiments/my_project/my_run
   mlworkflows make_config ~/mlworkflows/mlworkflows/workflows/train_sequence_labeling.py > train_sequence_labeling.yaml
   ```
4. **Edit the YAML**: at minimum the `dataset`, `model` and the `training` section (batch size, learning rate,
   evaluation/saving strategy, `output_dir`, `run_name`, ...).
5. **Add `accelerate.yaml`** (for multi-GPU) and a small launcher script `train.sh`:
   ```bash
   #!/bin/bash
   accelerate launch --config_file accelerate.yaml ~/mlworkflows/run.py \
       run ~/mlworkflows/mlworkflows/workflows/train_sequence_labeling.py \
       -c train_sequence_labeling.yaml 2>&1 | tee log.txt
   ```
6. **Run** `./train.sh`. To continue an interrupted run, set `resume_from_checkpoint: true` (the latest
   checkpoint in `output_dir`) or a path to a specific checkpoint.

## Examples

The YAML snippets below show the relevant parts only; keys that are not listed keep their defaults from the
generated template.

### 1. Sequence labeling (NER-like) with ModernBERT

Fine-tuning `answerdotai/ModernBERT-base` for software mention recognition on whole documents of
[SoFairOA/sofair_softcite_somesci](https://huggingface.co/datasets/SoFairOA/sofair_softcite_somesci), with
a 16k context window, on 2 GPUs.

**Input data.** Each sample must contain the raw text, its (word-level) tokens, character start offsets of
the tokens in the text and one label per token:

```json
{
  "text": "We used SPSS 24 for the analysis.",
  "tokens": ["We", "used", "SPSS", "24", "for", "the", "analysis", "."],
  "start_offsets": [0, 3, 8, 13, 16, 20, 24, 32],
  "labels": [0, 0, 1, 3, 0, 0, 0, 0]
}
```

The workflow tokenizes `text` with the model tokenizer and aligns the sub-word tokens to the given tokens
via the offsets; only the first sub-word of each token receives the label, the rest is ignored in the loss.
Label names are read from the dataset features (`Sequence(ClassLabel(names=[...]))`), so with Hub datasets
nothing more is needed. With plain JSON files that store label ids, list the label names in
`model.config.labels` (position = id), e.g. `labels: [O, B-SOFTWARE, I-SOFTWARE, B-VERSION, I-VERSION]`.
The field names are configurable (`text_field`, `tokens_field`, `start_offsets_field`, `labels_field`).
A fast tokenizer is required.

**Experiment directory**

```
sofair_softcite_somesci/
├── accelerate.yaml
├── train_sequence_labeling.yaml
└── train.sh
```

`accelerate.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
gpu_ids: all
mixed_precision: bf16
num_machines: 1
num_processes: 2
machine_rank: 0
main_training_function: main
rdzv_backend: static
same_network: true
use_cpu: false
```

`train_sequence_labeling.yaml` (relevant parts):

```yaml
dataset:
  cls: DatasetFactory
  config:
    path: SoFairOA/sofair_softcite_somesci
    name: sofair_softcite_somesci_documents
    train_split: train
    validation_split: validation
    test_split: test
model:
  cls: TokenClassificationModelFactory
  config:
    model_path: answerdotai/ModernBERT-base
    attn_implementation: flash_attention_2
    labels:            # empty = read from the dataset
    config:            # overrides of the model configuration
      max_position_embeddings: 16384
      global_rope_theta: 510000
tokenizer:
  cls: TokenizerFactory
  config:
    name_or_path: answerdotai/ModernBERT-base
    max_length: 16384
training:
  output_dir: ./training
  logging_dir: logs
  run_name: softcite_whole_documents_modern_bert
  do_train: true
  do_eval: true
  eval_strategy: steps
  eval_steps: 100
  save_strategy: steps
  save_steps: 100
  logging_steps: 10
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 8
  max_steps: 10_000
  lr_scheduler_type: cosine
  warmup_steps: 100
  bf16: true
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
compute_metrics:
  cls: ComputeMetrics
  config:
    path: evaluate-metric/seqeval
early_stopping:
  cls: EarlyStopping
  config:
    early_stopping_patience: 3
    early_stopping_threshold: 0.0
optimizer_factory:      # has priority over the optimizer settings in `training`
  cls: LionFactory
  config:
    learning_rate: 2e-05
    beta_1: 0.9
    beta_2: 0.99
    weight_decay: 0.0
text_field: text
labels_field: labels
tokens_field: tokens
start_offsets_field: start_offsets
truncation: true
save_model: best        # the best model is saved here in `from_pretrained` format
```

`train.sh`:

```bash
#!/bin/bash
accelerate launch --config_file accelerate.yaml ~/mlworkflows/run.py \
    run ~/mlworkflows/mlworkflows/workflows/train_sequence_labeling.py \
    -c train_sequence_labeling.yaml > log.txt
```

After the run, the directory contains `training/checkpoint-*`, TensorBoard logs in `logs/`, the training
log with seqeval metrics per entity type in `log.txt`, and the final best model in `best/`, which can be
used directly as `model_path` of the `predict_sequence_labeling.py` workflow.

### 2. Sequence classification on local JSONL files

Sentiment classification from local files. The model input is assembled by a **Jinja2 template**, so any
combination of fields can be used without changing the code (e.g. `"{{title}}\n\n{{abstract}}"`).

`data/train.jsonl` (and analogously `data/validation.jsonl`):

```json
{"id": 0, "text": "I love this product! It's amazing.", "sentiment": "positive"}
{"id": 1, "text": "This is the worst experience I've ever had.", "sentiment": "negative"}
{"id": 4, "text": "It's okay, not great but not terrible either.", "sentiment": "neutral"}
```

If `dataset.config.path` points directly to a single `.jsonl` file, it is loaded as the `train` split only;
in that case set `do_eval: false` or provide a validation split as below.

`train_sequence_classification.yaml` (relevant parts):

```yaml
dataset:
  cls: DatasetFactory
  config:
    path: json
    data_dir: data            # contains train.jsonl and validation.jsonl
model:
  cls: SequenceClassificationModelFactory
  config:
    model_path: answerdotai/ModernBERT-base
    labels: [negative, neutral, positive]   # empty = collected from the training data
input_template: "{{text}}"
labels_field: sentiment
transform_labels: true        # labels in the data are names, convert them to ids
training:
  output_dir: ./training
  num_train_epochs: 3
  per_device_train_batch_size: 16
  eval_strategy: epoch
  save_strategy: epoch
  load_best_model_at_end: true
  metric_for_best_model: accuracy
compute_metrics:
  cls: ComputeMetrics
  config:
    path: accuracy
optimizer_factory:
  cls: AdamWFactory
  config:
    learning_rate: 2e-05
save_model: best
```

```bash
mlworkflows run ~/mlworkflows/mlworkflows/workflows/train_sequence_classification.py -c train_sequence_classification.yaml
```

### 3. Pseudo-perplexity evaluation of a masked LM

Computes pseudo-perplexity of `chandar-lab/NeoBERT` on a random subset of English Wikipedia and stores the
per-sample values for later analysis.

`config.yaml` (relevant parts):

```yaml
dataset:
  cls: DatasetFactory
  config:
    path: wikimedia/wikipedia
    name: 20231101.en
split: train
model:
  cls: MaskedLanguageModelFactory
  config:
    model_path: chandar-lab/NeoBERT
    trust_remote_code: true
input:
  template: "{{text}}"
random_seed: 42
random_subset_size: 100        # number of samples, or a fraction in (0, 1)
batch_size_per_device: 4
pseudo_perplexity_per_sample: ppl_per_sample.jsonl
```

`run.sh`:

```bash
#!/bin/bash
accelerate launch --config_file accelerate.yaml ~/mlworkflows/run.py \
    run ~/mlworkflows/mlworkflows/workflows/evaluations/pseudo_perplexity.py -c config.yaml 2>&1 | tee log.txt
```

### 4. Pre-tokenizing a corpus and masked language modeling

For large corpora, tokenize once with `tokenize_dataset.py` and train on the saved result:

```yaml
# tokenize.yaml
dataset:
  cls: DatasetFactory
  config:
    path: wikimedia/wikipedia
    name: 20231101.en
split: train
tokenizer:
  cls: TokenizerFactory
  config:
    name_or_path: answerdotai/ModernBERT-base
input:
  template: "{{title}}\n\n{{text}}"
num_proc: 16
result_path: tokenized_wiki      # saved with `save_to_disk`
columns: [input_ids]
```

Then, in the configuration of `train_masked_language_modeling.py`, load it with:

```yaml
dataset:
  cls: DatasetFactory
  config:
    path: tokenized_wiki
    load_saved: true      # use `load_from_disk`
    preprocessed: true    # skip tokenization in the training workflow
selection_prob: 0.3       # fraction of tokens selected for MLM
masking_prob: 1.0         # selected tokens replaced by [MASK]
random_prob: 0.0          # selected tokens replaced by a random token
warmup_masked_lm_steps: 0 # linear warm-up of selection_prob
```

The saved dataset is a single split, so it is used entirely for training (set `do_eval: false`).

## Running tests

```bash
python -m unittest discover -s tests
```
