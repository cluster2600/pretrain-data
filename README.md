# Pretrain Data Pipeline

A scalable data processing pipeline for preparing high-quality pretraining datasets. Built on [DataTrove](https://github.com/huggingface/datatrove), it provides multi-stage filtering, annotation, and formatting for large-scale text corpora — designed to run on HPC clusters via SLURM.

## Pipeline Stages

Documents flow through configurable stages in sequence:

```
ParquetReader
  → Robots.txt / ID Filtering
  → Embedding-Based Quality Filtering
  → Toxicity Scoring & Filtering
  → PII Redaction
  → (Optional) Sampling
  → ParquetWriter
```

Each stage is optional and independently configurable per dataset.

## Project Structure

```
pretrain-data/
├── src/data_pipeline_pretrain/
│   ├── executor/
│   │   └── slurm_nodes.py            # SLURM array job orchestration
│   ├── pipeline/
│   │   ├── annotators/
│   │   │   └── xlmr_embedding_annotator.py  # XLM-RoBERTa embedding generation
│   │   ├── filters/
│   │   │   ├── code_quality_filter.py       # Code metrics & quality thresholds
│   │   │   ├── embeddings_filter.py         # Binary classifier on embeddings
│   │   │   ├── robots_filter.py             # Robots.txt compliance & ID filtering
│   │   │   └── toxic_filter.py              # Multilingual toxicity detection
│   │   ├── formatters/
│   │   │   └── pii_formatter.py             # PII detection & replacement
│   │   └── tokens/
│   │       └── megatron_tokenizer.py        # Megatron .bin/.idx tokenisation
│   ├── utils/                               # File listing & asset path helpers
│   └── assets/pii/eu_regex.xlsx             # EU-specific PII regex patterns
├── pipelines/                               # Per-dataset processing scripts
│   ├── fineweb/
│   ├── fineweb-2/
│   ├── fineweb-edu/
│   ├── finemath/
│   ├── megamath/
│   ├── euroblocks/
│   ├── europarl/
│   ├── dclm-edu/
│   ├── paradocs/
│   ├── gutenberg/
│   └── provenance-flan/
└── examples/                                # Standalone example pipelines
    ├── code_pipeline/                       # LLM-assisted code quality annotation
    ├── xlmr_embedding_annotator/            # Embedding generation examples
    └── tokenize_megatron/                   # Megatron format tokenisation
```

## Core Components

### Quality Filtering

A two-layer neural network (`BinaryClassifier`: 768 → 256 → 1) scores documents using pre-computed XLM-RoBERTa embeddings. The threshold is estimated from the top-*p* percentile of a sample — e.g. `p=0.1` retains the top 10% by quality score.

### Toxicity Detection

`RobertaClassifier` wraps XLM-RoBERTa-base with a classification head for multilingual toxicity scoring. Language-specific models and thresholds are supported (e.g. 0.999 for English, 0.595 for Chinese). The `ToxicScorer` annotates documents; `ToxicityBinaryClassifierFilter` removes those above a given threshold.

### PII Redaction

`PIIFormatter` detects and replaces:

| Type | Replacement Token |
|------|------------------|
| Email addresses | `<email-pii>` |
| IPv4 addresses | `<ip-pii>` |
| IBANs | `<iban-pii>` |
| EU-specific identifiers | `[REDACTED]` |

EU patterns are loaded from an Excel spreadsheet with configurable priority tiers (P0/P1/P2). PII counts are tracked in document metadata.

### Robots.txt Compliance

`RobotsTxtFilter` checks URLs against cached robots.txt rules for a configurable set of user agents (AI2Bot, Bytespider, CCBot, GPTBot, and others). Documents violating any agent's rules are filtered out.

### Embedding Generation

`XLMRobertaEmbeddingAnnotator` generates document-level embeddings using mean pooling over XLM-RoBERTa token outputs (bfloat16 inference, max 512 tokens). Embeddings are stored in document metadata for downstream quality classification.

### Megatron Tokenisation

`MegatronDocumentTokenizer` converts text to Megatron-compatible `.bin`/`.idx` binary format with configurable tokenisers and special tokens (`<BOS>`, `<EOS>`).

## Supported Datasets

| Dataset | Description | Languages |
|---------|-------------|-----------|
| **FineWeb** | Web-crawled text (v1.3.0) | English |
| **FineWeb 2** | Multilingual web text (v2.0.1) | 20 languages incl. German, French, Chinese, Japanese |
| **FineWeb-EDU** | Educational web content | English |
| **FineMath** | Mathematical content (3+ and 4+ subsets) | English |
| **MegaMath** | Mathematical content | English |
| **EuroBlocks** | European language blocks | Multiple |
| **Europarl** | European Parliament proceedings | Bidirectional pairs |
| **DCLM-EDU** | Educational content | English |
| **Paradocs** | Parallel document collections | Multiple |
| **Gutenberg** | Project Gutenberg books | English |
| **Provenance-FLAN** | Instruction-tuning data | English |

## Configuration

Pipeline scripts use a `CONFIGS` dictionary to define which stages to apply. Common configurations for FineWeb:

| Config | Quality Filter | Robots.txt | Toxicity |
|--------|---------------|------------|----------|
| `quality_10-keeprobots` | Top 10% | Disabled | Yes |
| `quality_10-filterrobots` | Top 10% | Enabled | Yes |
| `quality_33-keeprobots` | Top 33% | Disabled | Yes |
| `quality_33-filterrobots` | Top 33% | Enabled | Yes |
| `only-quality_10` | Top 10% | Disabled | No |
| `keeprobots` | None | Disabled | No |
| `filterrobots` | None | Enabled | No |

## Usage

### Running a Pipeline

```bash
# FineWeb with top-10% quality filter, robots.txt disabled
python pipelines/fineweb/main.py quality_10-keeprobots 0

# FineWeb 2 (multilingual) with robots.txt filtering
python pipelines/fineweb-2/main.py quality_10-filterrobots deu_Latn 0

# Code quality annotation for Python
python examples/code_pipeline/code_pipeline.py --language python

# Generate XLM-RoBERTa embeddings for FineWeb
python examples/xlmr_embedding_annotator/main_fineweb.py

# Tokenise a HuggingFace dataset to Megatron format
python examples/tokenize_megatron/preprocess_megatron.py \
    --tokenizer-name-or-path meta-llama/Meta-Llama-3-8B \
    --output-folder datasets/emotion --n-tasks 16 \
    hf --dataset dair-ai/emotion
```

### SLURM Execution

`SlurmPipelineNodeExecutor` handles distributed execution across cluster nodes:

- Automatic task array splitting for large jobs (configurable `max_array_size`)
- Job dependency management (`afterok`/`afterany` semantics)
- Signal-based requeuing on node preemption (`SIGUSR1`)
- Configurable staggered launches, randomised start delays, and email notifications
- Per-rank logging and post-processing statistics merging

### Custom Configuration

```python
CONFIGS = {
    "custom_config": {
        "robots_filter": True,
        "quality_filter": {"p": 0.2},
        "toxicity_filter": {"threshold": 0.95},
    }
}
```

## Output

Processed datasets are written in Parquet format:

- **Filtered content** — documents that passed all pipeline stages
- **Excluded content** — separately stored for analysis, with exclusion reasons
- **Metadata** — quality scores, toxicity scores, PII counts, and processing statistics
- **Logs** — per-task execution logs and merged statistics

## Dependencies

- [DataTrove](https://github.com/huggingface/datatrove) — core pipeline framework
- [PyTorch](https://pytorch.org/) — model inference
- [Transformers](https://github.com/huggingface/transformers) — XLM-RoBERTa tokeniser and model
- [FastText](https://fasttext.cc/) — text classification (code quality pipeline)
- [Pandas](https://pandas.pydata.org/) — data manipulation and EU PII pattern loading
- [vLLM](https://github.com/vllm-project/vllm) — LLM inference for code annotation

## Licence

[Apache Licence 2.0](LICENSE)
