# Pretrain Data Pipeline

A comprehensive data processing pipeline for preparing high-quality pretraining datasets. This project provides tools for filtering, annotating, and processing large-scale text datasets using quality classifiers, toxicity filters, PII removal, and embedding generation.

## Overview

The pretrain-data pipeline is designed to process large text corpora for language model training. It implements multiple filtering and annotation stages to ensure high data quality:

- **Quality Filtering**: Binary classifiers to filter content based on educational value, clarity, practice relevance, and difficulty
- **Toxicity Detection**: Multilingual toxicity classifiers to remove harmful content
- **PII Sanitization**: Detection and removal of personally identifiable information (emails, IPs, IBANs, EU-specific identifiers)
- **Embedding Generation**: XLM-RoBERTa embeddings for similarity-based filtering
- **Code Quality Assessment**: Specialized filters for code content evaluation
- **Robots.txt Compliance**: Filtering based on web crawling permissions

## Project Structure

```
pretrain-data/
├── src/data_pipeline_pretrain/          # Core pipeline components
│   ├── executor/                        # SLURM cluster execution
│   ├── pipeline/
│   │   ├── annotators/                  # Embedding and feature annotation
│   │   ├── filters/                     # Quality, toxicity, and content filters
│   │   ├── formatters/                  # PII removal and text formatting
│   │   └── tokens/                      # Tokenization utilities
│   └── utils/                          # File handling and asset utilities
├── pipelines/                          # Dataset-specific processing scripts
│   ├── fineweb/                        # FineWeb dataset processing
│   ├── finemath/                       # FineMath dataset processing
│   ├── euroblocks/                     # EuroBlocks dataset processing
│   └── ...                            # Other dataset pipelines
└── examples/                           # Example implementations
    ├── code_pipeline/                  # Code quality assessment pipeline
    ├── toxicity_filter/                # Toxicity detection examples
    └── xlmr_embedding_annotator/       # Embedding annotation examples
```

## Key Features

### 1. Multi-Stage Filtering Pipeline
- **Quality Classification**: Filters content based on educational value and clarity metrics
- **Toxicity Detection**: Removes harmful content using multilingual classifiers
- **Robots.txt Filtering**: Respects web crawling permissions
- **Code Quality Assessment**: Specialized evaluation for programming content

### 2. PII Detection and Removal
- Email addresses, IP addresses, and IBAN detection
- EU-specific identifier patterns (configurable via Excel spreadsheet)
- Configurable replacement tokens
- Metadata tracking of PII occurrences

### 3. Embedding-Based Processing
- XLM-RoBERTa embeddings for semantic similarity
- Batch processing for efficient GPU utilization
- Mean pooling for document-level representations

### 4. Scalable Execution
- SLURM cluster integration for distributed processing
- Configurable resource allocation (CPUs, memory, time limits)
- Job dependency management
- Automatic requeuing on node failures

## Configuration

The pipeline supports multiple predefined configurations:

- `quality_10-keeprobots`: 10% quality threshold, keep robots.txt filtered content
- `quality_33-filterrobots`: 33% quality threshold, apply robots.txt filtering
- `only-quality_10`: Only apply quality filtering (10% threshold)
- `keeprobots`/`filterrobots`: Only robots.txt filtering options

## Dataset Support

Current pipelines support processing:
- **FineWeb**: Web-crawled text data
- **FineMath**: Mathematical content
- **EuroBlocks**: European language blocks
- **DCLM-EDU**: Educational content
- **Paradocs**: Document collections
- **Europarl**: Parliamentary proceedings

## Dependencies

The project uses:
- **DataTrove**: Core pipeline framework
- **PyTorch**: Deep learning models
- **Transformers**: Hugging Face models (XLM-RoBERTa)
- **FastText**: Text classification
- **Pandas**: Data manipulation
- **SLURM**: Cluster job scheduling

## Usage

### Basic Pipeline Execution

```bash
# Run quality filtering pipeline on FineWeb
python pipelines/fineweb/main.py quality_10-keeprobots 0

# Process code quality assessment
python examples/code_pipeline/code_pipeline.py --language python
```

### Custom Configuration

Pipeline configurations can be customized by modifying the `CONFIGS` dictionary in the main scripts:

```python
CONFIGS = {
    "custom_config": {
        "robots_filter": True,
        "quality_filter": {"p": 0.2},
        "toxicity_filter": {"threshold": 0.95},
    }
}
```

### SLURM Integration

The pipeline automatically generates SLURM job scripts with configurable parameters:

- Time limits, partition selection
- CPU/memory allocation per task
- Job dependencies and requeuing
- Logging and monitoring

## Output

Processed datasets are saved in Parquet format with:
- **Filtered content**: High-quality text after all processing stages
- **Removed content**: Separately stored filtered-out data for analysis
- **Metadata**: Quality scores, PII counts, and processing statistics
- **Logs**: Detailed execution logs and statistics

## License

Apache License 2.0
