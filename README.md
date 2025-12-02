# EID-CoT: Entity ID-centric Chain-of-Thought

This repository contains the official implementation and the full text of the paper **"EID-CoT: Entity ID-centric Chain-of-Thought for Grounded Biomedical Reasoning"**.

## Abstract

Large Language Models (LLMs) have demonstrated significant potential in the biomedical domain but often suffer from hallucinations and ambiguity. **EID-CoT** addresses these challenges by grounding the reasoning process in standardized biomedical entity IDs (e.g., MeSH, HGNC). By constraining the model's intermediate reasoning steps to unique IDs rather than ambiguous natural language, we achieve:

- **Ambiguity Resolution**: Eliminating confusion between homonyms and synonyms (e.g., 'APC' as a gene vs. a medical procedure).
- **Grounded Reasoning**: Ensuring every reasoning step is backed by a verifiable knowledge base.
- **Hallucination Mitigation**: Structurally preventing the generation of non-existent entities.

## Repository Structure

The repository is organized as follows:

- **`src/`**: Core implementation of the EID-CoT pipeline, including modules for normalization, retrieval, and graph-based reasoning.
- **`paper/`**: The full text of the paper in Markdown format.
- **`experiments/`**: Scripts for running experiments and reproducing the results presented in the paper.
- **`resources/`**: Minimal datasets (Mini-sets) and resources required for demonstration and testing.
- **`configs/`**: Configuration files for the pipeline parameters.

## Installation

We recommend using **Python 3.13** or higher.

```bash
# 1. Clone the repository
git clone https://github.com/GYEONGMIN-CHO/EID-CoT.git
cd EID-CoT

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Configuration

This project utilizes OpenAI's GPT models as the reasoning backbone. You must configure your API key before running the pipeline.

Create a `.env` file in the root directory and add your `OPENAI_API_KEY`:

```bash
# Create .env file
echo "OPENAI_API_KEY=your_sk_key_here" > .env
```

> **Note**: The `.env` file is excluded from version control for security.

## Usage

### 1. Running the Pipeline (Demo)

To execute the EID-CoT pipeline on a single biomedical query:

```bash
PYTHONPATH=. python experiments/run_pipeline.py "TP53 is frequently mutated in colon cancer."
```

This will perform entity extraction, ID linking, and grounded reasoning, outputting the final answer along with the reasoning trace.

### 2. Evaluation (Mini-set)

To evaluate the model's performance on the provided mini-sets (e.g., BC5CDR, NCBI):

```bash
PYTHONPATH=. python experiments/run_eval_miniset.py bc5cdr
```

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@article{cho2024eidcot,
  title={EID-CoT: Entity ID-centric Chain-of-Thought for Grounded Biomedical Reasoning},
  author={Cho, Gyeongmin},
  year={2024}
}
```

## License

This project is licensed under the MIT License.
