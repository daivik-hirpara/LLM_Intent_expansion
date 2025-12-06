# Intent Expansion Pipeline

A scalable, AI-powered pipeline to discover new intents from customer support messages.

## Features
- **Hybrid Approach**: Combines `SentenceTransformers` + `AgglomerativeClustering` for scalability with an LLM-based "Critic" workflow for high-precision validation.
- **Automated Reporting**: Generates a JSON report of Overloaded, Missing, and Ambiguous intents.
- **Documentation**: Includes a script to generate a comprehensive `.docx` report.

## Setup
1. `pip install -r requirements.txt`
2. Set `GEMINI_API_KEY` in `.env`
3. Run `python intent_expansion_pipeline.py`

## Output
- `intent_expansion_report.json`: Structured findings.
- `Intent_Expansion_Approach.docx`: Executive report.
