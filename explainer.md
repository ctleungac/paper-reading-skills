# Paper Analysis Agent

This is a **Paper Analysis Agent** - an AI-powered system that reads, analyzes, and explains academic papers.

## Core Capabilities

The agent has 5 skills for processing research papers:

| Skill | Purpose |
|-------|---------|
| **paper-reader** | Downloads papers from arXiv, URLs, or local PDFs and extracts text |
| **paper-summarizer** | Generates summaries at brief, standard, or detailed levels |
| **paper-analyzer** | Evaluates methodology, contributions, strengths and weaknesses |
| **paper-explainer** | Translates complex content into plain language (ELI5 style) |
| **paper-reproducer** | Generates code to implement the paper's methods |

## How It Works

The system uses a **skill-based architecture** where:
1. Skills are defined as markdown files (`SKILL.md`) with instructions
2. The LLM dynamically selects which skill to use based on user requests
3. Skills can run scripts (like `download_paper.py` for PDF extraction)

## Two Modes

- **Interactive**: Chat with the agent, ask for specific analyses
- **Pipeline**: Run all 5 steps automatically on a paper and save a report

```bash
# Interactive
python skill_agent.py

# Full automated analysis
python skill_agent.py arxiv:2301.00001
```

## Tech Stack

- Uses Gemini (via OpenAI-compatible API)
- PyMuPDF for PDF text extraction
- arxiv library for paper downloads
- Reports saved as markdown to `paper_reports/`
