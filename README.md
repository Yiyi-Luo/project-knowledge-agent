# Project Knowledge Agent

A stateful LLM-powered system that turns AI conversation exports into structured, persistent project knowledge.

The agent processes newly added conversation content, detects and routes topics by project, and maintains planning, learning, architecture, and code-state documentation in an Obsidian-compatible vault. It combines deterministic Python processing with LLM-based classification and summarization, while keeping the user in control through human review checkpoints.

## Why this project exists

Long-running AI-assisted projects often lose context across sessions. Important decisions, implementation details, open questions, and lessons can become scattered across conversation histories.

Project Knowledge Agent addresses that problem by converting conversation history into a maintained project memory that both humans and AI assistants can reuse.

## Current capabilities

- Incremental conversation processing using session markers with anchor-based fallback
- LLM-based project and topic detection
- Human-in-the-loop approval, reclassification, and reassignment
- Project-level planning, learning, current-state, and project-map generation
- Merge-based knowledge updates that preserve previously documented context
- Structured change-record extraction
- Pre-write project snapshots for rollback
- Exploration-only project workflows and transition into technical projects
- AST-assisted source-code indexing
- Human-readable repository and function trees
- File summarization for Markdown, text, Python, and PDF documents
- Per-call token usage and estimated cost logging

## Architecture

```text
Conversation export
        |
        v
Incremental content extraction
        |
        v
LLM topic and project detection
        |
        v
Human review checkpoint
        |
        v
Project routing and grouping
        |
        v
+-------------------------------+
| Project-level summarization   |
| - Planning                    |
| - Learning                    |
| - Current state               |
| - Project map                 |
| - Exploration notes           |
| - Structured changes          |
+-------------------------------+
        |
        v
Snapshot, merge, and vault write
        |
        v
Obsidian-compatible knowledge base
```

## Repository structure

```text
project-knowledge-agent/
├── src/
│   └── project_knowledge_agent/
│       ├── __init__.py
│       ├── pipeline.py
│       ├── extractors.py
│       ├── summarizers.py
│       ├── indexing.py
│       ├── vault.py
│       └── config.py
├── tests/
├── examples/
│   └── sample_vault/
├── docs/
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

The current working implementation may initially remain in a single script. The modular layout above is the intended public repository structure as the code is cleaned and separated.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/project-knowledge-agent.git
cd project-knowledge-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

### 3. Install the package

```bash
pip install -e ".[dev]"
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and add your own API key and local vault path.

```bash
cp .env.example .env
```

Never commit `.env` or any file containing credentials.

## Example workflow

1. Export or paste an AI conversation into the inbox file.
2. Run the processing pipeline.
3. Review detected projects and topics.
4. Approve, reclassify, or reassign topics.
5. Allow the agent to update the project knowledge base.
6. Review generated planning, learning, code-state, and project-map files.

## Privacy and security

This repository should contain only synthetic or user-owned examples.

Do not commit:

- API keys, tokens, or credentials
- Private conversation histories
- Employer or client data
- Taxpayer or personally identifiable information
- Paid-source content or proprietary documentation
- Absolute local file paths
- Internal server, proxy, or network details

## Project status

Active development.

The core pipeline is functional. Current work focuses on repository cleanup, modularization, testing, documentation, and creation of synthetic examples suitable for public demonstration.

## Roadmap

- [ ] Move the current implementation into the public package structure
- [ ] Add unit tests for extraction, parsing, routing, and merge safeguards
- [ ] Add a synthetic end-to-end example
- [ ] Add a command-line interface
- [ ] Add architecture diagrams and screenshots
- [ ] Improve configuration management
- [ ] Add structured logging
- [ ] Add continuous integration

## Author

Yiyi Luo
