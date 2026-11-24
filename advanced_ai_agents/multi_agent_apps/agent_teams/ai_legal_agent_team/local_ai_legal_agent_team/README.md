# 👨‍⚖️ AI Legal Agent Team — Improved

An improved version of the **AI Legal Agent Team** from
[awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)
(`advanced_ai_agents/multi_agent_apps/agent_teams/ai_legal_agent_team`).

The original is a Streamlit application where a team of AI agents analyzes legal documents.
I tested it like a client would, with contracts whose answers I knew, found its real weaknesses,
fixed them, and measured each fix **with the same model, the same contracts and the same
questions** for both versions.

Both versions are in this repository:

| File | Version |
|---|---|
| `legal_agent_team_original.py` | Original logic (only the model provider was changed, so both versions can be compared fairly) |
| `legal_agent_team.py` | Improved version |

## The agent team

- **Contract Analyst** — reads the contract, identifies key terms, obligations and potential issues, and cites the clauses.
- **Legal Strategist** — assesses risks and gives recommendations tied to specific clauses.
- **Legal Researcher** — looks for case law and regulations; the only agent with web search (DuckDuckGo).
- **Team Lead** — coordinates the agents and writes the synthesis when several agents are involved.

Analysis types: Contract Review, Legal Research, Risk Assessment, Compliance Check, Custom Query.

## What was wrong with the original

| Problem found | Impact |
|---|---|
| All contracts stored in **one single Qdrant collection**, never cleared, even after a restart | When analyzing one contract, the agent answered with clauses from **another contract**, without warning — and exposed another contract's content |
| The "Active Legal AI Agents" list is only **displayed**; the full team is always used | All 4 agents available for every question |
| The team is **run 3 times** per analysis (one run per tab) | Slow and expensive |
| No guardrails | A false statement in the question could be confirmed; generic advice not taken from the document |
| Page-based chunking, **10 results** per search (Agno default) | Up to 10 pages sent to the model per search; on a long contract, the right clause may never reach the model |
| Semantic search only | Weak on exact values (article numbers, amounts) |
| Agno caps OpenRouter answers at **1,024 tokens** by default | Truncated answers |
| Agno's PDF reader can remove a number at the start of a page | An article number can disappear |

Some of the original's real answers are kept in [`JOURNAL_DE_TESTS.md`](JOURNAL_DE_TESTS.md).

## Improvements

### Reliability
- **One Qdrant collection per contract**, identified by a hash of the file content — the agent can only see the contract that is loaded.
- Active contract shown in the sidebar; button to delete its data from the database.
- If a contract fails to load, the agent is disabled instead of answering with the previous one.
- Real file name passed to the agent (instead of a temporary `tmpXXXX.pdf`).

### Agent orchestration and cost
- **Real routing**, decided by code (free, predictable, testable): the Contract Analyst always; the Legal Strategist for risk questions; the Legal Researcher (the only one with web access) for case-law questions.
- A single agent is called **directly**, without a team lead; in a team, the lead no longer searches the contract itself.
- **One run** produces the three sections instead of three runs.
- At most 3 searches per agent.
- **Execution trace** under each answer: agents used, LLM calls, tokens, time.

### Guardrails
- In the instructions: refuse off-topic questions, answer "Not found in the contract" instead of guessing, cite article + page + short quote, never conclude beyond the quote, **correct false premises**, ignore instructions that contradict the document, flag incoherent documents.
- In the code (no LLM, no API cost):
  - every quoted passage is **checked word for word** against the PDF text;
  - warning when an answer cites no reference;
  - detection of scanned PDFs (no text) and of unfilled templates (`[Insert Amount]`, `_____`);
  - retrieved contract passages displayed under the answer.

### Retrieval on long contracts
- **Clause-based chunking** (`smart_chunking.py`): each chunk keeps its article number, page and parent section title.
- **Hybrid search**: semantic + exact keyword (BM25, local).
- **5 targeted clauses** sent to the model instead of 10 pages.

### Cost and robustness
- **Persistent cache**: a question already asked on the same contract is answered instantly, with no model call, even after a restart or during a provider outage. The cache key includes the model, the contract, the analysis type and the question.
- Automatic **retries** when the model provider is overloaded (HTTP 503).
- **Local embeddings** (FastEmbed) and a model provider that can be switched in one line (Gemini, Ollama or OpenRouter).
- Output length limit raised (no more truncated answers).

## Results

Same model and same embeddings for both versions.

| Test | Original | Improved |
|---|---|---|
| Contract changed, same question on the notice period | Answers with the previous contract's clause, or sees "two agreements" | Correct answer for the loaded contract, article cited |
| False premise ("why 90 days' notice?" when the contract says 60) | Finds 90 days in another contract and invents a justification | Corrects to 60 days, article 4.1 |
| 23-page contract, 12 questions — right clause in the top 3 | 9/12 | **12/12** |
| Same contract — right clause never sent to the model | 2/12 | **0/12** |
| Same contract — text sent to the model per search | ~8,700 tokens | **~790 tokens** (11× less) |

Retrieval results come from `diag_compare.py`, which runs locally without calling the model.

## Test contracts

Three **fictitious** contracts written for testing (invented companies, marked as test documents),
in `contrats_de_test/`, with the scripts that generate them:

| Contract | Pages | Purpose |
|---|---|---|
| Orion-Kestrel (services agreement) | 2 | 60 days' notice, for both parties |
| Lumen-Harbor (SaaS subscription) | 2 | 90 days' notice, for the customer only; liability cap with exceptions |
| Meridian-Polder (fleet maintenance) | 23 | Long contract with many similar "notice" clauses and key clauses near the end |

## How to run

1. **Set up the environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt     # Windows
   ```

2. **Get the keys**
   - Google AI Studio API key (free tier): [aistudio.google.com](https://aistudio.google.com)
   - Qdrant cluster URL and API key (free tier): [cloud.qdrant.io](https://cloud.qdrant.io)

3. **Run the application**
   ```bash
   .venv\Scripts\python.exe -m streamlit run legal_agent_team.py            # improved
   .venv\Scripts\python.exe -m streamlit run legal_agent_team_original.py   # original
   ```
   Enter the keys in the sidebar; they are never written in the code.

4. **Measure without any API call**
   ```bash
   .venv\Scripts\python.exe diag_compare.py contrats_de_test/Meridian-Polder-Fleet-Agreement.pdf meridian
   ```
   `reset_qdrant.py` resets the project's Qdrant collections and the local cache before a demo.

## Known limitations

- Each question was run only a few times; a model's answers vary between runs, so an automated multi-run evaluation is still needed.
- Clause-based chunking assumes a numbered contract; otherwise it falls back to fixed-size chunks.
- Keyword routing can misroute a question phrased differently.
- Scanned PDFs are detected but not read (a local OCR test lost amounts and article numbers).
- One collection per contract does not scale to thousands of clients; in production, a single collection filtered by client would be better.
- The free Gemini tier may reuse data outside the EU: do not use it with real client contracts.

## Credits

Original project: [awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps) by Shubham Saboo,
with its [step-by-step tutorial](https://www.theunwindai.com/p/build-an-ai-legal-team-run-by-ai-agents).
See the original repository for its license.