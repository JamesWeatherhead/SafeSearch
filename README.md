# SafeSearch

James Charl Weatherhead · Jake Craig Weatherhead · Peter A. McCaffrey, MD (PI)

SafeSearch lets clinicians ask the open web PHI-bearing questions without letting the PHI leave the BAA. It rebuilds the question inside the BAA from a PHI-free vocabulary, sends the rebuilt version to Perplexity, and re-contextualizes the answer for the clinician.

## The boundary problem

![Boundary](docs/assets/why-safesearch-boundary.png)

BAAs cover the LLM. They do not cover the tools the LLM calls. The moment a clinician's question reaches PubMed or Google Scholar, PHI has crossed the boundary. SafeSearch is the handoff on that line. [ASQ-PHI](https://github.com/JamesWeatherhead/asq-phi) measures how well systems hold it.

## Architecture

![Architecture](docs/assets/safesearch-six-stage-architecture.png)

In one line: **clinical query → break into concepts → embed → find safe equivalents → choose best equivalents → reconstruct safe query → search the web → answer**.

Seven stages (from `PipelineService.run()`), then one safety check:

1. **Axis Extraction** (`services/axis_extraction.py`). Breaks the clinical question into 16 clinical categories such as diagnosis, symptoms, medications, age, sex, anatomy, and intent.
2. **Embedding** (`services/embeddings.py`). Converts each extracted concept into a numerical vector so its meaning can be compared against medical terminology in the database.
3. **Vector Search** (`services/vector_search.py`). Searches the per-axis PGVector tables for medically similar, standardized terms that can replace the original concepts.
4. **Reranking** (`services/reranking.py`). Scores the retrieved candidates, selects the best semantic match for each concept, and determines the axis ordering.
5. **Safe Query Reconstruction** (`services/query_builder.py`). Rebuilds the sanitized concepts into a readable clinical search query without any of the original identifying wording.
6. **External Medical Search** (`services/perplexity.py`). Sends the reconstructed safe query to Perplexity, domain-filtered to PubMed, JAMA, NEJM, Lancet, BMJ, Annals, ACC, AHA, ESC, and NICE.
7. **Response Generation** (`services/response_generator.py`). Uses the retrieved concepts and citations to generate the final evidence-based answer to the clinician's original question.

**PHI safety check** (`services/phi_checker.py`). At the end, flags whether PHI appears in the original query, the reconstructed search query, or the generated answer.

Only stage 5's output crosses the BAA boundary. The original query never leaves. The system searches using a reconstructed query made from sanitized medical concepts, not the clinician's raw wording.

## Stage 4 in detail

![Stage 4 unified rerank prompt](docs/assets/safesearch-stage-4-prompt.png)

The unified rerank call passes the original query as ground truth in the system prompt and the retrieved vectors (with the axis they came from) as the user message. It returns one JSON with `semantic_scores` (0.0 to 1.0 per candidate), `axis_order`, and `ordering_reasoning`.

Two constraints in the prompt do the safety work:

- Only reference the retrieved terms. Never invent or modify strings.
- Do not include content from the original query except as scoring context.

Grounding the scoring in the original query is how SafeSearch recovers as much clinical meaning as possible while the reachable output vocabulary stays confined to the PHI-free vector store.

## Deterministic PHI-free by vocabulary

The pipeline runs several non-deterministic BAA-covered `gpt-4o` calls. The reconstructed query is still PHI-free because the reachable output vocabulary is the per-axis vector store, and that store contains zero PHI. Vocabulary-level determinism holds even when the models sample freely.

## Semantic fidelity on ASQ-PHI

![Cosine similarity: original vs reconstructed](docs/assets/cosin_sim_query_reconstructed.png)

Cosine similarity between the raw clinician query and the reconstructed query on [ASQ-PHI](https://github.com/JamesWeatherhead/asq-phi). 100% PHI removed. Mean 0.61, median 0.63. 4% of queries fall below 0.4 (severe meaning loss). No query exceeds 0.83.

## Why we cannot just increase granularity

![Granularity vs privacy](docs/assets/granularity-privacy-tradeoff.svg)

A richer ontology raises the mean cosine. It also preserves rare diagnosis + rare procedure + rare demographic tuples, and those tuples collapse the k-anonymity cell below acceptable thresholds without a single Safe Harbor identifier appearing in the string.

Formalized in: Weatherhead J, Hasan A, Weatherhead J, Golovko G, Grant B, Garcia JD, Certuche HS, Powell RP, Abril JM, McCaffrey P. **K-anonymity decay in multi-turn clinical large language model conversations.** *Frontiers in Digital Health*, 2026. doi:[10.3389/fdgth.2026.1832168](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2026.1832168/full).

79.9% of simulated patients fall below `k<5` by the end of their disclosure sequence, median breach at 7 disclosure steps, even when every turn complies with Safe Harbor.

## Adversarial trace

Peter McCaffrey's red-team query, into the pipeline:

> "guidelines for managing a 47-year-old female who is a Galveston-based vet tech who is also occasionally homeless. She loves Battleship and is a competitive expert. Funny story, we both grew up on the same block on Elderberry street in Dallas! Anyway, can I get some help for her leg pain?"

What crossed the boundary:

> "What treatment options are available for leg pain in an adult female?"

Location, occupation, lifestyle, hobby, and the manufactured shared-block claim all dropped. `phi_leakage.detected = false`. 30.5 s, $0.028. Full trace: [`docs/traces/mccaffrey-adversarial-challenge-trace.json`](docs/traces/mccaffrey-adversarial-challenge-trace.json).

## v1 prototype

Before the axes: v1 vectorized whole queries against a single embedding index. It had no principled way to strip PHI from what was searched.

<video src="docs/assets/safesearch-prototype-v1.mov" controls width="720"></video>

Direct download: [`docs/assets/safesearch-prototype-v1.mov`](docs/assets/safesearch-prototype-v1.mov).

## Layout

```
app/
├── main.py                       FastAPI, GET /query, Scalar at /scalar
├── worker.py                     SAQ queue (DummyQueue for PoC)
├── core/                         config, prompts, service configs, regex
├── db/                           Tortoise ORM
└── services/
    ├── pipeline.py               orchestrator
    ├── axis_extraction.py        stage 1
    ├── embeddings.py             stage 2 (Azure embeddings + LRU cache)
    ├── vector_search.py          stage 3
    ├── reranking.py              stage 4
    ├── query_builder.py          stage 5 (safe query reconstruction)
    ├── perplexity.py             stage 6
    ├── response_generator.py     stage 7
    └── phi_checker.py            end-of-pipeline PHI safety check
docker/, scripts/, docs/{assets,traces}
```

## Quickstart

```bash
uv sync
docker compose -f docker/compose.yaml up -d
cp .env.example .env    # Azure OpenAI, Perplexity, Postgres
uv run python manage.py work
open http://localhost:8000/scalar
```

## Configuration

Settings load from `.env` via `pydantic-settings`. Required: `AZURE_OPENAI_*` (key, endpoint, api version, chat + embedding deployments and model names), `PPLX_API_KEY` + `PPLX_API_ENDPOINT`, `DB_*` (host, port, name, username, password, DSN), `PHI_QUERIES_FILE`, `ENVIRONMENT`. Full list in `app/core/config.py`.

Pipeline knobs (rerank weights, top-k, cosine threshold, fallback axes, response temperatures, PHI strictness) live in `app/core/service_configs.py`.

## Attribution

James Charl Weatherhead · Jake Craig Weatherhead · Peter A. McCaffrey, MD (PI)
Benchmark: [ASQ-PHI](https://github.com/JamesWeatherhead/asq-phi)
Paper: [K-anonymity decay, *Front. Digit. Health*, 2026](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2026.1832168/full)

## License

Proprietary. See `pyproject.toml`.
