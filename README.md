# SafeSearch

**SafeSearch is a privacy-preserving clinical search architecture for using external retrieval from a BAA-covered clinical LLM environment.** A PHI-bearing clinician query is interpreted inside the trusted environment, decomposed into clinical axes, and mapped by embedding retrieval to an allowlisted vocabulary derived from clinical ontologies such as UMLS, SNOMED CT, and RxNorm. The selected concepts are reranked, placed in clinically meaningful order, and passed to a separate query builder that never receives the original PHI-bearing text.

The key privacy property is **separation of interpretation from release**. The original query can be used inside the BAA-covered environment to preserve clinical meaning, but the outbound query is reconstructed only from concepts retrieved from the controlled vocabulary. Because source identifiers are not present in that vocabulary and the reconstruction model never receives the original text, there is no direct information path for the patient's original PHI to propagate into the external query.

> **SafeSearch does not redact the original query and send a modified copy. It constructs a new query from an allowlisted semantic representation.**

Conceptually, SafeSearch performs **retrieval for representation, followed by retrieval for evidence**:

`PHI-bearing query → clinical axes → allowlisted concepts → reconstructed query → external evidence → original context restored`

## Why the boundary matters

The [Health Insurance Portability and Accountability Act of 1996 (HIPAA)](https://www.hhs.gov/hipaa/index.html) governs protected health information (**PHI**) handled by covered healthcare organizations and their business associates. A **Business Associate Agreement (BAA)** can authorize a service to process PHI on behalf of a covered entity, but that authorization does not automatically extend to every search engine, API, or downstream tool the service may call. [[HHS: Business Associates](https://www.hhs.gov/hipaa/for-professionals/privacy/guidance/business-associates/index.html)] [[HHS: Business Associate Contracts](https://www.hhs.gov/hipaa/for-professionals/covered-entities/sample-business-associate-agreement-provisions/index.html)]

This creates the architectural problem SafeSearch addresses:

> **A clinical LLM may be allowed to see PHI. The external tool it calls may not be. What representation of the clinical question should cross that boundary?**

![The BAA boundary problem](docs/assets/why-safesearch-boundary.png)

HIPAA recognizes two methods of de-identification under [45 CFR § 164.514(b)](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html): **Safe Harbor**, which removes specified identifiers, and **Expert Determination**, which uses statistical and scientific analysis to establish a very small risk of identification.

SafeSearch addresses a related but different engineering problem. Instead of asking only **which parts of the source text should be removed or replaced**, it asks **whether the source representation needs to leave the trusted environment at all**.

That is the basis of **constrained semantic reconstruction**.

## Constrained semantic reconstruction

Conventional de-identification generally produces a modified copy of the source text. SafeSearch produces a new task-oriented representation.

**Redaction**  
`John Smith has leg pain → [NAME] has leg pain`

**Surrogate substitution**  
`John Smith has leg pain → Maria Lopez has leg pain`

**SafeSearch**  
`PHI-bearing clinical query → clinical concepts → controlled clinical representation → newly reconstructed query`

> **The objective is not textual preservation, but preservation of task-relevant clinical semantics across the privacy boundary.**

**Definition.** Constrained semantic reconstruction de-identifies a clinical query by decomposing it into structured clinical concepts, mapping those concepts onto a predefined controlled clinical vocabulary, and reconstructing a new query from the resulting representation rather than modifying PHI spans within the original text.

## Architecture at a glance

Seven stages implement this transformation: interpretation inside the trusted environment, controlled semantic reconstruction at the boundary, and evidence retrieval outside it.

1. **Clinical axis extraction**  
   Azure OpenAI, trusted. Decomposes the clinical query into 16 clinical axes.

2. **Embeddings**  
   Azure OpenAI, trusted. Converts each extracted concept into a semantic vector.

3. **Controlled vocabulary retrieval**  
   PostgreSQL + pgvector, trusted. Retrieves the nearest controlled clinical terms for each axis.

4. **Semantic reranking and axis ordering**  
   Azure OpenAI + hybrid score, trusted. Selects the best candidates and determines clinically meaningful ordering.

5. **Safe query reconstruction**  
   Azure OpenAI, trusted. Reconstructs a natural-language search query from the selected controlled terms.

6. **External evidence retrieval**  
   Perplexity `sonar-pro`, external. Retrieves current clinical evidence using the reconstructed query.

7. **Clinical re-contextualization**  
   Azure OpenAI, trusted. Generates the clinician-facing answer from original context, selected concepts, and returned evidence.

**PHI safety check.** An end-of-pipeline observability layer flags whether known PHI patterns appear in the original query, reconstructed query, or generated answer. See [`app/services/phi_checker.py`](app/services/phi_checker.py).

## Two retrievals

### 1. Retrieval for representation

Inside the trusted environment:

`clinical concept → embedding → pgvector → controlled clinical term`

The purpose is not to retrieve medical evidence. It is to find a controlled clinical representation of the source concept from the curated per-axis vocabulary.

### 2. Retrieval for evidence

After reconstruction:

`reconstructed query → external search → current medical evidence`

This is conventional information retrieval.

> [!NOTE]
> **Retrieval for representation, followed by retrieval for evidence.** Conventional retrieval-augmented generation (RAG) retrieves documents or chunks to provide additional knowledge to an LLM. SafeSearch's internal vector retrieval instead retrieves candidate representations used to transform the query itself. Only after that transformation does SafeSearch perform external retrieval for clinical evidence.

## Technical walkthrough

<details>
<summary><strong>Stage 1: Clinical axis extraction</strong></summary>

The original clinical query is processed inside the configured Azure OpenAI environment. The chat model decomposes the query into 16 clinical axes:

`age_bins`, `allergy_terms`, `anatomy_terms`, `comorbidity_terms`, `diagnosis_terms`, `family_history_terms`, `intent_terms`, `lifestyle_terms`, `procedure_terms`, `race_ethnicity`, `rxnorm_terms`, `severity_status`, `sex_terms`, `symptom_terms`, `temporal_context`, `wordlist_terms`.

Whole-query embedding would collapse everything into a single vector and search for a globally similar sentence. Axis decomposition separates different kinds of clinical information so each can be independently mapped to a controlled representation.

*Example.* A query about "guidelines for a 47-year-old female with leg pain, asking about treatment" is decomposed into roughly `age_bins: [adult]`, `sex_terms: [female]`, `anatomy_terms: [leg]`, `symptom_terms: [pain]`, `intent_terms: [treatment]`.

The extraction call runs at temperature 0.0 with JSON-mode enforced. Post-processing enriches `intent_terms` via a controlled synonym map, normalizes `age_bins` from age or DOB regexes, expands a small set of medication umbrellas into exemplar drugs, and drops substring duplicates. The original query is available at this stage because processing occurs on the trusted side of the boundary.

Sources: [`app/services/axis_extraction.py`](app/services/axis_extraction.py), [`app/core/prompts.py`](app/core/prompts.py) (`AxisExtractionPrompts.SYSTEM_EXTRACTION`), [`app/core/data_mappings.py`](app/core/data_mappings.py).

</details>

<details>
<summary><strong>Stage 2: Embeddings</strong></summary>

> An embedding converts a piece of text into a numerical vector representing features of its semantic meaning. Terms that the embedding model considers semantically similar tend to occupy nearby regions of the embedding space.

Each extracted concept `t` is embedded through the configured Azure OpenAI embedding deployment:

$$E(t) \in \mathbb{R}^{d}$$

Here `t` is an extracted clinical concept, `E` is the embedding model, and `d` is the vector dimensionality. The default configuration in [`.env.example`](.env.example) uses `text-embedding-3-large`; `d` is set by the configured deployment. This is an embedding API call, not generative LLM inference. The model returns a vector rather than a completion. Embedded terms are LRU-cached per process to avoid redundant round-trips.

Source: [`app/services/embeddings.py`](app/services/embeddings.py).

</details>

<details>
<summary><strong>Stage 3: Controlled vocabulary retrieval with PostgreSQL + pgvector</strong></summary>

> pgvector extends PostgreSQL with vector storage and nearest-neighbor search, allowing SafeSearch to ask which stored clinical terms are closest to an extracted concept in embedding space.

Each embedded concept is compared to the terms stored in the corresponding axis table. Each table contains precomputed embeddings for controlled clinical terms. Similarity is cosine similarity:

$$\cos(x,y)=\frac{x\cdot y}{\|x\|\|y\|}$$

Values closer to 1 indicate that the embedding model places the concepts in more similar semantic directions. pgvector implements this through its `<=>` cosine-distance operator; the code returns `1 - (vec <=> qvec)` as the similarity score.

SafeSearch retrieves the top **k = 3** nearest terms per axis at a **cosine threshold of 0.60**. For designated axes, low-similarity retrieval can broaden to the generic `wordlist_terms` table so unusual phrasings can still resolve to the controlled vocabulary.

**The controlled clinical vocabulary is organized into axis-specific vector tables containing clinical terms and their precomputed embeddings. At runtime, SafeSearch retrieves candidate representations from these tables rather than constructing the outbound query directly from the source text.**

Source: [`app/services/vector_search.py`](app/services/vector_search.py).

</details>

<details>
<summary><strong>Stage 4: Semantic reranking and axis ordering</strong></summary>

Vector similarity alone may return a geometrically nearby term that is not the best clinical replacement. SafeSearch therefore scores each candidate along three axes and selects the highest-scoring one per source concept.

$$S_{\text{hybrid}}(t, v \mid q) = w_c \cdot S_{\cos}(t, v) + w_l \cdot \frac{S_{\text{LLM}}(t, v \mid q)}{10} + w_x \cdot S_{\text{lex}}(t, v)$$

The weights in [`RerankingConfig`](app/core/service_configs.py) are `w_c = 0.65`, `w_l = 0.25`, `w_x = 0.10`.

- **Cosine similarity:** Are the concepts close in embedding space?
- **LLM semantic score:** Does this candidate make clinical sense as a representation of the source concept in the context of the original clinician query? The original query is supplied as scoring context; the requested output is a numeric semantic score from 0 to 10.
- **Lexical overlap:** Do the source and candidate literally share words? This is implemented as Jaccard overlap on whitespace-tokenized, lowercased tokens.

For most axes SafeSearch keeps the single top-scoring candidate per source concept. `intent_terms` and `rxnorm_terms` retain up to three because intents and medications can compose into search queries as sets.

Axis ordering is a separate Azure OpenAI chat call. Given the reranked candidates, it asks the model to return the clinically meaningful order of axes for reconstruction. Hybrid-score ordering provides the default ordering when needed.

Sources: [`app/services/reranking.py`](app/services/reranking.py), [`app/core/prompts.py`](app/core/prompts.py) (`RerankingPrompts.SYSTEM_RERANK`, `RerankingPrompts.AXIS_ORDER_SYSTEM`), [`app/core/service_configs.py`](app/core/service_configs.py).

</details>

<details>
<summary><strong>Stage 5: Safe query reconstruction</strong></summary>

This is the key boundary transformation.

**The query builder does not receive the original clinician query.** After candidate selection and axis ordering, it receives the selected clinical representations in axis order and converts them into a natural-language search query.

The Azure OpenAI chat deployment performs the conversion at temperature 0.05 with an 80-token budget and up to three retries. The selected terms can also be rendered directly as an ordered keyword query.

Sources: [`app/services/query_builder.py`](app/services/query_builder.py), [`app/core/prompts.py`](app/core/prompts.py) (`QueryBuilderPrompts.SYSTEM_REFINER`, `QueryBuilderPrompts.REFINEMENT_PROMPT`).

</details>

<details>
<summary><strong>Stage 6: External evidence retrieval</strong></summary>

The reconstructed query is sent to the Perplexity `sonar-pro` chat completion endpoint at temperature 0.1, filtered to a curated set of preferred clinical domains and a rolling one-year publication window. The preferred-domain filter is:

`pubmed.ncbi.nlm.nih.gov`, `jamanetwork.com`, `nejm.org`, `thelancet.com`, `bmj.com`, `annals.org`, `acc.org`, `ahajournals.org`, `escardio.org`, `nice.org.uk`.

The endpoint returns a summary and citation list.

> **Retrieval #1 chooses a representation. Retrieval #2 retrieves evidence.**

Source: [`app/services/perplexity.py`](app/services/perplexity.py).

</details>

<details>
<summary><strong>Stage 7: Clinical re-contextualization</strong></summary>

Evidence returns to the trusted environment. The configured Azure OpenAI chat deployment then has access to:

- the original clinician query,
- the selected controlled clinical concepts by axis, and
- the citations returned by external retrieval.

It generates the final clinician-facing markdown answer at temperature 0.15 with up to three retries and light validation. SafeSearch does not need to send the entire original context externally in order to use that context in the final answer.

> **Context is minimized for release, then restored for interpretation.**

Source: [`app/services/response_generator.py`](app/services/response_generator.py).

</details>

### PHI safety check

After reconstruction and response generation, SafeSearch performs an additional PHI safety check over pipeline outputs using configured PHI patterns and known-PHI terms when available. This operates as a secondary validation layer alongside the primary constrained semantic reconstruction architecture.

Source: [`app/services/phi_checker.py`](app/services/phi_checker.py).

<details>
<summary><strong>Model and trust-boundary map</strong></summary>

- **Axis extraction:** Azure OpenAI chat deployment. Sees the original query. Trusted.
- **Concept embedding:** Azure OpenAI embedding deployment. Sees extracted concepts. Trusted.
- **Semantic candidate rating:** Azure OpenAI chat deployment. Sees the original query as scoring context. Trusted.
- **Axis ordering:** Azure OpenAI chat deployment. Sees the original query as scoring context. Trusted.
- **Query reconstruction:** Azure OpenAI chat deployment. Receives ordered controlled terms only, not the original query. Trusted.
- **External evidence retrieval:** Perplexity `sonar-pro`. Receives the reconstructed query. External.
- **Final answer generation:** Azure OpenAI chat deployment. Sees the original query, selected concepts, and returned evidence. Trusted.

HIPAA compliance is not an intrinsic property of a model. The architecture assumes appropriately configured Azure OpenAI services operating under the institution's applicable BAA and safeguards.

</details>

<details>
<summary><strong>Prompt index</strong></summary>

- **Axis extraction:** [`AxisExtractionPrompts.SYSTEM_EXTRACTION`](app/core/prompts.py)
- **Semantic candidate rating:** [`RerankingPrompts.SYSTEM_RERANK`](app/core/prompts.py)
- **Axis ordering:** [`RerankingPrompts.AXIS_ORDER_SYSTEM`](app/core/prompts.py)
- **Query reconstruction system prompt:** [`QueryBuilderPrompts.SYSTEM_REFINER`](app/core/prompts.py)
- **Query reconstruction user template:** [`QueryBuilderPrompts.REFINEMENT_PROMPT`](app/core/prompts.py)
- **External retrieval:** inline in [`app/services/perplexity.py`](app/services/perplexity.py)
- **Final answer generation:** [`ResponseGeneratorPrompts.SYSTEM_RESPONSE`](app/core/prompts.py)

</details>

## Semantic fidelity on [ASQ-PHI](https://github.com/JamesWeatherhead/asq-phi)

![Cosine similarity between original and reconstructed queries on ASQ-PHI](docs/assets/cosin_sim_query_reconstructed.png)

Cosine similarity between the original clinician query and reconstructed query on the [ASQ-PHI](https://github.com/JamesWeatherhead/asq-phi) benchmark. **100% of Safe Harbor PHI was removed from the reconstructed queries.** Mean cosine 0.61, median 0.63. 4% of queries fall below 0.4. No reconstructed query exceeds 0.83.

This cosine is a proxy for how much task-relevant clinical meaning survives reconstruction. Fidelity and privacy trade against each other; the next section explains why.

## Why not just increase granularity?

![The SafeSearch granularity/privacy tension](docs/assets/granularity-privacy-tradeoff.svg)

Increasing vocabulary granularity can preserve more clinical specificity, but doing so can also preserve rarer combinations of quasi-identifying attributes. Individually none of those is necessarily a Safe Harbor identifier. Jointly they can collapse the k-anonymity cell of the disclosed representation below acceptable thresholds. Coarser vocabulary trades semantic fidelity for k-anonymity headroom. Richer vocabulary trades k-anonymity headroom for semantic fidelity.

This trade-off is formalized in:

> Weatherhead J, Hasan A, Weatherhead J, Golovko G, Grant B, Garcia JD, Certuche HS, Powell RP, Abril JM, McCaffrey P. **K-anonymity decay in multi-turn clinical large language model conversations.** *Frontiers in Digital Health*, 2026. doi:[10.3389/fdgth.2026.1832168](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2026.1832168/full).

Headline finding: **79.9% of simulated patients fall below the small-cell threshold ($k < 5$) by the end of their disclosure sequence**, with the median threshold breach at seven disclosure steps, even when every individual conversational turn complies with HIPAA Safe Harbor de-identification.

## Adversarial trace

Peter McCaffrey's red-team query into the pipeline:

> "guidelines for managing a 47-year-old female who is a Galveston-based vet tech who is also occasionally homeless. She loves Battleship and is a competitive expert. Funny story, we both grew up on the same block on Elderberry street in Dallas! Anyway, can I get some help for her leg pain?"

**What crossed the boundary:**

> "What treatment options are available for leg pain in an adult female?"

Location, occupation, lifestyle, hobby, and the manufactured shared-block claim were all dropped. Only clinically task-relevant axes survived: `intent_terms: treatment`, `anatomy_terms: leg`, `symptom_terms: pain`, `age_bins: adult`, `sex_terms: female`. `phi_leakage.detected = false` on the trace. 30.5 s end-to-end, $0.028.

[View the full adversarial trace](docs/traces/mccaffrey-adversarial-challenge-trace.json)

## v1 prototype

Before axis decomposition, v1 vectorized whole queries against a single embedding index. It had no principled way to separate clinical concepts from surrounding identifying context.

<video src="docs/assets/safesearch-prototype-v1.mov" controls preload="metadata" width="100%"></video>

**[Watch the v1 prototype demo](https://github.com/JamesWeatherhead/SafeSearch/blob/main/docs/assets/safesearch-prototype-v1.mov)**

If inline playback is unavailable in your GitHub client, the link above opens the repository-hosted video directly. GitHub supports `.mov` video files, although playback behavior can vary by client and browser.

## Repository layout

```text
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
    ├── reranking.py              stage 4 (rerank + axis ordering)
    ├── query_builder.py          stage 5 (safe query reconstruction)
    ├── perplexity.py             stage 6
    ├── response_generator.py     stage 7
    └── phi_checker.py            end-of-pipeline PHI safety check
docker/, scripts/init-db.sql, docs/{assets,traces}
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

Settings load from `.env` via `pydantic-settings`. See [`app/core/config.py`](app/core/config.py) for the full schema.

- **Azure OpenAI:** key, endpoint, API version, chat + embedding deployment names, chat + embedding model names.
- **Perplexity:** `PPLX_API_KEY`, `PPLX_API_ENDPOINT`.
- **PostgreSQL:** `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USERNAME`, `DB_PASSWORD`, `DB_CONNECT_TIMEOUT`, `DB_DSN`.
- **PHI dictionary path:** `PHI_QUERIES_FILE` (optional; regex-only if absent).
- **Environment name:** `ENVIRONMENT`.

Pipeline tunables such as rerank weights, top-k, cosine threshold, retrieval broadening, response temperatures, and PHI strictness live in [`app/core/service_configs.py`](app/core/service_configs.py).

## Attribution

**James Charl Weatherhead · Jake Craig Weatherhead · Peter A. McCaffrey, MD (PI)**

- Benchmark: [ASQ-PHI](https://github.com/JamesWeatherhead/asq-phi)
- Paper: [K-anonymity decay, *Frontiers in Digital Health*, 2026](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2026.1832168/full)

## License

Proprietary. See [`pyproject.toml`](pyproject.toml).