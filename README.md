Founder: James Charl Weatherhead
Co-Founder: Jake Craig Weatherhead

# SafeSearch

SafeSearch converts PHI-bearing clinician queries into semantically preserved,
PHI-free search queries using clinical-axis extraction, vector matching,
query resynthesis, external evidence retrieval, and BAA-side answer synthesis.

The consolidated local recovery is under [`recovered/`](recovered/README.md).
The centralized run/trace index is under
[`recovered/trace-index/`](recovered/trace-index/README.md).

![SafeSearch pipeline](docs/architecture/safesearch_pipeline.png)

To run:

`cd safesearch`

`source .venv/bin/activate` to activate uv virtual environment

`python manage.py work` to run the FastAPI application

Navigate to `localhost:8000/scalar` to view API definition and test `query` endpoint
