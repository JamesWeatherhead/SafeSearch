from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference
from .core.config import config
from .db.config import register_db
from .services.pipeline import PipelineService

_app = FastAPI(
    title="SafeSearch",
    description="Enabling medical professionals to make web-search, LLM-requests without leaking PHI.",
    debug=config.ENVIRONMENT == "dev",
)

@_app.get("/scalar", include_in_schema=False)
async def scalar_api_reference():
    return get_scalar_api_reference(
        openapi_url=_app.openapi_url,
        title=_app.title + " - Scalar",
    )

def get_application() -> FastAPI:
    register_db(_app)
    return _app

app = get_application()
pipeline = PipelineService()

@app.get("/query")
async def query(query: str):
    result = pipeline.run(query)
    return result
