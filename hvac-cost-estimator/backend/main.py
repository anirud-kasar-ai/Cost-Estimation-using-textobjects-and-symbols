"""FastAPI entrypoint.

Run from the backend/ directory:

    uvicorn main:app --reload
"""

from __future__ import annotations

import logging
import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import projects, reports, upload
from config import get_settings
from db.session import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.requirements_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    yield


app = FastAPI(
    title="Mechanical Device Cost Estimator",
    description=(
        "i-TAB HVAC proof of concept: extract title-block metadata and device "
        "symbol counts from HVAC layout PDFs and produce an editable costed report."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def handle_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """Like FastAPI's default 422 handler, but safe for non-finite floats.

    The default handler echoes the invalid input back in the response body;
    inf/NaN inputs (e.g. unit_cost=1e999) are not JSON-serializable and would
    turn the 422 into a 500.
    """

    def sanitize(value: object) -> object:
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value

    errors = [{**error, "input": sanitize(error.get("input"))} for error in exc.errors()]
    return JSONResponse(status_code=422, content=jsonable_encoder({"detail": errors}))


app.include_router(upload.router)
app.include_router(projects.router)
app.include_router(reports.router)


@app.get("/api/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}
