"""FastAPI app: POST /ask, the demo page, and the saved eval scorecards."""
import json
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import ASK_RATE_LIMIT, ASK_RATE_WINDOW_S, ROOT
from app.pipeline import AskResult, ask
from app.rag.search import get_index

app = FastAPI(title="Pharma Q&A", description="Cited answers over ClinicalTrials.gov and openFDA drug labels.")
RESULTS_DIR = ROOT / "eval" / "results"
WEB_DIR = ROOT / "web"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class RateLimiter:
    """Sliding-window limit per client IP. In-process, which is enough for one container."""

    def __init__(self, limit: int, window_s: int):
        self.limit, self.window = limit, window_s
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


limiter = RateLimiter(ASK_RATE_LIMIT, ASK_RATE_WINDOW_S)


@app.on_event("startup")
def _warm() -> None:
    get_index().search("warm up", k=1)  # load BM25, embeddings and the encoder before the first request


@app.post("/ask", response_model=AskResult)
def post_ask(body: AskRequest, request: Request) -> AskResult:
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
    if not limiter.allow(ip):
        raise HTTPException(429, f"Rate limit: {ASK_RATE_LIMIT} questions per {ASK_RATE_WINDOW_S}s")
    return ask(body.question)


@app.get("/api/scorecards")
def scorecards() -> list[dict]:
    out = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        run = json.loads(path.read_text())
        out.append({"file": path.name, **{k: run[k] for k in ("tag", "started_at", "metrics", "by_type", "config")}})
    return out


@app.get("/api/scorecards/{name}")
def scorecard(name: str) -> dict:
    path = (RESULTS_DIR / name).resolve()
    if path.parent != RESULTS_DIR.resolve() or not path.exists():
        raise HTTPException(404)
    return json.loads(path.read_text())


@app.get("/api/examples")
def examples() -> list[str]:
    return [
        "Which Phase 3 trials is Keytruda being tested in for lung cancer?",
        "What are the boxed warnings on the warfarin label?",
        "Contraindications for Ozempic?",
        "How many Phase 2 trials test nivolumab in melanoma?",
        "What will pembrolizumab's 2030 revenue be?",
    ]


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "chunks": len(get_index().chunks)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
