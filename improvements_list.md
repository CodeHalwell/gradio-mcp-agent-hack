# Improvements ongoing

### Kick off the three pipelines simultaneously

async def _run_subquestion(self, sq: str):
    raw = await self.web_search.search(sq)
    snippet = self._format_search_results(raw["results"])
    summary = await self.llm_processor.process(snippet, "summarize")
    cites   = await self.citation_formatter.format_citations(snippet)
    return summary, cites

async def orchestrate_async(self, user_request):
    subs = await self.question_enhancer.enhance_question(user_request)
    tasks = [asyncio.create_task(self._run_subquestion(sq))
             for sq in subs["sub_questions"]]
    results = await asyncio.gather(*tasks, return_exceptions=True)


## Implementation Guide for Up-Levelling CodeRunnerAgent — v2

### Build the reusable execution image
Create a Modal Image that pre-installs ~10 of the most-used Python libraries so every sandbox starts “batteries included”.

python
Copy
Edit
common_packages = (
    "numpy",
    "pandas",
    "polars",
    "matplotlib",
    "seaborn",
    "plotly",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "nltk",
    "requests",
    "beautifulsoup4",
    "scrapy",
    "flask",
    "fastapi",
    "starlette",
    "pillow",
    "imageio",
    "tqdm",
    "pytest",
    "python-dateutil",
    "pydantic",
    "click",
    "rich",
    "httpx",
    "duckdb",
    "networkx",
    "schedule",
    "watchdog",
    "sqlalchemy",
)

image = modal.Image.debian_slim().pip_install(*common_packages)

Store the image on the CodeRunnerAgent instance (self.image) for reuse across calls.

### Wrap Sandbox lifecycle in a context-manager

@contextmanager
def sandbox_context(app: modal.App, **kwargs):
    sb = modal.Sandbox.create(app=app, **kwargs)
    try:
        yield sb
    finally:
        sb.terminate()
Guarantees clean-up; accepts timeout, CPU, memory, mounts, etc.

### Custom error taxonomy

class CodeExecutionError(RuntimeError): ...
class UserCodeError(RuntimeError): ...
UserCodeError → user script failed

CodeExecutionError → sandbox / infra failed


### Async + warm-pool upgrade
Maintain an asyncio.Queue of warm sandboxes.

Use await sb.exec_async(...) to stay non-blocking.

Health-check & recycle sandboxes periodically.

### Observability & ops

Performance	Wrap run_code with @with_performance_tracking.
Logging	Include sandbox object_id, request ID, elapsed time.
Tracing	sb.set_tags({"request_id": rid, "user": uid}).
Metrics	Counter: total runs & errors. Histogram: duration.

**It would be great if the user interface could be updated to see the progress.**
