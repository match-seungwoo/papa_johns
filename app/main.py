from fastapi import FastAPI

from app.api.routes import health, jobs

app = FastAPI(title="AI Poster Engine", version="0.1.0")

app.include_router(health.router)
app.include_router(jobs.router)
