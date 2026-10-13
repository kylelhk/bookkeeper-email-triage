from fastapi import FastAPI

app = FastAPI(title="Bookkeeper Email Triage API")


@app.get("/health")
def health():
    return {"status": "ok"}
