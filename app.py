import os
import secrets

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
# Enum limits a field to a fixed list. If the AI invents a category like “TAX_STUFF”, validation fails instead of passing junk downstream.
from enum import Enum
from pydantic import BaseModel, Field

# Reads .env into environment variables, so secrets stay out of the code.
load_dotenv()

# Reads the Authorisation: Bearer <token> header.
bearer = HTTPBearer(auto_error=False)


def require_token(creds: HTTPAuthorizationCredentials | None = Depends(bearer)):
    expected = os.getenv("API_TOKEN", "")
    # If no token is configured, the server refuses (500) instead of letting everyone in. Failing closed is deliberate.
    if not expected:
        raise HTTPException(status_code=500, detail="Server is not configured")
    supplied = creds.credentials if creds else ""
    # secrets.compare_digest compares in constant time, which stops attackers guessing the token by timing responses.
    if not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(
            status_code=401, detail="Invalid or missing bearer token")


class Category(str, Enum):
    BAS_GST = "BAS_GST"
    PAYROLL_SUPER = "PAYROLL_SUPER"
    INVOICE_BILL_QUERY = "INVOICE_BILL_QUERY"
    DOCUMENT_SUBMISSION = "DOCUMENT_SUBMISSION"
    DEADLINE_STATUS = "DEADLINE_STATUS"
    REPORTS_REQUEST = "REPORTS_REQUEST"
    FEES_ENGAGEMENT = "FEES_ENGAGEMENT"
    TAX_ADVICE_REQUEST = "TAX_ADVICE_REQUEST"
    ATO_NOTICE = "ATO_NOTICE"
    GENERAL_UNCLEAR = "GENERAL_UNCLEAR"
    SPAM_OR_NOT_A_CLIENT = "SPAM_OR_NOT_A_CLIENT"


class Urgency(str, Enum):
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"


class EmailIn(BaseModel):
    sender_name: str = Field(default="", max_length=200)
    subject: str = Field(default="", max_length=500)
    body: str = Field(min_length=1, max_length=8000)
    # default="" make a field optional.
    # max_length on the email stops someone sending a huge input that costs money or breaks things.
    # min_length=1 on the body rejects empty emails.


class TriageResult(BaseModel):
    category: Category
    urgency: Urgency
    summary: str = ""
    missing_information: list[str] = Field(default_factory=list)
    draft_reply: str = ""
    needs_human_review: bool
    review_reason: str = ""


app = FastAPI(title="Bookkeeper Email Triage API")


@app.get("/health")
def health():
    return {"status": "ok"}

# Takes an email and returns a plain dictionary:


def run_model(email: EmailIn) -> dict:
    """Placeholder for the AI call. Returns a mock result for now."""
    text = f"{email.subject} {email.body}".lower()
    category = "BAS_GST" if "bas" in text else "GENERAL_UNCLEAR"
    return {
        "category": category,
        "urgency": "NORMAL",
        "summary": "Mock result - no AI used yet",
        "missing_information": [],
        "draft_reply": "Hi, thanks for your email. [MOCK DRAFT]",
        "needs_human_review": True,
        "review_reason": "mock mode",
    }

# Run this when someone sends data to /triage (POST is used as it's for submitting email data):


# response_model=TriageResult tells FastAPI to validate the output of run_model() against the TriageResult model so that the output is always in the intended format.
@app.post("/triage", response_model=TriageResult, dependencies=[Depends(require_token)])
def triage(email: EmailIn):
    raw = run_model(email)
    # model_validate forces the result through the TriageResult schema. When the real AI replaces this mock, its answer goes through the same check, so a bad category fails loudly instead of slipping through.
    return TriageResult.model_validate(raw)
