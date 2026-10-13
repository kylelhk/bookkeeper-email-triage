from fastapi import FastAPI
# Enum limits a field to a fixed list. If the AI invents a category like “TAX_STUFF”, validation fails instead of passing junk downstream.
from enum import Enum
from pydantic import BaseModel, Field


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
