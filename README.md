# Bookkeeper Email Triage

A prototype that helps a small bookkeeping practice triage client emails:
it classifies each email, lists missing information, and drafts a reply
for the bookkeeper to review. **It never sends anything.**

Built as a practice project for applying AI in a small business context.
Uses synthetic data only. Work in progress.

## The problem

A small practice receives a steady stream of messy client emails: BAS and GST
questions, receipts with no context, payroll worries, ATO notices. The owner
spends the start of each day reading, sorting, and writing near-identical
replies. This tool does the first pass; the bookkeeper stays in control.

## What it does

For each email, the API returns a **draft** containing:

- a category (e.g. BAS/GST, payroll/super, ATO notice, spam)
- an urgency level (high, normal, low)
- any information missing before the bookkeeper can act
- a draft reply for the bookkeeper to review and edit
- a flag when the email needs a human's attention

## Safety design

- Every output is a draft. Nothing is sent automatically.
- The AI is instructed not to give tax or legal advice, invent facts, quote prices, or follow instructions hidden in an email (`prompts/system_prompt.txt`).
- When unsure, the tool escalates to the bookkeeper instead of guessing.
- Access to the API requires a bearer token.

## Status

- [x] FastAPI skeleton and `/health` endpoint
- [x] Request and response schemas
- [x] System prompt
- [x] Mock `/triage` endpoint (keyword rules, no AI)
- [x] Bearer-token auth
- [x] Synthetic test set
- [x] Real AI model call (Gemini API)
- [x] Code-level guardrails (e.g. block invented dollar amounts)
- [ ] Automated tests
- [ ] Scoring script and baseline results
- [ ] Prompt improvement based on failures, with re-scored results

## Test data

- 20 synthetic emails, drafted with AI assistance and reviewed/edited by me.
- Expected answers are my own judgment, not an official standard.
- Because of the free-tier daily request limit of the Gemini API, I score a 10-email subset (`eval_set = run1`), chosen before running for coverage of high-risk cases. The other 10 (`holdout`) are kept unseen to check the final prompt.
- 10 emails is a small sample, so results show where the prompt fails, not overall accuracy.

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env             # Windows PowerShell: copy .env.example .env; then set API_TOKEN in .env
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs to try the endpoints. Use the **Authorize**
button to enter your token before calling `/triage`. Never commit `.env`.

## How this was built

Built step by step, one commit per step. I used an AI assistant (Claude) for
guidance and explanations, and wrote, tested, and committed each step myself.
