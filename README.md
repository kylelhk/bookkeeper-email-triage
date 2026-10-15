# Bookkeeper Email Triage

A prototype that helps a small bookkeeping practice triage client emails:
it classifies each email, lists missing information, and drafts a reply
for the bookkeeper to review. **It never sends anything.**

Built as a practice project for applying AI in a small business context.
Uses synthetic data only. A prototype, not a production tool.

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
- The prompt lists when an email must be flagged for human review (for example ATO notices, tax questions, fee enquiries and sensitive situations).
- Code-level guardrails run after the AI and do not depend on it behaving: they force review of ATO notices and prompt-injection attempts, raise urgency for ATO problems, flag dollar amounts in a reply that the client never mentioned, flag risky wording (lodgement status, claim advice, waivers or refunds), and blank any reply to a non-client.
- Access to the API requires a bearer token (`API_TOKEN` in `.env`).

## Results

I tested the pipeline on 20 synthetic emails (no real client data), split into two sets of 10:

- **`run1`**: The set I looked at while building and tuning the prompt.
- **`holdout`**: Emails I did not look at until after the prompt was final. This is the fairer measure.

Each email was sent to the model once (Claude Haiku, `claude-haiku-5-5` via the Messages API). The model sets no temperature, so a rerun can differ by a point or two. All raw outputs are saved in `results/`, so every number below can be re-checked.

| Run       | Prompt   | Category | Urgency | Review flag | All three correct |
| --------- | -------- | -------- | ------- | ----------- | ----------------- |
| `run1`    | baseline | 7/10     | 6/10    | 6/10        | 3/10              |
| `run1`    | improved | 10/10    | 9/10    | 10/10       | 9/10              |
| `holdout` | improved | 6/10     | 8/10    | 6/10        | 3/10              |

Baseline figures count one unusable model reply as a failure (see "What the baseline found").

**How to read the table above:**

- The `run1` jump is partly tuning on cases I had already seen: I wrote the new prompt rules after reading those failures. It shows the failures I found were fixable. It does not show the tool works in general.
- The `holdout` result (3/10 fully correct) is the realistic number. I did not run the old prompt on the holdout set, so I cannot say how much the improved prompt helped there. I did not tune against the holdout set afterwards, so it stays an unseen test.
- 10 emails per set is a small sample. Treat these numbers as indicative only.
- The code-level guardrails changed none of these accuracy scores. They are a safety layer (forcing review, blanking replies to non-clients), not an accuracy booster.

### What the baseline found

- **Over-flagging.** The previous prompt (before improvement) said "when unsure, flag for human review", so the model flagged routine requests too. The improved prompt lists exactly when to flag.
- **Wrong urgency on tax and fee questions.** The model rated them LOW. The improved prompt defines them as NORMAL.
- **A cut-off reply.** One reply exceeded the previous `max_tokens` limit in the `call_claude` function in `app.py` and came back as incomplete JSON. I raised the limit and the code now fails with a clear error if it happens again.
- **A scoring bug.** My scoring script first stopped on that cut-off reply and hid the failure from the totals. It now records unusable replies as failures and carries on.

### What still goes wrong (in the holdout emails)

- **Tax versus invoice categories.** When an email mentions GST, the model often picks `TAX_ADVICE_REQUEST` instead of the more specific category (two cases).
- **Review flag.** It still flags some vague or very short emails and misses one deadline question.
- **Over-asking.** Despite Rule 5 in the prompt, several replies still ask the sender to confirm their name and business (and one asks about an ABN, which may be legitimate for a new business).
- Some expected labels are judgement calls (for example, a billing dispute could be an invoice query or a fees matter). I wrote the labels myself, so an independent bookkeeper might disagree on some.

### Reply safety

I read all 20 draft replies and marked whether each is safe to send as is (`reply_safe_to_send_Y_N` and `safety_note` in the two scored CSVs).

- 2 of the 20 replies were not safe to send as written. One told a client not to pay or reply to the ATO, which is advice the draft must not give (Rule 2 of the prompt). One asked a client to email a tax file number declaration, date of birth, and bank account details, which is a privacy and security risk (to be confirmed by a bookkeeper).
- Both of those were flagged for human review, so the bookkeeper would have seen them before anything was sent. The guardrails did not catch them, so they point to the next guardrail to add.
- The tool never sends anything. Every draft is for a human to approve.

## Limitations

- Synthetic data only; no real client emails were used.
- 20 emails, one run each, labelled by me. Not a statistically meaningful evaluation.
- Real client emails would go to a third-party AI provider. Before using real data, check privacy obligations and the provider's data terms.
- Only the email text is analysed; attachments are not read.
- The free Gemini tier was unreliable during this project (capacity errors and a 20-request daily limit), so scored runs used Claude. Gemini is supported through the `PROVIDER` setting but was not scored.

## Next steps / potential enhancement

- Test on a small set of real, consented, and anonymised emails from a local bookkeeper, labelled by that bookkeeper.
- Add guardrails for the unsafe replies found above (instructions about what to do with an ATO notice, requests for tax file numbers or bank details by email).
- Clarify the boundary between GST/BAS questions and tax-advice questions in the prompt, then re-test on a fresh set of emails.
- Compare different AI models and repeat each run several times to measure variation.

## Test data

- 20 synthetic emails, drafted with AI assistance and reviewed/edited by me.
- Expected answers are my own judgment, not an official standard.
- I first planned to score only 10 emails because of the Gemini free tier's 20-request daily limit. In the end I scored both sets using the Claude API. `run1` was chosen before running for coverage of high-risk cases; `holdout` was kept unseen to check the final prompt.
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
button to enter your `API_TOKEN` (in `.env`) before calling `/triage`. Never commit `.env`.

## Run the tests

They use the mock model, no API key needed:

```bash
python -m pytest
```

Score the test emails against the model named by `PROVIDER` in `.env` (`claude` needs `ANTHROPIC_API_KEY`):

```bash
python score.py --label myrun --set run1     # or --set holdout
python score.py --rescore results/myrun_raw.json   # re-score saved outputs, no API calls
```
