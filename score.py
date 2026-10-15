"""Run the test emails through the triage pipeline and score the results.

Examples:
    python score.py --label baseline                       # Call the model on the run1 emails
    python score.py --rescore results/baseline_raw.json    # Re-score saved outputs, NO API calls

Every email is sent to the model ONCE. The run stops at the first provider error,
so an outage cannot burn the whole daily quota. A reply that is not usable JSON is
counted as a failed email and the run continues. Raw model outputs are saved after
every email, so the guardrails can be re-scored offline at no cost.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from app import EmailIn, TriageResult, apply_guardrails, run_model

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "data" / "email_triage_test_set.csv"
RESULTS_DIR = ROOT / "results"


def load_rows(eval_set: str) -> list[dict]:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if eval_set == "all" or r["eval_set"] == eval_set]


def collect_raw(rows: list[dict], raw_path: Path, delay: float) -> dict:
    raw: dict = {}
    for i, row in enumerate(rows):
        email = EmailIn(subject=row["subject"], body=row["email_body"])
        try:
            raw[row["id"]] = {"ok": True, "output": run_model(email)}
        except ValueError as exc:
            # The model answered, but not in a usable form (e.g. no JSON). That is a result, not an outage: Record it as a failure and carry on with the next email.
            raw[row["id"]] = {"ok": False, "unusable": True,
                              "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            print(
                f"  email {row['id']:>2} UNUSABLE OUTPUT (counted as a failure)")
        except Exception as exc:  # Provider error: Stop, so an outage cannot burn the quota
            raw[row["id"]] = {
                "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            left = len(rows) - i - 1
            print(
                f"STOPPED at email {row['id']}: {raw[row['id']]['error']}\n{left} emails not run.")
            raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
            break
        else:
            print(f"  email {row['id']:>2} done")
        raw_path.write_text(json.dumps(raw, indent=2),
                            encoding="utf-8")  # Save after every email
        if i < len(rows) - 1:
            time.sleep(delay)
    return raw


def matches(row: dict, res: TriageResult) -> tuple[bool, bool, bool]:
    return (
        res.category.value == row["expected_category"],
        res.urgency.value == row["expected_urgency"],
        ("YES" if res.needs_human_review else "NO") == row["expected_needs_human_review"],
    )


def score(rows: list[dict], raw: dict, label: str) -> None:
    out_rows, mism = [], []
    # Correct-answer counts, kept separately for the raw model output and for the final output after guardrails, so the guardrails' effect can be compared.
    # Each list is [category, urgency, review flag, all three correct], indexed in the same order as the tuple returned by matches().
    tally = {"raw": [0, 0, 0, 0], "final": [0, 0, 0, 0]}
    scored = invalid = unusable = 0
    for row in rows:
        entry = raw.get(row["id"])
        if not entry:
            continue  # Not run
        if entry.get("unusable"):
            unusable += 1
            mism.append(
                (row["id"], f"{row['trap_or_challenge']}: model returned unusable output"))
            continue
        if not entry.get("ok"):
            continue  # Provider error: The run stopped here
        email = EmailIn(subject=row["subject"], body=row["email_body"])
        try:
            res_raw = TriageResult.model_validate(entry["output"])
        except ValueError:
            invalid += 1
            mism.append(
                (row["id"], "INVALID model output (failed schema validation)"))
            continue
        res_final, notes = apply_guardrails(email, res_raw)
        scored += 1
        flags = {}
        for name, res in (("raw", res_raw), ("final", res_final)):
            m = matches(row, res)
            flags[name] = m
            for k in range(3):
                tally[name][k] += m[k]
            tally[name][3] += all(m)
        if not all(flags["final"]):
            bad = [n for n, ok in zip(
                ("category", "urgency", "review flag"), flags["final"]) if not ok]
            mism.append(
                (row["id"], f"{row['trap_or_challenge']}: wrong " + ", ".join(bad)))
        out_rows.append(
            {
                "id": row["id"],
                "challenge": row["trap_or_challenge"],
                "expected_category": row["expected_category"],
                "raw_category": res_raw.category.value,
                "final_category": res_final.category.value,
                "expected_urgency": row["expected_urgency"],
                "raw_urgency": res_raw.urgency.value,
                "final_urgency": res_final.urgency.value,
                "expected_review": row["expected_needs_human_review"],
                "raw_review": "YES" if res_raw.needs_human_review else "NO",
                "final_review": "YES" if res_final.needs_human_review else "NO",
                "expected_missing_info": row["expected_missing_info"],
                "ai_missing_info": "; ".join(res_final.missing_information),
                "draft_reply": res_final.draft_reply,
                "guardrail_notes": "; ".join(notes),
                # TODO: reply_safe_to_send_Y_N to be manually filled after reading each draft reply, for checking if the guardrails were sufficient to prevent a bad reply from being sent.
                "reply_safe_to_send_Y_N": "",
            }
        )

    if not out_rows:
        print("Nothing to score.")
        return
    csv_path = RESULTS_DIR / f"{label}_scored.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    n = scored + invalid + unusable  # Failed outputs count as wrong, not skipped
    failed = invalid + unusable
    print(f"\nScored {n} of {len(rows)} emails" +
          (f" ({failed} with unusable output, counted as wrong)" if failed else ""))
    print(f"{'':24}{'raw model':>12}{'+ guardrails':>15}")
    for k, name in enumerate(("category correct", "urgency correct", "review flag correct", "all three correct")):
        print(
            f"{name:24}{tally['raw'][k]:>9}/{n:<3}{tally['final'][k]:>11}/{n}")
    if mism:
        print("\nStill wrong after guardrails:")
        for i, why in mism:
            print(f"  #{i} {why}")
    print(f"\nWrote {csv_path}")
    print("Next: read every draft_reply and fill in reply_safe_to_send_Y_N by hand.")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="baseline",
                   help="name for this run, e.g. baseline or improved")
    p.add_argument("--set", dest="eval_set", default="run1",
                   help="run1, holdout or all")
    p.add_argument("--delay", type=float, default=13.0,
                   help="seconds between calls (free tier: 5/min)")
    p.add_argument(
        "--rescore", help="path to a saved *_raw.json: score it again with NO API calls")
    args = p.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    rows = load_rows(args.eval_set)
    if args.rescore:
        raw_path = Path(args.rescore)
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        label = raw_path.stem.removesuffix("_raw") + "_rescored"
    else:
        raw_path = RESULTS_DIR / f"{args.label}_raw.json"
        if raw_path.exists():
            print(
                f"{raw_path} already exists. Choose a new --label (or delete it) so quota isn't wasted by accident.")
            return 1
        print(
            f"Running {len(rows)} emails ({args.eval_set}) with PROVIDER set in .env ...")
        raw = collect_raw(rows, raw_path, args.delay)
        label = args.label
    score(rows, raw, label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
