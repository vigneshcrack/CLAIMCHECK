# Lane 5 — Web App & Explainability

Owner: _(assign teammate name here)_

## What you're building

A Streamlit (or Gradio) app that wires together Lane 2/3's `retrieve()` (+ Lane 3's `rerank()`) and Lane 4's `verify()` into a working claim-checking UI. You never need to understand what's inside those functions — just call them by their shared signatures (see the other lane READMEs).

## Steps

1. Build the app: claim input box, "Check Claim" button.
2. Wire the pipeline: `retrieve()` → `rerank()` (if using Lane 3's modern pipeline) → `verify()`.
3. UI: colour-coded verdict badge (SUPPORTS/REFUTES/NOT ENOUGH INFO), confidence score, evidence list with source + relevance score, a plain-English "why" explanation, and explicit NEI messaging ("this does not mean the claim is false — insufficient evidence was found").
4. Nice-to-have if time allows: a toggle between the traditional (Lane 2) and modern (Lane 3) pipeline so the comparison is visible live in the app, not just in a report table.
5. Take screenshots of the working app on several example claims for the report.
6. **Set up the top-level repo README** (project overview, architecture diagram, who built what — this top-level `README.md` at the repo root already has a skeleton; flesh it out), and generate a **QR code linking to the GitHub repo URL** once it's pushed — this was an explicit gap flagged in both the professor's general feedback and the Part A grading, so don't skip it.
7. Write a short UI/UX rationale paragraph — this feeds directly into the "User Interaction focused" 40% criterion in the Part B rubric.

## Submit to the group

- `app.py` (Streamlit/Gradio)
- Screenshots (PNG) for the report
- Updated top-level README + QR code image
- UI/UX design rationale paragraph

## Commit discipline

Push to this folder daily, even small progress — commit history is evidence of individual contribution for SPARKPlus and the report's "contributions by team member" section.
