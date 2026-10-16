# Lane 5 — ClaimCheck Streamlit Web Application

## Overview

Lane 5 implements the interactive web application and explainability interface for **ClaimCheck**, an evidence-based claim verification research prototype developed for the Applied Natural Language Processing group project.

The application provides a unified interface for the retrieval and verification components developed across the project. Users enter a factual claim, select an evidence retrieval method, and inspect the resulting model prediction alongside ranked evidence sentences, Natural Language Inference (NLI) classifications, retrieval relevance scores and explanatory information.

ClaimCheck operates on a limited **FEVER-Symmetric-derived evidence corpus** rather than performing live web searches. Its outputs represent model predictions based on available evidence and should not be interpreted as verified real-world facts.

## 1. Application Features

The Streamlit interface provides:

- **Claim input:** Users can enter their own factual statement or choose an example claim.
- **Retrieval method selection:** Five traditional and modern evidence retrieval configurations.
- **Configurable evidence count:** Retrieve up to 3, 5 or 10 evidence sentences.
- **Evidence verification:** Classify each retrieved sentence as SUPPORTS, REFUTES or NOT ENOUGH INFO.
- **Overall model prediction:** Apply Lane 4's evaluated Top-1 aggregation strategy to produce a claim-level prediction.
- **Evidence explanations:** Display ranked evidence sentences, their classifications and confidence values.
- **Conflict detection:** Warn users when supporting and contradicting evidence appear together.
- **Technical details:** Display evidence IDs, retrieval scores, NLI confidence and the processing pipeline.
- **Input validation and error handling:** Handle empty or excessively long claims, missing evidence and processing failures.
- **Transparency:** Explain corpus coverage, model-confidence limitations and the distinction between retrieved evidence and factual truth.

## 2. System Architecture

The application integrates the work of four upstream project components.

```text
                    User enters a claim
                             |
                             v
                   Streamlit interface
                    lane5_webapp/app.py
                             |
                             v
                    Evidence retrieval
                             |
             +---------------+---------------+
             |                               |
             v                               v
     Lane 2: Traditional             Lane 3: Modern
      BM25 / TF-IDF                   SBERT + FAISS
                                    Cross-Encoder reranking
             |                               |
             +---------------+---------------+
                             |
                             v
                  Top-K evidence sentences
                 (ID, relevance score, text)
                             |
                             v
                     Lane 4: DeBERTa
                 Evidence-level NLI verification
                             |
                             v
                    SUPPORTS / REFUTES /
                      NOT ENOUGH INFO
                             |
                             v
                 Lane 4: Top-1 aggregation
                             |
                             v
                   Overall model prediction
                             |
                             v
                    Lane 5: Results UI
                +-------------------------+
                | Overall prediction      |
                | Evidence summary        |
                | Ranked evidence cards   |
                | Confidence and scores   |
                | Conflict warnings       |
                | Technical details       |
                +-------------------------+
```

The application is separated into two main Python modules:

| File | Responsibility |
|---|---|
| `app.py` | Streamlit interface, input controls, session state, result presentation, evidence explanations and error messages |
| `pipeline.py` | Retrieval integration, evidence-level NLI execution, prediction validation, label counts and claim-level aggregation |

This separation improves modularity, testing and maintainability. It also allows changes to retrieval or verification models without substantially redesigning the user interface.

## 3. Retrieval Methods

The application supports five retrieval configurations.

| Method | Implementation | Description |
|---|---|---|
| BM25 | Lane 2 | Lexical retrieval based on term matching and document relevance |
| TF-IDF | Lane 2 | Vector-space retrieval using TF-IDF representations |
| SBERT + FAISS | Lane 3 | Semantic sentence embeddings with similarity-based retrieval |
| SBERT + Cross-Encoder | Lane 3 | SBERT candidate retrieval followed by Cross-Encoder reranking |
| BM25 + Cross-Encoder | Lanes 2 and 3 | BM25 candidate retrieval followed by Cross-Encoder reranking |

All methods return ranked evidence sentences with associated evidence identifiers and relevance scores.

The application preserves evidence IDs throughout retrieval, verification and presentation.

Retrieval scores are displayed according to their respective methods. **BM25, TF-IDF, cosine similarity and Cross-Encoder relevance scores are not directly comparable**, and high retrieval relevance does not establish factual correctness.

## 4. Evidence Verification and Claim-Level Prediction

### Evidence-Level Verification

Lane 4 uses the pretrained checkpoint:

`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`

The verifier processes claim–evidence pairs and maps NLI labels to the project's output labels:

| NLI label | ClaimCheck label |
|---|---|
| Entailment | SUPPORTS |
| Contradiction | REFUTES |
| Neutral | NOT ENOUGH INFO |

The application calls `verify_batch()` to classify retrieved sentences and displays the resulting labels and NLI confidence values.

### Overall Model Prediction

The overall model prediction uses Lane 4's **Top-1 aggregation strategy**, selected after comparing aggregation strategies using development-set evaluation.

Top-1 takes the verification label associated with the highest-ranked retrieved evidence sentence as the overall prediction.

For example, if the top-ranked evidence is classified as SUPPORTS, the claim-level model prediction is SUPPORTS.

The application also displays the NLI classifications of the remaining retrieved sentences so users can inspect additional evidence and identify disagreements.

### Conflicting Evidence

If retrieved sentences include both SUPPORTS and REFUTES classifications, the interface displays an explicit warning.

This warning is retained even when the Top-1 strategy produces an overall model prediction.

The purpose is to make model disagreement visible rather than conceal contradictory evidence behind a single output label.

## 5. Installation and Local Execution

### Prerequisites

- Python 3.12
- A local clone of the complete ClaimCheck repository
- Access to the dataset and code from Lanes 1–4
- Required Python dependencies and model downloads
- An internet connection for the initial download of pretrained models, if not already cached

The application should be launched from the **repository root**, not from inside `lane5_webapp`.

### Set Up the Environment

On macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the Streamlit application dependency:

```bash
python -m pip install streamlit pytest
```

Install the modern retrieval dependencies:

```bash
python -m pip install -r lane3_retrieval_modern/requirements.txt
```

The application also requires the dependencies used by Lanes 2 and 4, including their retrieval, machine-learning and Transformers libraries. Install those according to the relevant lane documentation or the team's consolidated dependency configuration.

### Launch the Application

Run the following command from the `CLAIMCHECK` root directory:

```bash
python -m streamlit run lane5_webapp/app.py
```

Streamlit will display a local address, typically:

`http://localhost:8501`

Open that address in a browser.

On first use, SBERT, Cross-Encoder and DeBERTa models may take additional time to load. Model loading and inference performance depend on the machine and its available memory.

## 6. Using the Application

1. Launch Streamlit using the command above.
2. Enter a factual claim or select one of the provided example claims.
3. Choose a retrieval method.
4. Select the maximum number of evidence sentences: 3, 5 or 10.
5. Click **Check Claim**.
6. Review the overall model prediction and its supporting explanation.
7. Examine the SUPPORTS, REFUTES and NOT ENOUGH INFO counts.
8. Expand evidence cards to inspect individual sentences.
9. Open **Technical details** to examine retrieval scores, confidence values and the pipeline.

### Example

**Input claim:**

> Telemundo is a Spanish-language television network.

During application testing, BM25 retrieval returned supporting and contradicting sentences within the prepared corpus.

The model predicted SUPPORTS using the highest-ranked evidence sentence while also displaying a conflict warning because another retrieved sentence was classified as REFUTES.

This example demonstrates the importance of presenting evidence-level disagreements alongside the overall model prediction.

## 7. Automated Testing

Lane 5 includes automated unit and integration tests using `pytest`.

The test suite contains four files:

| Test file | Purpose |
|---|---|
| `test_app_logic.py` | Input validation, evidence filtering, prediction processing, label counts and error handling |
| `test_claim_level_prediction.py` | Top-1 aggregation, empty evidence and neutral predictions |
| `test_modern_retrieval.py` | SBERT integration, reranker interface compatibility and score handling |
| `test_pipeline.py` | Integration smoke tests using the actual Lane 2 retrieval functions and Lane 4 NLI model |

Run all Lane 5 tests from the repository root:

```bash
python -m pytest lane5_webapp/tests/ -v
```

**Latest local test result: 24 tests passed, 0 failures, in 6.19 seconds.**

The test output included a PyTorch deprecation warning, which did not affect the test results.

Lane 4's separate retrieval-verification and aggregation test suite also passed 23 tests during integration.

In addition to automated testing, all five retrieval configurations were successfully executed through the Streamlit interface using the Telemundo example.

These tests confirm functional behaviour for the tested scenarios; they do not establish general factual accuracy or eliminate all possible application failures.

## 8. Model Evaluation and Limitations

The selected end-to-end configuration is:

**SBERT + Cross-Encoder retrieval → DeBERTa NLI → Top-1 aggregation**

It was selected from the implemented configurations based on development-set comparisons.

### Selected Configuration Results

| Metric | Development | Held-out test |
|---|---:|---:|
| Evaluated claims | 354 | 346 |
| Accuracy | 42.94% | 38.73% |
| Macro-F1 (3 classes) | 0.301 | 0.278 |
| Macro-F1 (SUPPORTS/REFUTES) | 0.451 | 0.417 |

The model underperformed the majority-class baseline on the held-out test split. Consequently, it should be regarded as a research demonstration rather than a reliable fact-checking system.

These performance figures apply specifically to the selected SBERT + Cross-Encoder / DeBERTa / Top-1 configuration, not automatically to the other retrieval options.

### Limitations

**Restricted corpus:** The application searches the prepared FEVER-Symmetric-derived corpus rather than accessing current news, articles or live Wikipedia content.

**Contrasting evidence:** The research corpus can include intentionally contrasting claims and evidence. Retrieved sentences should not automatically be treated as trustworthy factual sources.

**Model accuracy:** Held-out verification accuracy remains limited, and the system should not be used for high-stakes or consequential decisions.

**Confidence interpretation:** NLI confidence represents the model's classification score for a claim–evidence pair, not a calibrated probability that a real-world claim is true.

**Top-1 sensitivity:** The claim-level prediction depends on the highest-ranked evidence sentence. Incorrect retrieval or contradictory evidence can cause incorrect predictions.

**Evidence sufficiency:** Filtering non-positive BM25 and TF-IDF retrieval scores is a heuristic, not an independently validated relevance threshold.

**Dataset composition:** The FEVER-Symmetric evaluation splits used for the reported development and test comparisons contain SUPPORTS and REFUTES labels but do not provide the same coverage of genuinely unresolved real-world claims.

**Deployment resources:** Semantic retrieval and transformer-based verification require additional memory and model loading time, which may constrain deployment on limited hosting platforms.

## 9. UI/UX and Explainability Design

The Lane 5 interface was designed around transparency, usability and consistency between research outputs and user-facing explanations.

Key design decisions include:

- **Progressive disclosure:** Essential information appears in the main results summary, while detailed evidence and technical metrics are available through expandable sections and tabs.
- **Model transparency:** The interface explicitly labels the output as a model prediction rather than a verified factual conclusion.
- **Visible uncertainty:** Conflicting evidence is prominently flagged instead of being hidden by the overall prediction.
- **Interpretability:** Evidence IDs, retrieved sentences, NLI labels and confidence scores allow users to understand how the displayed prediction was produced.
- **Method comparison:** A consistent interface lets users switch between traditional and modern retrieval methods.
- **Clear limitations:** Explanatory messages communicate dataset restrictions, retrieval-score limitations and the distinction between model confidence and factual certainty.
- **Error handling:** Input validation and readable error messages help users recover from invalid submissions and processing failures.

The interface therefore aims to make an otherwise complex multi-model NLP workflow accessible while retaining the technical information necessary for evaluation and demonstration.
