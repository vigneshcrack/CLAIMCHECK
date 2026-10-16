
"""ClaimCheck Lane 5 — Streamlit interface.

Run from the repository root:

    python -m streamlit run lane5_webapp/app.py

Integrations:
    Lane 2: BM25 / TF-IDF
    Lane 3: SBERT / FAISS / Cross-Encoder
    Lane 4: DeBERTa NLI
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from lane5_webapp.pipeline import (
    METHOD_LABELS,
    analyse_claim,
    explain_label,
    label_counts,
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ClaimCheck | Evidence-Based Claim Verification",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


EXAMPLE_CLAIMS = [
    "Telemundo is a Spanish-language television network.",
    "Damon Albarn's debut album was released in 2014.",
]


def on_example_change():

    example = st.session_state.example_choice

    if example != "Write my own claim":
        st.session_state.claim_input = example


# ============================================================
# 2. HEADER
# ============================================================

st.title("🔎 ClaimCheck")

st.caption(
    "Evidence-based claim verification · "
    "Applied Natural Language Processing · Research prototype"
)

st.write(
    "Enter a factual statement to retrieve evidence from a "
    "**limited FEVER-Symmetric-derived corpus**. "
    "ClaimCheck analyses how each retrieved sentence relates "
    "to the statement; it does not independently establish "
    "factual truth."
)


# ============================================================
# 3. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About ClaimCheck")

    st.write(
        "**Lane 1:** Prepared corpus and offline evaluation"
    )

    st.write(
        "**Lane 2:** BM25 / TF-IDF retrieval"
    )

    st.write(
        "**Lane 3:** SBERT + FAISS / Cross-Encoder retrieval"
    )

    st.write(
        "**Lane 4:** DeBERTa NLI classification"
    )

    st.write(
        "**Lane 5:** Streamlit application"
    )

    st.info(
        "The overall claim-level verdict awaits the "
        "team's evaluated aggregation rule."
    )

    st.caption(
        "A retrieved evidence ID is not necessarily "
        "a link to a published article."
    )


# ============================================================
# 4. INPUT SECTION
# ============================================================

st.subheader("Check a claim")

if "claim_input" not in st.session_state:
    st.session_state.claim_input = ""

if "analysis" not in st.session_state:
    st.session_state.analysis = None


st.selectbox(
    "Try a claim from the dataset (optional)",
    options=[
        "Write my own claim",
        *EXAMPLE_CLAIMS,
    ],
    key="example_choice",
    on_change=on_example_change,
    help=(
        "Examples come from claims observed in the "
        "team's prepared dataset. Retrieved evidence "
        "may contain contradictory statements."
    ),
)


with st.form("claim_form"):

    claim = st.text_area(
        "Factual claim",
        key="claim_input",
        height=105,
        placeholder="Type one factual statement...",
    )

    col_method, col_k = st.columns(2)

    with col_method:

        method = st.selectbox(
            "Retrieval method",
            options=list(METHOD_LABELS),
            format_func=lambda m: METHOD_LABELS[m],
        )

    with col_k:

        top_k = st.selectbox(
            "Maximum evidence sentences",
            [3, 5, 10],
            index=1,
        )

    submitted = st.form_submit_button(
        "🔍 Check Claim",
        type="primary",
        use_container_width=True,
    )


st.caption(
    "Lexical, semantic and reranker scores use different scales "
    "and cannot be compared directly. Retrieval relevance is "
    "not factual truth. Modern methods may load additional "
    "models on first use."
)


# ============================================================
# 5. RUN ANALYSIS
# ============================================================

if submitted:

    clean_claim = claim.strip()

    st.session_state.analysis = None

    if not clean_claim:

        st.warning(
            "Enter a claim before clicking Check Claim."
        )

    elif len(clean_claim) > 2000:

        st.warning(
            "Please enter a shorter claim "
            "(up to 2,000 characters)."
        )

    else:

        try:

            with st.spinner(
                "Retrieving evidence and running "
                "NLI verification..."
            ):

                results = analyse_claim(
                    clean_claim,
                    method,
                    top_k,
                )

            st.session_state.analysis = {
                "claim": clean_claim,
                "method": method,
                "top_k": top_k,
                "results": results,
            }

        except Exception as error:

            st.error(
                "Analysis could not be completed. "
                "Please check the retrieval index, "
                "corpus and model setup."
            )

            with st.expander("Technical error details"):
                st.exception(error)


# ============================================================
# 6. RESULTS
# ============================================================

analysis = st.session_state.analysis

if analysis is not None:

    results = analysis["results"]
    chosen_method = analysis["method"]

    st.divider()

    st.subheader("Analysis results")

    st.write(
        f"**Claim:** {analysis['claim']}"
    )

    st.caption(
        f"Last run: {METHOD_LABELS[chosen_method]} · "
        f"Requested top {analysis['top_k']} · "
        f"Analysed {len(results)} evidence sentence(s)"
    )

    # --------------------------------------------------------
    # NO EVIDENCE
    # --------------------------------------------------------

    if not results:

        st.warning(
            "No usable evidence was returned "
            "from the available corpus."
        )

        st.info(
            "This does not mean the claim is false. "
            "The required evidence may be missing."
        )

    else:

        counts = label_counts(results)

        supporting = counts["SUPPORTS"]
        refuting = counts["REFUTES"]

        # ----------------------------------------------------
        # RESULT SUMMARY
        # ----------------------------------------------------

        if supporting and refuting:

            st.warning(
                "**Conflicting evidence relationships detected.** "
                "Some retrieved sentences support the claim "
                "and others contradict it. This does not "
                "establish which sentences are trustworthy."
            )

        elif supporting:

            st.info(
                "Some retrieved sentences support the claim. "
                "This is not a final claim verdict."
            )

        elif refuting:

            st.info(
                "Some retrieved sentences contradict the claim. "
                "This is not a final claim verdict."
            )

        else:

            st.info(
                "No retrieved sentence was classified as "
                "supporting or refuting the claim."
            )

        cols = st.columns(3)

        cols[0].metric(
            "Supports",
            counts["SUPPORTS"],
        )

        cols[1].metric(
            "Refutes",
            counts["REFUTES"],
        )

        cols[2].metric(
            "Neutral / NEI",
            counts["NOT ENOUGH INFO"],
        )

        st.caption(
            "Counts are per-sentence relationships, "
            "**not votes on factual truth**."
        )

        # ----------------------------------------------------
        # EVIDENCE AND TECHNICAL TABS
        # ----------------------------------------------------

        evidence_tab, technical_tab = st.tabs(
            [
                "Evidence",
                "Technical details",
            ]
        )

        # ----------------------------------------------------
        # EVIDENCE TAB
        # ----------------------------------------------------

        with evidence_tab:

            st.markdown(
                "#### Retrieved sentences"
            )

            st.caption(
                "Ranked by the selected retrieval method. "
                "Expand each item to inspect its text "
                "and interpretation."
            )

            for rank, item in enumerate(
                results,
                start=1,
            ):

                with st.expander(
                    f"#{rank} · "
                    f"{item['label']} · "
                    f"{item['evidence_id']}",
                    expanded=(rank == 1),
                ):

                    st.write(
                        item["text"]
                    )

                    st.write(
                        f"**Interpretation:** "
                        f"{explain_label(item['label'])}"
                    )

                    if chosen_method == "sbert":
                        score_label = "Cosine similarity"

                    elif chosen_method.endswith("_rerank"):
                        score_label = "Cross-Encoder relevance"

                    else:
                        score_label = (
                            chosen_method.upper() + " score"
                        )

                    st.caption(
                        f"NLI classification confidence: "
                        f"{item['confidence']:.1%} · "
                        f"{score_label}: "
                        f"{item['retrieval_score']:.4f}"
                    )

                    st.caption(
                        f"Evidence ID: {item['evidence_id']} · "
                        "Source: team's FEVER-Symmetric-derived "
                        "evidence corpus"
                    )

        # ----------------------------------------------------
        # TECHNICAL DETAILS TAB
        # ----------------------------------------------------

        with technical_tab:

            st.markdown(
                "#### Pipeline"
            )

            st.code(
                f"Claim → {chosen_method.upper()} → "
                f"top-{analysis['top_k']} evidence → "
                "DeBERTa NLI → evidence labels"
            )

            if chosen_method == "sbert":
                score_label = "Cosine similarity"

            elif chosen_method.endswith("_rerank"):
                score_label = "Cross-Encoder relevance"

            else:
                score_label = (
                    chosen_method.upper() + " score"
                )

            table_rows = [
                {
                    "Rank": rank,
                    "Evidence ID": item["evidence_id"],
                    "Relationship": item["label"],
                    "NLI confidence": round(
                        item["confidence"],
                        4,
                    ),
                    score_label: round(
                        item["retrieval_score"],
                        4,
                    ),
                }
                for rank, item in enumerate(
                    results,
                    start=1,
                )
            ]

            st.dataframe(
                table_rows,
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "The retrieval score measures relevance "
                "to the query, not correctness. NLI "
                "confidence is not a calibrated probability "
                "that a claim is true."
            )

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    with st.expander(
        "Scope and limitations",
        expanded=False,
    ):

        st.markdown(
            """
            - **Corpus coverage:** ClaimCheck searches a
              small FEVER-Symmetric-derived corpus, not
              the live internet.

            - **Lexical retrieval safeguard:** BM25 and
              TF-IDF hide non-positive scores. This is
              not a validated evidence-sufficiency threshold.

            - **Semantic retrieval:** SBERT and Cross-Encoder
              results are not filtered using that heuristic.

            - **Evidence reliability:** The corpus contains
              intentionally contrasting statements for research.

            - **NLI interpretation:** SUPPORTS, REFUTES and
              NOT ENOUGH INFO describe the relationship
              between a claim and an evidence sentence.

            - **Final verdict:** A team-evaluated claim-level
              aggregation function will be integrated later.

            - **Provenance:** Internal evidence IDs do not
              establish a publicly accessible source URL.
            """
        )

    st.caption(
        "ClaimCheck is an educational research prototype. "
        "Independently verify consequential claims."
    )
