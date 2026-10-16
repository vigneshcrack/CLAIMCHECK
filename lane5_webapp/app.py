
import sys
from pathlib import Path

import streamlit as st

# ==================================================
# 1. PROJECT SETUP
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Lane 2: Traditional evidence retrieval
from lane2_retrieval_traditional.retrieval import (
    get_index,
    retrieve_with_scores,
)

# Lane 4: Claim verification
from lane4_verification.nli_model import verify_batch


# ==================================================
# 2. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="ClaimCheck",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 ClaimCheck")
st.caption("Evidence-Based Claim Verification")

st.write(
    "Enter a factual claim. ClaimCheck retrieves relevant "
    "evidence and uses Natural Language Inference (NLI) "
    "to analyse how each sentence relates to your claim."
)


# ==================================================
# 3. MODEL AND RETRIEVAL FUNCTIONS
# ==================================================

@st.cache_resource(show_spinner=False)
def load_retrieval_index():
    """Load the BM25 index once and reuse it."""
    return get_index("bm25")



def retrieve_evidence(claim: str, k: int = 5):
    """
    Retrieve evidence with BM25, excluding
    candidates with no positive relevance score.
    """
    load_retrieval_index()

    results = retrieve_with_scores(
        claim=claim,
        k=k,
        method="bm25",
    )

    return [
        (evidence_id, score, text)
        for evidence_id, score, text in results
        if score > 0
    ]


def classify_evidence(claim: str, results: list):
    """
    Classify all retrieved evidence sentences.

    Each result from Lane 2:
        (evidence_id, retrieval_score, evidence_text)

    Lane 4 returns one NLI prediction per sentence.
    """
    if not results:
        return []

    evidence_texts = [
        evidence_text
        for _, _, evidence_text in results
    ]

    claims = [claim] * len(evidence_texts)

    return verify_batch(
        claims=claims,
        evidences=evidence_texts,
        batch_size=8,
    )


# ==================================================
# 4. USER INPUT
# ==================================================

st.subheader("Check a Claim")

claim = st.text_area(
    "Enter your factual claim:",
    placeholder="e.g. Paris is the capital of Germany.",
    height=110,
)

top_k = st.selectbox(
    "Number of evidence sentences",
    options=[3, 5, 10],
    index=1,
)

st.caption(
    "Current retrieval method: BM25 (Lane 2). "
    "Semantic retrieval (Lane 3) will be added later."
)


# ==================================================
# 5. CLAIM VERIFICATION WORKFLOW
# ==================================================

if st.button("Check Claim", type="primary"):

    if not claim.strip():
        st.warning("Please enter a factual claim.")

    else:
        claim = claim.strip()

        try:
            # --------------------------------------
            # STEP A: RETRIEVE EVIDENCE
            # --------------------------------------

            with st.spinner("Retrieving relevant evidence..."):
                evidence_results = retrieve_evidence(
                    claim=claim,
                    k=top_k,
                )

            if not evidence_results:
                st.warning(
                    "No evidence was found in the available corpus. "
                    "The claim cannot be assessed."
                )

            else:
                # --------------------------------------
                # STEP B: VERIFY EACH EVIDENCE
                # --------------------------------------

                with st.spinner(
                    "Analysing evidence using DeBERTa..."
                ):
                    predictions = classify_evidence(
                        claim,
                        evidence_results,
                    )

                if len(predictions) != len(evidence_results):
                    raise ValueError(
                        "The number of predictions does not "
                        "match the number of evidence sentences."
                    )

                # --------------------------------------
                # STEP C: DISPLAY RESULTS
                # --------------------------------------

                st.divider()

                st.subheader("Claim")
                st.write(claim)

                st.subheader("Evidence Analysis")

                st.success(
                    f"Analysed {len(predictions)} "
                    "retrieved evidence sentences."
                )

                # Count predictions by category
                counts = {
                    "SUPPORTS": 0,
                    "REFUTES": 0,
                    "NOT ENOUGH INFO": 0,
                }

                for prediction in predictions:
                    label = prediction["label"]
                    if label in counts:
                        counts[label] += 1
                
                # Detect evidence-level disagreement
                has_support = counts["SUPPORTS"] > 0
                has_refute = counts["REFUTES"] > 0

                if has_support and has_refute:
                     st.warning(
                        "⚠️ Conflicting evidence detected: Some retrieved "
                        "sentences support the claim, while others contradict it. "
                        "The system cannot establish which statements are more "
                        "trustworthy from NLI predictions alone."
                    )

                elif has_support:
                    st.info(
                        "The retrieved evidence includes statements that "
                        "the model considers supportive of the claim. "
                        "This is not yet a final claim-level verdict."
                    )

                elif has_refute:
                    st.info(
                        "The retrieved evidence includes statements that "
                        "the model considers contradictory to the claim. "
                        "This is not yet a final claim-level verdict."
                    )

                else:
                    st.info(
                        "The retrieved sentences do not provide a clear "
                        "supporting or contradicting relationship. "
                        "The claim remains unverified using these results."
                    )


                col1, col2, col3 = st.columns(3)

                col1.metric("Supports", counts["SUPPORTS"])
                col2.metric("Refutes", counts["REFUTES"])
                col3.metric(
                    "Neutral / NEI",
                    counts["NOT ENOUGH INFO"],
                )

                st.caption(
                    "These counts describe how the model "
                    "classified individual retrieved sentences. "
                    "They are not votes on the claim's truth."
                )

                st.divider()

                # --------------------------------------
                # STEP D: EVIDENCE CARDS
                # --------------------------------------

                st.subheader("Retrieved Evidence")

                for rank, (result, prediction) in enumerate(
                    zip(evidence_results, predictions),
                    start=1,
                ):
                    evidence_id, retrieval_score, text = result

                    label = prediction["label"]
                    confidence = float(prediction["confidence"])

                    with st.expander(f"Evidence {rank} — {prediction['label']}",expanded=(rank == 1),):

                        if label == "SUPPORTS":
                            st.success("SUPPORTS")

                        elif label == "REFUTES":
                            st.error("REFUTES")

                        else:
                            st.warning("NOT ENOUGH INFO")

                        st.write(text)

                        col_a, col_b = st.columns(2)

                        with col_a:
                            st.metric(
                                "NLI confidence",
                                f"{confidence:.1%}",
                            )

                        with col_b:
                            st.metric(
                                "BM25 relevance score",
                                f"{retrieval_score:.2f}",
                            )

                        st.caption(
                            f"Evidence ID: {evidence_id}"
                        )

                        if label == "SUPPORTS":
                            st.write(
                                "**Interpretation:** The model "
                                "considers this evidence to "
                                "support the claim."
                            )

                        elif label == "REFUTES":
                            st.write(
                                "**Interpretation:** The model "
                                "considers this evidence to "
                                "contradict the claim."
                            )

                        else:
                            st.write(
                                "**Interpretation:** This evidence "
                                "does not clearly establish or "
                                "contradict the claim."
                            )

                # --------------------------------------
                # STEP E: LIMITATIONS
                # --------------------------------------

                st.divider()

                st.info(
                    "ClaimCheck currently reports evidence-level "
                    "predictions, not an aggregated claim verdict. "
                    "Results depend on the available evidence "
                    "corpus and model accuracy."
                )

                st.caption(
                    "NLI confidence describes the model's "
                    "classification confidence for a particular "
                    "claim-evidence pair. BM25 scores measure "
                    "lexical relevance, not factual certainty."
                )

        except Exception as error:
            st.error(
                "Something went wrong while checking the claim."
            )
            st.exception(error)
