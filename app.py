import os
import re
from typing import List

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from groq import Groq

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 0. ENVIRONMENT
# ============================================================

load_dotenv()


def get_groq_api_key():
    """
    Get Groq API key from:
    1. Environment variable
    2. Streamlit secrets
    """

    key = os.getenv("GROQ_API_KEY")

    if key:
        return key.strip()

    try:
        key = st.secrets.get("GROQ_API_KEY", "")

        if key:
            return key.strip()

    except Exception:
        pass

    return ""


groq_api_key = get_groq_api_key()


# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Aegis Financial — SEC Audit Engine",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }

    .metric-card {
        background-color: #161b22;
        padding: 1.2rem;
        border-radius: 0.5rem;
        border: 1px solid #30363d;
        text-align: center;
    }

    .stAlert {
        border-radius: 6px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. GET MODELS AVAILABLE TO THIS API KEY
# ============================================================

def get_available_groq_models(api_key):
    """
    Ask Groq which models are available to this API key.
    """

    try:

        client = Groq(
            api_key=api_key
        )

        models = client.models.list()

        model_ids = []

        for model in models.data:

            if getattr(model, "active", True):

                model_ids.append(
                    model.id
                )

        return sorted(model_ids), None

    except Exception as e:

        return [], str(e)


# ============================================================
# 4. SELECT A GOOD CHAT MODEL
# ============================================================

def select_best_model(available_models):
    """
    Select the best available model from the API key.

    Preference order:
    1. llama-3.1-8b-instant
    2. llama-3.3-70b-versatile
    3. openai/gpt-oss-20b
    4. openai/gpt-oss-120b
    5. qwen models
    """

    preferred_models = [

        "llama-3.1-8b-instant",

        "llama-3.3-70b-versatile",

        "openai/gpt-oss-20b",

        "openai/gpt-oss-120b",

        "qwen/qwen3.6-27b",

        "qwen/qwen3.8-27b",

    ]

    for preferred in preferred_models:

        if preferred in available_models:

            return preferred


    # Fallback
    chat_candidates = [
        model
        for model in available_models
        if not any(
            keyword in model.lower()
            for keyword in [
                "whisper",
                "guard",
                "safety",
                "tts",
                "speech"
            ]
        )
    ]

    if chat_candidates:

        return chat_candidates[0]


    return None


# ============================================================
# 5. SIDEBAR
# ============================================================

st.sidebar.title(
    "🛡️ Aegis Financial"
)


if not groq_api_key:

    groq_api_key = st.sidebar.text_input(
        "Enter Groq API Key:",
        type="password",
        help="Enter your Groq API key."
    )

    if groq_api_key:

        groq_api_key = groq_api_key.strip()

        os.environ["GROQ_API_KEY"] = groq_api_key


# ============================================================
# 6. MODEL DETECTION
# ============================================================

available_models = []
model_error = None
selected_model = None


if groq_api_key:

    with st.spinner(
        "Checking Groq model access..."
    ):

        available_models, model_error = (
            get_available_groq_models(
                groq_api_key
            )
        )


    if available_models:

        default_model = select_best_model(
            available_models
        )

        selected_model = st.sidebar.selectbox(
            "Groq Model:",
            options=available_models,
            index=(
                available_models.index(default_model)
                if default_model in available_models
                else 0
            ),
            help=(
                "These are the models visible to "
                "your Groq API key."
            )
        )


# ============================================================
# 7. SIDEBAR SYSTEM INFORMATION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "System Information"
)


if selected_model:

    st.sidebar.caption(
        f"🟢 **Inference:** `{selected_model}`"
    )

else:

    st.sidebar.caption(
        "🔴 **Inference:** No model detected"
    )


st.sidebar.caption(
    "🟢 **Database:** ChromaDB Vector Store"
)

st.sidebar.caption(
    "🟢 **Embeddings:** all-MiniLM-L6-v2"
)

st.sidebar.caption(
    "🟢 **Analysis:** SEC Grounded RAG"
)


# ============================================================
# 8. MODEL ACCESS INFORMATION
# ============================================================

if groq_api_key:

    if model_error:

        st.sidebar.error(
            "Unable to retrieve Groq models."
        )

    elif available_models:

        st.sidebar.success(
            f"{len(available_models)} Groq models available"
        )

        with st.sidebar.expander(
            "View Available Models"
        ):

            for model in available_models:

                st.write(
                    f"• `{model}`"
                )


st.sidebar.markdown("---")

st.sidebar.caption(
    "© 2026 Aegis Financial. All Rights Reserved."
)

st.sidebar.caption(
    "Institutional Due Diligence Platform."
)


# ============================================================
# 9. PYDANTIC OUTPUT SCHEMAS
# ============================================================

class FinancialMetric(BaseModel):

    metric_name: str = Field(
        description=(
            "Name of financial metric such as "
            "Revenue, Free Cash Flow, or Net Debt."
        )
    )

    current_period: str = Field(
        description="Current period financial value."
    )

    previous_period: str = Field(
        description="Previous period financial value."
    )

    trend: str = Field(
        description=(
            "Directional trend: UP, DOWN, or STABLE."
        )
    )


class AegisFinancialReport(BaseModel):

    company_name: str = Field(
        description="Company or organization name."
    )

    period: str = Field(
        description="Reporting period."
    )

    executive_summary: str = Field(
        description=(
            "Concise financial performance and "
            "risk summary."
        )
    )

    key_metrics: List[FinancialMetric] = Field(
        description=(
            "Important financial metrics extracted "
            "from the SEC context."
        )
    )

    key_risk_factors: List[str] = Field(
        description=(
            "Important financial, regulatory, "
            "guidance, or operational risks."
        )
    )

    growth_drivers: List[str] = Field(
        description=(
            "Primary business or revenue growth drivers."
        )
    )

    overall_sentiment: str = Field(
        description=(
            "BULLISH, BEARISH, or NEUTRAL."
        )
    )


# ============================================================
# 10. SAMPLE SEC FILINGS
# ============================================================

RAW_SEC_FILINGS = [

    """
    TechCorp Inc. (NASDAQ: TCHP)
    Form 10-Q Disclosure (Q2 2026)

    For the quarter ended June 30, 2026,
    TechCorp consolidated revenue reached $1.2B.

    This represents 15% YoY growth compared
    to $1.04B in Q2 2025.

    Operating margin expanded from 22% in
    the prior year period to 25%.

    The improvement was attributed to cloud
    software licensing momentum.
    """,

    """
    TechCorp Inc.
    Balance Sheet & Cash Flow Review (Q2 2026)

    Free cash flow for the second quarter
    contracted 8% YoY to $180M.

    Free cash flow was $195M in Q2 2025.

    The decline was driven by increased
    infrastructure capital expenditures
    related to AI data center deployment.

    Total outstanding debt is $600M.

    Cash equivalents are $150M.

    Net Debt is therefore $450M.

    Net Debt was $400M in Q2 2025.
    """,

    """
    TechCorp Inc.
    Guidance & Risk Factors Disclosure

    Management revised full-year FY2026
    revenue growth guidance downward
    from 18% to 12%.

    The revision is attributed to European
    supply chain bottlenecks in semiconductor
    hardware delivery.

    The company also faces pending regulatory
    approval delays involving cross-border
    AI data transfers.
    """
]


# ============================================================
# 11. TEXT CLEANING
# ============================================================

def clean_and_normalize_sec_text(
    raw_text: str
) -> str:

    cleaned = re.sub(
        r"\s+",
        " ",
        raw_text
    )

    cleaned = re.sub(
        r"[^\x00-\x7F]+",
        "",
        cleaned
    )

    return cleaned.strip()


# ============================================================
# 12. RAG ENGINE
# ============================================================

@st.cache_resource
def setup_rag_engine():

    cleaned_docs = [
        clean_and_normalize_sec_text(
            doc
        )
        for doc in RAW_SEC_FILINGS
    ]


    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )
    )


    combined_text = " ".join(
        cleaned_docs
    )


    doc_chunks = text_splitter.split_text(
        combined_text
    )


    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


    vectorstore = Chroma.from_texts(
        texts=doc_chunks,
        embedding=embeddings,
        collection_name="aegis_sec_audits"
    )


    return vectorstore.as_retriever(
        search_kwargs={
            "k": 3
        }
    )


retriever = setup_rag_engine()


# ============================================================
# 13. PROMPT
# ============================================================

AEGIS_PROMPT = """
You are Aegis Financial's Lead Regulatory Compliance
and Due Diligence Intelligence Engine.

Analyze ONLY the SEC disclosures supplied below.

============================================================
SEC CONTEXT
============================================================

{financial_context}

============================================================
USER AUDIT QUERY
============================================================

{user_query}

============================================================
STRICT RULES
============================================================

1. Use ONLY the provided SEC context.

2. Never invent financial numbers.

3. Never invent dates.

4. Never invent company information.

5. Never use external knowledge.

6. If information is unavailable, state:
   "Not available in the provided SEC context."

7. Identify financial metrics.

8. Identify trends.

9. Identify risks.

10. Identify growth drivers.

11. Identify management guidance changes.

12. Overall sentiment MUST be one of:

BULLISH
BEARISH
NEUTRAL

Return JSON matching the requested schema.
"""


prompt_template = ChatPromptTemplate.from_template(
    AEGIS_PROMPT
)


# ============================================================
# 14. MAIN UI
# ============================================================

st.title(
    "🛡️ Aegis Financial Intelligence"
)

st.caption(
    "Autonomous SEC Due Diligence & Regulatory Risk Audit Engine"
)

st.divider()


query = st.text_input(
    "Enter Audit Query:",
    value=(
        "Perform an Aegis audit on TechCorp Q2 2026 "
        "revenue expansion, free cash flow changes, "
        "net debt, and guidance risks."
    )
)


# ============================================================
# 15. RUN AUDIT
# ============================================================

if st.button(
    "Run Audit",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # API KEY CHECK
    # --------------------------------------------------------

    if not groq_api_key:

        st.error(
            "❌ Groq API key is missing."
        )

        st.info(
            "Enter your Groq API key in the sidebar."
        )

        st.stop()


    # --------------------------------------------------------
    # MODEL CHECK
    # --------------------------------------------------------

    if model_error:

        st.error(
            "❌ Could not retrieve models from Groq."
        )

        st.code(
            model_error
        )

        st.stop()


    if not available_models:

        st.error(
            "❌ Your Groq API key returned no available models."
        )

        st.info(
            "Check your Groq API key and project configuration."
        )

        st.stop()


    if not selected_model:

        st.error(
            "❌ No usable Groq chat model was detected."
        )

        st.stop()


    # ========================================================
    # AUDIT
    # ========================================================

    try:

        with st.spinner(
            f"Running Aegis audit using {selected_model}..."
        ):

            # ------------------------------------------------
            # STEP 1: RETRIEVE DOCUMENTS
            # ------------------------------------------------

            docs = retriever.invoke(
                query
            )


            if not docs:

                st.error(
                    "No relevant SEC context found."
                )

                st.stop()


            # ------------------------------------------------
            # STEP 2: BUILD CONTEXT
            # ------------------------------------------------

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )


            # ------------------------------------------------
            # STEP 3: CREATE GROQ LLM
            # ------------------------------------------------

            llm = ChatGroq(
                model=selected_model,
                api_key=groq_api_key,
                temperature=0
            )


            # ------------------------------------------------
            # STEP 4: STRUCTURED OUTPUT
            # ------------------------------------------------

            structured_llm = (
                llm.with_structured_output(
                    AegisFinancialReport,
                    method="json_mode"
                )
            )


            # ------------------------------------------------
            # STEP 5: FORMAT PROMPT
            # ------------------------------------------------

            formatted_prompt = (
                prompt_template.format(
                    financial_context=context,
                    user_query=query
                )
            )


            # ------------------------------------------------
            # STEP 6: CALL MODEL
            # ------------------------------------------------

            report: AegisFinancialReport = (
                structured_llm.invoke(
                    formatted_prompt
                )
            )


        # ====================================================
        # RESULTS
        # ====================================================

        st.divider()


        # ====================================================
        # TOP METRICS
        # ====================================================

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Company",
                report.company_name,
                delta=report.period
            )


        with col2:

            sentiment = (
                report.overall_sentiment.upper()
            )

            st.metric(
                "Overall Sentiment",
                sentiment
            )


        with col3:

            st.metric(
                "Audit Status",
                "GROUNDED",
                delta="SEC Context"
            )


        # ====================================================
        # EXECUTIVE SUMMARY
        # ====================================================

        st.subheader(
            "📋 Executive Summary"
        )

        st.info(
            report.executive_summary
        )


        # ====================================================
        # DASHBOARD
        # ====================================================

        col_left, col_right = st.columns(
            [1.3, 1]
        )


        # ====================================================
        # FINANCIAL METRICS
        # ====================================================

        with col_left:

            st.subheader(
                "📊 Extracted Financial Metrics"
            )


            table_data = []


            for metric in report.key_metrics:

                trend = (
                    metric.trend.upper()
                    if metric.trend
                    else "STABLE"
                )


                if trend == "UP":

                    display_trend = "▲ UP"

                elif trend == "DOWN":

                    display_trend = "▼ DOWN"

                else:

                    display_trend = "➔ STABLE"


                table_data.append(
                    {
                        "Metric": metric.metric_name,
                        "Current Period": (
                            metric.current_period
                        ),
                        "Previous Period": (
                            metric.previous_period
                        ),
                        "Trend": display_trend
                    }
                )


            if table_data:

                st.dataframe(
                    pd.DataFrame(
                        table_data
                    ),
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No financial metrics extracted."
                )


        # ====================================================
        # RISKS
        # ====================================================

        with col_right:

            st.subheader(
                "⚠️ Guidance & Risk Flags"
            )


            if report.key_risk_factors:

                for risk in (
                    report.key_risk_factors
                ):

                    st.warning(
                        f"**Risk Flag:** {risk}"
                    )

            else:

                st.success(
                    "No major risk factors identified."
                )


            # ------------------------------------------------
            # GROWTH DRIVERS
            # ------------------------------------------------

            st.subheader(
                "📈 Primary Growth Drivers"
            )


            if report.growth_drivers:

                for driver in (
                    report.growth_drivers
                ):

                    st.success(
                        f"**Driver:** {driver}"
                    )

            else:

                st.info(
                    "No growth drivers identified."
                )


        # ====================================================
        # SENTIMENT
        # ====================================================

        st.divider()

        st.subheader(
            "🎯 Investment Sentiment"
        )


        if sentiment == "BULLISH":

            st.success(
                "🟢 BULLISH — Positive financial momentum "
                "is indicated by the available SEC context."
            )

        elif sentiment == "BEARISH":

            st.error(
                "🔴 BEARISH — Significant financial or "
                "regulatory risks are indicated."
            )

        else:

            st.warning(
                "🟡 NEUTRAL — The available evidence "
                "contains mixed signals."
            )


        # ====================================================
        # RETRIEVED CONTEXT
        # ====================================================

        with st.expander(
            "🔍 View Retrieved SEC Context"
        ):

            st.write(
                context
            )


        # ====================================================
        # STRUCTURED REPORT
        # ====================================================

        with st.expander(
            "🧾 View Structured Report"
        ):

            st.json(
                report.model_dump()
            )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        error_text = str(e)

        if (
            "model_not_found"
            in error_text.lower()
        ):

            st.error(
                "❌ The selected model was rejected by Groq."
            )

            st.warning(
                f"Model attempted: `{selected_model}`"
            )

            st.info(
                "Try another model from the Groq Model "
                "dropdown in the sidebar."
            )


        elif "401" in error_text:

            st.error(
                "❌ Invalid or unauthorized Groq API key."
            )


        elif "403" in error_text:

            st.error(
                "❌ Your Groq project does not have "
                "permission to use this model."
            )


        elif "429" in error_text:

            st.warning(
                "⚠️ Groq rate limit reached. "
                "Please wait and try again."
            )


        else:

            st.error(
                f"❌ Audit Processing Error: {error_text}"
            )


        with st.expander(
            "🔧 Technical Error Details"
        ):

            st.exception(e)
