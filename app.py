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
# 0. ENVIRONMENT / SECRET MANAGEMENT
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
# GROQ MODEL CONFIGURATION
# ============================================================

# Preferred model.
# The application will verify which models are actually
# available to the supplied Groq API key.
PREFERRED_GROQ_MODEL = "llama-3.1-8b-instant"


# ============================================================
# 1. STREAMLIT PAGE CONFIGURATION
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
# 3. GROQ MODEL DISCOVERY
# ============================================================

def get_available_groq_models(api_key):
    """
    Retrieve the models available to the current Groq API key.
    """

    try:

        client = Groq(
            api_key=api_key
        )

        response = client.models.list()

        models = []

        for model in response.data:

            model_id = getattr(
                model,
                "id",
                None
            )

            if model_id:
                models.append(model_id)

        return sorted(models), None

    except Exception as e:

        return [], str(e)


# ============================================================
# 4. SELECT BEST MODEL
# ============================================================

def select_best_model(available_models):
    """
    Select a suitable model.

    Priority:
    1. Llama 3.1 8B Instant
    2. Llama 3.3 70B Versatile
    3. OpenAI OSS models
    4. Other available models
    """

    preferred_models = [

        "llama-3.1-8b-instant",

        "llama-3.3-70b-versatile",

        "openai/gpt-oss-20b",

        "openai/gpt-oss-120b",

    ]

    for model in preferred_models:

        if model in available_models:

            return model


    # Exclude models that are clearly not chat models
    excluded_keywords = [
        "whisper",
        "tts",
        "speech",
        "guard",
        "safety",
        "audio",
        "embedding"
    ]


    chat_candidates = [

        model
        for model in available_models
        if not any(
            keyword in model.lower()
            for keyword in excluded_keywords
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
# 6. CHECK GROQ MODELS
# ============================================================

available_models = []
model_error = None
selected_model = None


if groq_api_key:

    with st.spinner(
        "Checking Groq model access..."
    ):

        (
            available_models,
            model_error
        ) = get_available_groq_models(
            groq_api_key
        )


    if available_models:

        default_model = select_best_model(
            available_models
        )


        if default_model:

            default_index = (
                available_models.index(
                    default_model
                )
            )

        else:

            default_index = 0


        selected_model = st.sidebar.selectbox(
            "Groq Model:",
            options=available_models,
            index=default_index,
            help=(
                "Models returned by your Groq API key."
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


if available_models:

    st.sidebar.success(
        f"{len(available_models)} Groq model(s) available"
    )


    with st.sidebar.expander(
        "View Available Models"
    ):

        for model in available_models:

            st.write(
                f"• `{model}`"
            )


elif model_error:

    st.sidebar.error(
        "Could not retrieve Groq models."
    )


st.sidebar.markdown("---")

st.sidebar.caption(
    "© 2026 Aegis Financial. All Rights Reserved."
)

st.sidebar.caption(
    "Institutional Due Diligence Platform."
)


# ============================================================
# 8. PYDANTIC OUTPUT SCHEMAS
# ============================================================

class FinancialMetric(BaseModel):

    metric_name: str = Field(
        description=(
            "Name of the financial metric. "
            "Examples: Consolidated Revenue, "
            "Operating Margin, Free Cash Flow, Net Debt."
        )
    )

    current_period: str = Field(
        description=(
            "Value for the current reporting period."
        )
    )

    previous_period: str = Field(
        description=(
            "Value for the previous/comparison period."
        )
    )

    trend: str = Field(
        description=(
            "Financial direction. Must be UP, DOWN, or STABLE."
        )
    )


class AegisFinancialReport(BaseModel):

    company_name: str = Field(
        description=(
            "Company name extracted from the SEC context."
        )
    )

    period: str = Field(
        description=(
            "Reporting period, such as Q2 2026."
        )
    )

    executive_summary: str = Field(
        description=(
            "Concise summary of financial performance, "
            "capital allocation, guidance and risks."
        )
    )

    key_metrics: List[FinancialMetric] = Field(
        description=(
            "List of important financial metrics "
            "extracted from the SEC context."
        )
    )

    key_risk_factors: List[str] = Field(
        description=(
            "Important financial, operational, regulatory, "
            "supply chain or guidance risks."
        )
    )

    growth_drivers: List[str] = Field(
        description=(
            "Business or revenue growth drivers "
            "explicitly supported by the SEC context."
        )
    )

    overall_sentiment: str = Field(
        description=(
            "Overall sentiment. Must be exactly "
            "BULLISH, BEARISH, or NEUTRAL."
        )
    )


# ============================================================
# 9. SAMPLE SEC FILINGS
# ============================================================

RAW_SEC_FILINGS = [

    """
    TechCorp Inc. (NASDAQ: TCHP)
    Form 10-Q Disclosure (Q2 2026)

    ITEM 1. FINANCIAL STATEMENTS & MANAGEMENT DISCUSSION

    For the quarter ended June 30, 2026, TechCorp
    consolidated revenue reached $1.2B, reflecting
    a 15% YoY growth compared to $1.04B in Q2 2025.

    Operating margin expanded from 22% in the prior
    year period to 25% due to cloud software
    licensing momentum.
    """,

    """
    TechCorp Inc.
    Balance Sheet & Cash Flow Review (Q2 2026)

    Free cash flow for the second quarter contracted
    8% YoY to $180M, down from $195M in Q2 2025.

    The decline was driven by increased infrastructure
    capital expenditures in AI data center deployment.

    Total outstanding debt stands at $600M offset by
    $150M in cash equivalents.

    This results in a Net Debt position of $450M,
    compared to $400M in Q2 2025.
    """,

    """
    TechCorp Inc.
    Guidance & Risk Factors Disclosure

    MANAGEMENT REVISION & REGULATORY WARNING:

    Management revised full-year FY2026 revenue growth
    guidance downward from 18% to 12%.

    The revision is attributed to European supply chain
    bottlenecks in semiconductor hardware delivery.

    There are also pending regulatory approval delays
    in cross-border AI data transfers.
    """
]


# ============================================================
# 10. TEXT CLEANING
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
# 11. RAG ENGINE
# ============================================================

@st.cache_resource
def setup_rag_engine():

    cleaned_docs = [
        clean_and_normalize_sec_text(
            document
        )
        for document in RAW_SEC_FILINGS
    ]


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            " ",
            ""
        ]
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


    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3
        }
    )


    return retriever


# Initialize RAG
retriever = setup_rag_engine()


# ============================================================
# 12. AEGIS PROMPT
# ============================================================

AEGIS_PROMPT = """
You are Aegis Financial's Lead Regulatory Compliance
and Due Diligence Intelligence Engine.

Analyze ONLY the SEC context provided below.

============================================================
RETRIEVED SEC CONTEXT
============================================================

{financial_context}

============================================================
AUDIT QUERY
============================================================

{user_query}

============================================================
CRITICAL OUTPUT INSTRUCTIONS
============================================================

You MUST fill the AegisFinancialReport schema.

Do NOT create a custom JSON structure.

Do NOT return these as top-level fields:

- revenue
- operating_margin
- free_cash_flow
- net_debt
- guidance
- risks

Instead, map the information into the required
AegisFinancialReport fields:

- company_name
- period
- executive_summary
- key_metrics
- key_risk_factors
- growth_drivers
- overall_sentiment

============================================================
KEY METRICS
============================================================

Every key metric MUST contain:

- metric_name
- current_period
- previous_period
- trend

The trend MUST be:

UP
DOWN
STABLE

============================================================
EXAMPLE
============================================================

If the SEC context states:

Revenue reached $1.2B compared to $1.04B,
representing 15% YoY growth.

Then create:

metric_name:
"Consolidated Revenue"

current_period:
"$1.2B"

previous_period:
"$1.04B"

trend:
"UP"

============================================================
ANOTHER EXAMPLE
============================================================

If the SEC context states:

Free cash flow declined from $195M to $180M.

Then create:

metric_name:
"Free Cash Flow"

current_period:
"$180M"

previous_period:
"$195M"

trend:
"DOWN"

============================================================
NET DEBT
============================================================

If the SEC context states:

Net Debt increased from $400M to $450M.

Create:

metric_name:
"Net Debt"

current_period:
"$450M"

previous_period:
"$400M"

trend:
"UP"

============================================================
OPERATING MARGIN
============================================================

If the SEC context states:

Operating margin increased from 22% to 25%.

Create:

metric_name:
"Operating Margin"

current_period:
"25%"

previous_period:
"22%"

trend:
"UP"

============================================================
RISKS
============================================================

Put identified risks into:

key_risk_factors

Examples from this context include:

- European semiconductor supply chain bottlenecks
- Cross-border AI data transfer regulatory delays
- Downward revision of FY2026 revenue guidance

============================================================
GROWTH DRIVERS
============================================================

Put positive business drivers into:

growth_drivers

For example:

- Cloud software licensing momentum

============================================================
SENTIMENT
============================================================

overall_sentiment MUST be exactly one of:

BULLISH
BEARISH
NEUTRAL

Base the sentiment only on the supplied context.

============================================================
GROUNDING RULES
============================================================

Use ONLY the supplied SEC context.

Never invent financial numbers.

Never invent dates.

Never invent companies.

Never invent financial metrics.

Never use outside knowledge.

If a requested fact is unavailable, write:

"Not available in the provided SEC context."

============================================================
FINAL REQUIREMENT
============================================================

Return the AegisFinancialReport structure.
"""


prompt_template = ChatPromptTemplate.from_template(
    AEGIS_PROMPT
)


# ============================================================
# 13. MAIN APPLICATION
# ============================================================

st.title(
    "🛡️ Aegis Financial Intelligence"
)

st.caption(
    "Autonomous SEC Due Diligence & Regulatory Risk Audit Engine"
)

st.divider()


# ============================================================
# 14. USER QUERY
# ============================================================

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

    # ========================================================
    # API KEY VALIDATION
    # ========================================================

    if not groq_api_key:

        st.error(
            "❌ Groq API key is missing."
        )

        st.info(
            "Enter your Groq API key in the sidebar."
        )

        st.stop()


    # ========================================================
    # MODEL API CHECK
    # ========================================================

    if model_error:

        st.error(
            "❌ Unable to communicate with Groq."
        )

        st.code(
            model_error
        )

        st.stop()


    # ========================================================
    # AVAILABLE MODEL CHECK
    # ========================================================

    if not available_models:

        st.error(
            "❌ No Groq models were returned for "
            "your API key."
        )

        st.info(
            "Please verify your Groq API key and project."
        )

        st.stop()


    # ========================================================
    # SELECTED MODEL CHECK
    # ========================================================

    if not selected_model:

        st.error(
            "❌ No usable chat model was detected."
        )

        st.stop()


    # ========================================================
    # AUDIT PROCESSING
    # ========================================================

    try:

        with st.spinner(
            f"Running Aegis audit using {selected_model}..."
        ):

            # ------------------------------------------------
            # STEP 1
            # Retrieve relevant SEC chunks
            # ------------------------------------------------

            docs = retriever.invoke(
                query
            )


            if not docs:

                st.error(
                    "No relevant SEC documents were found."
                )

                st.stop()


            # ------------------------------------------------
            # STEP 2
            # Build context
            # ------------------------------------------------

            context = "\n\n".join(
                [
                    document.page_content
                    for document in docs
                ]
            )


            # ------------------------------------------------
            # STEP 3
            # Create Groq model
            # ------------------------------------------------

            llm = ChatGroq(
                model=selected_model,
                api_key=groq_api_key,
                temperature=0
            )


            # ------------------------------------------------
            # STEP 4
            # STRUCTURED OUTPUT
            #
            # IMPORTANT:
            # function_calling is used instead of json_mode.
            # This allows LangChain to provide the Pydantic
            # schema to the model.
            # ------------------------------------------------

            structured_llm = (
                llm.with_structured_output(
                    AegisFinancialReport,
                    method="function_calling"
                )
            )


            # ------------------------------------------------
            # STEP 5
            # Format prompt
            # ------------------------------------------------

            formatted_prompt = (
                prompt_template.format(
                    financial_context=context,
                    user_query=query
                )
            )


            # ------------------------------------------------
            # STEP 6
            # Call Groq
            # ------------------------------------------------

            report: AegisFinancialReport = (
                structured_llm.invoke(
                    formatted_prompt
                )
            )


        # ====================================================
        # SUCCESS
        # ====================================================

        st.success(
            "✅ Aegis audit completed successfully."
        )

        st.divider()


        # ====================================================
        # TOP METRICS
        # ====================================================

        col1, col2, col3 = st.columns(3)


        # ----------------------------------------------------
        # COMPANY
        # ----------------------------------------------------

        with col1:

            st.metric(
                "Company",
                report.company_name,
                delta=report.period
            )


        # ----------------------------------------------------
        # SENTIMENT
        # ----------------------------------------------------

        with col2:

            sentiment = (
                report.overall_sentiment
                .upper()
                .strip()
            )

            st.metric(
                "Overall Sentiment",
                sentiment
            )


        # ----------------------------------------------------
        # AUDIT STATUS
        # ----------------------------------------------------

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
                    metric.trend.upper().strip()
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
        # RISK FACTORS
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
        # SENTIMENT ANALYSIS
        # ====================================================

        st.divider()

        st.subheader(
            "🎯 Investment Sentiment"
        )


        if sentiment == "BULLISH":

            st.success(
                "🟢 BULLISH — The available SEC context "
                "indicates positive financial momentum."
            )

        elif sentiment == "BEARISH":

            st.error(
                "🔴 BEARISH — The available SEC context "
                "indicates significant financial or "
                "regulatory risks."
            )

        else:

            st.warning(
                "🟡 NEUTRAL — The available SEC context "
                "contains mixed financial signals."
            )


        # ====================================================
        # RETRIEVED SEC CONTEXT
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
            "🧾 View Structured Aegis Report"
        ):

            st.json(
                report.model_dump()
            )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        error_text = str(e)

        error_lower = (
            error_text.lower()
        )


        # ----------------------------------------------------
        # MODEL NOT FOUND
        # ----------------------------------------------------

        if "model_not_found" in error_lower:

            st.error(
                "❌ Groq rejected the selected model."
            )

            st.warning(
                f"Selected model: `{selected_model}`"
            )

            st.info(
                "Choose another model from the "
                "Groq Model dropdown in the sidebar."
            )


        # ----------------------------------------------------
        # API KEY
        # ----------------------------------------------------

        elif (
            "401" in error_text
            or "authentication" in error_lower
            or "invalid api key" in error_lower
        ):

            st.error(
                "❌ Groq API authentication failed."
            )

            st.info(
                "Check your GROQ_API_KEY."
            )


        # ----------------------------------------------------
        # PERMISSION
        # ----------------------------------------------------

        elif (
            "403" in error_text
            or "permission" in error_lower
        ):

            st.error(
                "❌ Your Groq project does not have "
                "permission to use this model."
            )


        # ----------------------------------------------------
        # RATE LIMIT
        # ----------------------------------------------------

        elif (
            "429" in error_text
            or "rate limit" in error_lower
        ):

            st.warning(
                "⚠️ Groq rate limit reached."
            )

            st.info(
                "Wait a moment and try again."
            )


        # ----------------------------------------------------
        # STRUCTURED OUTPUT ERROR
        # ----------------------------------------------------

        elif (
            "outputparser" in error_lower
            or "validation error" in error_lower
            or "pydantic" in error_lower
            or "failed to parse" in error_lower
        ):

            st.error(
                "❌ The Groq response could not be "
                "converted into the AegisFinancialReport schema."
            )

            st.info(
                "The model responded, but its output did "
                "not match the required structured format."
            )


        # ----------------------------------------------------
        # GENERAL ERROR
        # ----------------------------------------------------

        else:

            st.error(
                f"❌ Audit Processing Error: {error_text}"
            )


        # ----------------------------------------------------
        # TECHNICAL DETAILS
        # ----------------------------------------------------

        with st.expander(
            "🔧 Technical Error Details"
        ):

            st.exception(e)
