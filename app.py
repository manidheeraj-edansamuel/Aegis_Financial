import os
import re
from typing import List

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain
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
        return key.strip() if key else ""
    except Exception:
        return ""


groq_api_key = get_groq_api_key()


# ============================================================
# GROQ MODEL CONFIGURATION
# ============================================================

# Current Groq production model
GROQ_MODEL = "llama-3.1-8b-instant"


# ============================================================
# 1. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Aegis Financial — SEC Audit Engine",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
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
# SIDEBAR
# ============================================================

st.sidebar.title("🛡️ Aegis Financial")

if not groq_api_key:

    groq_api_key = st.sidebar.text_input(
        "Enter Groq API Key:",
        type="password",
        help="Get your API key from GroqCloud."
    )

    if groq_api_key:
        groq_api_key = groq_api_key.strip()
        os.environ["GROQ_API_KEY"] = groq_api_key


st.sidebar.markdown("---")

st.sidebar.subheader("System Information")

st.sidebar.caption(
    "🟢 **Inference:** Groq LLaMA 3.1 8B Instant"
)

st.sidebar.caption(
    "🟢 **Database:** ChromaDB Vector Store"
)

st.sidebar.caption(
    "🟢 **Embeddings:** all-MiniLM-L6-v2"
)

st.sidebar.caption(
    "🟢 **Compliance:** Grounded SEC Analysis"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "© 2026 Aegis Financial. All Rights Reserved."
)

st.sidebar.caption(
    "Institutional Due Diligence Platform."
)


# ============================================================
# 2. PYDANTIC OUTPUT SCHEMAS
# ============================================================

class FinancialMetric(BaseModel):

    metric_name: str = Field(
        description=(
            "Name of financial metric, such as "
            "Consolidated Revenue, Free Cash Flow, or Net Debt."
        )
    )

    current_period: str = Field(
        description=(
            "Current period value, for example $1.2B or $180M."
        )
    )

    previous_period: str = Field(
        description=(
            "Previous/comparison period value."
        )
    )

    trend: str = Field(
        description=(
            "Directional trend: UP, DOWN, or STABLE."
        )
    )


class AegisFinancialReport(BaseModel):

    company_name: str = Field(
        description="Organization name."
    )

    period: str = Field(
        description="Reporting period."
    )

    executive_summary: str = Field(
        description=(
            "Concise financial performance, "
            "capital allocation, and risk summary."
        )
    )

    key_metrics: List[FinancialMetric] = Field(
        description=(
            "Important extracted financial metrics."
        )
    )

    key_risk_factors: List[str] = Field(
        description=(
            "Important guidance, supply chain, "
            "regulatory, or financial risks."
        )
    )

    growth_drivers: List[str] = Field(
        description=(
            "Primary business or revenue growth drivers."
        )
    )

    overall_sentiment: str = Field(
        description=(
            "Overall investment sentiment: "
            "BULLISH, BEARISH, or NEUTRAL."
        )
    )


# ============================================================
# 3. SAMPLE SEC DATA
# ============================================================

RAW_SEC_FILINGS = [

    """
    TechCorp Inc. (NASDAQ: TCHP) - Form 10-Q Disclosure (Q2 2026)

    ITEM 1. FINANCIAL STATEMENTS & MANAGEMENT DISCUSSION

    For the quarter ended June 30, 2026, TechCorp consolidated
    revenue reached $1.2B, reflecting a 15% YoY growth compared
    to $1.04B in Q2 2025.

    Operating margin expanded from 22% in the prior year period
    to 25% due to cloud software licensing momentum.
    """,

    """
    TechCorp Inc. - Balance Sheet & Cash Flow Review (Q2 2026)

    Free cash flow for the second quarter contracted 8% YoY
    to $180M, down from $195M in Q2 2025.

    The decline was driven by increased infrastructure
    capital expenditures related to AI data center deployment.

    Total outstanding debt stands at $600M.

    Cash equivalents stand at $150M.

    This results in Net Debt of $450M compared to
    $400M in Q2 2025.
    """,

    """
    TechCorp Inc. - Guidance & Risk Factors Disclosure

    MANAGEMENT REVISION & REGULATORY WARNING:

    Management revised full-year FY2026 revenue growth guidance
    downward from 18% to 12%.

    The revision is attributed to European supply chain
    bottlenecks in semiconductor hardware delivery.

    The company also faces pending regulatory approval delays
    involving cross-border AI data transfers.
    """
]


# ============================================================
# 4. CLEAN SEC DOCUMENTS
# ============================================================

def clean_and_normalize_sec_text(raw_text: str) -> str:

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
# 5. RAG ENGINE
# ============================================================

@st.cache_resource
def setup_rag_engine():

    cleaned_docs = [
        clean_and_normalize_sec_text(doc)
        for doc in RAW_SEC_FILINGS
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

    combined_text = " ".join(cleaned_docs)

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
# 6. AEGIS PROMPT
# ============================================================

AEGIS_PROMPT = """
You are Aegis Financial's Lead Regulatory Compliance
and Due Diligence Intelligence Engine.

Your job is to analyze the retrieved SEC disclosures
and produce an accurate financial due diligence report.

============================================================
RETRIEVED SEC DISCLOSURES
============================================================

{financial_context}

============================================================
AUDIT QUERY
============================================================

{user_query}

============================================================
STRICT ANALYSIS RULES
============================================================

1. Use ONLY information contained in the retrieved SEC context.

2. Do NOT invent financial numbers.

3. Do NOT invent dates.

4. Do NOT invent companies.

5. Do NOT invent financial metrics.

6. Do NOT use outside knowledge.

7. If information is unavailable, say:
   "Not available in the provided SEC context."

8. Calculate trends only when the relevant values
   are explicitly present in the context.

9. Identify important financial risks.

10. Identify revenue and business growth drivers.

11. Identify changes in management guidance.

12. Determine overall sentiment as exactly one of:

    BULLISH
    BEARISH
    NEUTRAL

============================================================
OUTPUT REQUIREMENT
============================================================

Return valid JSON that matches the requested
AegisFinancialReport structure.

Do not include Markdown code fences.

Do not include explanations outside the JSON.
"""


prompt_template = ChatPromptTemplate.from_template(
    AEGIS_PROMPT
)


# ============================================================
# 7. STREAMLIT MAIN UI
# ============================================================

st.title(
    "🛡️ Aegis Financial Intelligence"
)

st.caption(
    "Autonomous SEC Due Diligence & Regulatory Risk Audit Engine"
)

st.divider()


# ============================================================
# QUERY INPUT
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
# RUN AUDIT BUTTON
# ============================================================

if st.button(
    "Run Audit",
    type="primary",
    use_container_width=True
):

    # ========================================================
    # API KEY CHECK
    # ========================================================

    if not groq_api_key:

        st.error(
            "⚠️ Groq API Key is missing. "
            "Enter your Groq API key in the sidebar "
            "or configure GROQ_API_KEY."
        )

        st.stop()


    try:

        # ====================================================
        # PROCESSING
        # ====================================================

        with st.spinner(
            "Processing SEC disclosures with Groq..."
        ):

            # ==================================================
            # STEP 1: RETRIEVE DOCUMENTS
            # ==================================================

            docs = retriever.invoke(
                query
            )

            if not docs:

                st.error(
                    "No relevant SEC documents were found."
                )

                st.stop()


            # ==================================================
            # STEP 2: BUILD CONTEXT
            # ==================================================

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )


            # ==================================================
            # STEP 3: CREATE GROQ MODEL
            # ==================================================

            llm = ChatGroq(
                model=GROQ_MODEL,
                api_key=groq_api_key,
                temperature=0
            )


            # ==================================================
            # STEP 4: STRUCTURED OUTPUT
            # ==================================================

            structured_llm = llm.with_structured_output(
                AegisFinancialReport,
                method="json_mode"
            )


            # ==================================================
            # STEP 5: FORMAT PROMPT
            # ==================================================

            formatted_prompt = prompt_template.format(
                financial_context=context,
                user_query=query
            )


            # ==================================================
            # STEP 6: CALL GROQ
            # ==================================================

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
        # MAIN DASHBOARD
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
                        "Current Period": metric.current_period,
                        "Previous Period": metric.previous_period,
                        "Trend": display_trend
                    }
                )


            if table_data:

                df = pd.DataFrame(
                    table_data
                )

                st.dataframe(
                    df,
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

                for risk in report.key_risk_factors:

                    st.warning(
                        f"**Risk Flag:** {risk}"
                    )

            else:

                st.success(
                    "No major risk factors identified."
                )


            # =================================================
            # GROWTH DRIVERS
            # =================================================

            st.subheader(
                "📈 Primary Growth Drivers"
            )


            if report.growth_drivers:

                for driver in report.growth_drivers:

                    st.success(
                        f"**Driver:** {driver}"
                    )

            else:

                st.info(
                    "No growth drivers identified."
                )


        # ====================================================
        # OVERALL SENTIMENT
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
                "indicates significant financial or regulatory risks."
            )

        else:

            st.warning(
                "🟡 NEUTRAL — The available SEC context "
                "contains mixed or insufficient signals."
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
        # RAW STRUCTURED REPORT
        # ====================================================

        with st.expander(
            "🧾 View Structured JSON Report"
        ):

            try:

                st.json(
                    report.model_dump()
                )

            except Exception:

                st.write(
                    report
                )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        error_message = str(e)


        if "model_not_found" in error_message.lower():

            st.error(
                f"""
                ❌ Groq model access error.

                The application is configured to use:

                `{GROQ_MODEL}`

                Groq currently lists this model as:
                **Llama 3.1 8B Instant**

                Check that your Groq API key/project has
                access to this model.
                """
            )

            st.code(
                GROQ_MODEL
            )


        elif "401" in error_message:

            st.error(
                """
                ❌ Invalid Groq API key.

                Please check your GROQ_API_KEY and make sure
                it is active.
                """
            )


        elif "429" in error_message:

            st.error(
                """
                ⚠️ Groq rate limit reached.

                Please wait a moment and try again.
                """
            )


        else:

            st.error(
                f"❌ Audit Processing Error: {error_message}"
            )


        # Show technical traceback in an expandable section
        with st.expander(
            "🔧 Technical Error Details"
        ):

            st.exception(e)
