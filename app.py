import os
import re
import json
from typing import List

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from groq import Groq

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 0. ENVIRONMENT / SECRET MANAGEMENT
# ============================================================

load_dotenv()

try:
    secret_groq_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    secret_groq_key = ""

groq_api_key = os.getenv("GROQ_API_KEY") or secret_groq_key


# ============================================================
# 1. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Aegis Financial — SEC Audit Engine",
    page_icon="🛡️",
    layout="wide",
)

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
    unsafe_allow_html=True,
)


# ============================================================
# 2. SIDEBAR
# ============================================================

st.sidebar.title("🛡️ Aegis Financial")

if not groq_api_key:
    entered_key = st.sidebar.text_input(
        "Enter Groq API Key:",
        type="password",
        help="Get your API key from the Groq Console.",
    )

    if entered_key:
        groq_api_key = entered_key
        os.environ["GROQ_API_KEY"] = entered_key

st.sidebar.markdown("---")
st.sidebar.subheader("System Information")
st.sidebar.caption("🟢 Inference: Groq OpenAI GPT-OSS 20B")
st.sidebar.caption("🟢 Database: ChromaDB Vector Store")
st.sidebar.caption("🟢 Embeddings: all-MiniLM-L6-v2")
st.sidebar.caption("🟢 Output: Strict JSON Schema + Pydantic")

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Aegis Financial")
st.sidebar.caption("Institutional Due Diligence Platform")


# ============================================================
# 3. PYDANTIC OUTPUT MODELS
# ============================================================

class FinancialMetric(BaseModel):
    metric_name: str = Field(
        description="Name of the financial metric."
    )
    current_period: str = Field(
        description="Current period value."
    )
    previous_period: str = Field(
        description="Previous/comparison period value."
    )
    trend: str = Field(
        description="Directional trend: UP, DOWN, or STABLE."
    )


class AegisFinancialReport(BaseModel):
    company_name: str
    period: str
    executive_summary: str
    key_metrics: List[FinancialMetric]
    key_risk_factors: List[str]
    growth_drivers: List[str]
    overall_sentiment: str


# ============================================================
# 4. GROQ JSON SCHEMA
#
# IMPORTANT:
# Groq Structured Outputs requires additionalProperties:false
# on EVERY object, including nested FinancialMetric objects.
# ============================================================

AEGIS_FINANCIAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "company_name": {
            "type": "string"
        },
        "period": {
            "type": "string"
        },
        "executive_summary": {
            "type": "string"
        },
        "key_metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "metric_name": {
                        "type": "string"
                    },
                    "current_period": {
                        "type": "string"
                    },
                    "previous_period": {
                        "type": "string"
                    },
                    "trend": {
                        "type": "string",
                        "enum": [
                            "UP",
                            "DOWN",
                            "STABLE"
                        ]
                    }
                },
                "required": [
                    "metric_name",
                    "current_period",
                    "previous_period",
                    "trend"
                ]
            }
        },
        "key_risk_factors": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "growth_drivers": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
        "overall_sentiment": {
            "type": "string",
            "enum": [
                "BULLISH",
                "BEARISH",
                "NEUTRAL"
            ]
        }
    },
    "required": [
        "company_name",
        "period",
        "executive_summary",
        "key_metrics",
        "key_risk_factors",
        "growth_drivers",
        "overall_sentiment"
    ]
}


# ============================================================
# 5. SAMPLE SEC DATA
# ============================================================

RAW_SEC_FILINGS = [
    """
    TechCorp Inc. (NASDAQ: TCHP) - Form 10-Q Disclosure (Q2 2026)

    ITEM 1. FINANCIAL STATEMENTS & MANAGEMENT DISCUSSION

    For the quarter ended June 30, 2026, TechCorp consolidated revenue
    reached $1.2B, reflecting a 15% YoY growth compared to $1.04B
    in Q2 2025.

    Operating margin expanded from 22% in the prior year period
    to 25% due to cloud software licensing momentum.
    """,

    """
    TechCorp Inc. - Balance Sheet & Cash Flow Review (Q2 2026)

    Free cash flow for the second quarter contracted 8% YoY to $180M
    down from $195M in Q2 2025, driven by increased infrastructure
    capital expenditures in AI data center deployment.

    Total outstanding debt stands at $600M offset by $150M in cash
    equivalents, resulting in a Net Debt position of $450M compared
    to $400M in Q2 2025.
    """,

    """
    TechCorp Inc. - Guidance & Risk Factors Disclosure

    MANAGEMENT REVISION & REGULATORY WARNING:

    Management revised full-year FY2026 revenue growth guidance
    downward from 18% to 12%.

    The revision is attributed to European supply chain bottlenecks
    in semiconductor hardware delivery and pending regulatory approval
    delays in cross-border AI data transfers.
    """,
]


# ============================================================
# 6. TEXT CLEANING
# ============================================================

def clean_and_normalize_sec_text(raw_text: str) -> str:
    cleaned = re.sub(r"\s+", " ", raw_text)
    cleaned = re.sub(r"[^\x00-\x7F]+", "", cleaned)
    return cleaned.strip()


# ============================================================
# 7. CHROMADB RAG ENGINE
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
        separators=["\n\n", "\n", " ", ""],
    )

    doc_chunks = text_splitter.split_text(
        " ".join(cleaned_docs)
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_texts(
        texts=doc_chunks,
        embedding=embeddings,
        collection_name="aegis_sec_audits_v2",
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )


retriever = setup_rag_engine()


# ============================================================
# 8. PROMPT
# ============================================================

AEGIS_PROMPT = """
You are Aegis Financial's Lead Regulatory Compliance and
Due Diligence Intelligence Engine.

Analyze ONLY the retrieved SEC disclosure context.

IMPORTANT RULES:

1. Do not invent financial numbers.
2. Do not use information outside the supplied context.
3. Return ONLY the requested structured JSON object.
4. company_name must come from the supplied context.
5. period must come from the supplied context.
6. key_metrics should contain the important financial metrics
   supported by the evidence.
7. trend must be exactly one of:
   UP, DOWN, STABLE.
8. overall_sentiment must be exactly one of:
   BULLISH, BEARISH, NEUTRAL.
9. If a field cannot be supported by the context, use an empty
   string or an empty list instead of inventing information.

--- RETRIEVED SEC DISCLOSURES ---
{financial_context}

--- AUDIT QUERY ---
{user_query}
"""


# ============================================================
# 9. GROQ AUDIT FUNCTION
#
# This deliberately uses the Groq SDK directly instead of
# LangChain's with_structured_output().
#
# That avoids the previous:
# "Unknown tool type: functions.AegisFinancialReport"
# error.
# ============================================================

def run_groq_audit(
    api_key: str,
    financial_context: str,
    user_query: str,
) -> AegisFinancialReport:

    client = Groq(api_key=api_key)

    formatted_prompt = AEGIS_PROMPT.format(
        financial_context=financial_context,
        user_query=user_query,
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial due diligence analysis engine. "
                    "Follow the supplied JSON schema exactly."
                ),
            },
            {
                "role": "user",
                "content": formatted_prompt,
            },
        ],
        temperature=0,
        max_completion_tokens=4096,
        reasoning_effort="low",
        include_reasoning=False,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "aegis_financial_report",
                "strict": True,
                "schema": AEGIS_FINANCIAL_SCHEMA,
            },
        },
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError(
            "Groq returned an empty response."
        )

    try:
        parsed_json = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Groq returned invalid JSON: {exc}"
        ) from exc

    try:
        report = AegisFinancialReport.model_validate(
            parsed_json
        )
    except ValidationError as exc:
        raise ValueError(
            f"Groq returned JSON that failed Pydantic validation: {exc}"
        ) from exc

    return report


# ============================================================
# 10. STREAMLIT DASHBOARD
# ============================================================

st.title("🛡️ Aegis Financial Intelligence")
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
    ),
)

run_audit = st.button(
    "Run Audit",
    type="primary",
    use_container_width=True,
)


# ============================================================
# 11. RUN AUDIT
# ============================================================

if run_audit:

    if not groq_api_key:
        st.error(
            "⚠️ Groq API Key missing! "
            "Enter your key in the sidebar or configure "
            "GROQ_API_KEY in Streamlit Secrets."
        )
        st.stop()

    try:

        with st.spinner(
            "Retrieving SEC evidence and analyzing with Groq..."
        ):

            # --------------------------------------------
            # STEP 1: RAG RETRIEVAL
            # --------------------------------------------

            docs = retriever.invoke(query)

            if not docs:
                st.warning(
                    "No relevant SEC disclosure context was retrieved."
                )
                st.stop()

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            # --------------------------------------------
            # STEP 2: GROQ STRUCTURED OUTPUT
            # --------------------------------------------

            report = run_groq_audit(
                api_key=groq_api_key,
                financial_context=context,
                user_query=query,
            )

        # ====================================================
        # 12. DASHBOARD
        # ====================================================

        st.success(
            "✅ Audit completed successfully with Groq Structured Output."
        )

        st.divider()

        # --------------------------------------------
        # TOP METRICS
        # --------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Company",
                report.company_name,
                delta=report.period,
            )

        with col2:
            sentiment = report.overall_sentiment.upper()

            st.metric(
                "Overall Sentiment",
                sentiment,
            )

        with col3:
            st.metric(
                "Audit Status",
                "GROUNDED",
                delta="Validated",
            )

        # --------------------------------------------
        # EXECUTIVE SUMMARY
        # --------------------------------------------

        st.subheader("📋 Executive Summary")
        st.info(report.executive_summary)

        # --------------------------------------------
        # MAIN DASHBOARD
        # --------------------------------------------

        col_left, col_right = st.columns([1.35, 1])

        with col_left:

            st.subheader("📊 Extracted Financial Metrics")

            table_data = []

            for metric in report.key_metrics:

                trend = metric.trend.upper()

                if trend == "UP":
                    trend_display = "▲ UP"
                elif trend == "DOWN":
                    trend_display = "▼ DOWN"
                else:
                    trend_display = "➔ STABLE"

                table_data.append(
                    {
                        "Metric": metric.metric_name,
                        "Current Period": metric.current_period,
                        "Previous Period": metric.previous_period,
                        "Trend": trend_display,
                    }
                )

            if table_data:

                st.dataframe(
                    pd.DataFrame(table_data),
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                st.warning(
                    "No financial metrics were extracted."
                )

        with col_right:

            st.subheader("⚠️ Guidance & Risk Flags")

            if report.key_risk_factors:

                for risk in report.key_risk_factors:
                    st.warning(
                        f"**Risk Flag:** {risk}"
                    )

            else:
                st.success(
                    "No material risk factors were identified "
                    "in the retrieved context."
                )

            st.subheader("📈 Primary Growth Drivers")

            if report.growth_drivers:

                for driver in report.growth_drivers:
                    st.success(
                        f"**Driver:** {driver}"
                    )

            else:
                st.info(
                    "No growth drivers were identified."
                )

        # --------------------------------------------
        # RETRIEVED EVIDENCE
        # --------------------------------------------

        st.divider()

        with st.expander(
            "🔎 View Retrieved SEC Evidence"
        ):

            for i, doc in enumerate(docs, start=1):

                st.markdown(
                    f"### Evidence Chunk {i}"
                )

                st.write(
                    doc.page_content
                )

        # --------------------------------------------
        # STRUCTURED JSON
        # --------------------------------------------

        with st.expander(
            "🧩 View Validated Structured JSON"
        ):

            st.json(
                report.model_dump()
            )

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            f"❌ Audit Processing Error: {str(e)}"
        )

        with st.expander(
            "🔧 Technical Error Details"
        ):
            st.exception(e)
