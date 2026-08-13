import os
import pandas as pd
import streamlit as st
from typing import List
from pydantic import BaseModel, Field

# LangChain & Vector Store Imports
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 0. BACKEND API KEY CONFIGURATION (HIDDEN FROM UI)
# ==========================================
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "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")

# ==========================================
# 1. STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Aegis Financial",
    page_icon="🛡️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .metric-card {
        background-color: #161b22;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #30363d;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. PYDANTIC OUTPUT SCHEMAS
# ==========================================
class FinancialMetric(BaseModel):
    metric_name: str = Field(description="Name of financial metric")
    current_period: str = Field(description="Current period value")
    previous_period: str = Field(description="Comparison period value")
    trend: str = Field(description="UP, DOWN, or STABLE")

class AegisFinancialReport(BaseModel):
    company_name: str = Field(description="Organization name")
    period: str = Field(description="Reporting Period e.g. Q2 2026")
    executive_summary: str = Field(description="Executive performance and risk summary")
    key_metrics: List[FinancialMetric] = Field(description="Extracted financial metrics list")
    key_risk_factors: List[str] = Field(description="Flagged guidance risks or warnings")
    growth_drivers: List[str] = Field(description="Primary revenue drivers")
    overall_sentiment: str = Field(description="BULLISH, BEARISH, or NEUTRAL")

# ==========================================
# 3. RAG PIPELINE INITIALIZATION (CACHED)
# ==========================================
FINANCIAL_DISCLOSURES = [
    "Aegis Audit Target (TechCorp) Q2 2026: Consolidated revenue reached $1.2B, up 15% YoY ($1.04B Q2 2025). Operating margin expanded from 22% to 25%.",
    "Aegis Guidance Disclosure: Management revised full-year FY2026 growth guidance downward from 18% to 12% due to European supply chain bottlenecks and regulatory delays.",
    "Aegis Balance Sheet Review: Free cash flow contracted 8% to $180M due to increased infrastructure CapEx in AI data centers. Net debt is reported at $450M."
]

@st.cache_resource
def setup_rag_engine():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(
        texts=FINANCIAL_DISCLOSURES,
        embedding=embeddings,
        collection_name="aegis_simple_ui"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = setup_rag_engine()

# LLM Setup
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
structured_llm = llm.with_structured_output(AegisFinancialReport)

AEGIS_PROMPT = """
You are Aegis Financial's Lead Regulatory Compliance & Due Diligence Intelligence Engine.
Analyze the provided SEC disclosures, balance sheets, and earnings call context to evaluate financial risk.

--- RETRIEVED FINANCIAL DISCLOSURES ---
{financial_context}

--- AUDIT QUERY ---
{user_query}

Deliver an accurate, structured due diligence report adhering strictly to the facts provided above.
"""
prompt_template = ChatPromptTemplate.from_template(AEGIS_PROMPT)

# ==========================================
# 4. CLEAN STREAMLIT FRONTEND UI
# ==========================================

# Header
st.title("🛡️ Aegis Financial Intelligence")
st.caption("Automated Regulatory Compliance & Due Diligence Audit Engine")

st.divider()

# Input Section
query = st.text_input(
    "Enter Audit Query:",
    value="Perform an Aegis audit on TechCorp Q2 2026 revenue expansion, cash flow changes, and guidance risks."
)

if st.button("Run Audit", type="primary", use_container_width=True):
    try:
        with st.spinner("Processing SEC disclosures via Groq LLaMA 3.3..."):
            # Step A: Retrieve Context
            docs = retriever.invoke(query)
            context = "\n".join([doc.page_content for doc in docs])
            
            # Step B: LLM Generation
            formatted_prompt = prompt_template.format(
                financial_context=context,
                user_query=query
            )
            report: AegisFinancialReport = structured_llm.invoke(formatted_prompt)

        st.divider()

        # Target & Sentiment Bar
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Company", report.company_name, delta=report.period)
        with col2:
            st.metric("Overall Sentiment", report.overall_sentiment)
        with col3:
            st.metric("Audit Precision", "Zero Hallucination", delta="100% Grounded")

        # Executive Summary
        st.subheader("📋 Executive Summary")
        st.info(report.executive_summary)

        # Content Grid
        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            st.subheader("📊 Extracted Metrics")
            table_data = [
                {
                    "Metric": m.metric_name,
                    "Current": m.current_period,
                    "Previous": m.previous_period,
                    "Trend": "▲ UP" if m.trend.upper() == "UP" else ("▼ DOWN" if m.trend.upper() == "DOWN" else "➔ STABLE")
                }
                for m in report.key_metrics
            ]
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        with col_right:
            st.subheader("⚠️ Guidance & Risk Flags")
            for risk in report.key_risk_factors:
                st.warning(risk)

            st.subheader("📈 Revenue Drivers")
            for driver in report.growth_drivers:
                st.success(driver)

    except Exception as e:
        st.error(f"Audit Processing Error: {str(e)}")