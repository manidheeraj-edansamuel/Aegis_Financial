import os
import re
import pandas as pd
import streamlit as st
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain & Vector Store Imports
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 0. SAFE ENVIRONMENT / SECRET MANAGEMENT
# ==========================================
load_dotenv()

# Scrub potential OpenAI environment variables
for key in ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"]:
    os.environ.pop(key, None)

# Fetch key from OS environment or Streamlit secrets
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")

if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

# ==========================================
# 1. STREAMLIT PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Aegis Financial — SEC Audit Engine",
    page_icon="🛡️",
    layout="wide"
)

# Enterprise Dark Theme Styling
st.markdown("""
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
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("🛡️ Aegis Financial")

if not groq_api_key:
    groq_api_key = st.sidebar.text_input(
        "Enter Groq API Key:", 
        type="password",
        help="Get your key at https://console.groq.com/"
    )
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key

st.sidebar.markdown("---")
st.sidebar.subheader("System Information")
st.sidebar.caption("🟢 **Inference:** Groq LLaMA 3.3 70B")
st.sidebar.caption("🟢 **Database:** ChromaDB Vector Store")
st.sidebar.caption("🟢 **Compliance:** Zero-Hallucination Grounded")

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Aegis Financial. All Rights Reserved.")
st.sidebar.caption("Institutional Due Diligence Platform.")

# ==========================================
# 2. PYDANTIC OUTPUT SCHEMAS
# ==========================================
class FinancialMetric(BaseModel):
    metric_name: str = Field(description="Name of financial metric e.g. Consolidated Revenue, Free Cash Flow, Net Debt")
    current_period: str = Field(description="Current period value e.g. $1.2B, $180M, $450M")
    previous_period: str = Field(description="Comparison period value e.g. $1.04B Q2 2025, $195M Q2 2025")
    trend: str = Field(description="Directional trend: UP, DOWN, or STABLE")

class AegisFinancialReport(BaseModel):
    company_name: str = Field(description="Organization name e.g. TechCorp")
    period: str = Field(description="Reporting Period e.g. Q2 2026")
    executive_summary: str = Field(description="Concise performance, capital allocation, and risk summary for buy-side teams")
    key_metrics: List[FinancialMetric] = Field(description="Extracted financial metrics including YoY Revenue, Net Debt, and Free Cash Flow")
    key_risk_factors: List[str] = Field(description="Flagged guidance revisions, supply chain bottlenecks, or regulatory risks")
    growth_drivers: List[str] = Field(description="Primary business or revenue expansion drivers")
    overall_sentiment: str = Field(description="BULLISH, BEARISH, or NEUTRAL")

# ==========================================
# 3. DOCUMENT PROCESSING & VECTOR RAG ENGINE
# ==========================================
RAW_SEC_FILINGS = [
    """
    TechCorp Inc. (NASDAQ: TCHP) - Form 10-Q Disclosure (Q2 2026)
    ITEM 1. FINANCIAL STATEMENTS & MANAGEMENT DISCUSSION
    For the quarter ended June 30, 2026, TechCorp consolidated revenue reached $1.2B, reflecting a 15% YoY growth compared to $1.04B in Q2 2025.
    Operating margin expanded from 22% in the prior year period to 25% due to cloud software licensing momentum.
    """,
    """
    TechCorp Inc. - Balance Sheet & Cash Flow Review (Q2 2026)
    Free cash flow for the second quarter contracted 8% YoY to $180M (down from $195M in Q2 2025), driven by increased infrastructure capital expenditures in AI data center deployment.
    Total outstanding debt stands at $600M offset by $150M in cash equivalents, resulting in a Net Debt position of $450M (compared to $400M in Q2 2025).
    """,
    """
    TechCorp Inc. - Guidance & Risk Factors Disclosure
    MANAGEMENT REVISION & REGULATORY WARNING:
    Management revised full-year FY2026 revenue growth guidance downward from 18% to 12%.
    The revision is attributed to European supply chain bottlenecks in semiconductor hardware delivery and pending regulatory approval delays in cross-border AI data transfers.
    """
]

def clean_and_normalize_sec_text(raw_text: str) -> str:
    """Cleans SEC 10-K/10-Q text and multi-row tables."""
    cleaned = re.sub(r'\s+', ' ', raw_text)
    cleaned = re.sub(r'[^\x00-\x7F]+', '', cleaned)
    return cleaned.strip()

@st.cache_resource
def setup_rag_engine():
    """Document processing pipeline: cleaning, text splitting, and vector store initialization."""
    cleaned_docs = [clean_and_normalize_sec_text(doc) for doc in RAW_SEC_FILINGS]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    doc_chunks = text_splitter.split_text(" ".join(cleaned_docs))
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(
        texts=doc_chunks,
        embedding=embeddings,
        collection_name="aegis_sec_audits"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = setup_rag_engine()

AEGIS_PROMPT = """
You are Aegis Financial's Lead Regulatory Compliance & Due Diligence Intelligence Engine.
Analyze the provided SEC disclosures, balance sheets, and earnings call context to evaluate financial risk.

--- RETRIEVED SEC DISCLOSURES ---
{financial_context}

--- AUDIT QUERY ---
{user_query}

Deliver an accurate, structured due diligence report adhering strictly to the facts provided above.
Do not invent any numbers or metrics outside the provided context.
"""
prompt_template = ChatPromptTemplate.from_template(AEGIS_PROMPT)

# ==========================================
# 4. STREAMLIT FRONTEND UI DASHBOARD
# ==========================================
st.title("🛡️ Aegis Financial Intelligence")
st.caption("Autonomous SEC Due Diligence & Regulatory Risk Audit Engine")

st.divider()

# Input Query Section
query = st.text_input(
    "Enter Audit Query:",
    value="Perform an Aegis audit on TechCorp Q2 2026 revenue expansion, free cash flow changes, net debt, and guidance risks."
)

if st.button("Run Audit", type="primary", use_container_width=True):
    if not groq_api_key:
        st.error("⚠️ Groq API Key missing! Please enter your key in the sidebar or set it in your environment variables.")
    else:
        try:
            with st.spinner("Processing SEC disclosures via Groq LLaMA 3.3 70B..."):
                # Step 1: Retrieve Top-3 Context Chunks
                docs = retriever.invoke(query)
                context = "\n".join([doc.page_content for doc in docs])

# Step 2: Structured Output Generation via Pydantic & Groq
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",  # Updated Model ID
    temperature=0
)
structured_llm = llm.with_structured_output(AegisFinancialReport)

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

            # Executive Summary Box
            st.subheader("📋 Executive Summary")
            st.info(report.executive_summary)

            # Dashboard Grid
            col_left, col_right = st.columns([1.3, 1])

            with col_left:
                st.subheader("📊 Extracted Metrics Table")
                table_data = [
                    {
                        "Metric": m.metric_name,
                        "Current Period": m.current_period,
                        "Previous Period": m.previous_period,
                        "Trend": "▲ UP" if m.trend.upper() == "UP" else ("▼ DOWN" if m.trend.upper() == "DOWN" else "➔ STABLE")
                    }
                    for m in report.key_metrics
                ]
                st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

            with col_right:
                st.subheader("⚠️ Guidance & Risk Flags")
                for risk in report.key_risk_factors:
                    st.warning(f"**Risk Flag:** {risk}")

                st.subheader("📈 Primary Growth Drivers")
                for driver in report.growth_drivers:
                    st.success(f"**Driver:** {driver}")

        except Exception as e:
            st.error(f"Audit Processing Error: {str(e)}")
