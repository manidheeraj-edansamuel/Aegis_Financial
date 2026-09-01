import os
import re
import json
from typing import List, Literal

import pandas as pd
import streamlit as st

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from groq import Groq

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
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
# 1. PAGE CONFIGURATION
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
# 3. PYDANTIC OUTPUT SCHEMA
# ============================================================

class FinancialMetric(BaseModel):

    metric_name: str = Field(
        description="Name of the financial metric."
    )

    current_period: str = Field(
        description="Value for the current reporting period."
    )

    previous_period: str = Field(
        description="Value for the previous comparison period."
    )

    trend: Literal[
        "UP",
        "DOWN",
        "STABLE"
    ] = Field(
        description="Financial direction."
    )


class AegisFinancialReport(BaseModel):

    company_name: str = Field(
        description="Company name."
    )

    period: str = Field(
        description="Reporting period, for example Q2 2026."
    )

    executive_summary: str = Field(
        description=(
            "Concise summary of financial performance, "
            "capital allocation, guidance and risk."
        )
    )

    key_metrics: List[FinancialMetric] = Field(
        description="Important financial metrics."
    )

    key_risk_factors: List[str] = Field(
        description="Important risks supported by the SEC context."
    )

    growth_drivers: List[str] = Field(
        description=(
            "Business or revenue growth drivers "
            "supported by the SEC context."
        )
    )

    overall_sentiment: Literal[
        "BULLISH",
        "BEARISH",
        "NEUTRAL"
    ] = Field(
        description="Overall investment sentiment."
    )


# ============================================================
# 4. CREATE GROQ JSON SCHEMA
# ============================================================

def make_strict_schema(schema):
    """
    Groq strict JSON Schema requires objects to use
    additionalProperties: false.

    This function recursively applies that requirement.
    """

    if isinstance(schema, dict):

        schema = dict(schema)

        if schema.get("type") == "object":

            schema["additionalProperties"] = False

            properties = schema.get(
                "properties",
                {}
            )

            schema["required"] = list(
                properties.keys()
            )

            for key, value in properties.items():

                properties[key] = make_strict_schema(
                    value
                )

        elif schema.get("type") == "array":

            if "items" in schema:

                schema["items"] = make_strict_schema(
                    schema["items"]
                )

        else:

            for key, value in list(schema.items()):

                if isinstance(value, (dict, list)):

                    schema[key] = make_strict_schema(
                        value
                    )

        return schema

    elif isinstance(schema, list):

        return [
            make_strict_schema(item)
            for item in schema
        ]

    return schema


AEGIS_JSON_SCHEMA = (
    AegisFinancialReport
    .model_json_schema()
)

AEGIS_JSON_SCHEMA = make_strict_schema(
    AEGIS_JSON_SCHEMA
)


# ============================================================
# 5. GROQ MODEL DISCOVERY
# ============================================================

def get_available_groq_models(api_key):

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

                models.append(
                    model_id
                )

        return sorted(models), None

    except Exception as e:

        return [], str(e)


# ============================================================
# 6. STRUCTURED OUTPUT MODEL PRIORITY
# ============================================================

STRUCTURED_OUTPUT_MODELS = [

    # Current Groq documented structured-output models
    "meta-llama/llama-4-scout-17b-16e-instruct",

    "meta-llama/llama-4-maverick-17b-128e-instruct",

    "moonshotai/kimi-k2-instruct",

    # Newer OpenAI OSS models if available
    "openai/gpt-oss-20b",

    "openai/gpt-oss-120b",

]


def select_structured_model(
    available_models
):

    for model in STRUCTURED_OUTPUT_MODELS:

        if model in available_models:

            return model

    return None


# ============================================================
# 7. SIDEBAR
# ============================================================

st.sidebar.title(
    "🛡️ Aegis Financial"
)


# ============================================================
# API KEY INPUT
# ============================================================

if not groq_api_key:

    groq_api_key = st.sidebar.text_input(
        "Enter Groq API Key:",
        type="password",
        help="Get your API key from the Groq console."
    )

    if groq_api_key:

        groq_api_key = (
            groq_api_key.strip()
        )

        os.environ[
            "GROQ_API_KEY"
        ] = groq_api_key


# ============================================================
# 8. MODEL DISCOVERY
# ============================================================

available_models = []

model_error = None

structured_model = None


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

        structured_model = (
            select_structured_model(
                available_models
            )
        )


# ============================================================
# 9. SIDEBAR SYSTEM INFORMATION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "System Information"
)


if structured_model:

    st.sidebar.caption(
        f"🟢 **Structured AI:** `{structured_model}`"
    )

    st.sidebar.caption(
        "🟢 **Output:** Native Groq JSON Schema"
    )

else:

    st.sidebar.caption(
        "🔴 **Structured AI:** Not available"
    )


st.sidebar.caption(
    "🟢 **Database:** ChromaDB"
)

st.sidebar.caption(
    "🟢 **Embeddings:** all-MiniLM-L6-v2"
)

st.sidebar.caption(
    "🟢 **Analysis:** SEC Grounded RAG"
)


# ============================================================
# AVAILABLE MODELS
# ============================================================

if available_models:

    st.sidebar.success(
        f"{len(available_models)} Groq model(s) available"
    )

    with st.sidebar.expander(
        "View Available Groq Models"
    ):

        for model in available_models:

            if model == structured_model:

                st.write(
                    f"✅ `{model}` — Structured Output"
                )

            else:

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
# 10. SAMPLE SEC FILINGS
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
# 12. CHROMA RAG ENGINE
# ============================================================

@st.cache_resource
def setup_rag_engine():

    cleaned_docs = [
        clean_and_normalize_sec_text(
            document
        )
        for document in RAW_SEC_FILINGS
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


    doc_chunks = (
        text_splitter.split_text(
            combined_text
        )
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
# 13. AEGIS SYSTEM PROMPT
# ============================================================

AEGIS_SYSTEM_PROMPT = """
You are Aegis Financial's Lead Regulatory Compliance
and Due Diligence Intelligence Engine.

Your job is to analyze the supplied SEC disclosures.

IMPORTANT:

Use ONLY the supplied SEC context.

Never invent financial information.

Never use external knowledge.

Never invent numbers.

Never invent dates.

Never invent companies.

The response will be validated against a strict JSON schema.

You MUST populate these fields:

company_name
period
executive_summary
key_metrics
key_risk_factors
growth_drivers
overall_sentiment

For key_metrics, every metric must contain:

metric_name
current_period
previous_period
trend

trend must be:

UP
DOWN
STABLE

overall_sentiment must be:

BULLISH
BEARISH
NEUTRAL

Financial mapping example:

Revenue:
Current = $1.2B
Previous = $1.04B
Trend = UP

Free Cash Flow:
Current = $180M
Previous = $195M
Trend = DOWN

Net Debt:
Current = $450M
Previous = $400M
Trend = UP

Operating Margin:
Current = 25%
Previous = 22%
Trend = UP

Put financial and regulatory concerns into
key_risk_factors.

Put positive business drivers into
growth_drivers.

If a requested item is unavailable from the supplied
SEC context, state:

"Not available in the provided SEC context."

Return only the information requested by the schema.
"""


# ============================================================
# 14. GROQ AUDIT FUNCTION
# ============================================================

def run_groq_audit(
    api_key,
    model,
    context,
    user_query
):

    client = Groq(
        api_key=api_key
    )


    user_prompt = f"""
AUDIT QUERY:

{user_query}


SEC CONTEXT:

{context}


Perform the requested financial audit.

Base every conclusion strictly on the SEC context.
"""


    response = client.chat.completions.create(

        model=model,

        messages=[

            {
                "role": "system",
                "content": AEGIS_SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": user_prompt
            }

        ],

        temperature=0,

        response_format={

            "type": "json_schema",

            "json_schema": {

                "name": "aegis_financial_report",

                "strict": True,

                "schema": AEGIS_JSON_SCHEMA
            }
        }
    )


    # ========================================================
    # GET MODEL CONTENT
    # ========================================================

    content = (
        response
        .choices[0]
        .message
        .content
    )


    if not content:

        raise ValueError(
            "Groq returned an empty response."
        )


    # ========================================================
    # PARSE JSON
    # ========================================================

    try:

        parsed = json.loads(
            content
        )

    except json.JSONDecodeError as e:

        raise ValueError(
            "Groq returned invalid JSON: "
            + str(e)
        )


    # ========================================================
    # PYDANTIC VALIDATION
    # ========================================================

    report = (
        AegisFinancialReport
        .model_validate(parsed)
    )


    return report


# ============================================================
# 15. PAGE HEADER
# ============================================================

st.title(
    "🛡️ Aegis Financial Intelligence"
)

st.caption(
    "Autonomous SEC Due Diligence & Regulatory Risk Audit Engine"
)

st.divider()


# ============================================================
# 16. MODEL STATUS
# ============================================================

if groq_api_key:

    if structured_model:

        st.success(
            f"🟢 Groq Structured Output Ready — `{structured_model}`"
        )

    elif available_models:

        st.error(
            "❌ Your Groq API key has no detected model "
            "with the required Structured Output support."
        )

        st.info(
            "See the available models in the sidebar."
        )

    elif model_error:

        st.error(
            "❌ Unable to retrieve models from Groq."
        )


# ============================================================
# 17. AUDIT QUERY
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
# 18. RUN AUDIT BUTTON
# ============================================================

if st.button(
    "Run Audit",
    type="primary",
    use_container_width=True
):


    # ========================================================
    # CHECK API KEY
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
    # CHECK MODEL
    # ========================================================

    if not structured_model:

        st.error(
            "❌ No Groq Structured Output model "
            "is available for this API key."
        )

        st.write(
            "Available models:"
        )

        for model in available_models:

            st.code(
                model
            )

        st.stop()


    # ========================================================
    # RUN AUDIT
    # ========================================================

    try:

        with st.spinner(
            f"Running Aegis audit using {structured_model}..."
        ):


            # ------------------------------------------------
            # STEP 1 — RAG RETRIEVAL
            # ------------------------------------------------

            docs = retriever.invoke(
                query
            )


            if not docs:

                raise ValueError(
                    "No relevant SEC context found."
                )


            # ------------------------------------------------
            # STEP 2 — BUILD CONTEXT
            # ------------------------------------------------

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )


            # ------------------------------------------------
            # STEP 3 — GROQ STRUCTURED OUTPUT
            # ------------------------------------------------

            report = run_groq_audit(
                api_key=groq_api_key,
                model=structured_model,
                context=context,
                user_query=query
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

        col1, col2, col3 = st.columns(
            3
        )


        with col1:

            st.metric(
                "Company",
                report.company_name,
                delta=report.period
            )


        with col2:

            st.metric(
                "Overall Sentiment",
                report.overall_sentiment
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

                if metric.trend == "UP":

                    display_trend = "▲ UP"

                elif metric.trend == "DOWN":

                    display_trend = "▼ DOWN"

                else:

                    display_trend = "➔ STABLE"


                table_data.append(
                    {
                        "Metric": metric.metric_name,

                        "Current Period":
                            metric.current_period,

                        "Previous Period":
                            metric.previous_period,

                        "Trend":
                            display_trend
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


        if report.overall_sentiment == "BULLISH":

            st.success(
                "🟢 BULLISH — The available SEC context "
                "indicates positive financial momentum."
            )

        elif report.overall_sentiment == "BEARISH":

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

        if (
            "model_not_found"
            in error_lower
        ):

            st.error(
                "❌ Groq model access error."
            )

            st.warning(
                f"Model used: `{structured_model}`"
            )

            st.info(
                "Your Groq project does not appear to "
                "have access to this model."
            )


        # ----------------------------------------------------
        # STRUCTURED OUTPUT NOT SUPPORTED
        # ----------------------------------------------------

        elif (
            "response_format"
            in error_lower
            or "json_schema"
            in error_lower
            or "structured output"
            in error_lower
        ):

            st.error(
                "❌ Groq Structured Output error."
            )

            st.info(
                "The selected Groq model may not support "
                "JSON Schema Structured Outputs."
            )

            st.code(
                structured_model
            )


        # ----------------------------------------------------
        # AUTHENTICATION
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
        # PYDANTIC VALIDATION
        # ----------------------------------------------------

        elif (
            "validationerror"
            in error_lower
            or "validation error"
            in error_lower
        ):

            st.error(
                "❌ Groq returned data that did not "
                "match the Aegis schema."
            )


        # ----------------------------------------------------
        # GENERAL ERROR
        # ----------------------------------------------------

        else:

            st.error(
                f"❌ Audit Processing Error: {error_text}"
            )


        # ====================================================
        # TECHNICAL ERROR DETAILS
        # ====================================================

        with st.expander(
            "🔧 Technical Error Details"
        ):

            st.exception(e)
