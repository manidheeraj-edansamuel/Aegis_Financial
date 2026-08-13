# 🛡️ Aegis Financial — Regulatory Compliance & Due Diligence Intelligence Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3%2070B-orange.svg)](https://groq.com/)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Aegis Financial** is an enterprise-grade, zero-hallucination Retrieval-Augmented Generation (RAG) platform designed to automate financial document auditing, extract structured balance sheet metrics, and flag regulatory & guidance risks from corporate SEC disclosures and earnings transcripts.

---

## 🌟 Key Features

- 🎯 **Zero-Hallucination Extraction**: Uses dense vector retrieval combined with strict Pydantic schemas to ensure 100% grounded financial outputs.
- ⚡ **Ultra-Low Latency Inference**: Powered by the **Groq LLaMA 3.3 (70B)** engine, delivering structured financial audits in under 2.5 seconds.
- 📊 **Automated Metric & Trend Calculation**: Automatically calculates period-over-period metric deltas (`UP`, `DOWN`, `STABLE`).
- ⚠️ **Guidance & Risk Flagging**: Automatically scans management disclosures for supply chain bottlenecks, regulatory delays, and full-year guidance revisions.
- 🖥️ **Interactive Financial Dashboard**: Built using Streamlit with dark-mode optimized components for equity research and compliance teams.

---

## 🏗️ System Architecture
┌────────────────────────────────────────────────────────────────────────┐
│                        📱 STREAMLIT DASHBOARD                          │
│   • Input Audit Query    • Executive Summary    • Metrics & Risk Flags │
└───────────────────────────────────┬────────────────────────────────────┘
│ 1. Submit Query
▼
┌────────────────────────────────────────────────────────────────────────┐
│                     ⚡ LOCAL RAG VECTOR STORE                          │
│   • ChromaDB Vector Store                                              │
│   • HuggingFace Embeddings (all-MiniLM-L6-v2)                          │
│   • Top-K Dense Document Retrieval                                     │
└───────────────────────────────────┬────────────────────────────────────┘
│ 2. Context Chunks
▼
┌────────────────────────────────────────────────────────────────────────┐
│                     🧠 GROQ INFERENCE ENGINE                           │
│   • Groq LLaMA 3.3 (70B Versatile)                                     │
│   • Zero-Temperature Structured Output Extraction                      │
│   • Enforced AegisFinancialReport Pydantic Schema                      │
└────────────────────────────────────────────────────────────────────────┘


---

## 🛠️ Tech Stack

- **Language & Framework:** Python 3.10+, Streamlit
- **LLM Engine:** Groq API (`llama-3.3-70b-versatile`)
- **Orchestration & Validation:** LangChain, Pydantic (v2)
- **Vector Database:** ChromaDB (In-Memory / Local Disk)
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Data Manipulation:** Pandas

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure you have Python 3.10 or higher installed on your system.

### 2. Clone the Repository
```bash
git clone [https://github.com/your-username/aegis-financial.git](https://github.com/your-username/aegis-financial.git)
cd aegis-financial
