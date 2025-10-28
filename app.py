# app.py
import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

# ----- Streamlit UI -----
st.set_page_config(page_title="🌙 Umeeda — Assistant", layout="centered")
st.title("🌙 Umeeda — Your Secret Friend")
st.caption("Umeeda answers using your uploaded knowledge base. Admin: use sidebar to upload or reindex.")

# Import KB helpers (from kb_loader.py)
from kb_loader import load_index, retrieve, ingest_csv, build_index

# ----- Config / Guardrails -----
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in .env or Streamlit secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")

CULTURE_SYSTEM_PROMPT = """
You are Umeeda, an assistant for users in Pakistan. Answer respectfully and according to Islamic and local cultural norms.
Follow these rules:
1. Use modest, family-friendly, and respectful language.
2. Avoid sexual/explicit content or harmful/illegal advice.
3. Never insult or dehumanize anyone.
4. If asked something inappropriate, politely refuse.
5. Keep responses concise and optionally refer to Islamic values where suitable.
"""

# ----- Load Index (if available) -----
INDEX = None
METADATA = None
try:
    INDEX, METADATA = load_index("kb_index.faiss", "metadata.pkl")
    st.sidebar.success("✅ Knowledge base index loaded.")
except Exception:
    INDEX, METADATA = None, None

# ----- Decision Logic -----
HIGH_CONFIDENCE_THRESHOLD = 0.78

def decide_reply(query: str):
    """Return appropriate reply using KB or fallback to model."""
    if INDEX is None or METADATA is None:
        # Fallback: only model reply
        prompt = CULTURE_SYSTEM_PROMPT + "\n\nUser: " + query
        try:
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", None)
            if not text and hasattr(resp, "candidates") and resp.candidates:
                text = getattr(resp.candidates[0], "content", None)
            return text.strip() if text else "Sorry — I couldn't produce an answer."
        except Exception as e:
            return f"⚠️ Model error: {e}"

    # Retrieval-based response
    results = retrieve(query, INDEX, METADATA, top_k=4)
    if not results:
        return "I couldn't find a relevant answer. Please rephrase your question."

    top = results[0]
    score = float(top.get("score", 0.0))
    short_answer = top.get("short_answer") or top.get("text") or ""
    detailed_answer = top.get("detailed_answer") or top.get("text") or ""
    entry_id = top.get("id") or top.get("chunk_id") or "unknown"
    source = top.get("source", "unknown")
    risk = top.get("risk_level", top.get("risk", "Info"))

    # High confidence → short answer
    if score >= HIGH_CONFIDENCE_THRESHOLD and short_answer:
        prefix = ""
        if any((r.get("risk_level","").lower()=="urgent") for r in results):
            prefix = "⚠️ This appears urgent. Please consult a qualified person.\n\n"
        return f"{prefix}{short_answer}\n\nSource: [{source} | ID:{entry_id}]"

    # Otherwise → RAG prompt
    rag_context = ""
    for r in results:
        r_src = r.get("source", "unknown")
        r_id = r.get("id", r.get("chunk_id", "unknown"))
        r_risk = r.get("risk_level", r.get("risk", "Info"))
        r_detail = r.get("detailed_answer") or r.get("text") or ""
        rag_context += f"[{r_src} | ID:{r_id} | Risk:{r_risk}]\n{r_detail}\n\n"

    prompt = (
        CULTURE_SYSTEM_PROMPT
        + "\n\nRetrieved knowledge:\n"
        + rag_context
        + "\nUser question:\n"
        + query
        + "\n\nAnswer using the retrieved knowledge and cite sources."
    )
    try:
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None)
        if not text and hasattr(resp, "candidates") and resp.candidates:
            text = getattr(resp.candidates[0], "content", None)
        final = text.strip() if text else "Sorry — I couldn't produce an answer."
        sources = ", ".join(sorted({f"{r.get('source','unknown')} (ID {r.get('id', r.get('chunk_id','?'))})" for r in results}))
        return final + "\n\nSources: " + sources
    except Exception as e:
        return f"⚠️ Error: {e}"

# ----- Session & UI -----
if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.header("🛠️ Umeeda Admin Panel")

# Upload PDF(s)
uploaded_pdf = st.sidebar.file_uploader("Upload PDF (then click Rebuild Index)", type=["pdf"])
if uploaded_pdf:
    os.makedirs("sources", exist_ok=True)
    save_path = os.path.join("sources", uploaded_pdf.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.sidebar.success(f"Saved {uploaded_pdf.name} to /sources. Click 'Rebuild Index' to include it.")

# Upload CSV
uploaded_csv = st.sidebar.file_uploader("Upload CSV KB (id,theme,short_answer,detailed_answer,...)", type=["csv"])
if uploaded_csv:
    csv_path = "kb_import.csv"
    with open(csv_path, "wb") as f:
        f.write(uploaded_csv.getbuffer())
    st.sidebar.success("CSV uploaded successfully. Click 'Import CSV' to process it.")

# Import CSV Button
if st.sidebar.button("📥 Import CSV (Build KB Index)"):
    if not os.path.exists("kb_import.csv"):
        st.sidebar.error("kb_import.csv not found. Upload it first.")
    else:
        try:
            st.sidebar.info("Building index from CSV...")
            ingest_csv("kb_import.csv", index_path="kb_index.faiss", meta_path="metadata.pkl")
            INDEX, METADATA = load_index("kb_index.faiss", "metadata.pkl")
            st.sidebar.success("✅ CSV KB imported successfully.")
        except Exception as e:
            st.sidebar.error(f"Error importing CSV: {e}")

# Rebuild Index Button
if st.sidebar.button("🔄 Rebuild KB Index (from PDFs)"):
    try:
        st.sidebar.info("Building index from PDFs in /sources...")
        build_index("sources", index_path="kb_index.faiss", meta_path="metadata.pkl")
        INDEX, METADATA = load_index("kb_index.faiss", "metadata.pkl")
        st.sidebar.success("✅ PDF index rebuilt successfully.")
    except Exception as e:
        st.sidebar.error(f"Rebuild failed: {e}")

# ----- Chat Section -----
def _on_submit():
    query = st.session_state.get("user_input", "").strip()
    if not query:
        st.session_state.setdefault("_show_warning", True)
        return
    reply = decide_reply(query)
    st.session_state.history.append((query, reply))
    st.session_state["user_input"] = ""
    st.session_state.pop("_show_warning", None)

user_input = st.text_input("Ask Umeeda something...", key="user_input", on_change=_on_submit)
if st.button("Send"):
    _on_submit()
if st.session_state.get("_show_warning"):
    st.warning("Please type your question first.")

# Display chat history
for u, b in st.session_state.history:
    st.markdown(f"<div style='background:#f0f2f6;border-radius:12px;padding:8px;margin:6px 0;'><b>You:</b> {u}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='background:#fff8e1;border-radius:12px;padding:8px;margin:6px 0;'><b>Umeeda:</b> {b}</div>", unsafe_allow_html=True)
