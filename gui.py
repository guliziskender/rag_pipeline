import json
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Enterprise RAG Assistant (Advanced)",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 Enterprise RAG Assistant")

# Sidebar: Document Ingestion Portal
with st.sidebar:
    st.header("📄 Ingestion Portal")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file and st.button("Index Document", type="primary"):
        with st.spinner("Processing & indexing PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                response = requests.post(f"{API_URL}/ingest", files=files)
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Indexed **{uploaded_file.name}** ({data.get('chunks')} chunks).")
                else:
                    st.error(f"Failed to index file: {response.text}")
            except Exception as e:
                st.error(f"Backend connection error: {e}")

# Session State: Maintain Conversation History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Prior Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if prompt := st.chat_input("Ask a question based on your uploaded PDFs..."):
    # Render and store user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Format history payload for the contextual memory chain
    history_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    # Append current user prompt to history state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Render streaming assistant response
    with st.chat_message("assistant"):
        payload = {
            "question": prompt,
            "history": history_payload
        }

        try:
            res = requests.post(
                f"{API_URL}/query/advanced",
                json=payload,
                stream=True
            )

            if res.status_code != 200:
                st.error(f"API Error ({res.status_code}): {res.text}")
            else:
                def response_generator():
                    lines = res.iter_lines(decode_unicode=True)
                    sources = []
                    standalone_query = None

                    first_line = next(lines, None)
                    if first_line and first_line.startswith("METADATA:"):
                        meta_data = json.loads(first_line.replace("METADATA:", ""))
                        sources = meta_data.get("sources", [])
                        standalone_query = meta_data.get("standalone_query")
                    elif first_line:
                        yield first_line + " "

                    for line in lines:
                        if line:
                            yield line + " "

                    # Append metadata summary at the end of the response
                    footer_items = []
                    if standalone_query and standalone_query != prompt:
                        footer_items.append(f"\n\n🔍 **Search reformulation:** *\"{standalone_query}\"*")
                    if sources:
                        footer_items.append(f"📚 **Sources:** `{', '.join(sources)}`")

                    if footer_items:
                        yield "\n\n" + "\n\n".join(footer_items)

                full_response = st.write_stream(response_generator)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Failed to communicate with backend: {e}")