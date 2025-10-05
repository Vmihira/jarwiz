import streamlit as st
import os
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from serpapi import GoogleSearch
import PyPDF2
from io import BytesIO
import time
import json
import re
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Page config
st.set_page_config(
    page_title="Voice RAG Assistant",
    page_icon="🎤",
    layout="wide"
)

# API Keys
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize clients
@st.cache_resource
def init_clients():
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "rag-chatbot"
    
    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    index = pc.Index(index_name)
    
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    
    return deepgram, index, gemini_model

deepgram, pinecone_index, gemini_model = init_clients()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

# Helper Functions
def transcribe_audio(audio_bytes):
    """Transcribe audio using Deepgram"""
    try:
        payload: FileSource = {
            "buffer": audio_bytes,
        }
        
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
        )
        
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        transcript = response.results.channels[0].alternatives[0].transcript
        return transcript
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file with page numbers"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        pages_text = []
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                pages_text.append({
                    'page': page_num,
                    'text': text
                })
        return pages_text
    except Exception as e:
        st.error(f"Error extracting PDF text: {str(e)}")
        return None

def chunk_text_with_overlap(pages_text, chunk_size=400, overlap=100):
    """Split text into chunks with overlap and metadata"""
    chunks = []
    
    for page_data in pages_text:
        page_num = page_data['page']
        text = page_data['text']
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append({
                    'text': chunk,
                    'page': page_num,
                    'chunk_id': len(chunks)
                })
    
    return chunks

def get_embedding(text):
    """Get embedding using Gemini"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return None

def upload_to_pinecone(chunks, pdf_name):
    """Upload text chunks to Pinecone with metadata"""
    try:
        vectors = []
        for chunk_data in chunks:
            embedding = get_embedding(chunk_data['text'])
            if embedding:
                vectors.append({
                    "id": f"doc_{int(time.time())}_{chunk_data['chunk_id']}",
                    "values": embedding,
                    "metadata": {
                        "text": chunk_data['text'],
                        "page": chunk_data['page'],
                        "source": pdf_name,
                        "timestamp": int(time.time())
                    }
                })
        
        if vectors:
            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                pinecone_index.upsert(vectors=batch)
            return True
        return False
    except Exception as e:
        st.error(f"Error uploading to Pinecone: {str(e)}")
        return False

def expand_query(query):
    """Expand query with related terms for better retrieval"""
    try:
        prompt = f"""Given this search query: "{query}"
        
Generate 2-3 related search terms or phrasings that would help find relevant information. 
Return ONLY the alternative queries, one per line, without numbering or explanation."""
        
        response = gemini_model.generate_content(prompt)
        expansions = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
        return [query] + expansions[:2]  # Original + 2 expansions
    except:
        return [query]

def search_pinecone(query, top_k=10, score_threshold=0.55, max_chunks_per_page=3):
    """
    Retrieve RAG results with:
    - Deduplication
    - Page-wise chunk limit
    - Sorted by relevance and page
    """
    try:
        expanded_queries = expand_query(query)
        page_chunks = {}  # store chunks per page
        seen_ids = set()

        for expanded_query in expanded_queries:
            embedding = get_embedding(expanded_query)
            if not embedding:
                continue

            results = pinecone_index.query(
                vector=embedding,
                top_k=top_k*5,  # fetch more to filter and sort
                include_metadata=True
            )

            for match in results.matches:
                score = match.score
                page = match.metadata.get('page', 'N/A')

                if score < score_threshold or match.id in seen_ids:
                    continue

                seen_ids.add(match.id)
                if page not in page_chunks:
                    page_chunks[page] = []

                # Add up to max_chunks_per_page per page
                if len(page_chunks[page]) < max_chunks_per_page:
                    page_chunks[page].append({
                        'text': match.metadata.get('text', ''),
                        'score': score,
                        'page': page,
                        'source': match.metadata.get('source', 'Unknown'),
                        'id': match.id
                    })

        # Flatten results and sort: first by score descending, then by page ascending
        results_list = []
        for page, chunks in page_chunks.items():
            # Sort chunks per page by score descending
            chunks_sorted = sorted(chunks, key=lambda x: x['score'], reverse=True)
            results_list.extend(chunks_sorted)

        # Sort all by page number for coherent reading
        results_list.sort(key=lambda x: int(x['page'] if x['page'] != 'N/A' else 0))

        return results_list[:top_k]

    except Exception as e:
        st.error(f"Error searching Pinecone: {str(e)}")
        return []


def search_serp(query, num_results=5):
    """Search using SerpAPI with enhanced metadata"""
    try:
        if not SERP_API_KEY:
            st.error("❌ SERP_API_KEY is missing")
            return []
        
        params = {
            "q": query,
            "api_key": SERP_API_KEY,
            "engine": "google",
            "gl": "us",
            "hl": "en",
            "google_domain": "google.com",
            "num": num_results
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Debug: Print full response
        #st.write("🔹 Full SerpAPI Response:", results)
        
        search_results = []
        organic = results.get("organic_results", [])
        if not organic:
            st.warning("⚠️ No web results found")
            return []
        
        for idx, result in enumerate(organic[:num_results], 1):
            search_results.append({
                'title': result.get('title', 'No title'),
                'snippet': result.get('snippet', ''),
                'link': result.get('link', ''),
                'position': idx
            })
        
        return search_results
    except Exception as e:
        st.error(f"Error searching web: {str(e)}")
        return []


def format_sources(rag_results, web_results):
    """Format sources for display"""
    sources_text = "\n\n### 📚 Sources Used\n\n"
    
    if rag_results:
        sources_text += "**📄 Document Sources (RAG):**\n"
        for idx, result in enumerate(rag_results, 1):
            sources_text += f"{idx}. **{result['source']}** (Page {result['page']}) - Relevance: {result['score']:.2%}\n"
            sources_text += f"   > _{result['text'][:150]}..._\n\n"
    
    if web_results:
        sources_text += "**🌐 Web Sources:**\n"
        for idx, result in enumerate(web_results, 1):
            sources_text += f"{idx}. [{result['title']}]({result['link']})\n"
            sources_text += f"   > _{result['snippet'][:150]}..._\n\n"
    
    return sources_text

def generate_answer(query, rag_results, web_results):
    """Generate answer using Gemini with proper citations"""
    try:
        # Build context
        rag_context = ""
        if rag_results:
            rag_context = "**Context from Uploaded Documents:**\n\n"
            for idx, result in enumerate(rag_results, 1):
                rag_context += f"[DOC{idx}] From '{result['source']}' (Page {result['page']}, Relevance: {result['score']:.2%}):\n"
                rag_context += f"{result['text']}\n\n"
        
        web_context = ""
        if web_results:
            web_context = "**Context from Web Search:**\n\n"
            for idx, result in enumerate(web_results, 1):
                web_context += f"[WEB{idx}] {result['title']}:\n"
                web_context += f"{result['snippet']}\n\n"
        
        prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context. 

IMPORTANT INSTRUCTIONS:
1. Always include citations inline where you use the information
2. If information comes from multiple sources, cite all of them
3. Be specific and accurate
4. If the context doesn't fully answer the question, say so clearly
5. Synthesize information from multiple sources when relevant

User Question: {query}

{rag_context}

{web_context}

Please provide a comprehensive, well-cited answer:"""

        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "I apologize, but I encountered an error generating the response."

def process_query(query, use_rag, use_web):
    """Process a query and return answer with sources"""
    with st.spinner("🔍 Searching for relevant information..."):
        # Get RAG context
        rag_results = []
        if use_rag and st.session_state.pdf_uploaded:
            rag_results = search_pinecone(query, top_k=30, score_threshold=0.65)
            if rag_results:
                st.info(f"✅ Found {len(rag_results)} relevant document chunks")
        
        # Get web context
        web_results = []
        if use_web:
            web_results = search_serp(query, num_results=5)
            if web_results:
                st.info(f"✅ Found {len(web_results)} relevant web results")
    
    with st.spinner("🤔 Generating answer..."):
        # Generate answer
        answer = generate_answer(query, rag_results, web_results)
        
        # Format sources
        sources = format_sources(rag_results, web_results)
        
        return answer, sources, rag_results, web_results

# UI Layout
st.title("🎤 Voice RAG Assistant")
st.markdown("Ask questions using your voice or text, and get answers with proper citations!")

# Sidebar for PDF upload
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if uploaded_file and not st.session_state.pdf_uploaded:
        with st.spinner("Processing PDF..."):
            pages_text = extract_text_from_pdf(uploaded_file)
            if pages_text:
                chunks = chunk_text_with_overlap(pages_text, chunk_size=400, overlap=100)
                success = upload_to_pinecone(chunks, uploaded_file.name)
                if success:
                    st.success(f"✅ PDF uploaded! ({len(chunks)} chunks indexed)")
                    st.session_state.pdf_uploaded = True
                    st.session_state.pdf_name = uploaded_file.name
                else:
                    st.error("Failed to upload PDF to vector database")
    
    if st.session_state.pdf_uploaded:
        st.info(f"📚 **{st.session_state.pdf_name}** is loaded")
        if st.button("🗑️ Clear PDF"):
            st.session_state.pdf_uploaded = False
            st.session_state.pdf_name = ""
            st.rerun()
    
    st.divider()
    st.markdown("### ⚙️ Settings")
    use_rag = st.checkbox("📄 Use Document Context", value=True)
    use_web = st.checkbox("🌐 Use Web Search", value=True)
    
    st.divider()
    st.markdown("### 📊 Stats")
    if st.session_state.pdf_uploaded:
        st.metric("Document", "Loaded ✅")
    st.metric("Messages", len(st.session_state.messages))

# Main chat interface
st.markdown("### 💬 Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 View Sources", expanded=False):
                st.markdown(message["sources"])

# Create two columns for input options
col1, col2 = st.columns([3, 1])


    # Text input
prompt = st.chat_input("Type your question here...")


    # Audio input button
audio_value = st.audio_input("🎤 Voice")

# Handle audio input
if audio_value:
    # Check if this is a new audio input (not processed yet)
    audio_bytes = audio_value.getvalue()
    audio_hash = hash(audio_bytes)
    
    if "last_audio_hash" not in st.session_state:
        st.session_state.last_audio_hash = None
    
    if audio_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = audio_hash
        
        with st.spinner("🎙️ Transcribing audio..."):
            transcript = transcribe_audio(audio_bytes)
            
            if transcript:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": transcript})
                
                # Generate response
                answer, sources, rag_results, web_results = process_query(transcript, use_rag, use_web)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
                
                # Force rerun to display new messages
                st.rerun()

# Handle text input
if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    answer, sources, rag_results, web_results = process_query(prompt, use_rag, use_web)
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": sources
    })
    
    # Force rerun to display new messages
    st.rerun()

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()