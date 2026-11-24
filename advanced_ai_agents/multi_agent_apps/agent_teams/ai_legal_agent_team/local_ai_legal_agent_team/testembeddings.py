# legal_agent_team.py — version OpenRouter + CORRECTIF n°1 : isolation des documents
# Problème corrigé : dans l'original, tous les PDF allaient dans la même collection Qdrant
# ("legal_documents"), conservée d'une session à l'autre. Un nouveau contrat était
# mélangé avec les anciens -> l'agent répondait avec les clauses d'un autre contrat.
# Correctif : une collection par document (hash du contenu) + document actif affiché.
# Les lignes modifiées sont marquées "AMÉLIORATION".
import streamlit as st
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.team import Team
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.openrouter import OpenRouter
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
import tempfile
import os
import hashlib
from qdrant_client import QdrantClient

# --- Configuration OpenRouter (seule adaptation par rapport au projet d'origine) ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_MODEL_ID = "nvidia/nemotron-3-ultra-550b-a55b:free"  # modèle de CHAT gratuit (ancien : openai/gpt-4o-mini)
EMBEDDING_MODEL_ID = "openai/text-embedding-3-small"  # modèle d'EMBEDDINGS : ne pas remplacer par un modèle de chat
MAX_TOKENS = 4096
# Embeddings LOCAUX et gratuits (FastEmbed, sur ton PC, sans API ni crédit).
# Mettre False pour revenir aux embeddings OpenRouter (payants).
USE_LOCAL_EMBEDDINGS = True
LOCAL_EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"  # 384 dimensions, ~70 Mo téléchargés au 1er lancement  # Agno limite OpenRouter à 1024 par défaut : réponses tronquées, et vides avec un modèle qui raisonne

def init_session_state():
    """Initialize session state variables"""
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = None
    if 'qdrant_api_key' not in st.session_state:
        st.session_state.qdrant_api_key = None
    if 'qdrant_url' not in st.session_state:
        st.session_state.qdrant_url = None
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'legal_team' not in st.session_state:
        st.session_state.legal_team = None
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = None
    # AMÉLIORATION : on suit le document actif par son empreinte (hash), plus par son nom
    if 'active_doc_hash' not in st.session_state:
        st.session_state.active_doc_hash = None

COLLECTION_PREFIX = "legal_documents"  # AMÉLIORATION : une collection par document


def file_hash(uploaded_file) -> str:
    """Empreinte unique du contenu du fichier : même fichier -> même hash."""
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()[:12]


def collection_name(doc_hash: str) -> str:
    # Suffixe "_local" : les vecteurs locaux (384 dim.) ne sont pas compatibles
    # avec les anciennes collections OpenRouter (1536 dim.) -> collections séparées
    suffix = "_local" if USE_LOCAL_EMBEDDINGS else ""
    return f"{COLLECTION_PREFIX}_{doc_hash}{suffix}"


def make_embedder():
    if USE_LOCAL_EMBEDDINGS:
        return FastEmbedEmbedder(id=LOCAL_EMBEDDING_MODEL_ID, dimensions=384)
    return OpenAIEmbedder(
        id=EMBEDDING_MODEL_ID,
        api_key=st.session_state.openai_api_key,
        base_url=OPENROUTER_BASE_URL,
        dimensions=1536,
    )


def check_qdrant_connection() -> bool:
    """Vérifie que l'URL et la clé Qdrant fonctionnent."""
    try:
        QdrantClient(url=st.session_state.qdrant_url,
                     api_key=st.session_state.qdrant_api_key).get_collections()
        return True
    except Exception as e:
        st.error(f"🔴 Qdrant connection failed: {str(e)}")
        return False


def init_qdrant(doc_hash: str):
    """Crée l'accès Qdrant pour la collection propre à CE document."""
    return Qdrant(
        collection=collection_name(doc_hash),   # AMÉLIORATION : isolation des documents
        url=st.session_state.qdrant_url,
        api_key=st.session_state.qdrant_api_key,
        embedder=make_embedder(),
    )

def process_document(uploaded_file, vector_db: Qdrant):
    """
    Process document, create embeddings and store in Qdrant vector database
    
    Args:
        uploaded_file: Streamlit uploaded file object
        vector_db (Qdrant): Initialized Qdrant instance from Agno
    
    Returns:
        Knowledge: Initialized knowledge base with processed documents
    """
    if not st.session_state.openai_api_key:
        raise ValueError("OpenRouter API key not provided")
        
    os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
    
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        st.info("Loading and processing document...")
        
        # Create a Knowledge base with the vector_db
        knowledge_base = Knowledge(
            vector_db=vector_db
        )

        # AMÉLIORATION : si ce document est déjà indexé (même hash), on ne refait rien
        if vector_db.exists() and vector_db.get_count() > 0:
            os.unlink(temp_file_path)
            st.success("✅ Document déjà indexé : réutilisation sans nouveau calcul")
            return knowledge_base
        
        # Add the document to the knowledge base
        with st.spinner('📤 Loading documents into knowledge base...'):
            try:
                knowledge_base.add_content(path=temp_file_path)
                st.success("✅ Documents stored successfully!")
            except Exception as e:
                st.error(f"Error loading documents: {str(e)}")
                raise
        
        # Clean up the temporary file
        try:
            os.unlink(temp_file_path)
        except Exception:
            pass
            
        return knowledge_base
            
    except Exception as e:
        st.error(f"Document processing error: {str(e)}")
        raise Exception(f"Error processing document: {str(e)}")

def build_team(knowledge_base):
    """Construit l'équipe d'agents (inchangée par rapport à l'original)."""
    # Initialize agents
    legal_researcher = Agent(
        name="Legal Researcher",
        role="Legal research specialist",
        model=OpenRouter(id=CHAT_MODEL_ID, api_key=st.session_state.openai_api_key, max_tokens=MAX_TOKENS),
        tools=[DuckDuckGoTools()],
        knowledge=knowledge_base,
        search_knowledge=True,
        instructions=[
            "Find and cite relevant legal cases and precedents",
            "Provide detailed research summaries with sources",
            "Reference specific sections from the uploaded document",
            "Always search the knowledge base for relevant information"
        ],
        debug_mode=True,
        markdown=True
    )

    contract_analyst = Agent(
        name="Contract Analyst",
        role="Contract analysis specialist",
        model=OpenRouter(id=CHAT_MODEL_ID, api_key=st.session_state.openai_api_key, max_tokens=MAX_TOKENS),
        knowledge=knowledge_base,
        search_knowledge=True,
        instructions=[
            "Review contracts thoroughly",
            "Identify key terms and potential issues",
            "Reference specific clauses from the document"
        ],
        markdown=True
    )

    legal_strategist = Agent(
        name="Legal Strategist", 
        role="Legal strategy specialist",
        model=OpenRouter(id=CHAT_MODEL_ID, api_key=st.session_state.openai_api_key, max_tokens=MAX_TOKENS),
        knowledge=knowledge_base,
        search_knowledge=True,
        instructions=[
            "Develop comprehensive legal strategies",
            "Provide actionable recommendations",
            "Consider both risks and opportunities"
        ],
        markdown=True
    )

    # Legal Agent Team
    return Team(
        name="Legal Team Lead",
        model=OpenRouter(id=CHAT_MODEL_ID, api_key=st.session_state.openai_api_key, max_tokens=MAX_TOKENS),
        members=[legal_researcher, contract_analyst, legal_strategist],
        knowledge=knowledge_base,
        search_knowledge=True,
        instructions=[
            "Coordinate analysis between team members",
            "Provide comprehensive responses",
            "Ensure all recommendations are properly sourced",
            "Reference specific parts of the uploaded document",
            "Always search the knowledge base before delegating tasks"
        ],
        debug_mode=True,
        markdown=True
    )



def main():
    st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
    init_session_state()

    st.title("Improved AI Legal Agent Team 👨‍⚖️")

    with st.sidebar:
        st.header("🔑 API Configuration")
   
        openai_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=st.session_state.openai_api_key if st.session_state.openai_api_key else "",
            help="Enter your OpenRouter API key (sk-or-...)"
        )
        if openai_key:
            st.session_state.openai_api_key = openai_key.strip()
            os.environ['OPENROUTER_API_KEY'] = openai_key.strip()  # secours si un composant ne reçoit pas la clé
            # AMÉLIORATION : contrôle du format de la clé (une clé tronquée provoque
            # l'erreur OpenRouter « Missing Authentication header »)
            k = openai_key.strip()
            if not k.startswith("sk-or-v1-") or len(k) != 73:
                st.warning(f"⚠️ Clé OpenRouter suspecte : {len(k)} caractères "
                           "(attendu : 73, commençant par sk-or-v1-). Elle est peut-être tronquée.")

        qdrant_key = st.text_input(
            "Qdrant API Key",
            type="password",
            value=st.session_state.qdrant_api_key if st.session_state.qdrant_api_key else "",
            help="Enter your Qdrant API key"
        )
        if qdrant_key:
            st.session_state.qdrant_api_key = qdrant_key.strip()

        qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state.qdrant_url if st.session_state.qdrant_url else "",
            help="Enter your Qdrant instance URL"
        )
        if qdrant_url:
            st.session_state.qdrant_url = qdrant_url.strip()

        if all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
            if not st.session_state.get("qdrant_ok"):
                st.session_state.qdrant_ok = check_qdrant_connection()
                if st.session_state.qdrant_ok:
                    st.success("Successfully connected to Qdrant!")

        st.divider()

        if all([st.session_state.openai_api_key, st.session_state.get('qdrant_ok')]):
            st.header("📄 Document Upload")
            uploaded_file = st.file_uploader("Upload Legal Document", type=['pdf'])
            
            if uploaded_file:
                # AMÉLIORATION : on identifie le document par son contenu (hash), pas par son nom
                doc_hash = file_hash(uploaded_file)
                if st.session_state.active_doc_hash != doc_hash:
                    with st.spinner("Processing document..."):
                        try:
                            vector_db = init_qdrant(doc_hash)
                            knowledge_base = process_document(uploaded_file, vector_db)
                            if knowledge_base:
                                st.session_state.vector_db = vector_db
                                st.session_state.knowledge_base = knowledge_base
                                st.session_state.legal_team = build_team(knowledge_base)
                                st.session_state.active_doc_hash = doc_hash
                                st.success("✅ Document processed and team initialized!")
                        except Exception as e:
                            st.session_state.legal_team = None
                            st.error(f"Error processing document: {str(e)}")
                else:
                    st.success("✅ Document already processed and team ready!")

                # AMÉLIORATION : on affiche toujours sur quel document l'agent travaille
                st.info(f"📄 Document actif : {uploaded_file.name}\n\nCollection : {collection_name(doc_hash)}")

                # AMÉLIORATION (confidentialité) : supprimer les données de ce document
                if st.button("🗑️ Supprimer ce document de la base"):
                    QdrantClient(url=st.session_state.qdrant_url,
                                 api_key=st.session_state.qdrant_api_key
                                 ).delete_collection(collection_name(doc_hash))
                    st.session_state.active_doc_hash = None
                    st.session_state.legal_team = None
                    st.success("Données du document supprimées de Qdrant.")

            st.divider()
            st.header("🔍 Analysis Options")
            analysis_type = st.selectbox(
                "Select Analysis Type",
                [
                    "Contract Review",
                    "Legal Research",
                    "Risk Assessment",
                    "Compliance Check",
                    "Custom Query"
                ]
            )
        else:
            st.warning("Please configure all API credentials to proceed")

    # Main content area
    if not all([st.session_state.openai_api_key, st.session_state.get('qdrant_ok')]):
        st.info("👈 Please configure your API credentials in the sidebar to begin")
    elif not uploaded_file:
        st.info("👈 Please upload a legal document to begin analysis")
    elif st.session_state.legal_team:
        # Create a dictionary for analysis type icons
        analysis_icons = {
            "Contract Review": "📑",
            "Legal Research": "🔍",
            "Risk Assessment": "⚠️",
            "Compliance Check": "✅",
            "Custom Query": "💭"
        }

        # Dynamic header with icon
        st.header(f"{analysis_icons[analysis_type]} {analysis_type} Analysis")
  
        analysis_configs = {
            "Contract Review": {
                "query": "Review this contract and identify key terms, obligations, and potential issues.",
                "agents": ["Contract Analyst"],
                "description": "Detailed contract analysis focusing on terms and obligations"
            },
            "Legal Research": {
                "query": "Research relevant cases and precedents related to this document.",
                "agents": ["Legal Researcher"],
                "description": "Research on relevant legal cases and precedents"
            },
            "Risk Assessment": {
                "query": "Analyze potential legal risks and liabilities in this document.",
                "agents": ["Contract Analyst", "Legal Strategist"],
                "description": "Combined risk analysis and strategic assessment"
            },
            "Compliance Check": {
                "query": "Check this document for regulatory compliance issues.",
                "agents": ["Legal Researcher", "Contract Analyst", "Legal Strategist"],
                "description": "Comprehensive compliance analysis"
            },
            "Custom Query": {
                "query": None,
                "agents": ["Legal Researcher", "Contract Analyst", "Legal Strategist"],
                "description": "Custom analysis using all available agents"
            }
        }

        st.info(f"📋 {analysis_configs[analysis_type]['description']}")
        st.write(f"🤖 Active Legal AI Agents: {', '.join(analysis_configs[analysis_type]['agents'])}")  #dictionary!!

        # Replace the existing user_query section with this:
        if analysis_type == "Custom Query":
            user_query = st.text_area(
                "Enter your specific query:",
                help="Add any specific questions or points you want to analyze"
            )
        else:
            user_query = None  # Set to None for non-custom queries


        if st.button("Analyze"):
            if analysis_type == "Custom Query" and not user_query:
                st.warning("Please enter a query")
            else:
                with st.spinner("Analyzing document..."):
                    try:
                        # Ensure OpenAI API key is set
                        os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
                        os.environ['OPENROUTER_API_KEY'] = st.session_state.openai_api_key
                        
                        # Combine predefined and user queries
                        if analysis_type != "Custom Query":
                            combined_query = f"""
                            Using the uploaded document as reference:
                            
                            Primary Analysis Task: {analysis_configs[analysis_type]['query']}
                            Focus Areas: {', '.join(analysis_configs[analysis_type]['agents'])}
                            
                            Please search the knowledge base and provide specific references from the document.
                            """
                        else:
                            combined_query = f"""
                            Using the uploaded document as reference:
                            
                            {user_query}
                            
                            Please search the knowledge base and provide specific references from the document.
                            Focus Areas: {', '.join(analysis_configs[analysis_type]['agents'])}
                            """

                        response: RunOutput = st.session_state.legal_team.run(combined_query)
                        
                        # Display results in tabs
                        tabs = st.tabs(["Analysis", "Key Points", "Recommendations"])
                        
                        with tabs[0]:
                            st.markdown("### Detailed Analysis")
                            if response.content:
                                st.markdown(response.content)
                            else:
                                for message in response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)
                        
                        with tabs[1]:
                            st.markdown("### Key Points")
                            key_points_response: RunOutput = st.session_state.legal_team.run(
                                f"""Based on this previous analysis:    
                                {response.content}
                                
                                Please summarize the key points in bullet points.
                                Focus on insights from: {', '.join(analysis_configs[analysis_type]['agents'])}"""
                            )
                            if key_points_response.content:
                                st.markdown(key_points_response.content)
                            else:
                                for message in key_points_response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)
                        
                        with tabs[2]:
                            st.markdown("### Recommendations")
                            recommendations_response: RunOutput = st.session_state.legal_team.run(
                                f"""Based on this previous analysis:
                                {response.content}
                                
                                What are your key recommendations based on the analysis, the best course of action?
                                Provide specific recommendations from: {', '.join(analysis_configs[analysis_type]['agents'])}"""
                            )
                            if recommendations_response.content:
                                st.markdown(recommendations_response.content)
                            else:
                                for message in recommendations_response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)

                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
    else:
        st.info("Please upload a legal document to begin analysis")

if __name__ == "__main__":
    main()