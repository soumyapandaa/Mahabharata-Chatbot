import streamlit as st
from groq import Groq
import fitz # this is pymupdf
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()  # reads the .env file

groq_client = Groq(api_key= os.getenv("GROQ_API_KEY"))

# page config
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("AI Document Assistant 🤖")

@st.cache_resource
def load_rag_system():
    try :
        if not os.path.exists("mahabharata.pdf"):
            st.error("❌ mahabharata.pdf not found. Please add it to the project folder.")
            st.stop()
        
        ### load your PDF
        doc = fitz.open("mahabharata.pdf")

        # extract text from all pages : 
        text = ''
        for page in doc:
            text += page.get_text()



        ### Chunking :

        # cut text into chunks of 500 characters 
        # with character overlap between chunks

        def chunk_text(text, chunk_size = 500, overlap = 50):
            chunks = []
            start = 0

            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                chunks.append(chunk)
                start = end - overlap # overlap so we don't miss context

            return chunks

        chunks = chunk_text(text)
        print(f"Total chunks: {len(chunks)}")
        print(f"\nFirst chunk: {chunks[0]}")
        print(f"\nSecond chunk: {chunks[1]}")    

        ### Convert Chunks into Vectors :

        # load the emebedding model 
        # this runs locally on your laptop - no API needed
        embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # convert all 73 chunks into vectors 
        embeddings = embedder.encode(chunks)

        print(f'Total embeddings: {len(embeddings)}')
        print(f'Each vector size: {len(embeddings[0])} numbers')

        ### Storing all this vectors in ChromaDB so we can search them :

        # create a local database
        # saves to your laptop
        client_db = chromadb.Client()

        # create a collection (like a table in database)
        
        try:
            client_db.delete_collection('Mahabharata_chunks')
        except:
            pass
        collection = client_db.create_collection('Mahabharata_chunks')
        
        # store all chunks with their vectors
        collection.add(
            documents = chunks,                               # original text chunks
            embeddings = embeddings.tolist(),                 # their vectors 
            ids = [f"chunk{i}" for i in range(len(chunks))]   # unique id for each 
        )

        print(f"Stored {collection.count()} chunks in database ✅")
        return collection, embedder


    
    except Exception as e:
        st.error(f"❌ Failed to load document :{str(e)}")


collection, embedder = load_rag_system()


# initialize chat history
# session_state survives reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# chat input at bottom
user_input = st.chat_input("Ask anything...")

if user_input:
    
    try :
            # search chromadb
        input_embedding = embedder.encode([user_input]).tolist()
        results = collection.query(
            query_embeddings=input_embedding,
            n_results=3
        )

        # Better way (works for any n_results):
        context = "\n".join(results['documents'][0])
        
    except Exception as e :
        st.error("❌ Search failed. Please try again.")
        st.stop()
    
    # add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
     # build messages = system prompt + full history
    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant that answers questions 
            based on the provided context only.
            If answer not in context say 'I could not find this in the document'
            
            Context from document:
            {context}"""   # ← context goes in system prompt, not user message
        }
    ] + st.session_state.messages  # ← attach entire history here ✅             
    
    try :          
        # send to groq
        with st.spinner("Thinking..."):   
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages
            )
        
        reply = response.choices[0].message.content
        
    except Exception as e:
        reply = "❌ I couldn't get a response. Please try again."
        st.error(f"API Error : {str(e)}")
    
    
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": reply
    })
    
    st.rerun()