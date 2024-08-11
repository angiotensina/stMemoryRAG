import streamlit as st
from pymilvus import connections, db, CollectionSchema, FieldSchema, DataType, Collection, MilvusClient
import os
from dotenv import load_dotenv, find_dotenv


from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# import Langchain modules
from langchain.agents import tool
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFDirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus.vectorstores import Milvus

# Load docs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate

######################### VARIABLES DE ENTORNO #########################
QUERY = "Tratamiento del parkinson segun el documento aportado"

LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-large"
DIMENSION_EMBEDDING = 3072

INPUT_FILE = "./colonoscopy.txt"
OUTPUT_FILE = f"{INPUT_FILE}_output.txt"
BD_NAME = "EPID"
COLLECTION_NAME = "EPID"

URI_CONNECTION = "http://localhost:19530"
HOST = "localhost"
PORT = 19530


######################### FUNCIONES #########################

def getVectorStoreMilvus(dbName, collectionName, api_key_openAI):
    ######################### CONEXIÓN A MILVUS #########################

    uri = URI_CONNECTION

    client = MilvusClient(
        uri=uri,
        token="joaquin:chamorro"
    )

    connections.connect(alias="default", host=HOST, port=PORT)

    ######################### CREAR LA BASE DE DATOS EN MILVUS #########################

    db_name = dbName
    if db_name not in db.list_database():
        db.create_database(db_name, using="default")
        db.using_database(db_name, using="default")
    else:
        db.using_database(db_name, using="default")

    print(f"Conectado a la base de datos {db_name}")

    ######################### GUARDAR LOS VECTORES EN MILVUS - VECTORSTORE #########################
    #vector_store = Milvus()

    # Crear la función de embeddings de código abierto
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key_openAI)

    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 8, "efConstruction": 64}
    }

    vector_store = Milvus(
        embeddings,
        collection_name=collectionName,
        connection_args={"uri": uri},
        enable_dynamic_field=True,
        primary_field="pk",
        text_field="text",
        vector_field="vector",
        index_params=index_params
    )
    print("Colección ya existe")

    print(f"Conexión a Milvus-VectorStore establecida.\nConectado a colleccion: {collectionName}\n")
    
    return vector_store


def getAnswer(query, vector_store, api_key_openAI):
    
    model = ChatOpenAI(api_key=api_key_openAI, model=LLM_MODEL)

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 10, "filter": {"chapter": "30"}})
        #search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50})


    ######################### EJECUTAR EL PIPELINE #########################

    template =  """
                - Contesta como un profesional medico: {context}
                - Si no se aportan documentos:
                    - Menciona que no se aportan documentos
                    - Responde con tu conocimiento
                - Question: {question}
                """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()})
    chain = setup_and_retrieval | prompt | model | output_parser
    respuesta=chain.invoke(query)

    return respuesta
    
    
def main():
    ######################### OBTENER API KEY DE OPENAI #########################
    # Carga las variables de entorno desde un archivo .env
    load_dotenv(find_dotenv(), override=True)

    # Obtiene la API key de OpenAI desde las variables de entorno
    api_key_openAI = os.environ.get("OPENAI_API_KEY")
    print(api_key_openAI)

    vector_store = getVectorStoreMilvus(BD_NAME, COLLECTION_NAME, api_key_openAI)

    
    st.title("Consultar sobre los documentos")
    st.sidebar.markdown("<H1 style='text-align: center'> Panel principal </H1>", unsafe_allow_html=True)
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = getAnswer(prompt, vector_store, api_key_openAI)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        
if __name__ == "__main__":
    main()