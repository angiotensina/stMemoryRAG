import os
from dotenv import load_dotenv, find_dotenv
from pymilvus import connections, db, CollectionSchema, FieldSchema, DataType, Collection, MilvusClient

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

######################### VARIABLES DE ENTORNO #########################
QUERY = "Tecnicas para evitar la formacion de asas durante la colonoscopia"

LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-large"
DIMENSION_EMBEDDING = 3072

INPUT_FILE = "./colonoscopy.txt"
OUTPUT_FILE = f"{INPUT_FILE}_output.txt"
BD_NAME = "CottonWilliams"
COLLECTION_NAME = "Colonoscopy"

URI_CONNECTION = "http://localhost:19530"
HOST = "localhost"
PORT = 19530

######################### OBTENER API KEY DE OPENAI #########################
# Carga las variables de entorno desde un archivo .env
load_dotenv(find_dotenv(), override=True)

# Obtiene la API key de OpenAI desde las variables de entorno
api_key_openAI = os.environ.get("OPENAI_API_KEY")
print(api_key_openAI)

######################### CARGAR EL DOCUMENTO #########################

# load the document and split it into chunks

# loader = PyPDFDirectoryLoader("./folder")
loader = TextLoader(INPUT_FILE)
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(
    separator="\n\n", 
    chunk_overlap=0
    )
docs = text_splitter.split_documents(documents)

######################## ADDING METADATA TO DOCUMENTS ########################
# Adding metadata to documents
for i, doc in enumerate(docs):
    doc.metadata["chunk"] = i
    doc.metadata["Title"] = "Cotton & Williams"
    doc.metadata["Capitulo"] = "Colonoscopia"
    doc.metadata["author"] = "Joaquin Chamorro"

######################### CONECTAR A MILVUS #########################

uri = URI_CONNECTION

client = MilvusClient(
    uri=uri,
    token="joaquin:chamorro"
)

connections.connect(alias="default", host=HOST, port=PORT)

######################### CREAR LA BASE DE DATOS EN MILVUS #########################

db_name = BD_NAME
if db_name not in db.list_database():
    db.create_database(db_name, using="default")
    db.using_database(db_name, using="default")
else:
    db.using_database(db_name, using="default")

print(f"Conectado a la base de datos {db_name}")

######################### CREAR LA COLECCIÓN EN MILVUS #########################
collection_name = COLLECTION_NAME
dimension = DIMENSION_EMBEDDING  # Especifica la dimensión del vector

######################### GUARDAR LOS VECTORES EN MILVUS - VECTORSTORE #########################
#vector_store = Milvus()

# Crear la función de embeddings de código abierto
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key_openAI)

index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 8, "efConstruction": 64}
}

if collection_name not in client.list_collections():
    vector_store = Milvus.from_documents(
        docs,
        embedding=embeddings,
        collection_name=collection_name,
        connection_args={"uri": uri},
        primary_field="pk",
        text_field="text",
        vector_field="vector",
        index_params=index_params,
        enable_dynamic_field=True,
        drop_old=True
    )
    print("Colección creada")
else:
    vector_store = Milvus(
        embeddings,
        collection_name=collection_name,
        connection_args={"uri": uri},
        enable_dynamic_field=True,
        primary_field="pk",
        text_field="text",
        vector_field="vector",
        index_params=index_params
    )
    print("Colección ya existe")

print(f"Conexión a Milvus-VectorStore establecida.\nConectado a colleccion: {collection_name}\n")

######################### RECUPERAR LOS VECTORES SIMILARES - RETRIEVER #########################

query = QUERY

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

print(respuesta)

filename = "colonoscopy.txt"
if documents:  # Verifica si la lista 'documents' no está vacía
    with open(filename, "w", encoding="utf-8") as file:
    # Iterar sobre cada documento
        for i, document in enumerate(documents):
            file.write(respuesta)

else:
    print("No se encontraron documentos.")

