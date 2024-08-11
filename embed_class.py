import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility, MilvusClient, db
from langchain_milvus.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings

class EmbeddingProcessor:
    def __init__(self, input_file, uri_connection, host, port, db_name, collection_name, dimension_embedding):
        self.input_file = input_file
        self.uri_connection = uri_connection
        self.host = host
        self.port = port
        self.db_name = db_name
        self.collection_name = collection_name
        self.dimension_embedding = dimension_embedding
        self.api_key_openAI = None
        self.docs = None
        self.client = None
        self.vector_store = None

    def load_api_key(self):
        load_dotenv(find_dotenv(), override=True)
        self.api_key_openAI = os.environ.get("OPENAI_API_KEY")
        print(self.api_key_openAI)

    def load_document(self):
        loader = TextLoader(self.input_file)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(separator="\n\n", chunk_overlap=0)
        self.docs = text_splitter.split_documents(documents)
        self.add_metadata_to_documents()

    def add_metadata_to_documents(self):
        for i, doc in enumerate(self.docs):
            doc.metadata["chunk"] = i
            doc.metadata["Title"] = "Cotton & Williams"
            doc.metadata["Capitulo"] = "Colonoscopia"
            doc.metadata["author"] = "Joaquin Chamorro"

    def connect_to_milvus(self):
        self.client = MilvusClient(uri=self.uri_connection, token="joaquin:chamorro")
        connections.connect(alias="default", host=self.host, port=self.port)

    def create_database(self):
        if self.db_name not in db.list_database():
            db.create_database(self.db_name, using="default")
            db.using_database(self.db_name, using="default")
        else:
            db.using_database(self.db_name, using="default")

        print(f"Conectado a la base de datos {self.db_name}")

    def create_collection(self):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=self.api_key_openAI)
        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 8, "efConstruction": 64}
        }

        if self.collection_name not in self.client.list_collections():
            self.vector_store = Milvus.from_documents(
                self.docs,
                embedding=embeddings,
                collection_name=self.collection_name,
                connection_args={"uri": self.uri_connection},
                primary_field="pk",
                text_field="text",
                vector_field="vector",
                index_params=index_params,
                enable_dynamic_field=True,
                drop_old=True
            )
            print("Colección creada")
        else:
            self.vector_store = Milvus(
                embeddings,
                collection_name=self.collection_name,
                connection_args={"uri": self.uri_connection},
                enable_dynamic_field=True,
                primary_field="pk",
                text_field="text",
                vector_field="vector",
                index_params=index_params
            )
            print("Colección ya existe")

        print(f"Conexión a Milvus-VectorStore establecida.\nConectado a coleccion: {self.collection_name}\n")
        
    def process(self):
        self.load_api_key()
        self.load_document()
        self.connect_to_milvus()
        self.create_database()
        self.create_collection()
        
        

"""
# Usage
input_file = 'your_input_file.txt'
uri_connection = 'your_uri_connection'
host = 'your_host'
port = 'your_port'
db_name = 'your_db_name'
collection_name = 'your_collection_name'
dimension_embedding = 3072  # Example dimension

processor = EmbeddingProcessor(input_file, uri_connection, host, port, db_name, collection_name, dimension_embedding)
processor.process()
"""