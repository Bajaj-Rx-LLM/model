import chromadb
from chromadb.utils import embedding_functions

class ChromaDBManager:
    """
    A manager class to handle all interactions with a ChromaDB server.
    This class is compatible with ChromaDB's /v2 API when used with an updated client library.
    """

    def __init__(self, host: str = "localhost", port: int = 8000, tenant: str = "default_tenant", database: str = "default_database"):
        """
        Initializes the client to connect to the ChromaDB server.

        Args:
            host (str): The hostname of the ChromaDB server.
            port (int): The port of the ChromaDB server.
            tenant (str): The name of the tenant to use.
            database (str): The name of the database to use.
        """
        try:
            # With an updated chromadb library, this client will correctly
            # communicate with the /v2 API endpoints.
            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                tenant=tenant,
                database=database
            )
            self.client.heartbeat()
            print(f"✅ Successfully connected to ChromaDB server at {host}:{port}")
            print(f"   Tenant: {tenant}, Database: {database}")
        except Exception as e:
            print(f"❌ Failed to connect to ChromaDB server: {e}")
            print("Please ensure the ChromaDB container is running and accessible.")
            self.client = None

    def get_or_create_collection(self, collection_name: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Gets an existing collection or creates a new one if it doesn't exist.
        """
        if not self.client:
            return None

        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            print(f"✅ Collection '{collection_name}' is ready.")
            return collection
        except Exception as e:
            print(f"❌ Could not get or create collection '{collection_name}': {e}")
            return None

    def add_documents(self, collection: chromadb.Collection, documents: list[str], metadatas: list[dict], ids: list[str]):
        """
        Adds documents to a specified collection.
        """
        if not collection:
            print("❌ Cannot add documents, collection is not valid.")
            return
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ Successfully added {len(documents)} documents to '{collection.name}'.")
        except Exception as e:
            print(f"❌ Error adding documents: {e}")

    def query_collection(self, collection: chromadb.Collection, query_text: str, n_results: int = 2):
        """
        Queries a collection to find documents similar to the query text.
        """
        if not collection:
            print("❌ Cannot query, collection is not valid.")
            return None
        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"❌ Error querying collection: {e}")
            return None

    def count_items(self, collection: chromadb.Collection) -> int:
        """
        Counts the total number of items in a collection.
        """
        if not collection:
            return 0
        return collection.count()

