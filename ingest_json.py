import json
from chroma_manager import ChromaDBManager

def ingest_data_from_json(json_file_path: str, collection_name: str):
    """
    Reads data from a JSON file and ingests it into a ChromaDB collection.

    Args:
        json_file_path (str): The path to the input JSON file.
        collection_name (str): The name of the collection to ingest data into.
    """
    # 1. Load the data from the JSON file
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded {len(data)} records from '{json_file_path}'.")
    except FileNotFoundError:
        print(f"❌ Error: The file '{json_file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: The file '{json_file_path}' is not a valid JSON file.")
        return

    # 2. Prepare data for ChromaDB
    ids = [item['id'] for item in data]
    documents = [item['document'] for item in data]
    metadatas = [item['metadata'] for item in data]

    # 3. Connect to ChromaDB and get the collection
    db_manager = ChromaDBManager()
    if not db_manager.client:
        return

    collection = db_manager.get_or_create_collection(collection_name)
    if not collection:
        return

    # 4. Add the documents to the collection
    print(f"🚀 Starting ingestion into '{collection_name}' collection...")
    db_manager.add_documents(
        collection=collection,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    # 5. Verify the ingestion
    item_count = db_manager.count_items(collection)
    print(f"\n✅ Verification complete. Collection '{collection_name}' now has {item_count} items.")


# --- Main execution block ---
if __name__ == "__main__":
    # Define the path to your JSON file and the target collection name
    INPUT_JSON_FILE = "data.json"
    COLLECTION_NAME = "rfc_documents"

    ingest_data_from_json(INPUT_JSON_FILE, COLLECTION_NAME)
