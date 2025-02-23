from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from qdrant_client import QdrantClient, models

from decouple import config

qdrant_api_key = config("QDRANT_API_KEY")
qdrant_url = config("QDRANT_URL")
collection_name = "Websites"


def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], 
model=model).data[0].embedding


client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key
)


def check_collection(collection_name):
    existing_collections = client.get_collections().collections
    collection_names = [col.name for col in existing_collections]

    if collection_name not in collection_names:
        print(f"Collection '{collection_name}' not found. Creating...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )
        print(f"Collection '{collection_name}' created successfully.")
    else:
        print(f"Collection '{collection_name}' already exists.")


check_collection(collection_name)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=OpenAIEmbeddings(
        api_key=config("OPENAI_API_KEY")
    )
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=20, 
    length_function=len
)

def create_collection(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
    )
    print(f"Collection {collection_name} created successfully")


def is_url_indexed(url: str) -> bool:
    try:
        existing_docs = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source_url",
                        match=models.MatchValue(value=url)
                    )
                ]
            ),
            limit=1
        )
        return len(existing_docs[0]) > 0  # Check if any document exists

    except Exception as e:
        print(f"Error checking if URL is indexed: {str(e)}")
        return False


def upload_website_to_collection(url:str):
    if not client.collection_exists(collection_name=collection_name):
        create_collection(collection_name)

    if is_url_indexed(url):
        return f"URL {url} is already indexed, skipping."

    loader = WebBaseLoader(url)
    docs = loader.load_and_split(text_splitter)
    for doc in docs:
        doc.metadata = {"source_url": url}
    
    print(f"Indexing {len(docs)} documents for URL: {url}")
    
    vector_store.add_documents(docs)
    return f"Successfully uploaded {len(docs)} documents to collection {collection_name} from {url}"



def qdrant_search(query: str):
    vector_search = get_embedding(query)
    docs = client.search(
        collection_name=collection_name,
        query_vector=vector_search,
        limit=4
    )
    return docs


# create_collection(collection_name)
# upload_website_to_collection("https://hamel.dev/blog/posts/evals/")

