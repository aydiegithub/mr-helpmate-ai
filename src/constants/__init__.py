import os
import dotenv
dotenv.load_dotenv()

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL: str = "models/text-embedding-004"

CHROMA_API_KEY: str = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT: str = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE: str = os.getenv("CHROMA_DATABASE")
CHROMA_SERVER_HTTP_PORT: int = 443
CHROMA_SERVER_HOST: str = "api.trychroma.com"

COLLECTION_NAME: str = "insurance_data_mr_helpmate"
CACHE_COLLECTION_NAME: str = "query_insurance_data_mr_helpmate"

DISTANCE_THRESHOLD: float = 0.1