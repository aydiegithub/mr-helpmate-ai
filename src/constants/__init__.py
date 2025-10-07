import os
import dotenv
dotenv.load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = "models/text-embedding-004"

CHROMA_DB_API_KEY = os.getenv("CHROMA_DB_API_KEY")
COLLECTION_NAME = "insurance_data_mr_helpmate"
CACHE_COLLECTION_NAME = "query_insurance_data_mr_helpmate"

DISTANCE_THRESHOLD = 0.1