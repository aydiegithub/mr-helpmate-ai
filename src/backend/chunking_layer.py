from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.logging import Logger

logging = Logger()

class TextChunking:
    def __init__(self):
        pass
    
    
    def create_chunks(self, input_text: str, 
                      chunk_size: int = 1200, 
                      overlap: int = 250) -> list[str]:
        """
        Splits the input text into chunks of specified size with overlap.

        Args:
            input_text (str): The text to be chunked.
            chunk_size (int, optional): The size of each chunk. Defaults to 1200.
            overlap (int, optional): The number of overlapping characters between chunks. Defaults to 250.

        Returns:
            list[str]: A list of text chunks.
        """
        try:
            logging.info(f"Starting text chunking with chunk_size={chunk_size}, overlap={overlap}")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap
            )
            chunks = text_splitter.split_text(input_text)
            logging.info(f"Text chunking successful. Number of chunks created: {len(chunks)}")
            if not chunks:
                logging.warning("No chunks were created from the input text.")
            return chunks

        except Exception as e:
            logging.error(f"Error during text chunking: {e}")
            return []
