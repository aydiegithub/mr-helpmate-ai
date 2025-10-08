import pdfplumber
from typing import Dict, Any
from src.logging import Logger

logging = Logger()

class Extractor:
    """
    Extractor class for extracting text and tables from PDF files.

    Methods
    -------
    content_extractor_node(file_path: str) -> Dict[str, Any]
        Extracts text and tables from each page of the PDF and returns the content as a dictionary.
    """

    def __init__(self):
        pass

    def content_extractor(self, file_path: str) -> str:
        """
        Extracts text and tables from each page of the PDF file.

        Parameters
        ----------
        file_path : str
            The path to the PDF file.

        Returns
        -------
        str
            A string containing the extracted content.
        """
        content = []
        try:
            logging.info(f"Opening PDF file: {file_path}")
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            content.append(page_text)
                        tables = page.extract_tables()
                        for t_idx, table in enumerate(tables, 1):
                            table_string = "\n".join(
                                ["\t".join(cell if cell is not None else "" for cell in row) for row in table]
                            )
                            content.append(table_string)
                    except Exception as e:
                        logging.warning(f"Failed to extract content from page {page_num}: {e}")
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return ""
        except Exception as e:
            logging.error(f"Error opening or processing PDF file: {e}")
            return ""

        logging.info(f"Extraction completed for file: {file_path}")
        return "\n\n".join(content)