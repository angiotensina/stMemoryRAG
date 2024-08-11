import nest_asyncio
import os
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv, find_dotenv

class PDFParser:
    def __init__(self, archivo_pdf, archivo_salida):
        self.archivo_pdf = archivo_pdf
        self.archivo_salida = archivo_salida
        self.documents = []
        self.instructions = """
            - Elimina encabezados y pies de pagina de cada pagina del PDF.
            - Organiza tablas en lineas de texto.
            - organiza todo el texto en una sola pagina separada por parrafos.
            - Los párrafos no finalizados complétalo con la información existente en la página siguiente, o en el párrafo siguiente.
            - Organiza en párrafos separados por dos retornos de carro.
            - Traduce el texto al español.
            """

    def apikey(self):
        nest_asyncio.apply()
        load_dotenv(find_dotenv(), override=True)

        self.llama_key = os.environ.get("LLAMA_CLOUD_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")

    def load_parser(self):
        self.parser = LlamaParse(
            api_key=self.llama_key,
            result_type="text",
            parsing_instruction=self.instructions,
            language="es",
            skip_diagonal_text=False,
            invalidate_cache=False,
            gpt4o_mode=False,
            gpt4o_api_key=self.openai_key,
            verbose=True,
            show_progress=True,
        )

    def parse_pdf(self):
        self.apikey()
        self.load_parser()
        self.documents = self.parser.load_data(self.archivo_pdf)
        self._save_output()

    def _save_output(self):
        if self.documents:  # Verifica si la lista 'documents' no está vacía
            with open(self.archivo_salida, "w", encoding="utf-8") as file:
            # Iterar sobre cada documento
                for i, self.document in enumerate(self.documents):
                    file.write(self.documents[i].text)
        else:
            print("No se encontraron documentos.")

        return self.archivo_salida

"""
# Uso de la clase
archivo_pdf = "./PK/CURRENT NEURO PARKINSON.pdf"
archivo_salida = "./PK/CURRENT NEURO PARKINSON.txt"

parser = PDFParser(archivo_pdf, archivo_salida)
ARCHIVO_SALIDA = parser.parse_pdf()
"""