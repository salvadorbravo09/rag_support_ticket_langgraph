import hashlib # para generar hashes únicos de los documentos (detectar duplicados/cambios)
from typing import List # tipo genérico para anotar listas en type hints
from pathlib import Path # manejo de rutas de archivos de forma multiplataforma (Windows/Linux/Mac)
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import shutil # para eliminar directorios (limpiar ChromaDB antes de crear uno nuevo)

from config import *


class DocumentProcessor:
  """Procesador de documentos para el sistema RAG.
  
  Esta clase se encarga de:
  1. Cargar documentos desde un directorio (docs/)
  2. Dividirlos en chunks más pequeños para mejor búsqueda
  3. Generar embeddings y almacenarlos en ChromaDB (base de datos vectorial)
  
  Flujo: Documentos (.md) -> Chunks de texto -> Embeddings (vectores) -> ChromaDB
  """
  
  # Funcion de inicializacion del procesador, recibe rutas para documentos y ChromaDB
  def __init__(self, docs_path: str = DOCS_PATH, chroma_path: str = CHROMADB_PATH):
    # Ruta al directorio que contiene los documentos fuente (FAQs, manuales, guías)
    self.docs_path = Path(docs_path)
    # Ruta donde se almacenará la base de datos vectorial ChromaDB
    self.chroma_path = Path(chroma_path)
    
    # Modelo de embeddings de Google - convierte texto en vectores de 768 dimensiones
    # Estos vectores capturan el "significado semántico" del texto,
    # permitiendo buscar por similitud de contenido (no solo palabras exactas)
    self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
    # Splitter que divide documentos largos en chunks manejables
    # Es "recursivo" porque intenta dividir por los separadores en orden de prioridad
    self.text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200,
      # Función para medir el largo del texto (usa len = cantidad de caracteres)
      length_function=len,
      # Separadores ordenados por prioridad: primero intenta cortar por doble salto de línea,
      # luego por salto simple, luego por puntuación, y como último recurso por espacio o carácter
      separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
  
  # Funcion para cargar documentos desde el directorio docs. Devuelve una lista de objetos Document (texto + metadatos)
  def load_documents(self) -> List[Document]:
    """Carga documentos markdown (.md) desde el directorio docs."""
    print(f"Cargando documentos desde: {self.docs_path}")
    
    # Cargar archivos markdown usando DirectoryLoader de LangChain
    loader = DirectoryLoader(
      str(self.docs_path), # Ruta al directorio de documentos
      glob="*.md", # Solo cargar archivos con extensión .md (Markdown)
      loader_cls=TextLoader, # Usar TextLoader para cargar el contenido como texto plano
      loader_kwargs={"encoding": "utf-8"} # Asegurar que se lean con codificación UTF-8 para evitar errores de caracteres
    )
    
    # Cargar los documentos
    documents = loader.load()
    
    # Enriquece metadatos de cada documento
    # Por cada documento cargado, agregamos metadatos adicionales
    """ return
      "filename": "faq"
      "doc_type": "faq",
      "doc_id": "a3f8c2d1..."
    """
    for doc in documents:
      filename = Path(doc.metadata["source"]).stem # Obtener el nombre del archivo sin extensión
      doc.metadata.update({
        "filename": filename,
        "doc_type": self._get_doc_type(filename), # Inferir tipo de documento (FAQ, manual, guía) basado en el nombre del archivo
        "doc_id": self._generate_doc_id(doc.page_content) # Generar un ID único basado en el contenido del documento
      })
      
    print(f"Cargado {len(documents)} documentos.")
    return documents
  
  # Funcion privada auxiliar para inferir el tipo de documento basado en el nombre del archivo
  def _get_doc_type(self, filename: str) -> str:
    """Determina el tipo de documento basado en el nombre del archivo."""
    
    # Si el nombre del archivo contiene "faq || manual || etc" entonces asignamos un tipo específico, sino lo clasificamos como "general"
    if "faq" in filename.lower():
      return "faq"
    elif "manual" in filename.lower():
      return "manual"
    elif "guia" in filename.lower() or "resolucion" in filename.lower():
      return "troubleshooting"
    else:
      return "general"
    
  # Funcion privada auxiliar para generar un ID unico para cada documento basado en su contenido
  def _generate_doc_id(self, content: str) -> str:
    """Genera un ID unico para el documento usando un hash de su contenido."""
    return hashlib.md5(content.encode()).hexdigest()[:8] # Usamos MD5 para generar un hash del contenido, y tomamos solo los primeros 8 caracteres para un ID compacto
  
  # Funcion para dividir documentos en chunks usando el text splitter configurado
  def split_documents(self, documents: List[Document]) -> List[Document]:
    """Divide documentos en chunks mas pequeños"""
    print("Dividiendo documentos en chunks...")
    
    # El splitter de texto toma cada documento y lo divide en partes más pequeñas (chunks) según la configuración de chunk_size y chunk_overlap.
    chunks = self.text_splitter.split_documents(documents)
    
    # Agregar metadatos adicionales a cada chunk para rastrear su origen
    # A cada chunk resultante se le añaden dos campos
    for i, chunk in enumerate(chunks):
      chunk.metadata.update({
        "chunk_id": i, # Indice del chunk (0, 1, 2, ...) para saber su posicion
        "chunk_size": len(chunk.page_content) # Longitud real en caracteres de ese chunk especifico
      })
    
    print(f"Creado {len(chunks)} chunks de texto.")
    return chunks
  
  # Funcion para crear la base de datos vectorial ChromaDB a partir de los chunks de texto
  # recibe "documents": lista de chunks generada por split_documents
  def create_vectorstore(self, documents: List[Document]) -> Chroma:
    """Crea el vectorstore con ChromaDB a partir de los documentos procesados."""
    print("Creando vectorstore con ChromaDB...")
    
    # Limpiar el directorio anterior de ChromaDB si existe y para evitar conflictos con datos antiguos
    if self.chroma_path.exists():
      shutil.rmtree(self.chroma_path)
      
    # Crear el vectorstore usando Chroma, que se encargará de generar los embeddings y almacenarlos
    vectorstore = Chroma.from_documents(
      documents=documents, # Lista de documentos (chunks) a indexar
      embedding=self.embeddings, # Modelo de embeddings para convertir texto en vectores
      persist_directory=str(self.chroma_path), # Ruta donde se guardará la base de datos de ChromaDB (vectores + metadatos)
      collection_name="support_docs" # Nombre de la colección dentro de ChromaDB para organizar los documentos relacionados al soporte
    )
    
    print(f"Vectorstore creado en: {self.chroma_path}")
    print(f"Total de vectores almacenados: {len(documents)}")
    
    return vectorstore
    
  # Funcion para cargar un vectorstore existente desde ChromaDB si ya fue creado previamente
  # Sirve para ir cargando documentos nuevos sin tener que volver a procesar todo desde cero, o para reiniciar el sistema sin perder los datos ya indexados
  def load_existing_vectorstore(self) -> Chroma:
    """Carga un vectorstore existente desde ChromaDB si ya fue creado previamente."""
    
    # Verificar si el directorio de ChromaDB existe, si no existe, lanzar un error indicando que no se ha creado aun
    if not self.chroma_path.exists():
      raise FileNotFoundError(f"Vectorstore no encontrado en: {self.chroma_path}")
    
    vectorstore = Chroma(
      persist_directory=str(self.chroma_path), # Ruta donde se encuentra la base de datos de ChromaDB
      embedding_function=self.embeddings, # Modelo de embeddings para convertir texto en vectores
      collection_name="support_docs" # Nombre de la colección dentro de ChromaDB para organizar los documentos relacionados al soporte
    )
    
    return vectorstore
  
  # Funcion principal para configurar todo el sistema RAG: carga, procesamiento y creación del vectorstore
  def setup_rag_system(self, force_rebuild: bool = False):
    """Configura el sistema RAG completo: carga, procesa y crea el vectorstore."""
    print("Configurando sistema RAG...")
    
    # Verificar si ya existe un vectorstore previamente y no forzar rebuild
    if self.chroma_path.exists() and not force_rebuild:
      print("Vectorstore existente encontrado")
      return self.load_existing_vectorstore()
    
    # Cargar y procesar los documentos para crear un nuevo vectorstore
    documents = self.load_documents()
    if not documents:
      print("No se encontraron documentos para procesar.")
      return None
    
    # Dividir documentos
    chunks = self.split_documents(documents)
    
    # Crear vectorstore con los chunks procesados
    vectorstore = self.create_vectorstore(chunks)
    
    print("Sistema RAG configurado exitosamente.")
    return vectorstore
  
  # Funcion de prueba para verificar que el vectorstore funciona correctamente realizando una busqueda de similitud con una consulta de ejemplo
  def test_search(self, vectorstore: Chroma, query: str = "resetear contraseña"):
    """Prueba la funcionalidad de busqueda."""
    print(f"\nProbando busqueda: '{query}'")
    
    results = vectorstore.similarity_search(query, k=3) # Buscar los 3 chunks más similares a la consulta
    
    for i, doc in enumerate(results, 1):
      print(f"\nResultado {i}:")
      print(f"Tipo: {doc.metadata.get('doc_type', 'unknown')}")
      print(f"Archivo: {doc.metadata.get('filename', 'unknown')}")
      print(f"Contenido: {doc.page_content[:200]}...")
    
    return results
    
def main():
  """Funcion principal para configurar el sistema RAG."""
  print("Configuracion RAG - Document Processor")
  print("=" * 40)
  
  # Configurar procesador
  processor = DocumentProcessor(docs_path=DOCS_PATH, chroma_path=CHROMADB_PATH) # Crear una instancia del DocumentProcessor con las rutas configuradas para los documentos y ChromaDB
  
  # Configurar sistema RAG
  vectorstore = processor.setup_rag_system(force_rebuild=True) # Forzar rebuild para asegurarnos de procesar los documentos actuales
  
  if vectorstore:
    # Probar busquedas
    test_queries = [
      "resetear contraseña",
      "error 500",
      "cancelar suscripción",
      "aplicacion lenta"
    ]
    
  for query in test_queries:
    processor.test_search(vectorstore, query)
    
  print("\nConfiguracion completada")
  
if __name__ == "__main__":
  main()