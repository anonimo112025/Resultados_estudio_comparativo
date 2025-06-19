import os
import re
import subprocess
import csv
from datetime import datetime

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from rouge_score import rouge_scorer

# Cargardo variables de entorno
load_dotenv()

# 1. Cargar la HU y su contexto
def load_documents():
    loaders = [
        TextLoader("data/contexto/SIN_CONTEXTO.md", encoding="utf-8"),
        PyMuPDFLoader("data/historias_usuario/ingles/HU01.pdf")
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs

# 2. Procesar documentos en PDF
def process_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)

# 3. Construir vectorstore
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local("vectorstore")
    return vectordb

# 4. Crear cadena de preguntas-respuestas según modelo seleccionado
def create_qa_chain(vectordb, model_name):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    if model_name == "llama":
        llm = OllamaLLM(model="llama3.2")
    elif model_name == "deepseek":
        llm = OllamaLLM(model="deepseek-r1")
    elif model_name == "mistral":
        llm = OllamaLLM(model="mistral")
    elif model_name == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-preview-05-06", google_api_key=api_key)
    elif model_name == "claude": 
        llm = ChatAnthropic(model="claude-sonnet-4-20250514", api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif model_name == "gpt":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(
            model_name="gpt-4",
            openai_api_key=openai_api_key
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Cargando reglas de los diagramas de clases o actividades
def load_rules(diagram_type):
    path = f"data/reglas/reglas_clases.txt"
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def generate_prompt(user_story, additional_info):
    reglas = load_rules("clases")
    
    valid_example = """
        @startuml
        class User {
            - username: String
            - password: String
            + login(): boolean
        }
        @enduml
        """

    invalid_example = """
        @startuml
        :User
        + User(username: String)
        @enduml

        @startuml
        class System {
            + createAccount()
        }
        @enduml
        """
    
    prompt = f"""
            You are a software analyst.

            Based on the previously provided context (in Spanish) and the following user story (in English), generate a class diagram at the **analysis level**.

            User story:
            {user_story}

            Additional information:
            {additional_info}

            STRICT INSTRUCTIONS:
            - Your output MUST consist ONLY of valid PlantUML code for a class diagram.
            - The output MUST start with "@startuml" and end with "@enduml", with NOTHING outside this block.
            - Each class MUST be declared using the syntax: `class ClassName { ... }`.
            - DO NOT use `:ClassName`, `interface`, or any other syntax different from `class`.
            - DO NOT include constructors like `+ ClassName(...)` inside the class.
            - DO NOT write comments (e.g., `// ...`, `/* ... */`) or explanation tags (e.g., <think>, <note>, <explanation>, etc.).
            - DO NOT include any class named "System", "Sistema", or placeholder entities.
            - Use only public (+), private (-), or protected (#) for visibility of attributes and methods.
            - Output MUST be syntactically valid PlantUML code suitable for direct rendering.
            - Use consistent indentation and formatting.

            Modeling rules:
            {reglas}

            Diagram requirements:
            - Focus on user responsibilities, not technical implementation.
            - Exclude infrastructure-related classes (controllers, databases, services, etc.).
            - Use class names, attributes, and methods in **English**, consistent with the user story.
            - Group responsibilities into meaningful and relevant classes.

            Output format:
            Only output valid PlantUML code between "@startuml" and "@enduml". Nothing else.

            EXAMPLES:

            ✅ VALID:
            {valid_example}

            ❌ INVALID:
            {invalid_example}
            
        """
    
    return prompt

# 5. Generar el prompt interno y preguntar
def ask_question(chain, model_name):
    """
    Genera un prompt para el modelo LLM, obtiene el diagrama UML en formato PlantUML
    y lo guarda en la estructura de carpetas definida.

    Args:
        chain (RetrievalQA): Cadena de preguntas y respuestas del modelo.
        model_name (str or list): Nombre del modelo utilizado o lista de modelos.
    """
    def process_model(model):
        print(f"\n🔄 Generando prompt para el modelo **{model}** con tipo de diagrama de clases ...")

        prompt = generate_prompt()
        print("🧠 Enviando prompt al modelo...")
        response = chain.invoke({"query": prompt})
        raw_output = response["result"]

        cleaned_diagram = clean_and_extract_plantuml(raw_output)

        if not cleaned_diagram:
            print(f"❌ Error: El modelo {model} no generó un bloque válido de PlantUML.")
            return

        print("\n✅ Respuesta recibida del LLM:")
        print(cleaned_diagram)
        print("\n💾 Guardando el diagrama en la estructura de carpetas...")
        save_diagram(cleaned_diagram, model, "clases")

    # Procesamiento para uno o varios modelos
    if isinstance(model_name, list):
        for model in model_name:
            process_model(model)
    else:
        process_model(model_name)


# 6. Guardar diagrama
def save_diagram(diagram_code, model_name=None, diagram_kind=None):
    """
    Guarda el diagrama generado en un archivo, usando el nombre del modelo y el tipo de diagrama.
    Además, se guarda en una carpeta estructurada por modelo y tipo de diagrama.

    Args:
        diagram_code (str): Código del diagrama en PlantUML.
        model_name (str): Nombre del modelo usado para generar el diagrama.
        diagram_kind (str): Tipo de diagrama (clases, actividades, etc.).
    """
    if not model_name or not diagram_kind:
        print("❌ Error: model_name y diagram_kind son obligatorios para guardar el diagrama.")
        return

    # Definir la estructura de carpetas
    base_dir = os.path.join("data","diagramas_generados", model_name, diagram_kind)

    # Generar el nombre del archivo
    filename = f"{model_name}_{diagram_kind}.puml"
    file_path = os.path.join(base_dir, filename)

    # Guardar el código PlantUML
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(diagram_code)
    
    print(f"✅ Diagrama guardado como: {file_path}")
    
    # Guardar también en el formato plano para evaluación ROUGE
    eval_file = f"{model_name}_{diagram_kind}.txt"
    eval_path = os.path.join(base_dir, eval_file)
    with open(eval_path, "w", encoding="utf-8") as file:
        file.write(diagram_code)
    
    print(f"✅ Salida del modelo guardada para evaluación ROUGE en: {eval_path}")

    # Llamar al renderizado con la nueva ruta generada
    render_diagram(file_path)

# 7. Renderizar diagrama
def render_diagram(file_path):
    """
    Renderiza el diagrama UML utilizando PlantUML y lo guarda en la misma ruta
    donde se encuentran los archivos .puml y .txt.

    Args:
        file_path (str): Ruta completa del archivo .puml a renderizar.
    """
    jar_path = "C:/plantuml/plantuml.jar"

    # Verificación de existencia del archivo
    if not os.path.exists(file_path):
        print(f"❌ Error: No se encontró el archivo {file_path} para renderizar.")
        return
 
    # Directorio para ejecutar el comando (donde está el .puml)
    output_dir = os.path.dirname(file_path)
    try:
        # Ejecutar PlantUML en el mismo directorio
        subprocess.run(["java", "-jar", jar_path, "-tpng", "-o", "", file_path], check=True)

        # Construir el nombre del archivo .png basado en el .puml
        image_name = os.path.splitext(os.path.basename(file_path))[0] + ".png"
        image_path = os.path.join(output_dir, image_name)
        
        print(f"✅ Diagrama renderizado correctamente en: {image_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al renderizar el diagrama: {e}")


# 8. Menú para elegir modelo y tipo de diagrama
def choose_model():
    opciones = ["llama", "deepseek", "gemini", "claude", "gpt", "mistral", "Todos"]
    print("\nSelecciona el modelo LLM a utilizar:")
    for i, modelo in enumerate(opciones, 1):
        print(f"{i}. {modelo}")

    while True:
        try:
            choice = int(input("Modelo: "))
            if 1 <= choice <= len(opciones):
                break
            else:
                print(f"Por favor ingresa un número entre 1 y {len(opciones)}.")
        except ValueError:
            print("Entrada inválida. Ingresa un número.")

    seleccionado = opciones[choice - 1]
    modelos = opciones[:-1] if seleccionado == "Todos" else [seleccionado]
    
    return modelos


# 9. Evaluación con la métrica de ROUGE
def evaluate_rouge(model_name, diagram_type):
    """
    Evalúa el diagrama generado por el modelo usando ROUGE contra un archivo de referencia
    y guarda los resultados en un archivo CSV.

    Args:
        model_name (str): Nombre del modelo utilizado o lista de modelos.
        diagram_type (str): Tipo de diagrama generado ("clases" o "actividades").
    """
    print("\n🔎 Iniciando evaluación ROUGE...")

    # Crear el archivo de resultados si no existe
    output_file = "resultados_rouge_todos_CON_RAG.csv"
    header = ["Modelo", "Diagrama", "ROUGE Tipo", "Precisión", "Cobertura", "F1", "Fecha"]
    
    # Verificar si el archivo ya existe para no sobreescribir el encabezado
    file_exists = os.path.exists(output_file)
    
    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(header)  # Escribir el encabezado si es la primera vez

        # Si model_name es una lista, se evalúan todos los modelos
        if isinstance(model_name, list):
            for model in model_name:
                print(f"\n🔄 Evaluando modelo: {model}")
                generated_path = f"data/diagramas_generados/{model}/{diagram_type}/{model}_{diagram_type}.txt"
                reference_path = "data/diagramas_esperados/PlantUML-HU10.txt"

                if not os.path.exists(generated_path):
                    print(f"❌ Error: No se encontró el archivo generado en {generated_path}")
                    continue

                if not os.path.exists(reference_path):
                    print(f"❌ Error: No se encontró el archivo de referencia en {reference_path}")
                    continue

                with open(generated_path, 'r', encoding='utf-8') as f:
                    generated_text = f.read()

                with open(reference_path, 'r', encoding='utf-8') as f:
                    reference_text = f.read()

                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                scores = scorer.score(reference_text, generated_text)

                print(f"\n🔍 Resultados de evaluación ROUGE para el modelo {model}:")
                for key, score in scores.items():
                    print(f"{key}:")
                    print(f"  - Precisión (P): {score.precision:.3f}")
                    print(f"  - Cobertura (R): {score.recall:.3f}")
                    print(f"  - Medida F1 (F1): {score.fmeasure:.3f}")

                    # Guardar en CSV
                    writer.writerow([
                        model, diagram_type, key, 
                        f"{score.precision:.3f}", 
                        f"{score.recall:.3f}", 
                        f"{score.fmeasure:.3f}", 
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ])

        else:
            # Evaluación normal para un único modelo
            print(f"\n🔄 Evaluando modelo: {model_name}")
            generated_path = f"data/diagramas_generados/{model_name}/{diagram_type}/{model_name}_{diagram_type}.txt"
            reference_path = "data/diagramas_esperados/PlantUML-HU10.txt"

            if not os.path.exists(generated_path):
                print(f"❌ Error: No se encontró el archivo generado en {generated_path}")
                return

            if not os.path.exists(reference_path):
                print(f"❌ Error: No se encontró el archivo de referencia en {reference_path}")
                return

            with open(generated_path, 'r', encoding='utf-8') as f:
                generated_text = f.read()

            with open(reference_path, 'r', encoding='utf-8') as f:
                reference_text = f.read()

            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference_text, generated_text)

            print("\n🔍 Resultados de evaluación ROUGE:")
            for key, score in scores.items():
                print(f"{key}:")
                print(f"  - Precisión (P): {score.precision:.3f}")
                print(f"  - Cobertura (R): {score.recall:.3f}")
                print(f"  - Medida F1 (F1): {score.fmeasure:.3f}")

                # Guardar en CSV
                writer.writerow([
                    model_name, diagram_type, key, 
                    f"{score.precision:.3f}", 
                    f"{score.recall:.3f}", 
                    f"{score.fmeasure:.3f}", 
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])

        print(f"\n✅ Resultados de evaluación guardados en {output_file}")
        print("-" * 50)


def evaluate_all_models_rouge(model_names, diagram_type):
    print("🔎 Iniciando evaluación ROUGE para todos los modelos...")

    resultados_csv = "resultados_rouge_todos_con_RAG.csv"
    encabezado = ["Historia", "Modelo", "Tipo Diagrama", "ROUGE Tipo", "Precisión", "Recall", "Medida F1"]
    
    # Crear archivo si no existe
    archivo_existe = os.path.exists(resultados_csv)
    with open(resultados_csv, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not archivo_existe:
            writer.writerow(encabezado)
        
        for i in range(1, 11):  # historias del 1 al 10
            diagrama_esperado_id = f"HU{i:02}"
            archivo_esperado = os.path.join("data", "diagramas_esperados", f"PlantUML-{diagrama_esperado_id}.txt")

            if not os.path.exists(archivo_esperado):
                print(f"⚠️  No se encontró el archivo esperado: {archivo_esperado}")
                continue

            with open(archivo_esperado, "r", encoding="utf-8") as f:
                referencia = f.read()

            for modelo in model_names:
                diagrama_generado_id = f"HU{i:02}_{modelo}_{diagram_type}"
                archivo_generado = os.path.join("data", "diagramas_generados", modelo, diagram_type, f"{diagrama_generado_id}.txt")

                if not os.path.exists(archivo_generado):
                    print(f"⚠️  No se encontró el archivo generado: {archivo_generado}")
                    continue

                with open(archivo_generado, "r", encoding="utf-8") as f:
                    generado = f.read()

                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                scores = scorer.score(referencia, generado)

                for clave, score in scores.items():
                    writer.writerow([
                        diagrama_esperado_id, modelo, diagram_type, clave,
                        f"{score.precision:.3f}",
                        f"{score.recall:.3f}",
                        f"{score.fmeasure:.3f}"
                    ])
    
    print("✅ Evaluación completada y guardada en 'resultados_rouge_todos_CON_RAG.csv'")

def evaluate_model_rouge(model_name, diagram_type):
    print(f"🔎 Iniciando evaluación ROUGE para el modelo '{model_name}'...")

    resultados_csv = f"resultados_rouge_{model_name}.csv"
    encabezado = ["Historia", "Modelo", "Tipo Diagrama", "ROUGE Tipo", "Precisión", "Recall", "Medida F1"]
    
    # Crear archivo si no existe
    archivo_existe = os.path.exists(resultados_csv)
    with open(resultados_csv, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not archivo_existe:
            writer.writerow(encabezado)
        
        for i in range(1, 16):  # historias del 1 al 15
            historia_id = f"HU{i:02}"
            archivo_esperado = os.path.join("data", "diagramas_esperados", f"PlantUML-{historia_id}.txt")

            if not os.path.exists(archivo_esperado):
                print(f"⚠️  No se encontró el archivo esperado: {archivo_esperado}")
                continue

            with open(archivo_esperado, "r", encoding="utf-8") as f:
                referencia = f.read()

            archivo_generado = os.path.join("data", "diagramas_generados", model_name, diagram_type, f"{historia_id}_{model_name}_{diagram_type}.txt")

            if not os.path.exists(archivo_generado):
                print(f"⚠️  No se encontró el archivo generado: {archivo_generado}")
                continue

            with open(archivo_generado, "r", encoding="utf-8") as f:
                generado = f.read()

            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(referencia, generado)

            for clave, score in scores.items():
                writer.writerow([
                    historia_id, model_name, diagram_type, clave,
                    f"{score.precision:.3f}",
                    f"{score.recall:.3f}",
                    f"{score.fmeasure:.3f}"
                ])
    
    print(f"✅ Evaluación completada para el modelo '{model_name}'. Resultados guardados en '{resultados_csv}'")

def process_all_HUs(model_name, diagram_type):
    contexto_path = f"data/contexto/CONTEXTO_RAG_ORIGINAL.md"
    for i in range(1, 11):  # De HU1 a HU10
        print(f"\n📄 Procesando HU{i}...")
       
        hu_path = f"data/historias_usuario/ingles/HU{i}.pdf"

        if not os.path.exists(contexto_path) or not os.path.exists(hu_path):
            print(f"⚠️  Archivos para HU{i} no encontrados. Se omite.")
            continue

        loaders = [TextLoader(contexto_path, encoding="utf-8"), PyMuPDFLoader(hu_path)]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())

        chunks = process_documents(docs)
        vectordb = build_vectorstore(chunks)
        chain = create_qa_chain(vectordb, model_name)
        ask_question2(chain, model_name, diagram_type, hu_number=i)  # <- Nuevo parámetro

def process_all_HUs_only_HU_prompt(model_name, diagram_type):
    contexto_path = "data/contexto/CONTEXTO_RAG_ORIGINAL.md"

    if not os.path.exists(contexto_path):
        print("⚠️  Archivo de contexto no encontrado. Se aborta la ejecución.")
        return

    # Cargar y procesar solo una vez el contexto
    context_docs = TextLoader(contexto_path, encoding="utf-8").load()
    context_chunks = process_documents(context_docs)
    vectordb = build_vectorstore(context_chunks)

    for i in range(1, 11):  # De HU1 a HU10
        print(f"\n📄 Procesando HU{i}...")

        hu_path = f"data/historias_usuario/ingles/HU{i}.pdf"
        inf_complementaria_path = f"data/informacion_complementaria/INF_ADICIONAL_HU{i}.pdf"

        if not os.path.exists(hu_path):
            print(f"⚠️  Archivo HU{i} no encontrado. Se omite.")
            continue

        # Cargar historia de usuario individualmente
        hu_docs = PyMuPDFLoader(hu_path).load()
        inf_doc = PyMuPDFLoader(inf_complementaria_path).load()

        # Crear la cadena con el vectorstore del contexto original
        chain = create_qa_chain(vectordb, model_name)

        # Pasar la historia como pregunta al modelo (el prompt es la HU)
        ask_question22(chain, model_name, diagram_type, hu_number=i, user_story_docs=hu_docs, inf_compl_doc = inf_doc)

def ask_question22(chain, model_name, diagram_type, hu_number, user_story_docs, inf_compl_doc):
    prompt = generate_prompt(user_story_docs, inf_compl_doc)
    response = chain.invoke({"query": prompt})
    raw_response = response["result"]

    # Limpieza y extracción
    diagram_code = clean_and_extract_plantuml(raw_response)

    if not diagram_code:
        print("❌ Error: El modelo no generó un bloque válido de PlantUML.")
        return

    print(f"\n✅ Respuesta recibida del LLM ({model_name}):")
    print(diagram_code)

    print("\n💾 Guardando el diagrama en la estructura de carpetas...")
    save_diagram2(diagram_code, model_name, diagram_type, hu_number)

def clean_and_extract_plantuml(salida_modelo: str) -> str:
    """
    Elimina bloques <think>...</think> y extrae solo el código entre @startuml y @enduml.
    """
    # 1. Eliminar cualquier bloque <think>...</think>
    sin_think = re.sub(r"<think>.*?</think>", "", salida_modelo, flags=re.DOTALL)

    # 2. Extraer bloque PlantUML
    patron = re.compile(r"@startuml.*?@enduml", re.DOTALL)
    coincidencias = patron.findall(sin_think)

    if coincidencias:
        return coincidencias[0].strip()
    else:
        return None

def ask_question2(chain, model_name, diagram_type, hu_number):
    prompt = generate_prompt()
    response = chain.invoke({"query": prompt})
    raw_response = response["result"]

    # Limpieza y extracción
    diagram_code = clean_and_extract_plantuml(raw_response)

    if not diagram_code:
        print("❌ Error: El modelo no generó un bloque válido de PlantUML.")
        return

    print(f"\n✅ Respuesta recibida del LLM ({model_name}):")
    print(diagram_code)

    print("\n💾 Guardando el diagrama en la estructura de carpetas...")
    save_diagram2(diagram_code, model_name, diagram_type, hu_number)

def save_diagram2(diagram_code, model_name, diagram_kind, hu_number):
    base_dir = os.path.join("data", "diagramas_generados", model_name, diagram_kind)
    os.makedirs(base_dir, exist_ok=True)

    filename = f"HU{hu_number:02}_{model_name}_{diagram_kind}.puml"
    file_path = os.path.join(base_dir, filename)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(diagram_code)

    eval_file = f"HU{hu_number:02}_{model_name}_{diagram_kind}.txt"
    eval_path = os.path.join(base_dir, eval_file)
    with open(eval_path, "w", encoding="utf-8") as file:
        file.write(diagram_code)

    render_diagram(file_path)

# 10. Main
def main():
    
    print("Cargando documentos...")
    docs = load_documents()
    print("Procesando documentos...")
    chunks = process_documents(docs)
    print("Construyendo base de datos FAISS...")
    vectordb = build_vectorstore(chunks)

    model_name = choose_model()
    
    print(f"Usando los modelos: {model_name} y el tipo de diagrama de clases:")

    # Si es una lista (evaluación múltiple), se crean las cadenas para cada modelo
    if isinstance(model_name, list):
        for model in model_name:
            print(f"\n🔄 Creando QA Chain para el modelo **{model}**...")
            chain = create_qa_chain(vectordb, model)
            ask_question(chain, model)
        # Evaluación para todos los modelos generados
        evaluate_rouge(model_name, "clases")
    else:
        # Flujo normal para un solo modelo
        chain = create_qa_chain(vectordb, model_name)
        ask_question(chain, model_name)
        evaluate_rouge(model_name, "clases")

    print("\n🚀 Proceso completado para todos los modelos seleccionados.")

def main2():
    print("\nSelecciona el modelo LLM a utilizar:")
    opciones_mostrar = ["llama", "deepseek", "gemini", "claude", "gpt", "mistral", "Todos"]
    opciones_internas = {
        "llama": "llama",
        "deepseek": "deepseek",
        "gemini": "gemini",
        "claude": "claude",
        "gpt": "gpt",
        "mistral": "mistral"
    }

    for i, modelo in enumerate(opciones_mostrar, 1):
        print(f"{i}. {modelo}")
    
    choice_model = int(input("Modelo: "))
    selected_model = opciones_mostrar[choice_model - 1]

    print("\nSelecciona el tipo de diagrama:")
    tipos_diagrama = ["clases", "actividades"]
    for i, tipo in enumerate(tipos_diagrama, 1):
        print(f"{i}. {tipo}")
    
    choice_diagram = int(input("Diagrama: "))
    diagram_type = tipos_diagrama[choice_diagram - 1]

    if selected_model == "Todos":
        for modelo in opciones_internas.values():
            process_all_HUs_only_HU_prompt(modelo, diagram_type)
        evaluate_all_models_rouge(opciones_internas.values(), diagram_type)
    else:
        model_key = opciones_internas[selected_model]
        process_all_HUs(model_key, diagram_type)
        evaluate_model_rouge(model_key, diagram_type)
    
def main_evaluacion():
    opciones_internas = {
        "llama": "llama",
        "deepseek": "deepseek",
        "gemini": "gemini",
        "claude": "claude",
        "gpt": "gpt",
        "mistral": "mistral"
    }

    diagram_type = "clases"

    evaluate_all_models_rouge(opciones_internas.values(), diagram_type)

if __name__ == "__main__":
    #main()
    main2()
    #main_evaluacion()
