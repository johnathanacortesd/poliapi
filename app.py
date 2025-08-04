# app.py
import streamlit as st
import pandas as pd
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, NamedStyle
from copy import deepcopy
import datetime
import io
import openai
import re
import time
from unidecode import unidecode
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# --- Configuración de la página, Modelos y Constantes ---
st.set_page_config(page_title="Análisis de Noticias de la Policía", layout="wide")
OPENAI_MODEL_EMBEDDING = 'text-embedding-3-small'
OPENAI_MODEL_CLASIFICACION = "gpt-4.1-nano-2025-04-14" # Usamos gpt-4o-mini por su excelente relación coste/rendimiento/velocidad para esta tarea
MARCA_ANALIZAR = "Policía Nacional de Colombia"
TEMAS_FIJOS = {
    "Entorno": "Menciones referenciales o incidentales a la policía, donde no es el actor principal.",
    "Gestión": "Actividades administrativas, campañas de prevención, jornadas de acompañamiento, consejos de seguridad, deportaciones, gestión de líneas de emergencia, trata de personas.",
    "Operativos": "Operativos policiales, capturas, incautaciones, desarticulación de bandas, rescates.",
    "Institucional": "Nombramientos, cambios de mando, presupuesto, discursos, polémicas internas, hechos de corrupción o irregularidades.",
    "Delitos": "Noticias sobre crímenes (homicidios, robos, etc.) donde la policía actúa como fuente de información o llega después del hecho, no en un operativo en curso.",
    "Centros de reclusión": "Noticias específicamente sobre la situación en CAI, estaciones de policía o centros de detención transitoria."
}

# ==============================================================================
# SECCIÓN DE AUTENTICACIÓN
# ==============================================================================
def check_password():
    """Verifica la contraseña almacenada en los secrets de Streamlit."""
    if st.session_state.get("password_correct", False):
        return True
    st.header("🔐 Acceso Protegido")
    with st.form("password_form"):
        password = st.text_input("Ingresa la contraseña para continuar:", type="password")
        submitted = st.form_submit_button("Ingresar")
        if submitted:
            # Compara la contraseña con el secret de Streamlit
            if password == st.secrets.get("APP_PASSWORD", "INVALID_DEFAULT"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("La contraseña es incorrecta.")
    return False

# ==============================================================================
# CLASE DE ANÁLISIS IA (Tono, Tema y Subtema)
# ==============================================================================
class AnalizadorContenidoIA:
    def __init__(self, marca, temas_fijos):
        self.marca = marca
        self.temas_fijos_prompt = "\n".join([f"- {k}: {v}" for k, v in temas_fijos.items()])
        self.cache = {}
        self.high_similarity_threshold = 0.95

    def _limpiar(self, t):
        """Limpia y normaliza el texto."""
        return re.sub(r'\s+', ' ', unidecode(str(t)).strip()) if pd.notna(t) else ""

    def _get_embedding(self, texto_corto):
        """Obtiene el embedding de un texto, usando caché."""
        if not texto_corto: return None
        if texto_corto in self.cache and 'embedding' in self.cache[texto_corto]:
            return self.cache[texto_corto]['embedding']
        try:
            time.sleep(0.02)
            response = openai.embeddings.create(input=[texto_corto], model=OPENAI_MODEL_EMBEDDING)
            embedding = response.data[0].embedding
            if texto_corto not in self.cache: self.cache[texto_corto] = {}
            self.cache[texto_corto]['embedding'] = embedding
            return embedding
        except Exception as e:
            st.warning(f"API Embedding Warn: {e}")
            return None

    def _call_gpt(self, prompt, max_tokens, temperature=0.1):
        """Función genérica para llamar a la API de ChatCompletion."""
        try:
            time.sleep(0.05)
            response = openai.chat.completions.create(
                model=OPENAI_MODEL_CLASIFICACION,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.warning(f"API Call Warn: {str(e)[:100]}...")
            return None

    def _analizar_contenido(self, texto_corto):
        """Obtiene Tono, Tema y Subtema para un texto, usando caché."""
        if not texto_corto: return "Neutro", "Entorno", "Sin Información"
        if texto_corto in self.cache and 'analisis' in self.cache[texto_corto]:
            return self.cache[texto_corto]['analisis']

        # 1. Clasificar Tono
        prompt_tono = f"""Eres un analista de medios experto en la reputación de la "{self.marca}". Clasifica el TONO de la siguiente noticia.
        - POSITIVO: Exalta labores, gestiones, campañas, operativos exitosos y acciones positivas.
        - NEGATIVO: Críticas, corrupción, irregularidades, ataques contra la policía o sus miembros.
        - NEUTRO: Menciones referenciales, incidentales o informativas sin juicio de valor.
        NOTICIA: --- {texto_corto} ---
        Responde ÚNICAMENTE con una de las siguientes tres palabras: POSITIVO, NEGATIVO o NEUTRO."""
        tono_raw = self._call_gpt(prompt_tono, 5, 0.0)
        if tono_raw and 'POSITIVO' in tono_raw.upper(): tono = 'Positivo'
        elif tono_raw and 'NEGATIVO' in tono_raw.upper(): tono = 'Negativo'
        else: tono = 'Neutro'

        # 2. Clasificar Tema Fijo
        prompt_tema = f"""Clasifica la siguiente noticia en UNO de los temas definidos. Responde solo con la palabra del tema.
        TEMAS:
        {self.temas_fijos_prompt}
        NOTICIA: --- {texto_corto} ---
        TEMA:"""
        tema_raw = self._call_gpt(prompt_tema, 10, 0.0)
        tema = next((t for t in TEMAS_FIJOS if t.lower() in (tema_raw or "").lower()), "Entorno")

        # 3. Generar Subtema
        prompt_subtema = f"""Analiza esta noticia y genera un SUBTEMA que capture su esencia.
        REGLAS: Máximo 4 palabras, frase nominal coherente (ej: "Captura líder de banda"), NO terminar con 'de', 'la', 'en'.
        NOTICIA: --- {texto_corto} ---
        SUBTEMA:"""
        subtema = self._call_gpt(prompt_subtema, 15, 0.2) or "Análisis Específico"
        subtema = subtema.replace('"', '').replace('.', '')

        resultado = (tono, tema, subtema)
        if texto_corto not in self.cache: self.cache[texto_corto] = {}
        self.cache[texto_corto]['analisis'] = resultado
        return resultado

    def procesar_lote(self, df_columna_resumen, progress_bar):
        """Procesa un lote de resúmenes, agrupando similares para eficiencia."""
        resumenes_cortos = [self._limpiar(r)[:800] for r in df_columna_resumen] # Truncar para ser más eficiente
        n = len(resumenes_cortos)
        tonos = [""] * n; temas = [""] * n; subtemas = [""] * n

        embeddings = [self._get_embedding(r) for r in resumenes_cortos]
        valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
        
        if not valid_indices: # Si no hay embeddings, procesar individualmente
             for i, r in enumerate(resumenes_cortos):
                tonos[i], temas[i], subtemas[i] = self._analizar_contenido(r)
                progress_bar.progress((i + 1) / n, text=f"Analizando Contenido: {i+1}/{n}")
             return tonos, temas, subtemas

        # Agrupar por similitud para reducir llamadas a la API
        emb_matrix = np.array([embeddings[i] for i in valid_indices])
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - self.high_similarity_threshold, metric='cosine', linkage='complete').fit(emb_matrix)
        
        df_cluster = pd.DataFrame({'resumen_corto': [resumenes_cortos[i] for i in valid_indices], 'cluster_id': clustering.labels_}, index=valid_indices)

        processed_count = 0
        for cluster_id, group in df_cluster.groupby('cluster_id'):
            # Usar el resumen más largo del cluster como representante
            representante_idx = group['resumen_corto'].str.len().idxmax()
            texto_representante = df_cluster.loc[representante_idx, 'resumen_corto']
            
            # Analizar el representante una sola vez
            tono_grupo, tema_grupo, subtema_grupo = self._analizar_contenido(texto_representante)
            
            # Asignar el mismo resultado a todo el grupo
            for idx in group.index:
                tonos[idx], temas[idx], subtemas[idx] = tono_grupo, tema_grupo, subtema_grupo
            
            processed_count += len(group)
            progress_bar.progress(processed_count / n, text=f"Analizando clusters: {processed_count}/{n}")

        # Rellenar los que no obtuvieron embedding (si los hay)
        for i in range(n):
            if not tonos[i]:
                tonos[i], temas[i], subtemas[i] = self._analizar_contenido(resumenes_cortos[i])

        return tonos, temas, subtemas

# ==============================================================================
# FUNCIONES DE PROCESAMIENTO DE EXCEL
# ==============================================================================
def norm_key(text):
    """Normaliza un texto para usarlo como clave."""
    return re.sub(r'\W+', '', str(text).lower().strip()) if text else ""

def clean_title_for_output(title):
    """Limpia el título eliminando partes no deseadas."""
    return re.sub(r'\s*\|\s*[\w\s]+$', '', str(title)).strip()

def corregir_texto(text):
    """Corrige y formatea el texto del resumen."""
    if not isinstance(text, str): return text
    text = re.sub(r'(<br>|\[\.\.\.\]|\s+)', ' ', text).strip()
    match = re.search(r'[A-Z]', text)
    if match: text = text[match.start():]
    if text and not text.endswith('...'): text = text.rstrip('.') + '...'
    return text

def are_duplicates(row1, row2, key_map, title_similarity_threshold=0.85):
    """Determina si dos filas son duplicadas basadas en reglas específicas."""
    if norm_key(row1.get(key_map['medio'])) != norm_key(row2.get(key_map['medio'])):
        return False

    titulo1 = normalize_title_for_comparison(row1.get(key_map['titulo']))
    titulo2 = normalize_title_for_comparison(row2.get(key_map['titulo']))
    
    # Si los títulos son idénticos y no están vacíos, es un duplicado fuerte
    if titulo1 == titulo2 and titulo1 != "":
        return True
    
    # Si no, comprobar similitud del título
    if SequenceMatcher(None, titulo1, titulo2).ratio() >= title_similarity_threshold:
        return True

    return False

def normalize_title_for_comparison(title):
    if not isinstance(title, str): return ""
    # Elimina todo después de un "|" y normaliza para comparación
    cleaned_title = re.sub(r'\s*\|\s*.+$', '', title).strip()
    return re.sub(r'\W+', ' ', cleaned_title).lower().strip()

def run_dossier_logic(sheet):
    """Lógica principal para leer, limpiar y deduplicar el archivo de dossier."""
    headers = [cell.value for cell in sheet[1] if cell.value]
    norm_keys = [norm_key(h) for h in headers]
    
    # Mapeo de columnas esperadas
    key_map = {nk: nk for nk in norm_keys}
    key_map.update({
        'titulo': norm_key('Título'), 'resumen': norm_key('Resumen - Aclaracion'),
        'tipo_medio': norm_key('Tipo de Medio'), 'medio': norm_key('Medio'),
        'fecha': norm_key('Fecha'), 'menciones': norm_key('Menciones - Empresa'),
        'tono': norm_key('Tono'), 'tonoai': norm_key('Tono AI'), 
        'temaai': norm_key('Tema AI'), 'subtemaai': norm_key('Subtema AI')
    })
    
    processed_rows = []
    for row_idx, row in enumerate(sheet.iter_rows(min_row=2, values_only=True)):
        if all(c is None for c in row): continue
        base_data = {norm_keys[i]: cell for i, cell in enumerate(row) if i < len(norm_keys)}
        
        # Separar por menciones si es necesario
        menciones = [m.strip() for m in str(base_data.get(key_map['menciones']) or '').split(';') if m.strip()]
        if not menciones:
            processed_rows.append(base_data)
        else:
            for mencion in menciones:
                new_row = deepcopy(base_data)
                new_row[key_map['menciones']] = mencion
                processed_rows.append(new_row)

    for idx, row in enumerate(processed_rows):
        row.update({'original_index': idx, 'is_duplicate': False})

    # Deduplicación
    for i in range(len(processed_rows)):
        if processed_rows[i]['is_duplicate']: continue
        for j in range(i + 1, len(processed_rows)):
            if processed_rows[j]['is_duplicate']: continue
            if are_duplicates(processed_rows[i], processed_rows[j], key_map):
                processed_rows[j]['is_duplicate'] = True

    for row in processed_rows:
        if row['is_duplicate']:
            row[key_map['tono']] = "Duplicada"
            row[key_map['tonoai']] = "-"
            row[key_map['temaai']] = "-"
            row[key_map['subtemaai']] = "-"
            
    return processed_rows, key_map

def generate_output_excel(all_processed_rows, key_map):
    """Genera el archivo Excel de salida con los datos procesados y analizados."""
    out_wb = Workbook()
    out_sheet = out_wb.active
    out_sheet.title = "Resultado Analisis IA"

    final_order = [
        "ID Noticia", "Fecha", "Hora", "Medio", "Tipo de Medio", "Sección - Programa",
        "Título", "Autor - Conductor", "Nro. Pagina", "Dimensión",
        "Duración - Nro. Caracteres", "CPE", "Tier", "Audiencia",
        "Tono", "Tono AI", "Tema AI", "Subtema AI", # Nuevas columnas de IA
        "Resumen - Aclaracion", "Link Nota",
        "Link (Streaming - Imagen)", "Menciones - Empresa"
    ]
    out_sheet.append(final_order)

    for row_data in all_processed_rows:
        # Limpieza final antes de escribir
        if not row_data.get('is_duplicate'):
            row_data[key_map['titulo']] = clean_title_for_output(row_data.get(key_map['titulo']))
        row_data[key_map['resumen']] = corregir_texto(row_data.get(key_map['resumen']))
        
        row_to_append = []
        for header in final_order:
            key = norm_key(header)
            val = row_data.get(key, "") # Usar "" como default
            row_to_append.append(val)
        out_sheet.append(row_to_append)

    output = io.BytesIO()
    out_wb.save(output)
    output.seek(0)
    return output.getvalue()

# ==============================================================================
# LÓGICA PRINCIPAL DE LA APLICACIÓN
# ==============================================================================
def run_full_process(dossier_file):
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        st.error("Error: 'OPENAI_API_KEY' no encontrado en los Secrets de Streamlit.")
        st.stop()

    with st.status("Paso 1/3: Limpiando y deduplicando noticias...", expanded=True) as status:
        wb = load_workbook(dossier_file, data_only=True)
        all_processed_rows, key_map = run_dossier_logic(wb.active)
        num_duplicates = sum(1 for r in all_processed_rows if r.get('is_duplicate'))
        to_analyze_count = len(all_processed_rows) - num_duplicates
        status.update(label=f"✅ Limpieza completada. {to_analyze_count} noticias únicas de {len(all_processed_rows)} filas.", state="complete")

    rows_to_analyze = [row for row in all_processed_rows if not row.get('is_duplicate')]
    
    if rows_to_analyze:
        for row in rows_to_analyze:
            # Crear resumen corto para la API
            titulo = str(row.get(key_map.get('titulo', ''), ''))
            resumen = str(row.get(key_map.get('resumen', ''), ''))
            row['resumen_api'] = f"{titulo}. {resumen[:150]}"

        df_temp_api = pd.DataFrame(rows_to_analyze)
        
        with st.status(f"Paso 2/3: Analizando Tono, Tema y Subtema para {len(rows_to_analyze)} noticias...", expanded=True) as status:
            analizador = AnalizadorContenidoIA(MARCA_ANALIZAR, TEMAS_FIJOS)
            progress_bar_analisis = st.progress(0, "Iniciando análisis de contenido...")
            
            tonos, temas, subtemas = analizador.procesar_lote(df_temp_api['resumen_api'], progress_bar_analisis)
            
            df_temp_api[key_map['tonoai']] = tonos
            df_temp_api[key_map['temaai']] = temas
            df_temp_api[key_map['subtemaai']] = subtemas
            
            status.update(label="✅ Análisis de contenido completado.", state="complete")
        
        # Mapear los resultados de vuelta a la lista original
        results_map = df_temp_api.set_index('original_index')[[key_map['tonoai'], key_map['temaai'], key_map['subtemaai']]].to_dict('index')
        for row in all_processed_rows:
            if not row.get('is_duplicate'):
                result = results_map.get(row['original_index'])
                if result:
                    row[key_map['tonoai']] = result[key_map['tonoai']]
                    row[key_map['temaai']] = result[key_map['temaai']]
                    row[key_map['subtemaai']] = result[key_map['subtemaai']]

    with st.status("Paso 3/3: Generando archivo de salida...", expanded=True) as status:
        st.session_state['output_data'] = generate_output_excel(all_processed_rows, key_map)
        st.session_state['output_filename'] = f"Informe_IA_Policia_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        st.session_state['processing_complete'] = True
        status.update(label="✅ Archivo de salida listo para descargar.", state="complete")

# ==============================================================================
# ESTRUCTURA DE LA APP STREAMLIT
# ==============================================================================
if check_password():
    st.title("🤖 Análisis IA de Noticias sobre la Policía de Colombia")
    st.markdown("Esta herramienta **limpia, deduplica y analiza noticias** usando la API de OpenAI para clasificar el **Tono**, un **Tema** fijo y un **Subtema** dinámico.")

    if not st.session_state.get('processing_complete', False):
        with st.form("input_form"):
            st.header("Carga de Archivo")
            dossier_file = st.file_uploader("Carga el archivo de Dossier (formato .xlsx)", type=["xlsx"])
            
            submitted = st.form_submit_button("🚀 Iniciar Análisis Completo")
            
            if submitted:
                if not dossier_file:
                    st.warning("Por favor, carga el archivo de dossier para continuar.")
                else:
                    # Limpiar estado de sesión anterior antes de empezar un nuevo proceso
                    password_correct = st.session_state.get("password_correct", False)
                    st.session_state.clear()
                    st.session_state["password_correct"] = password_correct
                    
                    run_full_process(dossier_file)
                    st.rerun()
    else:
        st.success("🎉 ¡Proceso finalizado con éxito!")
        st.info("El informe con los datos procesados y analizados por la IA está listo para ser descargado.")
        st.download_button(
            label="📥 Descargar Informe Completo",
            data=st.session_state['output_data'],
            file_name=st.session_state['output_filename'],
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        if st.button("🧹 Empezar un Nuevo Análisis"):
            # Limpiar estado para un nuevo ciclo
            password_correct = st.session_state.get("password_correct", False)
            st.session_state.clear()
            st.session_state["password_correct"] = password_correct
            st.rerun()
