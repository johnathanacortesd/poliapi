# app.py
import streamlit as st
import pandas as pd
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, NamedStyle
from openpyxl.utils import get_column_letter
from copy import deepcopy
import datetime
import io
import openai
import re
import time
from unidecode import unidecode
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from difflib import SequenceMatcher

# --- Configuración de la página, Modelos y Constantes ---
st.set_page_config(page_title="Análisis de Noticias de la Policía", layout="wide")
OPENAI_MODEL_EMBEDDING = 'text-embedding-3-small'
OPENAI_MODEL_CLASIFICACION = "gpt-4.1-nano-2025-04-14" # <-- MODELO RESTAURADO
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
    if st.session_state.get("password_correct", False): return True
    st.header("🔐 Acceso Protegido")
    with st.form("password_form"):
        password = st.text_input("Ingresa la contraseña para continuar:", type="password")
        submitted = st.form_submit_button("Ingresar")
        if submitted:
            if password == st.secrets.get("APP_PASSWORD", "INVALID_DEFAULT"):
                st.session_state["password_correct"] = True; st.rerun()
            else: st.error("La contraseña es incorrecta.")
    return False

# ==============================================================================
# CLASE DE ANÁLISIS IA - VERSIÓN COMPATIBLE
# ==============================================================================
class AnalizadorContenidoIA:
    def __init__(self, marca, temas_fijos):
        self.marca = marca
        self.temas_fijos_prompt = "\n".join([f"- {k}: {v}" for k, v in temas_fijos.items()])
        self.cache = {}; self.high_similarity_threshold = 0.95
    def _limpiar(self, t): return re.sub(r'\s+', ' ', unidecode(str(t)).strip()) if pd.notna(t) else ""
    def _get_embedding(self, texto_corto):
        if not texto_corto: return None
        if texto_corto in self.cache and 'embedding' in self.cache[texto_corto]: return self.cache[texto_corto]['embedding']
        try:
            time.sleep(0.02); response = openai.Embedding.create(input=[texto_corto], model=OPENAI_MODEL_EMBEDDING)
            embedding = response['data'][0]['embedding']
            if texto_corto not in self.cache: self.cache[texto_corto] = {}
            self.cache[texto_corto]['embedding'] = embedding; return embedding
        except Exception as e: st.warning(f"API Embedding Warn: {e}"); return None
    def _call_gpt(self, prompt, max_tokens, temperature=0.1):
        try:
            time.sleep(0.05)
            response = openai.ChatCompletion.create(model=OPENAI_MODEL_CLASIFICACION, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=temperature)
            return response.choices[0].message['content'].strip()
        except Exception as e: st.warning(f"API Call Warn: {str(e)[:100]}..."); return None
    def _analizar_contenido(self, texto_corto):
        if not texto_corto: return "Neutro (Error)", "Entorno (Error)", "Sin Información"
        if texto_corto in self.cache and 'analisis' in self.cache[texto_corto]: return self.cache[texto_corto]['analisis']
        prompt_tono = f"""Eres un analista de medios experto en la reputación de la "{self.marca}". Clasifica el TONO de la siguiente noticia.
        - POSITIVO: Exalta labores, gestiones, campañas, operativos exitosos y acciones positivas.
        - NEGATIVO: Críticas, corrupción, irregularidades, ataques contra la policía o sus miembros.
        - NEUTRO: Menciones referenciales, incidentales o informativas sin juicio de valor.
        NOTICIA: --- {texto_corto} ---
        Responde ÚNICAMENTE con una de las siguientes tres palabras: POSITIVO, NEGATIVO o NEUTRO."""
        tono_raw = self._call_gpt(prompt_tono, 5, 0.0)
        if tono_raw and 'POSITIVO' in tono_raw.upper(): tono = 'Positivo'
        elif tono_raw and 'NEGATIVO' in tono_raw.upper(): tono = 'Negativo'
        else: tono = 'Neutro' if tono_raw else "Neutro (Error API)"
        prompt_tema = f"""Clasifica la noticia en UNO de los temas. Responde solo con la palabra del tema. TEMAS:\n{self.temas_fijos_prompt}\nNOTICIA: --- {texto_corto} ---\nTEMA:"""
        tema_raw = self._call_gpt(prompt_tema, 10, 0.0)
        tema = next((t for t in TEMAS_FIJOS if t.lower() in (tema_raw or "").lower()), "Entorno (Error API)")
        prompt_subtema = f"""Genera un SUBTEMA para esta noticia. REGLAS: Máximo 4 palabras, frase nominal (ej: "Captura líder de banda"), NO terminar con 'de', 'la', 'en'.\nNOTICIA: --- {texto_corto} ---\nSUBTEMA:"""
        subtema = self._call_gpt(prompt_subtema, 15, 0.2) or "Análisis Específico (Error API)"
        subtema = subtema.replace('"', '').replace('.', '')
        resultado = (tono, tema, subtema)
        if texto_corto not in self.cache: self.cache[texto_corto] = {}
        self.cache[texto_corto]['analisis'] = resultado; return resultado
    def procesar_lote(self, df_columna_resumen, progress_bar):
        resumenes_cortos = [self._limpiar(r)[:800] for r in df_columna_resumen]; n = len(resumenes_cortos)
        tonos, temas, subtemas = [""] * n, [""] * n, [""] * n
        embeddings = [self._get_embedding(r) for r in resumenes_cortos]
        valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
        if not valid_indices:
             for i, r in enumerate(resumenes_cortos):
                tonos[i], temas[i], subtemas[i] = self._analizar_contenido(r)
                progress_bar.progress((i + 1) / n, text=f"Analizando Contenido: {i+1}/{n}")
             return tonos, temas, subtemas
        emb_matrix = np.array([embeddings[i] for i in valid_indices])
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - self.high_similarity_threshold, metric='cosine', linkage='complete').fit(emb_matrix)
        df_cluster = pd.DataFrame({'resumen_corto': [resumenes_cortos[i] for i in valid_indices], 'cluster_id': clustering.labels_}, index=valid_indices)
        processed_count = 0
        for cluster_id, group in df_cluster.groupby('cluster_id'):
            representante_idx = group['resumen_corto'].str.len().idxmax()
            texto_representante = df_cluster.loc[representante_idx, 'resumen_corto']
            tono_grupo, tema_grupo, subtema_grupo = self._analizar_contenido(texto_representante)
            for idx in group.index: tonos[idx], temas[idx], subtemas[idx] = tono_grupo, tema_grupo, subtema_grupo
            processed_count += len(group); progress_bar.progress(processed_count / n, text=f"Analizando clusters: {processed_count}/{n}")
        for i in range(n):
            if not tonos[i]: tonos[i], temas[i], subtemas[i] = self._analizar_contenido(resumenes_cortos[i])
        return tonos, temas, subtemas

# ==============================================================================
# FUNCIONES DE PROCESAMIENTO DE EXCEL (CON HIPERVÍNCULOS)
# ==============================================================================
def norm_key(text): return re.sub(r'\W+', '', str(text).lower().strip()) if text else ""
def extract_link(cell): # <-- FUNCIÓN DE HIPERVÍNCULOS RESTAURADA
    if hasattr(cell, 'hyperlink') and cell.hyperlink: return {"value": "Link", "url": cell.hyperlink.target}
    if isinstance(cell.value, str):
        match = re.search(r'=HYPERLINK\("([^"]+)"', cell.value); 
        if match: return {"value": "Link", "url": match.group(1)}
    return {"value": cell.value, "url": None}
def clean_title_for_output(title): return re.sub(r'\s*\|\s*[\w\s]+$', '', str(title)).strip()
def corregir_texto(text):
    if not isinstance(text, str): return text
    text = re.sub(r'(<br>|\[\.\.\.\]|\s+)', ' ', text).strip()
    match = re.search(r'[A-Z]', text)
    if match: text = text[match.start():]
    if text and not text.endswith('...'): text = text.rstrip('.') + '...'
    return text
def are_duplicates(row1, row2, key_map, title_similarity_threshold=0.85):
    if norm_key(row1.get(key_map['medio'])) != norm_key(row2.get(key_map['medio'])): return False
    titulo1 = normalize_title_for_comparison(row1.get(key_map['titulo']))
    titulo2 = normalize_title_for_comparison(row2.get(key_map['titulo']))
    if titulo1 == titulo2 and titulo1 != "": return True
    if SequenceMatcher(None, titulo1, titulo2).ratio() >= title_similarity_threshold: return True
    return False
def normalize_title_for_comparison(title):
    if not isinstance(title, str): return ""
    cleaned_title = re.sub(r'\s*\|\s*.+$', '', title).strip()
    return re.sub(r'\W+', ' ', cleaned_title).lower().strip()
def run_dossier_logic(sheet): # <-- LÓGICA DE LECTURA CON HIPERVÍNCULOS RESTAURADA
    headers = [cell.value for cell in sheet[1] if cell.value]
    norm_keys = [norm_key(h) for h in headers]
    key_map = {nk: nk for nk in norm_keys}
    key_map.update({ 'titulo': norm_key('Título'), 'resumen': norm_key('Resumen - Aclaracion'), 'medio': norm_key('Medio'), 'menciones': norm_key('Menciones - Empresa'), 'tono': norm_key('Tono'), 'tonoai': norm_key('Tono AI'), 'temaai': norm_key('Tema AI'), 'subtemaai': norm_key('Subtema AI'), 'link_nota': norm_key('Link Nota'), 'link_streaming': norm_key('Link (Streaming - Imagen)') })
    processed_rows = []
    for row in sheet.iter_rows(min_row=2):
        if all(c.value is None for c in row): continue
        base_data = {norm_keys[i]: extract_link(cell) if norm_keys[i] in [key_map['link_nota'], key_map['link_streaming']] else cell.value for i, cell in enumerate(row) if i < len(norm_keys)}
        menciones = [m.strip() for m in str(base_data.get(key_map['menciones']) or '').split(';') if m.strip()]
        if not menciones: processed_rows.append(base_data)
        else:
            for mencion in menciones: new_row = deepcopy(base_data); new_row[key_map['menciones']] = mencion; processed_rows.append(new_row)
    for idx, row in enumerate(processed_rows): row.update({'original_index': idx, 'is_duplicate': False})
    for i in range(len(processed_rows)):
        if processed_rows[i]['is_duplicate']: continue
        for j in range(i + 1, len(processed_rows)):
            if processed_rows[j]['is_duplicate']: continue
            if are_duplicates(processed_rows[i], processed_rows[j], key_map): processed_rows[j]['is_duplicate'] = True
    for row in processed_rows:
        if row['is_duplicate']: row[key_map['tono']] = "Duplicada"; row[key_map['tonoai']], row[key_map['temaai']], row[key_map['subtemaai']] = "-", "-", "-"
    return processed_rows, key_map
def generate_output_excel(all_processed_rows, key_map): # <-- LÓGICA DE ESCRITURA CON HIPERVÍNCULOS RESTAURADA
    out_wb = Workbook(); out_sheet = out_wb.active; out_sheet.title = "Resultado Analisis IA"
    final_order = [ "ID Noticia", "Fecha", "Hora", "Medio", "Tipo de Medio", "Sección - Programa", "Región", "Título", "Autor - Conductor", "Nro. Pagina", "Dimensión", "Duración - Nro. Caracteres", "CPE", "Tier", "Audiencia", "Tono", "Tono AI", "Tema AI", "Subtema AI", "Resumen - Aclaracion", "Link Nota", "Link (Streaming - Imagen)", "Menciones - Empresa" ]
    out_sheet.append(final_order)
    link_style = NamedStyle(name="Hyperlink_Custom", font=Font(color="0000FF", underline="single"))
    if "Hyperlink_Custom" not in out_wb.named_styles: out_wb.add_named_style(link_style)
    link_nota_idx = final_order.index("Link Nota") + 1
    link_streaming_idx = final_order.index("Link (Streaming - Imagen)") + 1
    for row_data in all_processed_rows:
        if not row_data.get('is_duplicate'): row_data[key_map['titulo']] = clean_title_for_output(row_data.get(key_map['titulo']))
        row_data[key_map.get(norm_key('resumen'),'resumen')] = corregir_texto(row_data.get(key_map.get(norm_key('resumen'),'resumen')))
        row_to_append = []
        for header in final_order:
            val = row_data.get(norm_key(header))
            row_to_append.append(val['value'] if isinstance(val, dict) else val)
        out_sheet.append(row_to_append)
        current_row_idx = out_sheet.max_row
        link_nota_data = row_data.get(key_map['link_nota']); link_streaming_data = row_data.get(key_map['link_streaming'])
        if isinstance(link_nota_data, dict) and link_nota_data.get("url"):
            cell = out_sheet.cell(row=current_row_idx, column=link_nota_idx)
            cell.hyperlink = link_nota_data["url"]; cell.style = "Hyperlink_Custom"
        if isinstance(link_streaming_data, dict) and link_streaming_data.get("url"):
            cell = out_sheet.cell(row=current_row_idx, column=link_streaming_idx)
            cell.hyperlink = link_streaming_data["url"]; cell.style = "Hyperlink_Custom"
    output = io.BytesIO(); out_wb.save(output); output.seek(0); return output.getvalue()

# ==============================================================================
# LÓGICA PRINCIPAL DE LA APLICACIÓN (CON MAPEOS)
# ==============================================================================
def run_full_process(dossier_file, region_file, internet_file):
    try: openai.api_key = st.secrets["OPENAI_API_KEY"]
    except Exception: st.error("Error: 'OPENAI_API_KEY' no encontrado."); st.stop()

    with st.status("Paso 1/5: Limpiando y deduplicando noticias...", expanded=True) as status:
        wb = load_workbook(dossier_file, data_only=False) # data_only=False para leer fórmulas de hipervínculos
        all_processed_rows, key_map = run_dossier_logic(wb.active)
        num_duplicates = sum(1 for r in all_processed_rows if r.get('is_duplicate'))
        to_analyze_count = len(all_processed_rows) - num_duplicates
        status.update(label=f"✅ Limpieza completada. {to_analyze_count} noticias únicas de {len(all_processed_rows)} filas.", state="complete")
    
    with st.status("Paso 2/5: Aplicando mapeos (Región e Internet)...", expanded=True) as status: # <-- PASO DE MAPEOS RESTAURADO
        df_region = pd.read_excel(region_file); region_map = pd.Series(df_region.iloc[:, 1].values, index=df_region.iloc[:, 0].astype(str).str.lower().str.strip()).to_dict()
        df_internet = pd.read_excel(internet_file); internet_map = pd.Series(df_internet.iloc[:, 1].values, index=df_internet.iloc[:, 0].astype(str).str.lower().str.strip()).to_dict()
        for row in all_processed_rows:
            medio_key = str(row.get(key_map['medio'], '')).lower().strip()
            row[norm_key('Región')] = region_map.get(medio_key, "Error Mapeo Región")
            if row.get(norm_key('Tipo de Medio')) == 'Internet':
                row[key_map['medio']] = internet_map.get(medio_key, row.get(key_map['medio']))
        status.update(label="✅ Mapeos aplicados.", state="complete")

    rows_to_analyze = [row for row in all_processed_rows if not row.get('is_duplicate')]
    if rows_to_analyze:
        for row in rows_to_analyze:
            titulo_val = row.get(key_map.get('titulo', ''), '')
            titulo = titulo_val['value'] if isinstance(titulo_val, dict) else titulo_val
            resumen_val = row.get(key_map.get('resumen', ''), '')
            resumen = resumen_val['value'] if isinstance(resumen_val, dict) else resumen_val
            row['resumen_api'] = f"{titulo}. {resumen[:150]}"
        df_temp_api = pd.DataFrame(rows_to_analyze)
        with st.status(f"Paso 3/5: Analizando Tono, Tema y Subtema para {len(rows_to_analyze)} noticias...", expanded=True) as status:
            analizador = AnalizadorContenidoIA(MARCA_ANALIZAR, TEMAS_FIJOS)
            progress_bar_analisis = st.progress(0, "Iniciando análisis de contenido...")
            tonos, temas, subtemas = analizador.procesar_lote(df_temp_api['resumen_api'], progress_bar_analisis)
            df_temp_api[key_map['tonoai']] = tonos; df_temp_api[key_map['temaai']] = temas; df_temp_api[key_map['subtemaai']] = subtemas
            status.update(label="✅ Análisis de contenido completado.", state="complete")
        results_map = df_temp_api.set_index('original_index')[[key_map['tonoai'], key_map['temaai'], key_map['subtemaai']]].to_dict('index')
        for row in all_processed_rows:
            if not row.get('is_duplicate'):
                result = results_map.get(row['original_index'])
                if result: row.update(result)
    with st.status("Paso 4/5: Generando archivo de salida...", expanded=True) as status:
        st.session_state['output_data'] = generate_output_excel(all_processed_rows, key_map)
        st.session_state['output_filename'] = f"Informe_IA_Policia_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        st.session_state['processing_complete'] = True
        status.update(label="✅ Archivo de salida listo para descargar.", state="complete")

# ==============================================================================
# ESTRUCTURA DE LA APP STREAMLIT (CON MAPEOS)
# ==============================================================================
if check_password():
    st.title("🤖 Análisis IA de Noticias sobre la Policía de Colombia")
    st.markdown("Herramienta para limpiar, deduplicar, mapear y analizar noticias (Tono, Tema y Subtema) usando OpenAI.")
    if not st.session_state.get('processing_complete', False):
        with st.form("input_form"):
            st.header("Carga de Archivos")
            dossier_file = st.file_uploader("1. Archivo de Dossier (Principal)", type=["xlsx"])
            region_file = st.file_uploader("2. Archivo de Mapeo de Región", type=["xlsx"])
            internet_file = st.file_uploader("3. Archivo de Mapeo de Internet", type=["xlsx"])
            submitted = st.form_submit_button("🚀 Iniciar Análisis Completo")
            if submitted:
                if not all([dossier_file, region_file, internet_file]):
                    st.warning("Por favor, carga los tres archivos requeridos.")
                else:
                    password_correct = st.session_state.get("password_correct", False); st.session_state.clear()
                    st.session_state["password_correct"] = password_correct
                    run_full_process(dossier_file, region_file, internet_file); st.rerun()
    else:
        st.success("🎉 ¡Proceso finalizado con éxito!")
        st.info("El informe con los datos procesados y analizados por la IA está listo para ser descargado.")
        st.download_button(label="📥 Descargar Informe Completo", data=st.session_state['output_data'], file_name=st.session_state['output_filename'], mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        if st.button("🧹 Empezar un Nuevo Análisis"):
            password_correct = st.session_state.get("password_correct", False); st.session_state.clear()
            st.session_state["password_correct"] = password_correct; st.rerun()
