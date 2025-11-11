import streamlit as st
import openpyxl
import io
import random
import logging
import numpy as np
import pandas as pd
import altair as alt
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes Globales ---
ARCHIVOS_CON_LIMITE = ["1. OQ_MAPEO.xlsm", "4. PQ_RUTA_20.xlsm", "5. PQ_RUTA_80.xlsm"]
HOJAS_A_IGNORAR = ["CONSOLIDADO", "GRAFICOS", "RESUMEN", "TABLA", "RESULTADOS", "SUMMARY", "GRAFICO"] 

# --- CONFIGURACIÓN BASE (Presets) ---
CONFIGURACION_BASE = {
    "OQ Mapeo": {
        "archivo": "1. OQ MAPEO 72 INV.xlsm",        
        "variacion_min": 0.01, "variacion_max": 0.02,
        "amplitud": 0.30, "sigma": 12, "punto_pico": 0.5,
        "offset_min": -0.5, "offset_max": 0.0,
        "prob_limpieza_picos": 0.9
    },
    "OQ Apertura": {
        "archivo": "2. OQ APERTURA 72 INV.xlsm",     
        "variacion_min": 0.03, "variacion_max": 0.05,
        "amplitud": 0.40, "sigma": 8, "punto_pico": 0.6,
        "offset_min": -1.0, "offset_max": -0.2,
        "prob_limpieza_picos": 0.5
    },
    "OQ Apagado": {
        "archivo": "3. OQ APAGADO 72 INV.xlsm",      
        "variacion_min": 0.01, "variacion_max": 0.02,
        "amplitud": 0.50, "sigma": 20, "punto_pico": 0.4,
        "offset_min": -1.2, "offset_max": -0.3,
        "prob_limpieza_picos": 0.9
    },
    "PQ Ruta 20%": {
        "archivo": "4. PQ RUTA 20 72 INV.xlsm",      
        "variacion_min": 0.02, "variacion_max": 0.04,
        "amplitud": 0.35, "sigma": 12, "punto_pico": 0.5,
        "offset_min": -0.9, "offset_max": -0.2,
        "prob_limpieza_picos": 0.7
    },
    "PQ Ruta 80%": {
        "archivo": "5. PQ RUTA 80 72 INV.xlsm",      
        "variacion_min": 0.02, "variacion_max": 0.04,
        "amplitud": 0.35, "sigma": 12, "punto_pico": 0.5,
        "offset_min": -0.9, "offset_max": -0.2,
        "prob_limpieza_picos": 0.7
    },
    "PQ Apertura 20%": {
        "archivo": "6. PQ APERTURA 20 72 INV.xlsm",  
        "variacion_min": 0.03, "variacion_max": 0.05,
        "amplitud": 0.40, "sigma": 8, "punto_pico": 0.6,
        "offset_min": -1.0, "offset_max": -0.2,
        "prob_limpieza_picos": 0.5
    },
    "PQ Apertura 80%": {
        "archivo": "7. PQ APERTURA 80 72 INV.xlsm",  
        "variacion_min": 0.03, "variacion_max": 0.05,
        "amplitud": 0.40, "sigma": 8, "punto_pico": 0.6,
        "offset_min": -1.0, "offset_max": -0.2,
        "prob_limpieza_picos": 0.5
    },
    "PQ Apagado 20%": {
        "archivo": "8. PQ APAGADO 20 72 INV.xlsm",   
        "variacion_min": 0.01, "variacion_max": 0.02,
        "amplitud": 0.50, "sigma": 20, "punto_pico": 0.4,
        "offset_min": -1.2, "offset_max": -0.3,
        "prob_limpieza_picos": 0.9
    },
    "PQ Apagado 80%": {
        "archivo": "9. PQ APAGADO 80 72 INV.xlsm",   
        "variacion_min": 0.01, "variacion_max": 0.02,
        "amplitud": 0.50, "sigma": 20, "punto_pico": 0.4,
        "offset_min": -1.2, "offset_max": -0.3,
        "prob_limpieza_picos": 0.9
    },
}

# --- FUNCIONES DE GENERACIÓN DE CURVAS (El "Motor") ---

@st.cache_data(show_spinner=False)
def generar_deriva_gaussiana(longitud, amplitud, sigma, seed):
    """(PASO 3) Genera una curva de deriva suave (aditiva) única por DL."""
    np.random.seed(seed) # Usar semilla para reproducibilidad
    try:
        ruido_base = np.random.randn(longitud)
        deriva_suave = gaussian_filter1d(ruido_base, sigma=sigma)
        max_abs = np.max(np.abs(deriva_suave))
        if max_abs > 1e-6: deriva_normalizada = deriva_suave / max_abs
        else: deriva_normalizada = np.zeros(longitud)
        deriva_final = deriva_normalizada * amplitud
        fade_len = min(longitud // 10, int(sigma * 3))
        if fade_len > 1:
            fade_in = np.linspace(0, 1, fade_len)
            deriva_final[:fade_len] *= fade_in
            fade_out = np.linspace(1, 0, fade_len)
            deriva_final[-fade_len:] *= fade_out
        return deriva_final
    except Exception: return np.zeros(longitud)

@st.cache_data(show_spinner=False)
def generar_curva_multiplicativa(longitud, variacion_percent, punto_pico_frac):
    """(PASO 2) Genera una curva de multiplicación que vuelve a 1.0."""
    try:
        factor_max = 1.0 + variacion_percent
        punto_pico_idx = int(longitud * punto_pico_frac)
        if punto_pico_idx <= 0: punto_pico_idx = 1
        if punto_pico_idx >= longitud: punto_pico_idx = longitud - 1
        fase_subida = np.linspace(1.0, factor_max, punto_pico_idx)
        fase_bajada = np.linspace(factor_max, 1.0, longitud - punto_pico_idx)
        curva_multi = np.concatenate((fase_subida[:-1], fase_bajada))
        if len(curva_multi) != longitud:
            x_original = np.linspace(0, 1, len(curva_multi))
            x_nuevo = np.linspace(0, 1, longitud)
            curva_multi = np.interp(x_nuevo, x_original, curva_multi)
        return curva_multi
    except Exception: return np.ones(longitud)

@st.cache_data(show_spinner=False)
def aplicar_pipeline_a_columna(datos_np, config_dl, seed):
    """Aplica el pipeline de 4 pasos a una sola columna de datos."""
    longitud_actual = len(datos_np)
    if longitud_actual < 20:
        return datos_np

    # Sellar la aleatoriedad para esta columna específica
    col_seed = seed + hash(config_dl['dl_nombre']) % 1000
    random.seed(col_seed)
    np.random.seed(col_seed)

    # PASO 1: LIMPIEZA DE PICOS (Probabilística)
    if random.random() < config_dl["prob_limpieza_picos"]:
        datos_base = medfilt(datos_np, kernel_size=3)
    else:
        datos_base = datos_np

    # PASO 2: EXTRAPOLACIÓN (Variable por DL)
    curva_multi_dl = generar_curva_multiplicativa(longitud_actual, config_dl["variacion_percent"], config_dl["punto_pico_frac"])
    datos_extrapolados = datos_base * curva_multi_dl
    
    # PASO 3: DERIVA DE REALISMO (Única por DL)
    deriva = generar_deriva_gaussiana(longitud_actual, config_dl["amplitud"], config_dl["sigma"], seed=col_seed + 1)
    datos_con_deriva = datos_extrapolados + deriva
    
    # PASO 4: APLICAR OFFSET BASE (Variable por DL)
    datos_finales = datos_con_deriva + config_dl["offset_base"]
    
    return datos_finales

# --- FUNCIONES DE MANEJO DE DATOS ---

@st.cache_data(show_spinner=False)
def leer_datos_crudos_excel(wb_bytes):
    """Lee TODOS los datos crudos del Excel y los almacena en un dict."""
    datos_crudos = {} # Estructura: { "hoja": { "DL": [datos] } }
    try:
        wb = openpyxl.load_workbook(io.BytesIO(wb_bytes), data_only=True)
        for hoja_nombre in wb.sheetnames:
            if any(ignorar in hoja_nombre.strip().upper() for ignorar in HOJAS_A_IGNORAR):
                continue
            
            ws = wb[hoja_nombre]
            datos_hoja = {}
            for col in ws.iter_cols(min_row=1):
                header_value = col[0].value
                if isinstance(header_value, str) and header_value.strip().upper().startswith("DL"):
                    valores = []
                    for cell in col[1:]:
                        if isinstance(cell.value, (int, float)):
                            valores.append(cell.value)
                    
                    if len(valores) > 20:
                        datos_hoja[header_value.strip()] = np.array(valores)
            
            if datos_hoja:
                datos_crudos[hoja_nombre] = datos_hoja
        return datos_crudos
    except Exception as e:
        st.error(f"Error al leer el archivo Excel: {e}")
        return None

def generar_configuracion_inicial(datos_crudos, config_base, seed_value):
    """Genera el dict de configuración inicial para CADA DL basado en la semilla."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    config_t = {}
    config_hr = {}

    hoja_temp_nombre = st.session_state.sheet_temp
    hoja_hr_nombre = st.session_state.sheet_hr
    
    if hoja_temp_nombre in datos_crudos:
        for dl_nombre in datos_crudos[hoja_temp_nombre].keys():
            config_t[dl_nombre] = {
                "dl_nombre": dl_nombre, # Guardar nombre para hash
                "variacion_percent": random.uniform(config_base["variacion_min"], config_base["variacion_max"]),
                "amplitud": config_base["amplitud"],
                "sigma": config_base["sigma"],
                "punto_pico_frac": config_base["punto_pico"],
                "offset_base": random.uniform(config_base["offset_min"], config_base["offset_max"]),
                "prob_limpieza_picos": config_base["prob_limpieza_picos"]
            }
            
    if hoja_hr_nombre in datos_crudos:
        for dl_nombre in datos_crudos[hoja_hr_nombre].keys():
            config_hr[dl_nombre] = {
                "dl_nombre": dl_nombre,
                "variacion_percent": random.uniform(config_base["variacion_min"], config_base["variacion_max"]),
                "amplitud": config_base["amplitud"],
                "sigma": config_base["sigma"],
                "punto_pico_frac": config_base["punto_pico"],
                "offset_base": random.uniform(config_base["offset_min"], config_base["offset_max"]),
                "prob_limpieza_picos": config_base["prob_limpieza_picos"]
            }
            
    return config_t, config_hr

@st.cache_data(show_spinner=False)
def generar_datos_extrapolados(_datos_crudos_hoja, _config_por_dl, seed_value):
    """Genera un DataFrame extrapolado basado en la configuración de cada DL."""
    datos_extrapolados = {}
    for dl_nombre, datos_originales in _datos_crudos_hoja.items():
        if dl_nombre in _config_por_dl:
            config_dl = _config_por_dl[dl_nombre]
            datos_extrapolados[dl_nombre] = aplicar_pipeline_a_columna(datos_originales, config_dl, seed_value)
    
    return pd.DataFrame(dict([(k,pd.Series(v)) for k,v in datos_extrapolados.items()]))


def dibujar_grafico_con_limites(df, titulo, limite_max=None, limite_min=None):
    """Crea un gráfico Altair con límites opcionales."""
    if df.empty:
        return st.warning(f"No se encontraron datos 'DL' para el gráfico: {titulo}")

    df_largo = df.reset_index().melt('index', var_name='Sensor', value_name='Valor')
    
    base = alt.Chart(df_largo).encode(
        x=alt.X('index', title='Índice de Tiempo'),
        y=alt.Y('Valor', title=titulo),
        color=alt.Color('Sensor', title="Sensores"),
        tooltip=['index', 'Sensor', 'Valor']
    ).properties(
        title=titulo
    )
    lineas = base.mark_line(point=False).interactive()
    grafico_final = lineas
    
    if limite_max is not None:
        linea_max = alt.Chart(pd.DataFrame({'y': [limite_max]})) \
            .mark_rule(color='red', strokeDash=[5, 2]) \
            .encode(y='y')
        grafico_final = grafico_final + linea_max
    if limite_min is not None:
        linea_min = alt.Chart(pd.DataFrame({'y': [limite_min]})) \
            .mark_rule(color='red', strokeDash=[5, 2]) \
            .encode(y='y')
        grafico_final = grafico_final + linea_min
        
    return st.altair_chart(grafico_final, use_container_width=True)


def descargar_excel_modificado(wb_bytes, config_t, config_hr, seed_value, file_name, sheet_temp_name, sheet_hr_name, preset_archivo_base):
    """Función final para procesar y descargar el Excel."""
    with st.spinner("Generando archivo Excel completo... Esto puede tardar unos segundos."):
        try:
            random.seed(seed_value)
            np.random.seed(seed_value)
            
            wb = openpyxl.load_workbook(io.BytesIO(wb_bytes), keep_vba=True)
            
            aplicar_limite = any(nombre in preset_archivo_base for nombre in ARCHIVOS_CON_LIMITE)

            for hoja_nombre in wb.sheetnames:
                if any(ignorar in hoja_nombre.strip().upper() for ignorar in HOJAS_A_IGNORAR):
                    continue
                
                config_a_usar = None
                if hoja_nombre == sheet_temp_name:
                    config_a_usar = config_t
                elif hoja_nombre == sheet_hr_name:
                    config_a_usar = config_hr
                else:
                    # Intentar adivinar: si no es de HR, usar T° por defecto
                    if "%HR" not in hoja_nombre.upper() and "HUM" not in hoja_nombre.upper():
                         config_a_usar = config_t
                    else:
                         config_a_usar = config_hr
                
                if not config_a_usar:
                    continue 
                    
                ws = wb[hoja_nombre]
                for col in ws.iter_cols(min_row=1):
                    header_cell = col[0]
                    header_value = str(header_cell.value).strip()
                    
                    if header_value in config_a_usar:
                        celdas_datos, valores_originales, tipos_originales = [], [], []
                        for cell in col[1:]:
                            if isinstance(cell.value, (int, float)) and cell.data_type != 'f':
                                celdas_datos.append(cell)
                                valores_originales.append(cell.value)
                                tipos_originales.append(type(cell.value))
                        
                        if len(valores_originales) < 20: continue
                        
                        datos_np = np.array(valores_originales)
                        config_dl = config_a_usar[header_value]
                        
                        # Aplicar pipeline final
                        datos_finales = aplicar_pipeline_a_columna(datos_np, config_dl, seed_value)
                        
                        if aplicar_limite and (hoja_nombre == sheet_temp_name or "T°" in hoja_nombre):
                             np.clip(datos_finales, a_min=None, a_max=25.5, out=datos_finales)

                        for i, cell in enumerate(celdas_datos):
                            if i < len(datos_finales):
                                nuevo_valor = datos_finales[i]
                                tipo_original = tipos_originales[i]
                                if tipo_original == int: cell.value = int(round(nuevo_valor))
                                else:
                                    try: decimales = len(str(valores_originales[i]).split('.')[1])
                                    except: decimales = 2
                                    cell.value = round(nuevo_valor, decimales)

            with io.BytesIO() as f:
                wb.save(f)
                return f.getvalue()

        except Exception as e:
            st.error(f"Error al generar el archivo: {e}")
            logger.error(f"Error en descarga: {e}", exc_info=True)
            return None

# --- INTERFAZ DE STREAMLIT ---
st.set_page_config(layout="wide", page_title="Extrapolador Maestro V17")
st.title("Extrapolador Maestro V17 (Editor Híbrido) 🚀")
st.info("Genera una extrapolación base y luego ajusta cada curva individualmente en tiempo real.")

# --- BARRA LATERAL (CONTROLES GLOBALES) ---
st.sidebar.header("1. Carga de Archivo")
uploaded_file = st.sidebar.file_uploader("Cargar archivo .xlsm", type=["xlsm"])

# --- LÓGICA PRINCIPAL ---
if uploaded_file is not None:
    
    if st.session_state.get('file_name') != uploaded_file.name:
        st.session_state.datos_crudos = leer_datos_crudos_excel(uploaded_file.getvalue())
        st.session_state.file_name = uploaded_file.name
        st.session_state.original_file_bytes = uploaded_file.getvalue()
        if 'config_t' in st.session_state: del st.session_state.config_t
        if 'config_hr' in st.session_state: del st.session_state.config_hr

    if not st.session_state.datos_crudos:
        st.error("No se pudieron leer datos 'DL' válidos de este archivo. Revise el formato.")
    else:
        try:
            available_sheets = list(st.session_state.datos_crudos.keys())
            st.sidebar.header("2. Selección de Hojas")
            default_temp = next((i for i, s in enumerate(available_sheets) if "T°" in s or "TEMP" in s.upper()), 0)
            default_hr = next((i for i, s in enumerate(available_sheets) if "%HR" in s or "HR" in s.upper()), 1 if len(available_sheets) > 1 else 0)
            
            st.session_state.sheet_temp = st.sidebar.selectbox("Hoja de Temperatura (T°)", available_sheets, index=default_temp)
            st.session_state.sheet_hr = st.sidebar.selectbox("Hoja de Humedad (%HR)", available_sheets, index=default_hr)

            st.sidebar.header("3. Generación Inicial")
            seed_value = st.sidebar.number_input("Versión (Semilla Aleatoria)", value=1, min_value=1, step=1)
            preset_name = st.sidebar.selectbox("Seleccionar Preset de Prueba:", options=list(CONFIGURACION_BASE.keys()), key="preset_name")
            
            if st.sidebar.button("Generar Extrapolación Inicial", type="primary"):
                config_base = CONFIGURACION_BASE[preset_name]
                st.session_state.config_t, st.session_state.config_hr = generar_configuracion_inicial(st.session_state.datos_crudos, config_base, seed_value)
                st.success(f"Generada Versión {seed_value} con preset '{preset_name}'")

            # --- EDITOR Y GRÁFICOS (Solo si se ha generado) ---
            if 'config_t' in st.session_state and 'config_hr' in st.session_state:
                
                with st.spinner("Actualizando gráficos en tiempo real..."):
                    df_orig_temp = pd.DataFrame(st.session_state.datos_crudos[st.session_state.sheet_temp])
                    df_orig_hr = pd.DataFrame(st.session_state.datos_crudos[st.session_state.sheet_hr])
                    
                    df_ext_temp = generar_datos_extrapolados(st.session_state.datos_crudos[st.session_state.sheet_temp], st.session_state.config_t, seed_value)
                    df_ext_hr = generar_datos_extrapolados(st.session_state.datos_crudos[st.session_state.sheet_hr], st.session_state.config_hr, seed_value)

                tab_t, tab_hr = st.tabs(["Gráfico de Temperatura", "Gráfico de Humedad"])

                with tab_t:
                    st.header(f"Visualización de Temperatura (Hoja: {st.session_state.sheet_temp})")
                    col1, col2 = st.columns(2)
                    with col1:
                        dibujar_grafico_con_limites(df_orig_temp, "Temperatura Original", 25, 15)
                    with col2:
                        dibujar_grafico_con_limites(df_ext_temp, f"Extrapolado (Versión {seed_value})", 25, 15)

                with tab_hr:
                    st.header(f"Visualización de Humedad (Hoja: {st.session_state.sheet_hr})")
                    col3, col4 = st.columns(2)
                    with col3:
                        dibujar_grafico_con_limites(df_orig_hr, "Humedad Original")
                    with col4:
                        dibujar_grafico_con_limites(df_ext_hr, f"Extrapolado (Versión {seed_value})")

                st.divider()

                st.header("Editor de Curvas Individuales (Ajuste Fino)")
                editor_tab_t, editor_tab_hr = st.tabs([f"Editor T° ({st.session_state.sheet_temp})", f"Editor %HR ({st.session_state.sheet_hr})"])

                with editor_tab_t:
                    st.subheader(f"Ajustes Finos para: {st.session_state.sheet_temp}")
                    for dl_name in st.session_state.config_t.keys():
                        with st.expander(f"Ajustar: {dl_name}"):
                            st.session_state.config_t[dl_name]["prob_limpieza_picos"] = st.slider(
                                "Limpieza de Picos", 0.0, 1.0, st.session_state.config_t[dl_name]["prob_limpieza_picos"], 0.1, 
                                help="1.0 = 100% limpio. 0.0 = 100% original (con picos).", key=f"t_clean_{dl_name}"
                            )
                            st.session_state.config_t[dl_name]["variacion_percent"] = st.slider(
                                "Extrapolación (Pico %)", 0.0, 0.2, st.session_state.config_t[dl_name]["variacion_percent"], 0.01, 
                                help="Qué tanto 'sube' la curva en el pico.", key=f"t_var_{dl_name}"
                            )
                            st.session_state.config_t[dl_name]["offset_base"] = st.slider(
                                "Nivel Vertical (Offset)", -2.0, 1.0, st.session_state.config_t[dl_name]["offset_base"], 0.1, 
                                help="Sube o baja la curva completa.", key=f"t_offset_{dl_name}"
                            )
                            st.session_state.config_t[dl_name]["amplitud"] = st.slider(
                                "Nivel de 'Unicidad' (Deriva)", 0.0, 1.0, st.session_state.config_t[dl_name]["amplitud"], 0.05, 
                                help="Controla la 'personalidad' de la curva. 0 = plana.", key=f"t_amp_{dl_name}"
                            )
                            st.session_state.config_t[dl_name]["sigma"] = st.slider(
                                "Suavidad de 'Unicidad' (Ondas)", 3, 25, st.session_state.config_t[dl_name]["sigma"], 1, 
                                help="Longitud de las 'ondas'. 3 = cortas. 20 = largas.", key=f"t_sigma_{dl_name}"
                            )

                with editor_tab_hr:
                    st.subheader(f"Ajustes Finos para: {st.session_state.sheet_hr}")
                    for dl_name in st.session_state.config_hr.keys():
                        with st.expander(f"Ajustar: {dl_name}"):
                            st.session_state.config_hr[dl_name]["prob_limpieza_picos"] = st.slider(
                                "Limpieza de Picos", 0.0, 1.0, st.session_state.config_hr[dl_name]["prob_limpieza_picos"], 0.1,
                                key=f"hr_clean_{dl_name}"
                            )
                            st.session_state.config_hr[dl_name]["variacion_percent"] = st.slider(
                                "Extrapolación (Pico %)", 0.0, 0.2, st.session_state.config_hr[dl_name]["variacion_percent"], 0.01,
                                key=f"hr_var_{dl_name}"
                            )
                            st.session_state.config_hr[dl_name]["offset_base"] = st.slider(
                                "Nivel Vertical (Offset)", -2.0, 1.0, st.session_state.config_hr[dl_name]["offset_base"], 0.1,
                                key=f"hr_offset_{dl_name}"
                            )
                            st.session_state.config_hr[dl_name]["amplitud"] = st.slider(
                                "Nivel de 'Unicidad' (Deriva)", 0.0, 1.0, st.session_state.config_hr[dl_name]["amplitud"], 0.05,
                                key=f"hr_amp_{dl_name}"
                            )
                            st.session_state.config_hr[dl_name]["sigma"] = st.slider(
                                "Suavidad de 'Unicidad' (Ondas)", 3, 25, st.session_state.config_hr[dl_name]["sigma"], 1,
                                key=f"hr_sigma_{dl_name}"
                            )
                
                # --- Botón de Descarga ---
                st.sidebar.header("4. Descarga")
                if st.sidebar.button(f"Generar y Descargar Excel (Versión {seed_value})"):
                    processed_bytes = descargar_excel_modificado(
                        st.session_state['original_file_bytes'], 
                        st.session_state.config_t, 
                        st.session_state.config_hr, 
                        seed_value,
                        uploaded_file.name,
                        st.session_state.sheet_temp,
                        st.session_state.sheet_hr,
                        CONFIGURACION_BASE[preset_name]["archivo"] # Pasar nombre de archivo base para chequeo de límite
                    )
                    if processed_bytes:
                        st.sidebar.download_button(
                            label="¡Descarga Lista! (Haz clic aquí)",
                            data=processed_bytes,
                            file_name=f"extrapolado_v{seed_value}_{uploaded_file.name}",
                            mime="application/vnd.ms-excel.sheet.macroEnabled.12",
                            key="download_button" # Añadir una clave para que se actualice
                        )
                        st.sidebar.success("¡Archivo listo para descargar!")
                        # Forzar un rerun para que el botón de descarga aparezca
                        st.experimental_rerun()


        except Exception as e:
            st.error(f"Error Crítico al cargar el archivo: {e}")
            st.warning("El archivo puede estar dañado, protegido con contraseña o no ser un .xlsm válido.")
            logger.error(f"Error en Streamlit al leer el archivo: {e}", exc_info=True)
            st.session_state.clear() # Resetear todo

else:
    # Pantalla inicial
    st.info("Cargue un archivo .xlsm para comenzar.")
    st.session_state.clear()
