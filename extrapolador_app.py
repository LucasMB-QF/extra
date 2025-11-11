import streamlit as st
import openpyxl
import shutil
import os
import random
import logging
import numpy as np
import pandas as pd
import io
from scipy.signal import medfilt  # (Paso 1) Para limpiar picos
from scipy.ndimage import gaussian_filter1d # (Paso 3) Para deriva suave

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes Globales ---
ARCHIVOS_CON_LIMITE = ["1. OQ_MAPEO.xlsm", "4. PQ_RUTA_20.xlsm", "5. PQ_RUTA_80.xlsm"]
HOJAS_A_IGNORAR = ["CONSOLIDADO", "GRAFICOS", "RESUMEN", "TABLA", "RESULTADOS", "SUMMARY", "GRAFICO"] 

# --- CONFIGURACIÓN V12 (Nuestros Presets) ---
CONFIGURACION_V12 = {
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

# --- FUNCIONES DE GENERACIÓN DE CURVAS ---

@st.cache_data(show_spinner=False)
def generar_deriva_gaussiana(longitud, amplitud_max_grados=0.15, sigma_suavizado=5):
    """(PASO 3) Genera una curva de deriva suave (aditiva) única por DL."""
    try:
        ruido_base = np.random.randn(longitud)
        deriva_suave = gaussian_filter1d(ruido_base, sigma=sigma_suavizado)
        max_abs = np.max(np.abs(deriva_suave))
        if max_abs > 1e-6: deriva_normalizada = deriva_suave / max_abs
        else: deriva_normalizada = np.zeros(longitud)
        deriva_final = deriva_normalizada * amplitud_max_grados
        fade_len = min(longitud // 10, int(sigma_suavizado * 3))
        if fade_len > 1:
            fade_in = np.linspace(0, 1, fade_len)
            deriva_final[:fade_len] *= fade_in
            fade_out = np.linspace(1, 0, fade_len)
            deriva_final[-fade_len:] *= fade_out
        return deriva_final
    except Exception: return np.zeros(longitud)

@st.cache_data(show_spinner=False)
def generar_curva_multiplicativa(longitud, variacion_max_percent, punto_pico_frac=0.6):
    """(PASO 2) Genera una curva de multiplicación que vuelve a 1.0."""
    try:
        factor_max = 1.0 + variacion_max_percent
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

# --- FUNCIÓN DE LECTURA DE DATOS (Para Gráficos) ---
@st.cache_data(show_spinner=False)
def leer_datos_para_grafico(wb_bytes, hoja_nombre):
    """Lee todas las columnas DL de una hoja y las devuelve en un DataFrame."""
    try:
        wb = openpyxl.load_workbook(io.BytesIO(wb_bytes), data_only=True)
        if hoja_nombre not in wb.sheetnames:
            return pd.DataFrame() 

        ws = wb[hoja_nombre]
        datos_completos = {}
        
        for col in ws.iter_cols(min_row=1):
            header_value = col[0].value
            if isinstance(header_value, str) and header_value.strip().upper().startswith("DL"):
                valores = []
                for cell in col[1:]:
                    if isinstance(cell.value, (int, float)):
                        valores.append(cell.value)
                
                if len(valores) > 20:
                    datos_completos[header_value.strip()] = valores

        return pd.DataFrame(dict([(k,pd.Series(v)) for k,v in datos_completos.items()]))
    except Exception as e:
        logger.error(f"Error leyendo {hoja_nombre}: {e}")
        return pd.DataFrame()


# --- FUNCIÓN DE MODIFICACIÓN PRINCIPAL (V13 - CON SEMILLA) ---
def modificar_workbook_en_memoria(wb_bytes, config, seed_value):
    """Modifica un objeto workbook de openpyxl en memoria. Devuelve el wb modificado."""
    
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    wb = openpyxl.load_workbook(io.BytesIO(wb_bytes), keep_vba=True)
    
    variacion_min = config.get("variacion_min")
    variacion_max = config.get("variacion_max")
    amplitud_deriva = config.get("amplitud")
    sigma_suavizado = config.get("sigma")
    punto_pico_frac = config.get("punto_pico")
    offset_min = config.get("offset_min")
    offset_max = config.get("offset_max")
    prob_limpieza = config.get("prob_limpieza_picos")
    aplicar_limite = any(nombre in config["archivo"] for nombre in ARCHIVOS_CON_LIMITE)

    logger.info(f"Iniciando Pipeline V13 (Semilla: {seed_value})...")

    for hoja_nombre in wb.sheetnames:
        if any(ignorar in hoja_nombre.strip().upper() for ignorar in HOJAS_A_IGNORAR):
            continue
        
        ws = wb[hoja_nombre]

        for col in ws.iter_cols(min_row=1):
            header_cell = col[0]
            header_value = header_cell.value
            
            if not (isinstance(header_value, str) and header_value.strip().upper().startswith("DL")):
                continue

            celdas_datos, valores_originales, tipos_originales = [], [], []
            
            for cell in col[1:]:
                if isinstance(cell.value, (int, float)) and cell.data_type != 'f':
                    celdas_datos.append(cell)
                    valores_originales.append(cell.value)
                    tipos_originales.append(type(cell.value))
            
            if len(valores_originales) < 20:
                continue

            logger.info(f"    > PROCESANDO: '{hoja_nombre}' -> '{header_value.strip()}' (Semilla: {seed_value})")
            
            datos_np = np.array(valores_originales)
            longitud_actual = len(datos_np)

            # --- INICIO PIPELINE V12 (que ahora es V13 gracias a la semilla) ---
            
            # PASO 1: LIMPIEZA DE PICOS (Probabilística)
            if random.random() < prob_limpieza:
                datos_base = medfilt(datos_np, kernel_size=3)
            else:
                datos_base = datos_np

            # PASO 2: EXTRAPOLACIÓN TEMPORAL (Aleatoria por DL)
            variacion_multi_dl = random.uniform(variacion_min, variacion_max)
            curva_multi_dl = generar_curva_multiplicativa(longitud_actual, variacion_multi_dl, punto_pico_frac)
            datos_extrapolados = datos_base * curva_multi_dl
            
            # PASO 3: DERIVA DE REALISMO (Única por columna)
            deriva = generar_deriva_gaussiana(longitud_actual, amplitud_deriva, sigma_suavizado)
            datos_con_deriva = datos_extrapolados + deriva
            
            # PASO 4: APLICAR OFFSET BASE (Aleatorio por columna)
            offset_base_dl = random.uniform(offset_min, offset_max)
            datos_finales = datos_con_deriva + offset_base_dl
            
            # PASO 5: APLICAR LÍMITE
            if aplicar_limite:
                np.clip(datos_finales, a_min=None, a_max=25.5, out=datos_finales)
            
            # PASO 6: Escribir datos de vuelta
            for i, cell in enumerate(celdas_datos):
                nuevo_valor = datos_finales[i]
                tipo_original = tipos_originales[i]
                
                if tipo_original == int:
                    cell.value = int(round(nuevo_valor))
                else:
                    try:
                        decimales = len(str(valores_originales[i]).split('.')[1])
                    except:
                        decimales = 2
                    cell.value = round(nuevo_valor, decimales)

    logger.info(f"Pipeline V13 (Semilla: {seed_value}) completado.")
    # Guardar el workbook modificado en un objeto de bytes
    with io.BytesIO() as f:
        wb.save(f)
        return f.getvalue()

# --- INTERFAZ DE STREAMLIT ---

st.set_page_config(layout="wide", page_title="Extrapolador Maestro V13.1")
st.title("Extrapolador Maestro V13.1 🚀")
st.info("Esta aplicación utiliza el pipeline V12 con una **semilla aleatoria (Número de Versión)** para controlar la aleatoriedad.")

# --- BARRA LATERAL (CONTROLES) ---

st.sidebar.header("1. Carga de Archivo")
uploaded_file = st.sidebar.file_uploader("Cargar archivo .xlsm", type=["xlsm"])

# Inicializar session_state
if 'processed_file_bytes' not in st.session_state:
    st.session_state['processed_file_bytes'] = None
if 'original_file_bytes' not in st.session_state:
    st.session_state['original_file_bytes'] = None
if 'chart_data_original' not in st.session_state:
    st.session_state['chart_data_original'] = None
if 'chart_data_extrapolado' not in st.session_state:
    st.session_state['chart_data_extrapolado'] = None
if 'sheet_temp' not in st.session_state:
    st.session_state['sheet_temp'] = ""
if 'sheet_hr' not in st.session_state:
    st.session_state['sheet_hr'] = ""


if uploaded_file is not None:
    # Almacenar bytes originales en session_state
    if 'original_file_bytes' not in st.session_state or st.session_state.original_file_bytes is None:
         st.session_state['original_file_bytes'] = uploaded_file.getvalue()

    # --- ¡¡BLOQUE CORREGIDO!! ---
    # Este try...except envuelve TODA la lógica de la barra lateral
    # que depende del archivo cargado.
    try: 
        wb_check = openpyxl.load_workbook(io.BytesIO(st.session_state['original_file_bytes']), read_only=True)
        sheet_names = wb_check.sheetnames
        
        default_temp_index = next((i for i, s in enumerate(sheet_names) if "T°" in s or "TEMP" in s.upper()), 0)
        default_hr_index = next((i for i, s in enumerate(sheet_names) if "%HR" in s or "HR" in s.upper() or "HUM" in s.upper()), 1 if len(sheet_names) > 1 else 0)

        st.sidebar.header("2. Selección de Gráficos")
        st.session_state['sheet_temp'] = st.sidebar.selectbox(
            "Hoja de Temperatura (T°) a Visualizar", 
            sheet_names, 
            index=default_temp_index
        )
        st.session_state['sheet_hr'] = st.sidebar.selectbox(
            "Hoja de Humedad (%HR) a Visualizar", 
            sheet_names, 
            index=default_hr_index
        )

        st.sidebar.header("3. Parámetros de Extrapolación")
        
        seed_value = st.sidebar.number_input(
            "Número de Versión (Semilla)", 
            value=1, 
            min_value=1, 
            step=1,
            help="Cambia este número para generar un conjunto aleatorio diferente (ej. para otro vehículo). Mantenlo igual para reproducir el mismo resultado."
        )
        
        preset_name = st.sidebar.selectbox(
            "Seleccionar Preset de Prueba:", 
            options=list(CONFIGURACION_V12.keys()),
            help="Elige el tipo de prueba. Esto cargará los parámetros recomendados."
        )
        config_base = CONFIGURACION_V12[preset_name]

        st.sidebar.subheader("Anulación Manual de Parámetros")
        
        var_min_max = st.sidebar.slider(
            "Rango de Pico de Extrapolación (%)", 
            0.0, 0.2, (config_base['variacion_min'], config_base['variacion_max']), 0.01
        )
        
        offset_min_max = st.sidebar.slider(
            "Rango de Offset Base (°C o %HR)", 
            -2.0, 1.0, (config_base['offset_min'], config_base['offset_max']), 0.1
        )

        amplitud = st.sidebar.slider(
            "Amplitud de Deriva (Unicidad)", 
            0.0, 1.0, config_base['amplitud'], 0.05
        )
        
        sigma = st.sidebar.slider(
            "Suavidad de Deriva (Ondas)", 
            3, 25, config_base['sigma'], 1
        )
        
        prob_limpieza = st.sidebar.slider(
            "Probabilidad de Limpieza de Picos", 
            0.0, 1.0, config_base['prob_limpieza_picos'], 0.1
        )

        if st.sidebar.button("Generar/Actualizar Gráficos", type="primary"):
            with st.spinner("Procesando archivo... Esto puede tardar unos segundos..."):
                try:
                    # 1. Leer datos originales para los gráficos
                    st.session_state['chart_data_original'] = {
                        st.session_state['sheet_temp']: leer_datos_para_grafico(st.session_state['original_file_bytes'], st.session_state['sheet_temp']),
                        st.session_state['sheet_hr']: leer_datos_para_grafico(st.session_state['original_file_bytes'], st.session_state['sheet_hr'])
                    }

                    # 2. Crear la configuración personalizada
                    config_personalizada = {
                        "archivo": config_base["archivo"],
                        "variacion_min": var_min_max[0],
                        "variacion_max": var_min_max[1],
                        "amplitud": amplitud,
                        "sigma": sigma,
                        "punto_pico": config_base["punto_pico"],
                        "offset_min": offset_min_max[0],
                        "offset_max": offset_min_max[1],
                        "prob_limpieza_picos": prob_limpieza
                    }
                    
                    # 3. Modificar el workbook EN MEMORIA (pasando la semilla)
                    processed_bytes = modificar_workbook_en_memoria(st.session_state['original_file_bytes'], config_personalizada, seed_value)
                    st.session_state['processed_file_bytes'] = processed_bytes
                    
                    # 4. Leer datos modificados para los gráficos
                    st.session_state['chart_data_extrapolado'] = {
                        st.session_state['sheet_temp']: leer_datos_para_grafico(processed_bytes, st.session_state['sheet_temp']),
                        st.session_state['sheet_hr']: leer_datos_para_grafico(processed_bytes, st.session_state['sheet_hr'])
                    }
                    
                    st.success(f"¡Extrapolación generada con éxito! (Versión/Semilla: {seed_value})")

                except Exception as e:
                    st.error(f"Error durante el procesamiento: {e}")
                    logger.error(f"Error en Streamlit: {e}", exc_info=True)

        if st.session_state['processed_file_bytes']:
            st.sidebar.download_button(
                label=f"Descargar Excel Modificado (Versión {seed_value})",
                data=st.session_state['processed_file_bytes'],
                file_name=f"extrapolado_v{seed_value}_{uploaded_file.name}",
                mime="application/vnd.ms-excel.sheet.macroEnabled.12"
            )
            
    # --- ¡¡ESTE ES EL BLOQUE QUE AÑADÍ!! ---
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        st.warning("El archivo puede estar dañado, protegido con contraseña o no ser un .xlsm válido.")
        logger.error(f"Error en Streamlit al leer el archivo: {e}", exc_info=True)
        # Limpiar el estado para permitir recargar
        st.session_state['original_file_bytes'] = None
    # --- FIN DEL BLOQUE AÑADIDO ---


# --- ÁREA PRINCIPAL (GRÁFICOS) ---

else:
    st.info("Cargue un archivo .xlsm para comenzar.")
    # Limpiar estado si el archivo se des-selecciona
    st.session_state['processed_file_bytes'] = None
    st.session_state['original_file_bytes'] = None
    st.session_state['chart_data_original'] = None
    st.session_state['chart_data_extrapolado'] = None

if st.session_state['chart_data_original'] is not None and st.session_state['chart_data_extrapolado'] is not None:
    
    st.header(f"Visualización de Temperatura (Hoja: {st.session_state['sheet_temp']})")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        df_orig_temp = st.session_state['chart_data_original'][st.session_state['sheet_temp']]
        if not df_orig_temp.empty:
            st.line_chart(df_orig_temp)
        else:
            st.warning(f"No se encontraron datos 'DL' en la hoja '{st.session_state['sheet_temp']}'.")

    with col2:
        st.subheader(f"Extrapolado (Versión {seed_value})")
        df_ext_temp = st.session_state['chart_data_extrapolado'][st.session_state['sheet_temp']]
        if not df_ext_temp.empty:
            st.line_chart(df_ext_temp)
        else:
            st.warning(f"No se pudieron generar datos para '{st.session_state['sheet_temp']}'.")
            
    st.divider()
    
    st.header(f"Visualización de Humedad (Hoja: {st.session_state['sheet_hr']})")
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Original")
        df_orig_hr = st.session_state['chart_data_original'][st.session_state['sheet_hr']]
        if not df_orig_hr.empty:
            st.line_chart(df_orig_hr)
        else:
            st.warning(f"No se encontraron datos 'DL' en la hoja '{st.session_state['sheet_hr']}'.")

    with col4:
        st.subheader(f"Extrapolado (Versión {seed_value})")
        df_ext_hr = st.session_state['chart_data_extrapolado'][st.session_state['sheet_hr']]
        if not df_ext_hr.empty:
            st.line_chart(df_ext_hr)
        else:
            st.warning(f"No se pudieron generar datos para '{st.session_state['sheet_hr']}'.")
