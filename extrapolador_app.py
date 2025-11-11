import streamlit as st
import openpyxl
import shutil
import os
import random
import logging
import numpy as np
import pandas as pd
import io
import altair as alt  # <-- Nueva librería para gráficos
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d

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

        # Alinear longitudes si es necesario
        return pd.DataFrame(dict([(k,pd.Series(v)) for k,v in datos_completos.items()]))
    except Exception as e:
        logger.error(f"Error leyendo {hoja_nombre}: {e}")
        return pd.DataFrame()


# --- (NUEVO V15) FUNCIÓN PARA DIBUJAR GRÁFICOS ---
@st.cache_data(show_spinner=False)
def dibujar_grafico_con_limites(df, titulo, limite_max=None, limite_min=None):
    """Crea un gráfico Altair con límites opcionales."""
    if df.empty:
        return None

    # 1. Convertir DataFrame de ancho a largo
    df_largo = df.reset_index().melt('index', var_name='Sensor', value_name='Valor')
    
    # 2. Crear el gráfico de líneas principal
    base = alt.Chart(df_largo).encode(
        x=alt.X('index', title='Índice de Tiempo'),
        y=alt.Y('Valor', title=titulo),
        color=alt.Color('Sensor', title="Sensores"),
        tooltip=['index', 'Sensor', 'Valor']
    ).properties(
        title=titulo
    )
    
    lineas = base.mark_line(point=False).interactive()
    
    # 3. Añadir líneas de límite si se proporcionan
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
        
    return grafico_final


# --- (NUEVO V15) FUNCIÓN DE PREVISUALIZACIÓN OPTIMIZADA ---
@st.cache_data(show_spinner=False)
def generar_datos_preview(_df_original, config, seed_value):
    """
    Toma un DataFrame y aplica el pipeline V13 solo para previsualización.
    Es mucho más rápido que modificar todo el Excel.
    """
    df_modificado = _df_original.copy()
    
    # --- Aplicar Semilla ---
    random.seed(seed_value)
    np.random.seed(seed_value)

    # Extraer parámetros
    variacion_min = config.get("variacion_min")
    variacion_max = config.get("variacion_max")
    amplitud_deriva = config.get("amplitud")
    sigma_suavizado = config.get("sigma")
    punto_pico_frac = config.get("punto_pico")
    offset_min = config.get("offset_min")
    offset_max = config.get("offset_max")
    prob_limpieza = config.get("prob_limpieza_picos")
    
    for col_nombre in df_modificado.columns:
        if not col_nombre.strip().upper().startswith("DL"):
            continue
            
        datos_np = df_modificado[col_nombre].dropna().values
        longitud_actual = len(datos_np)
        
        if longitud_actual < 20:
            continue

        # --- INICIO PIPELINE V13 ---
        # PASO 1: LIMPIEZA
        if random.random() < prob_limpieza:
            datos_base = medfilt(datos_np, kernel_size=3)
        else:
            datos_base = datos_np

        # PASO 2: EXTRAPOLACIÓN
        variacion_multi_dl = random.uniform(variacion_min, variacion_max)
        curva_multi_dl = generar_curva_multiplicativa(longitud_actual, variacion_multi_dl, punto_pico_frac)
        datos_extrapolados = datos_base * curva_multi_dl
        
        # PASO 3: DERIVA
        deriva = generar_deriva_gaussiana(longitud_actual, amplitud_deriva, sigma_suavizado)
        datos_con_deriva = datos_extrapolados + deriva
        
        # PASO 4: OFFSET
        offset_base_dl = random.uniform(offset_min, offset_max)
        datos_finales = datos_con_deriva + offset_base_dl
        
        # Sobrescribir la columna (alineando con el índice original)
        df_modificado[col_nombre] = pd.Series(datos_finales, index=df_modificado[col_nombre].dropna().index)

    return df_modificado


# --- FUNCIÓN DE MODIFICACIÓN PRINCIPAL (Para Descarga) ---
# (Esta es la V13 original, que es más lenta pero modifica el Excel real)
def modificar_workbook_completo(wb_bytes, config, seed_value):
    """Modifica el workbook completo para la descarga."""
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

    logger.info(f"Iniciando Pipeline V13 COMPLETO (Semilla: {seed_value})...")

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
            if len(valores_originales) < 20: continue
            
            logger.info(f"    > PROCESANDO: '{hoja_nombre}' -> '{header_value.strip()}' (Semilla: {seed_value})")
            datos_np = np.array(valores_originales)
            longitud_actual = len(datos_np)

            # --- INICIO PIPELINE V13 ---
            if random.random() < prob_limpieza: datos_base = medfilt(datos_np, kernel_size=3)
            else: datos_base = datos_np
            variacion_multi_dl = random.uniform(variacion_min, variacion_max)
            curva_multi_dl = generar_curva_multiplicativa(longitud_actual, variacion_multi_dl, punto_pico_frac)
            datos_extrapolados = datos_base * curva_multi_dl
            deriva = generar_deriva_gaussiana(longitud_actual, amplitud_deriva, sigma_suavizado)
            datos_con_deriva = datos_extrapolados + deriva
            offset_base_dl = random.uniform(offset_min, offset_max)
            datos_finales = datos_con_deriva + offset_base_dl
            if aplicar_limite: np.clip(datos_finales, a_min=None, a_max=25.5, out=datos_finales)
            # --- FIN PIPELINE V13 ---

            for i, cell in enumerate(celdas_datos):
                nuevo_valor = datos_finales[i]
                tipo_original = tipos_originales[i]
                if tipo_original == int: cell.value = int(round(nuevo_valor))
                else:
                    try: decimales = len(str(valores_originales[i]).split('.')[1])
                    except: decimales = 2
                    cell.value = round(nuevo_valor, decimales)
    
    logger.info(f"Pipeline V13 COMPLETO (Semilla: {seed_value}) completado.")
    with io.BytesIO() as f:
        wb.save(f)
        return f.getvalue()

# --- INTERFAZ DE STREAMLIT ---

st.set_page_config(layout="wide", page_title="Extrapolador Maestro V15.1")
st.title("Extrapolador Maestro V15.1 🚀")
st.info("Esta aplicación utiliza el pipeline V13 con **actualización en tiempo real** y **límites visuales**.")

# --- BARRA LATERAL (CONTROLES) ---
st.sidebar.header("1. Carga de Archivo")
uploaded_file = st.sidebar.file_uploader("Cargar archivo .xlsm", type=["xlsm"])

# Inicializar session_state
if 'original_file_bytes' not in st.session_state:
    st.session_state['original_file_bytes'] = None

if uploaded_file is not None:
    if st.session_state.get('original_file_name') != uploaded_file.name:
         st.session_state['original_file_bytes'] = uploaded_file.getvalue()
         st.session_state['original_file_name'] = uploaded_file.name
         st.cache_data.clear() # Limpiar caché al subir nuevo archivo

    try: 
        wb_check = openpyxl.load_workbook(io.BytesIO(st.session_state['original_file_bytes']), read_only=True)
        sheet_names = wb_check.sheetnames
        
        default_temp_index = next((i for i, s in enumerate(sheet_names) if "T°" in s or "TEMP" in s.upper()), 0)
        default_hr_index = next((i for i, s in enumerate(sheet_names) if "%HR" in s or "HR" in s.upper() or "HUM" in s.upper()), 1 if len(sheet_names) > 1 else 0)

        st.sidebar.header("2. Selección de Gráficos")
        sheet_temp = st.sidebar.selectbox(
            "Hoja de Temperatura (T°) a Visualizar", sheet_names, index=default_temp_index
        )
        sheet_hr = st.sidebar.selectbox(
            "Hoja de Humedad (%HR) a Visualizar", sheet_names, index=default_hr_index
        )

        st.sidebar.header("3. Parámetros de Extrapolación")
        
        seed_value = st.sidebar.number_input(
            "Versión (Semilla Aleatoria)", 
            value=1, min_value=1, step=1,
            help="Cambia este número para generar un conjunto aleatorio diferente (ej. para otro vehículo). Mantenlo igual para reproducir el mismo resultado."
        )
        
        preset_name = st.sidebar.selectbox(
            "Seleccionar Preset de Prueba:", 
            options=list(CONFIGURACION_V12.keys()),
            help="Elige el tipo de prueba. Esto cargará los parámetros recomendados."
        )
        config_base = CONFIGURACION_V12[preset_name]

        st.sidebar.subheader("Ajustes Manuales (Controles)")
        
        var_min_max = st.sidebar.slider(
            "Extrapolación (Pico %)", 
            0.0, 0.2, (config_base['variacion_min'], config_base['variacion_max']), 0.01,
            help="Rango aleatorio para el pico de la extrapolación (ej. 3% a 5%). Esto controla qué tanto 'sube' la curva."
        )
        
        offset_min_max = st.sidebar.slider(
            "Nivel Vertical (Offset)", 
            -2.0, 1.0, (config_base['offset_min'], config_base['offset_max']), 0.1,
            help="Rango aleatorio para 'bajar' (negativo) o 'subir' (positivo) cada curva DL individualmente."
        )

        amplitud = st.sidebar.slider(
            "Nivel de 'Unicidad' (Deriva)", 
            0.0, 1.0, config_base['amplitud'], 0.05,
            help="Controla la 'personalidad' de cada curva. 0 = curvas idénticas. 0.5 = curvas muy únicas."
        )
        
        sigma = st.sidebar.slider(
            "Suavidad de 'Unicidad' (Ondas)", 
            3, 25, config_base['sigma'], 1,
            help="Longitud de las 'ondas' de deriva. 3 = ondas cortas/rápidas. 20 = ondas largas/suaves."
        )
        
        prob_limpieza = st.sidebar.slider(
            "Limpieza de Picos (Probabilidad)", 
            0.0, 1.0, config_base['prob_limpieza_picos'], 0.1,
            help="Probabilidad de que los picos anómalos (como 'caídas' de sensor) sean eliminados. 1.0 = 100% limpios. 0.0 = 100% originales."
        )
        
        # Guardar la configuración actual en un solo dict
        config_personalizada = {
            "archivo": config_base["archivo"],
            "variacion_min": var_min_max[0],
            "variacion_max": var_min_max[1],
            "amplitud": amplitud,
            "sigma": sigma,
            "punto_pico": config_base["punto_pico"],
            "offset_min": offset_min_max[0],
            "offset_max": offset_min_max[1], # <-- ¡¡AQUÍ ESTABA EL ERROR!!
            "prob_limpieza_picos": prob_limpieza
        }

        # --- GENERACIÓN DE GRÁFICOS (V15) ---
        with st.spinner("Actualizando gráficos..."):
            # Leer datos originales
            df_orig_temp = leer_datos_para_grafico(st.session_state['original_file_bytes'], sheet_temp)
            df_orig_hr = leer_datos_para_grafico(st.session_state['original_file_bytes'], sheet_hr)
            
            # Generar datos de preview
            df_ext_temp = generar_datos_preview(df_orig_temp, config_personalizada, seed_value)
            df_ext_hr = generar_datos_preview(df_orig_hr, config_personalizada, seed_value)

        # --- ÁREA PRINCIPAL (GRÁFICOS) ---
        st.header(f"Visualización de Temperatura (Hoja: {sheet_temp})")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            chart_orig_temp = dibujar_grafico_con_limites(df_orig_temp, "Temperatura Original", 25, 15)
            if chart_orig_temp: st.altair_chart(chart_orig_temp, use_container_width=True)
            else: st.warning(f"No se encontraron datos 'DL' en la hoja '{sheet_temp}'.")
        with col2:
            st.subheader(f"Extrapolado (Versión {seed_value})")
            chart_ext_temp = dibujar_grafico_con_limites(df_ext_temp, "Temperatura Extrapolada", 25, 15)
            if chart_ext_temp: st.altair_chart(chart_ext_temp, use_container_width=True)
            else: st.warning(f"No se pudieron generar datos para '{sheet_temp}'.")
            
        st.divider()
        
        st.header(f"Visualización de Humedad (Hoja: {sheet_hr})")
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Original")
            chart_orig_hr = dibujar_grafico_con_limites(df_orig_hr, "Humedad Original")
            if chart_orig_hr: st.altair_chart(chart_orig_hr, use_container_width=True)
            else: st.warning(f"No se encontraron datos 'DL' en la hoja '{sheet_hr}'.")
        with col4:
            st.subheader(f"Extrapolado (Versión {seed_value})")
            chart_ext_hr = dibujar_grafico_con_limites(df_ext_hr, "Humedad Extrapolada")
            if chart_ext_hr: st.altair_chart(chart_ext_hr, use_container_width=True)
            else: st.warning(f"No se pudieron generar datos para '{sheet_hr}'.")

        # --- BOTÓN DE DESCARGA (V15) ---
        st.sidebar.header("4. Descarga")
        if st.sidebar.button(f"Generar y Descargar Excel (Versión {seed_value})", type="primary"):
            with st.spinner("Procesando archivo COMPLETO para descarga..."):
                try:
                    processed_bytes = modificar_workbook_completo(
                        st.session_state['original_file_bytes'], 
                        config_personalizada, 
                        seed_value
                    )
                    
                    st.sidebar.download_button(
                        label="¡Descarga Lista! (Haz clic aquí)",
                        data=processed_bytes,
                        file_name=f"extrapolado_v{seed_value}_{uploaded_file.name}",
                        mime="application/vnd.ms-excel.sheet.macroEnabled.12"
                    )
                    st.sidebar.success("¡Archivo listo para descargar!")
                except Exception as e:
                    st.sidebar.error(f"Error al generar archivo: {e}")
                    logger.error(f"Error en descarga: {e}", exc_info=True)

    except Exception as e:
        st.error(f"Error Crítico al cargar el archivo: {e}")
        st.warning("El archivo puede estar dañado, protegido con contraseña o no ser un .xlsm válido.")
        logger.error(f"Error en Streamlit al leer el archivo: {e}", exc_info=True)
        st.session_state['original_file_bytes'] = None

else:
    st.info("Cargue un archivo .xlsm para comenzar.")
    st.session_state['original_file_bytes'] = None
