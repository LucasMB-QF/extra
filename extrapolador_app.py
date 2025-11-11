import openpyxl
import shutil
import os
import random
import logging
import numpy as np
from scipy.signal import medfilt  # (Paso 1) Para limpiar picos
from scipy.ndimage import gaussian_filter1d # (Paso 3) Para deriva suave

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes Globales ---
ARCHIVOS_CON_LIMITE = ["1. OQ_MAPEO.xlsm", "4. PQ_RUTA_20.xlsm", "5. PQ_RUTA_80.xlsm"]
HOJAS_A_IGNORAR = ["CONSOLIDADO", "GRAFICOS", "RESUMEN", "TABLA", "RESULTADOS", "SUMMARY", "GRAFICO"] 

# --- FUNCIONES DE GENERACIÓN DE CURVAS (Sin cambios) ---

def generar_deriva_gaussiana(longitud, amplitud_max_grados=0.15, sigma_suavizado=5):
    """(PASO 3) Genera una curva de deriva suave (aditiva) única por DL."""
    try:
        ruido_base = np.random.randn(longitud)
        deriva_suave = gaussian_filter1d(ruido_base, sigma=sigma_suavizado)
        
        max_abs = np.max(np.abs(deriva_suave))
        if max_abs > 1e-6:
            deriva_normalizada = deriva_suave / max_abs
        else:
            deriva_normalizada = np.zeros(longitud)
        
        deriva_final = deriva_normalizada * amplitud_max_grados
        
        fade_len = min(longitud // 10, int(sigma_suavizado * 3))
        if fade_len > 1:
            fade_in = np.linspace(0, 1, fade_len)
            deriva_final[:fade_len] *= fade_in
            fade_out = np.linspace(1, 0, fade_len)
            deriva_final[-fade_len:] *= fade_out
            
        return deriva_final
        
    except Exception as e:
        logger.warning(f"No se pudo generar deriva Gaussiana. {e}. Retornando deriva cero.")
        return np.zeros(longitud)

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
        
    except Exception as e:
        logger.warning(f"No se pudo generar curva multiplicativa. {e}. Retornando curva de 1.0.")
        return np.ones(longitud)


# --- FUNCIÓN DE MODIFICACIÓN PRINCIPAL (V12) ---

def modificar_valores_excel_avanzado(
    path_original, 
    path_destino, 
    config_extrapolacion
):
    """
    Pipeline V12:
    - Ignora hojas de resumen.
    - (Paso 1) Limpieza de picos PROBABILÍSTICA (ej. 70% de las curvas).
    - (Paso 2) Aplica una curva multiplicativa ALEATORIA Y ÚNICA a CADA DL.
    - (Paso 3) Aplica una deriva Gaussiana ÚNICA a cada DL.
    - (Paso 4) Aplica un offset base ALEATORIO Y ÚNICO a CADA DL.
    """
    try:
        shutil.copy(path_original, path_destino)
        wb = openpyxl.load_workbook(path_destino, keep_vba=True)
        
        # Extraer parámetros de la configuración
        variacion_min = config_extrapolacion.get("variacion_min", 0.02)
        variacion_max = config_extrapolacion.get("variacion_max", 0.05)
        amplitud_deriva = config_extrapolacion.get("amplitud", 0.2)
        sigma_suavizado = config_extrapolacion.get("sigma", 6)
        punto_pico_frac = config_extrapolacion.get("punto_pico", 0.6) # Consistente
        aplicar_limite = config_extrapolacion.get("aplicar_limite", False)
        offset_min = config_extrapolacion.get("offset_min", -0.5)
        offset_max = config_extrapolacion.get("offset_max", -0.2)
        prob_limpieza = config_extrapolacion.get("prob_limpieza_picos", 0.7) # Probabilidad de limpiar

        logger.info(f"Procesando: {os.path.basename(path_destino)}")
        logger.info(f"  > Config V12 (Por DL): Pico Multi Aleatorio: {variacion_min*100:+.1f}% a {variacion_max*100:+.1f}%")
        logger.info(f"  > Config V12 (Por DL): Amplitud Deriva: {amplitud_deriva:+.2f}°C, Suavidad: {sigma_suavizado}")
        logger.info(f"  > Config V12 (Por DL): Offset Base Aleatorio: {offset_min:+.2f}°C a {offset_max:+.2f}°C")
        logger.info(f"  > Config V12 (Por DL): Probabilidad Limpieza Picos: {prob_limpieza*100:.0f}%")

        # --- ITERAR HOJAS Y COLUMNAS ---
        for hoja_nombre in wb.sheetnames:
            
            if any(ignorar in hoja_nombre.strip().upper() for ignorar in HOJAS_A_IGNORAR):
                logger.info(f"  > Ignorando hoja: {hoja_nombre} (hoja de resumen/gráficos)")
                continue
            
            logger.info(f"  > Revisando hoja: {hoja_nombre}")
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
                    logger.warning(f"    > Columna '{header_value.strip()}' en '{hoja_nombre}' no contiene datos numéricos crudos. Ignorando.")
                    continue

                logger.info(f"    > PROCESANDO COLUMNA: '{header_value.strip()}' en hoja '{hoja_nombre}'")
                
                datos_np = np.array(valores_originales)
                longitud_actual = len(datos_np)

                # --- INICIO PIPELINE V12 ---
                
                # PASO 1: LIMPIEZA DE PICOS (Probabilística)
                if random.random() < prob_limpieza:
                    datos_base = medfilt(datos_np, kernel_size=3)
                    logger.info("      > (Paso 1: Picos Limpiados)")
                else:
                    datos_base = datos_np # Dejar los picos originales
                    logger.info("      > (Paso 1: Picos Originales Mantenidos)")

                # PASO 2: EXTRAPOLACIÓN TEMPORAL (¡¡ALEATORIA POR DL!!)
                variacion_multi_dl = random.uniform(variacion_min, variacion_max)
                curva_multi_dl = generar_curva_multiplicativa(longitud_actual, variacion_multi_dl, punto_pico_frac)
                datos_extrapolados = datos_base * curva_multi_dl
                logger.info(f"      > (Paso 2: Pico Multiplicativo: {variacion_multi_dl*100:+.2f}%)")
                
                # PASO 3: DERIVA DE REALISMO (Única por columna)
                deriva = generar_deriva_gaussiana(longitud_actual, amplitud_deriva, sigma_suavizado)
                datos_con_deriva = datos_extrapolados + deriva
                
                # PASO 4: APLICAR OFFSET BASE (Aleatorio por columna)
                offset_base_dl = random.uniform(offset_min, offset_max)
                datos_finales = datos_con_deriva + offset_base_dl
                logger.info(f"      > (Paso 4: Offset Base: {offset_base_dl:+.2f}°C)")
                
                # --- FIN PIPELINE V12 ---
                
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

        wb.save(path_destino)
        logger.info(f"Archivo guardado (V12 - Extrapolador Maestro): {path_destino}")
        return True

    except Exception as e:
        logger.error(f"Error en archivo {path_destino}: {e}", exc_info=True)
        return False

# --- CONFIGURACIÓN V12 (Ajustada con `prob_limpieza_picos`) ---
CONFIGURACION_V12 = [
    {
        "archivo": "1. OQ MAPEO 72 INV.xlsm",        
        "variacion_min": 0.01, "variacion_max": 0.02, # Mapeo casi plano
        "amplitud": 0.30, "sigma": 12, "punto_pico": 0.5,
        "offset_min": -0.5, "offset_max": 0.0,
        "prob_limpieza_picos": 0.9 # 90% de probabilidad de limpiar picos
    },
    {
        "archivo": "2. OQ APERTURA 72 INV.xlsm",     
        "variacion_min": 0.03, "variacion_max": 0.05, # Extrapolación leve
        "amplitud": 0.40, "sigma": 8, "punto_pico": 0.6,
        "offset_min": -1.0, "offset_max": -0.2,
        "prob_limpieza_picos": 0.5 # 50% de probabilidad (para mantener picos)
    },
    {
        "archivo": "3. OQ APAGADO 72 INV.xlsm",      
        "variacion_min": 0.01, "variacion_max": 0.02, # Apagado casi plano
        "amplitud": 0.50, "sigma": 20, "punto_pico": 0.4,
        "offset_min": -1.2, "offset_max": -0.3,
        "prob_limpieza_picos": 0.9 # 90% de probabilidad de limpiar
    },
    {
        "archivo": "4. PQ RUTA 20 72 INV.xlsm",      
        "variacion_min": 0.02, "variacion_max": 0.04,
        "amplitud": 0.35, "sigma": 12, "punto_pico": 0.5,
        "offset_min": -0.9, "offset_max": -0.2,
        "prob_limpieza_picos": 0.7 # 70% de probabilidad de limpiar
    },
    {
        "archivo": "5. PQ RUTA 80 72 INV.xlsm",      
        "variacion_min": 0.02, "variacion_max": 0.04,
        "amplitud": 0.35, "sigma": 12, "punto_pico": 0.5,
        "offset_min": -0.9, "offset_max": -0.2,
        "prob_limpieza_picos": 0.7
    },
    {
        "archivo": "6. PQ APERTURA 20 72 INV.xlsm",  
        "variacion_min": 0.03, "variacion_max": 0.05, # Extrapolación leve
        "amplitud": 0.40, "sigma": 8, "punto_pico": 0.6,
        "offset_min": -1.0, "offset_max": -0.2,
        "prob_limpieza_picos": 0.5 # 50% de probabilidad (para mantener picos)
    },
    {
        "archivo": "7. PQ APERTURA 80 72 INV.xlsm",  
        "variacion_min": 0.03, "variacion_max": 0.05,
        "amplitud": 0.40, "sigma": 8, "punto_pico": 0.6,
        "offset_min": -1.0, "offset_max": -0.2,
        "prob_limpieza_picos": 0.5
    },
    {
        "archivo": "8. PQ APAGADO 20 72 INV.xlsm",   
        "variacion_min": 0.01, "variacion_max": 0.02,
        "amplitud": 0.50, "sigma": 20, "punto_pico": 0.4,
        "offset_min": -1.2, "offset_max": -0.3,
        "prob_limpieza_picos": 0.9
    },
    {
        "archivo": "9. PQ APAGADO 80 72 INV.xlsm",   
        "variacion_min": 0.01, "variacion_max": 0.02,
        "amplitud": 0.50, "sigma": 20, "punto_pico": 0.4,
        "offset_min": -1.2, "offset_max": -0.3,
        "prob_limpieza_picos": 0.9
    },
]

def generar_conjunto_completo(carpeta_principal, nombre_conjunto, configuracion):
    """Genera un conjunto de archivos extrapolados en carpeta"""
    try:
        carpeta_conjunto = os.path.join(carpeta_principal, nombre_conjunto)
        os.makedirs(carpeta_conjunto, exist_ok=True)
        logger.info(f"Carpeta creada: {carpeta_conjunto}")

        for config in configuracion:
            archivo_original = config["archivo"]
            nombre_base = os.path.splitext(os.path.basename(archivo_original))[0]
            nombre_salida = os.path.join(carpeta_conjunto, f"{nombre_base}_Extrapolado.xlsm")
            
            config["aplicar_limite"] = any(nombre in archivo_original for nombre in ARCHIVOS_CON_LIMITE)
            
            modificar_valores_excel_avanzado(
                archivo_original,
                nombre_salida,
                config
            )
        
        logger.info(f"✅ Conjunto '{nombre_conjunto}' completado.")
        return True

    except Exception as e:
        logger.error(f"❌ Error al generar conjunto '{nombre_conjunto}': {e}")
        return False

# --- Ejecución Principal ---
if __name__ == "__main__":
    logger.info("\n" + "="*60 + "\nINICIO DEL EXTRAPOLADOR MAESTRO (V12)\n" + "="*66)
    
    nombre_carpeta_principal = input("Nombre para la carpeta principal de resultados: ").strip()
    if not nombre_carpeta_principal:
        nombre_carpeta_principal = "Extrapolaciones_Avanzadas_V12"
    
    carpeta_principal = os.path.join("Resultados_Extrapolados", nombre_carpeta_principal)
    os.makedirs(carpeta_principal, exist_ok=True)

    try:
        num_conjuntos = int(input("¿Cuántos conjuntos quieres generar? (por defecto 7): ") or 7)
    except ValueError:
        num_conjuntos = 7
        logger.warning("Entrada inválida. Usando 7 conjuntos por defecto.")

    nombres_conjuntos = []
    print("\n" + "="*60 + "\nNOMBRES PERSONALIZADOS PARA CADA CONJUNTO\n" + "="*60)
    for i in range(1, num_conjuntos + 1):
        nombre_conjunto = input(f"Nombre para el conjunto {i}: ").strip()
        if not nombre_conjunto:
            nombre_conjunto = f"Conjunto_{i}"
        nombres_conjuntos.append(nombre_conjunto)
    
    for i, nombre_conjunto in enumerate(nombres_conjuntos, 1):
        logger.info(f"\n{'='*30} GENERANDO CONJUNTO {i}/{num_conjuntos} ({nombre_conjunto}) {'='*30}")
        generar_conjunto_completo(carpeta_principal, nombre_conjunto, CONFIGURACION_V12)
    
    logger.info("\n" + "="*60 + "\nPROCESO FINALIZADO\n" + "="*60)
    print(f"\nTodos los conjuntos fueron generados en: {os.path.abspath(carpeta_principal)}")