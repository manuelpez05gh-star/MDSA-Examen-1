import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
import io

# ==============================================================================
# 0. CONFIGURACIÓN INICIAL, FUNCIONES AUXILIARES Y ESTILOS CSS
# ==============================================================================
st.set_page_config(
    page_title="Plataforma de Simulación Actuarial",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=3600, show_spinner=False)
def descargar_datos_financieros(ticker, anios):
    """
    Descarga el historial de precios de cierre desde Yahoo Finance para el
    horizonte seleccionado (1 a 5 años) y devuelve la serie de precios,
    la serie de rendimientos diarios (%), el precio spot y las fechas.
    """
    tkr = yf.Ticker(ticker.strip().upper())
    df_mkt = tkr.history(period=f"{int(anios)}y")
    if df_mkt is None or df_mkt.empty or 'Close' not in df_mkt.columns:
        return None, None, None, None, None
    
    precios_s = df_mkt['Close'].dropna()
    if len(precios_s) < 2:
        return None, None, None, None, None
        
    rendimientos_s = precios_s.pct_change().dropna() * 100.0
    precio_spot = float(precios_s.iloc[-1])
    fecha_inicio = precios_s.index[0].strftime('%Y-%m-%d')
    fecha_fin = precios_s.index[-1].strftime('%Y-%m-%d')
    return precios_s, rendimientos_s, precio_spot, fecha_inicio, fecha_fin

def construir_multinomial_1d(n_trials, probs):
    """
    Construye una variable aleatoria discreta compatible con SciPy (pmf, cdf, ppf, rvs)
    a partir de una distribución Multinomial(n, p) proyectada sobre el índice de sus categorías.
    """
    probs = np.array(probs, dtype=float)
    probs = np.maximum(probs, 1e-6)
    probs = probs / probs.sum()
    
    poly = np.array([1.0])
    for _ in range(max(int(n_trials), 1)):
        poly = np.convolve(poly, probs)
        
    xk = np.arange(len(poly), dtype=int)
    pk = poly / poly.sum()
    return stats.rv_discrete(name='multinomial_1d', values=(xk, pk)), probs

def calcular_bondad_chi2(data, cdf_func, num_params_estimados=0, n_bins=12, alpha=0.05):
    """
    Ejecuta la prueba de bondad de ajuste Chi-cuadrada de Pearson agrupando automáticamente
    intervalos con frecuencia esperada < 5 (Regla de Cochran) y ajustando grados de libertad.
    """
    n = len(data)
    f_obs_raw, bin_edges = np.histogram(data, bins=n_bins)
    
    cdf_vals = cdf_func(bin_edges)
    probs = np.zeros(len(f_obs_raw), dtype=float)
    probs[0] = max(0.0, cdf_vals[1])
    for i in range(1, len(f_obs_raw) - 1):
        probs[i] = max(0.0, cdf_vals[i+1] - cdf_vals[i])
    probs[-1] = max(0.0, 1.0 - cdf_vals[-2])
    
    if np.sum(probs) <= 1e-12:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / np.sum(probs)
        
    f_exp_raw = n * probs
    
    f_obs_list, f_exp_list, labels_list = [], [], []
    obs_acum = 0
    exp_acum = 0.0
    left_edge = bin_edges[0]
    
    for i, (o, e) in enumerate(zip(f_obs_raw, f_exp_raw)):
        obs_acum += o
        exp_acum += e
        right_edge = bin_edges[i+1]
        if exp_acum >= 5.0:
            f_obs_list.append(obs_acum)
            f_exp_list.append(exp_acum)
            labels_list.append(f"[{left_edge:.2f}, {right_edge:.2f}]")
            obs_acum = 0
            exp_acum = 0.0
            left_edge = right_edge
            
    if exp_acum > 0 or obs_acum > 0:
        if len(f_exp_list) > 0:
            f_obs_list[-1] += obs_acum
            f_exp_list[-1] += exp_acum
            prev_left = labels_list[-1].split(",")[0]
            labels_list[-1] = f"{prev_left}, {bin_edges[-1]:.2f}]"
        else:
            f_obs_list.append(obs_acum)
            f_exp_list.append(max(exp_acum, 1e-6))
            labels_list.append(f"[{bin_edges[0]:.2f}, {bin_edges[-1]:.2f}]")
            
    f_obs = np.array(f_obs_list, dtype=float)
    f_exp = np.array(f_exp_list, dtype=float)
    f_exp = f_exp * (np.sum(f_obs) / np.sum(f_exp))
    
    k_clases = len(f_obs)
    gl = max(1, k_clases - 1 - num_params_estimados)
    
    aportes_chi2 = ((f_obs - f_exp) ** 2) / np.maximum(f_exp, 1e-9)
    chi2_stat = float(np.sum(aportes_chi2))
    chi2_pval = float(1.0 - stats.chi2.cdf(chi2_stat, df=gl))
    chi2_crit = float(stats.chi2.ppf(1.0 - alpha, df=gl))
    
    df_tabla = pd.DataFrame({
        "Intervalo / Clase": labels_list,
        "Frec. Observada (Oi)": f_obs.astype(int),
        "Frec. Esperada (Ei)": np.round(f_exp, 2),
        "Estadístico (Oi - Ei)² / Ei": np.round(aportes_chi2, 4)
    })
    
    return chi2_stat, chi2_pval, chi2_crit, gl, k_clases, f_obs, f_exp, labels_list, df_tabla

def calcular_anderson_darling(data, cdf_func):
    """
    Calcula el estadístico general de Anderson-Darling (A²) y su p-value asintótico
    (Marsaglia & Marsaglia, 2004), otorgando mayor peso a las colas de la distribución.
    """
    x_ord = np.sort(data)
    n = len(x_ord)
    eps = 1e-12
    u = np.clip(cdf_func(x_ord), eps, 1.0 - eps)
    i = np.arange(1, n + 1)
    S = np.sum(((2.0 * i - 1.0) / n) * (np.log(u) + np.log(1.0 - u[::-1])))
    A2 = float(max(0.0, -n - S))
    
    if A2 < 2.0:
        pval = 1.0 - np.exp(-1.2337141 / max(A2, 1e-8)) / np.sqrt(max(A2, 1e-8)) * (
            2.00012 + (0.247105 - (0.0649821 - (0.0347962 - (0.011672 - 0.00168691 * A2) * A2) * A2) * A2) * A2
        )
    else:
        pval = 1.0 - np.exp(-np.exp(1.0776 - (2.30695 - (0.43424 - (0.082433 - (0.008056 - 0.0003146 * A2) * A2) * A2) * A2) * A2))
    return A2, float(np.clip(pval, 0.0, 1.0))

def test_christoffersen(data, var_limit, alpha=0.05):
    """
    Evalúa la independencia de las excepciones del VaR (Christoffersen Markov Test)
    para detectar agrupamiento de volatilidad (clusters) y cobertura condicional.
    """
    hits = (data < var_limit).astype(int)
    n = len(hits)
    if n < 2:
        return 0.0, 1.0, 0.0, 1.0, (0, 0, 0, 0)
        
    h_prev, h_curr = hits[:-1], hits[1:]
    n00 = int(np.sum((h_prev == 0) & (h_curr == 0)))
    n01 = int(np.sum((h_prev == 0) & (h_curr == 1)))
    n10 = int(np.sum((h_prev == 1) & (h_curr == 0)))
    n11 = int(np.sum((h_prev == 1) & (h_curr == 1)))
    
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi_all = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    
    eps = 1e-12
    def safe_ll(c0, c1, p):
        p = np.clip(p, eps, 1.0 - eps)
        return c0 * np.log(1.0 - p) + c1 * np.log(p)
        
    ll_null = safe_ll(n00 + n10, n01 + n11, pi_all)
    ll_alt = safe_ll(n00, n01, pi0) + safe_ll(n10, n11, pi1)
    lr_ind = float(max(0.0, -2.0 * (ll_null - ll_alt)))
    pval_ind = float(1.0 - stats.chi2.cdf(lr_ind, df=1))
    
    x = int(np.sum(hits))
    ll_uc_null = safe_ll(n - x, x, alpha)
    ll_uc_alt = safe_ll(n - x, x, x / max(n, 1))
    lr_uc = float(max(0.0, -2.0 * (ll_uc_null - ll_uc_alt)))
    
    lr_cc = lr_uc + lr_ind
    pval_cc = float(1.0 - stats.chi2.cdf(lr_cc, df=2))
    return lr_ind, pval_ind, lr_cc, pval_cc, (n00, n01, n10, n11)

def ejecutar_bootstrap_completo(data, dist_obj, cdf_evaluar, ks_obs, alpha=0.05, n_boot=1000, seed=42):
    """
    1. Bootstrap No Paramétrico: Intervalos de confianza para Media, Volatilidad, VaR y CVaR.
    2. Bootstrap Paramétrico: Distribución empírica del estadístico D* de K-S bajo H0.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    
    # A. Bootstrap No Paramétrico (remuestreo con reemplazo de los datos observados)
    idx = rng.integers(0, n, size=(n_boot, n))
    muestras_np = data[idx]
    boot_medias = muestras_np.mean(axis=1)
    boot_stds = muestras_np.std(axis=1, ddof=1)
    boot_vars = np.percentile(muestras_np, alpha * 100.0, axis=1)
    
    mask_cvar = muestras_np <= boot_vars[:, None]
    boot_cvars = np.array([
        muestras_np[i, mask_cvar[i]].mean() if np.any(mask_cvar[i]) else boot_vars[i]
        for i in range(n_boot)
    ])
    
    # B. Bootstrap Paramétrico sobre el estadístico K-S (D*)
    n_boot_ks = min(n_boot, 1000)
    sim_samples = dist_obj.rvs(size=(n_boot_ks, n), random_state=rng)
    u_sim = np.sort(cdf_evaluar(sim_samples), axis=1)
    ecdf_plus = np.arange(1, n + 1) / n
    ecdf_minus = np.arange(0, n) / n
    d_plus = np.max(np.abs(ecdf_plus - u_sim), axis=1)
    d_minus = np.max(np.abs(u_sim - ecdf_minus), axis=1)
    d_boot = np.maximum(d_plus, d_minus)
    
    pval_boot_ks = float(np.mean(d_boot >= ks_obs))
    crit_boot_ks = float(np.percentile(d_boot, (1.0 - alpha) * 100.0))
    
    return boot_medias, boot_stds, boot_vars, boot_cvars, d_boot, pval_boot_ks, crit_boot_ks

st.markdown("""
<style>
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    .stApp { background-color: #0A0A0A; color: #E2E8F0; }
    h1, h2, h3 { color: #4FC3F7 !important; font-weight: 700; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    h4, h5 { color: #90CAF9 !important; font-weight: 500; }
    hr { border-top: 1px solid #1E293B; }
    .stTabs [data-baseweb="tab-list"] { background-color: #0F172A; border-bottom: 2px solid #1E293B; }
    .stTabs [data-baseweb="tab"] { color: #64748B; font-weight: 600; padding: 10px 18px; }
    .stTabs [aria-selected="true"] { border-bottom: 2px solid #00B4D8 !important; color: #00B4D8 !important; background-color: rgba(0, 180, 216, 0.05) !important; }
    .stButton > button { background-color: #0284C7 !important; color: #ffffff !important; border: 1px solid #0369A1; font-weight: bold; width: 100%; border-radius: 6px; transition: 0.3s; }
    .stButton > button:hover { background-color: #0369A1 !important; box-shadow: 0 0 10px rgba(2, 132, 199, 0.5); }
    .well { background-color: #111827; border-left: 4px solid #00B4D8; padding: 15px; border-radius: 4px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .metric-card { background-color: #111827; border: 1px solid #1F2937; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.4); }
    .alert-danger { background-color: rgba(153, 27, 27, 0.2); color: #FCA5A5; border: 1px solid #7F1D1D; padding: 15px; border-radius: 6px; margin-top: 15px; }
    .alert-success { background-color: rgba(6, 78, 59, 0.2); color: #6EE7B7; border: 1px solid #064E3B; padding: 15px; border-radius: 6px; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h2>Análisis Estocástico y Simulación Monte Carlo</h2>", unsafe_allow_html=True)
st.markdown("<h5>Evaluación de Riesgos, Inferencia Bootstrap y Modelado de Series Financieras</h5>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ==============================================================================
# 1. INTERFAZ DE USUARIO (BARRA LATERAL)
# ==============================================================================
df = None
p0_default = 100.0
data = np.array([])
serie_precios_hist = None
serie_rend_hist = None
unidad_var = "%"
etiqueta_var = "Rendimiento (%)"

with st.sidebar:
    st.markdown("<h4 style='color: #4FC3F7;'>1. Carga de Datos</h4>", unsafe_allow_html=True)
    origen_datos = st.radio(
        "Fuente de información:",
        ["Conexión Bursátil (Tiempo Real)", "Cargar Archivo CSV", "Generar Datos Simulados"]
    )
    
    if origen_datos == "Conexión Bursátil (Tiempo Real)":
        catalogo_tickers = {
            "Palantir Technologies (PLTR)": "PLTR",
            "NVIDIA Corp. (NVDA)": "NVDA",
            "Apple Inc. (AAPL)": "AAPL",
            "Microsoft Corp. (MSFT)": "MSFT",
            "Tesla Inc. (TSLA)": "TSLA",
            "Amazon.com Inc. (AMZN)": "AMZN",
            "Alphabet Inc. (GOOGL)": "GOOGL",
            "Meta Platforms (META)": "META",
            "S&P 500 ETF (SPY)": "SPY",
            "Bitcoin USD (BTC-USD)": "BTC-USD",
            "Otro Ticker personalizado...": "CUSTOM"
        }
        seleccion_activo = st.selectbox("Seleccione el activo financiero:", list(catalogo_tickers.keys()), index=0)
        
        if catalogo_tickers[seleccion_activo] == "CUSTOM":
            ticker_activo = st.text_input("Ingrese el Ticker (ej. AMD, NFLX, WALMEX.MX):", value="AMD").strip().upper()
        else:
            ticker_activo = catalogo_tickers[seleccion_activo]
            
        anios_hist = st.slider(
            "Temporalidad histórica hacia atrás (Años):",
            min_value=1, max_value=5, value=3, step=1,
            help="Extrae los cierres diarios desde hace N años hasta el día de hoy."
        )
        
        tipo_variable_mkt = st.radio(
            "Variable a analizar en las pruebas de ajuste:",
            ["Rendimientos Diarios (%)", "Precio de Cierre ($)"],
            index=0
        )
        
        if ticker_activo:
            with st.spinner(f"Descargando historial de {ticker_activo}..."):
                precios_s, rend_s, spot_mkt, f_ini, f_fin = descargar_datos_financieros(ticker_activo, anios_hist)
                if precios_s is not None:
                    serie_precios_hist = precios_s
                    serie_rend_hist = rend_s
                    p0_default = spot_mkt
                    if tipo_variable_mkt == "Precio de Cierre ($)":
                        data = precios_s.values
                        unidad_var = "$"
                        etiqueta_var = "Precio de Cierre ($)"
                    else:
                        data = rend_s.values
                        unidad_var = "%"
                        etiqueta_var = "Rendimiento (%)"
                    st.success(f"**{ticker_activo}** conectado ({len(data)} observaciones)")
                    st.caption(f"📅 **Periodo:** {f_ini} al {f_fin}  \n💵 **Último Cierre (Spot):** ${p0_default:,.2f}")
                else:
                    st.error(f"No se encontraron datos para el ticker '{ticker_activo}'. Verifique el símbolo.")

    elif origen_datos == "Generar Datos Simulados":
        n_sim = st.number_input("Tamaño muestral (n):", min_value=10, max_value=5000, value=500, step=1)
        gl_sim = st.number_input("Grados de libertad (df):", min_value=1, value=7, step=1)
        st.caption("Datos generados mediante distribución Chi-cuadrada.")
        np.random.seed(42)
        data = np.random.chisquare(df=gl_sim, size=n_sim)
        serie_rend_hist = pd.Series(data)
        unidad_var = "u"
        etiqueta_var = "Variable Simulada (X)"
        
    else:
        file_csv = st.file_uploader("Seleccione archivo CSV:", type=["csv"])
        sep_csv = st.selectbox("Separador de columnas:", [",", ";"])
        col_csv = None
        if file_csv is not None:
            try:
                df = pd.read_csv(file_csv, sep=sep_csv)
                # Intentar construir serie histórica de precios si el CSV tiene Price y Date
                if 'Price' in df.columns:
                    p0_default = float(str(df['Price'].iloc[0]).replace(',', ''))
                    df_temp = df.copy()
                    if 'Date' in df_temp.columns:
                        df_temp['Date_dt'] = pd.to_datetime(df_temp['Date'], errors='coerce')
                        df_temp = df_temp.dropna(subset=['Date_dt']).sort_values('Date_dt')
                        idx_fechas = df_temp['Date_dt']
                    else:
                        idx_fechas = pd.RangeIndex(len(df_temp))
                    precios_csv = pd.to_numeric(df_temp['Price'].astype(str).str.replace(',', ''), errors='coerce')
                    serie_precios_hist = pd.Series(precios_csv.values, index=idx_fechas).dropna()
                    if len(serie_precios_hist) >= 2:
                        serie_rend_hist = serie_precios_hist.pct_change().dropna() * 100.0

                modo_col_csv = st.radio(
                    "Selección rápida de variable en CSV:",
                    ["Rendimientos (Change %)", "Precio de Cierre (Price)", "Seleccionar otra columna"],
                    index=0
                )
                if modo_col_csv == "Rendimientos (Change %)" and "Change %" in df.columns:
                    col_csv = "Change %"
                    unidad_var = "%"
                    etiqueta_var = "Rendimiento (%)"
                elif modo_col_csv == "Precio de Cierre (Price)" and "Price" in df.columns:
                    col_csv = "Price"
                    unidad_var = "$"
                    etiqueta_var = "Precio de Cierre ($)"
                else:
                    default_idx = df.columns.get_loc("Change %") if "Change %" in df.columns else (6 if len(df.columns) >= 7 else 0)
                    col_csv = st.selectbox("Columna del CSV:", options=df.columns, index=default_idx)
                    unidad_var = "$" if col_csv == "Price" else "%"
                    etiqueta_var = f"Variable ({col_csv})"
                    
                if col_csv is not None:
                    col_limpia = df[col_csv].astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False).str.replace('$', '', regex=False)
                    data = pd.to_numeric(col_limpia, errors='coerce').dropna().values
                    if serie_rend_hist is None and len(data) >= 2:
                        serie_rend_hist = pd.Series(data)
            except Exception as e:
                st.error(f"Error de lectura: {e}")
        st.caption("Auto-limpieza de caracteres especiales (%, $) activada.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #4FC3F7;'>2. Especificación del Modelo</h4>", unsafe_allow_html=True)
    
    dist_map = {
        "t-Student (Continua)": "t",
        "Normal / Gaussiana (Continua)": "norm",
        "Lognormal (Continua)": "lnorm",
        "Laplace (Continua)": "laplace",
        "Exponencial (Continua)": "exp",
        "Chi-cuadrada (Continua)": "chisq",
        "Erlang (Continua)": "erlang",
        "Poisson (Discreta)": "pois",
        "Geométrica (Discreta)": "geom",
        "Binomial (Discreta)": "binom",
        "Bernoulli (Discreta)": "bernoulli",
        "Binomial Negativa (Discreta)": "nbinom",
        "Hipergeométrica (Discreta)": "hypergeom",
        "Uniforme Discreta": "randint",
        "Multinomial (Discreta)": "multinomial"
    }
    dist_sel = st.selectbox("Distribución Teórica:", list(dist_map.keys()), index=0) 
    dist_test = dist_map[dist_sel]
    
    alpha = st.selectbox("Nivel de Confianza (α):", [0.10, 0.05, 0.01], index=1, format_func=lambda x: f"{(1-x)*100:.0f}% (α={x})")
    n_bins_chi2 = st.slider("Intervalos iniciales (Prueba Chi²):", min_value=6, max_value=30, value=12, step=1)
    criterio_desbloqueo = st.selectbox(
        "Criterio para desbloquear Monte Carlo:",
        [
            "Al menos una prueba (K-S, Chi² o AD)",
            "Solo Prueba K-S",
            "Solo Prueba Chi-cuadrada",
            "Solo Prueba Anderson-Darling",
            "Todas las pruebas simultáneamente"
        ],
        index=0
    )
    modo_estimacion = st.radio("Método de parametrización:", ["Automático (MLE)", "Manual"], horizontal=True)
    
    params_manuales = {}
    if modo_estimacion == "Manual":
        with st.expander("Parámetros Manuales", expanded=True):
            if dist_test == "t":
                params_manuales['df_t'] = st.number_input("Grados de libertad (ν):", min_value=0.1, value=3.0, step=0.5)
                params_manuales['loc_t'] = st.number_input("Ubicación (loc):", value=0.0, step=0.1)
                params_manuales['scale_t'] = st.number_input("Escala (scale):", min_value=0.0001, value=2.5, step=0.1)
            elif dist_test == "norm":
                params_manuales['mean'] = st.number_input("Media (μ):", value=0.0, step=0.1)
                params_manuales['sd'] = st.number_input("Desviación (σ):", min_value=0.0001, value=2.0, step=0.1)
            elif dist_test == "lnorm":
                params_manuales['s'] = st.number_input("Forma (s / sdlog):", min_value=0.0001, value=1.0, step=0.1)
                params_manuales['loc'] = st.number_input("Ubicación (loc):", value=0.0, step=0.1)
                params_manuales['scale'] = st.number_input("Escala (scale):", min_value=0.0001, value=1.0, step=0.1)
            elif dist_test == "laplace":
                params_manuales['loc'] = st.number_input("Ubicación (loc):", value=0.0, step=0.1)
                params_manuales['scale'] = st.number_input("Escala (scale):", min_value=0.0001, value=2.0, step=0.1)
            elif dist_test == "exp":
                params_manuales['loc'] = st.number_input("Ubicación (loc):", value=0.0, step=0.1)
                params_manuales['scale'] = st.number_input("Escala (1/λ):", min_value=0.0001, value=5.0, step=0.5)
            elif dist_test == "chisq":
                params_manuales['df'] = st.number_input("Grados de libertad (df):", min_value=0.1, value=7.0, step=1.0)
                params_manuales['loc'] = st.number_input("Ubicación (loc):", value=0.0, step=0.1)
                params_manuales['scale'] = st.number_input("Escala (scale):", min_value=0.0001, value=1.0, step=0.1)
            elif dist_test == "erlang":
                params_manuales['k_er'] = st.number_input("Forma entera (k):", min_value=1, value=3, step=1)
                params_manuales['loc_er'] = st.number_input("Ubicación (loc):", value=0.0, step=0.1)
                params_manuales['scale_er'] = st.number_input("Escala (1/λ):", min_value=0.0001, value=2.0, step=0.1)
            elif dist_test == "pois":
                params_manuales['lambda'] = st.number_input("Tasa de ocurrencia (λ):", min_value=0.0001, value=4.0, step=0.5)
            elif dist_test == "geom":
                params_manuales['p_geom'] = st.number_input("Probabilidad de éxito (p):", min_value=0.0001, max_value=1.0, value=0.25, step=0.05)
            elif dist_test == "binom":
                params_manuales['n_binom'] = st.number_input("Número de ensayos (n):", min_value=1, value=10, step=1)
                params_manuales['p_binom'] = st.number_input("Probabilidad de éxito (p):", min_value=0.0001, max_value=1.0, value=0.5, step=0.05)
            elif dist_test == "bernoulli":
                params_manuales['p_bern'] = st.number_input("Probabilidad de éxito (p):", min_value=0.0001, max_value=0.9999, value=0.5, step=0.05)
            elif dist_test == "nbinom":
                params_manuales['r_nb'] = st.number_input("Éxitos requeridos (r):", min_value=1, value=5, step=1)
                params_manuales['p_nb'] = st.number_input("Probabilidad de éxito (p):", min_value=0.0001, max_value=0.9999, value=0.5, step=0.05)
            elif dist_test == "hypergeom":
                params_manuales['M_hyp'] = st.number_input("Población total (M):", min_value=2, value=50, step=5)
                params_manuales['K_hyp'] = st.number_input("Éxitos en la población (K):", min_value=1, max_value=int(params_manuales['M_hyp']), value=min(20, int(params_manuales['M_hyp'])), step=1)
                params_manuales['N_hyp'] = st.number_input("Tamaño de extracción (N):", min_value=1, max_value=int(params_manuales['M_hyp']), value=min(10, int(params_manuales['M_hyp'])), step=1)
            elif dist_test == "randint":
                params_manuales['low_ud'] = st.number_input("Cota inferior inclusiva (a):", value=-5, step=1)
                params_manuales['high_ud'] = st.number_input("Cota superior inclusiva (b):", min_value=int(params_manuales['low_ud']) + 1, value=max(5, int(params_manuales['low_ud']) + 1), step=1)
            elif dist_test == "multinomial":
                params_manuales['n_multi'] = st.number_input("Número de ensayos (n):", min_value=1, value=1, step=1)
                params_manuales['p_multi_str'] = st.text_input("Probabilidades (separadas por coma):", value="0.2, 0.5, 0.3")
                
    st.markdown("<br>", unsafe_allow_html=True)
    ejecutar = st.button("▶ Procesar Modelo Matemático")

# ==============================================================================
# 2. ESTIMACIÓN DE PARÁMETROS DEL MODELO
# ==============================================================================
dist_obj = None
fit_params = ()
num_params_est = 0
es_discreta = dist_test in ["pois", "geom", "binom", "bernoulli", "nbinom", "hypergeom", "randint", "multinomial"]

if len(data) >= 2:
    if modo_estimacion == "Automático (MLE)":
        if dist_test == "t":
            fit_params = stats.t.fit(data)
            dist_obj = stats.t(*fit_params)
            num_params_est = 3
        elif dist_test == "norm":
            fit_params = stats.norm.fit(data)
            dist_obj = stats.norm(*fit_params)
            num_params_est = 2
        elif dist_test == "lnorm":
            shift = abs(min(data)) + 1e-4 if min(data) <= 0 else 0
            fit_params = stats.lognorm.fit(data + shift)
            dist_obj = stats.lognorm(*fit_params)
            num_params_est = 3
        elif dist_test == "laplace":
            fit_params = stats.laplace.fit(data)
            dist_obj = stats.laplace(*fit_params)
            num_params_est = 2
        elif dist_test == "exp":
            fit_params = stats.expon.fit(data)
            dist_obj = stats.expon(*fit_params)
            num_params_est = 2
        elif dist_test == "chisq":
            fit_params = stats.chi2.fit(data)
            dist_obj = stats.chi2(*fit_params)
            num_params_est = 3
        elif dist_test == "erlang":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a_g, loc_g, scale_g = stats.gamma.fit(data)
                k_er = max(int(round(a_g)), 1)
                _, loc_er, scale_er = stats.gamma.fit(data, fa=k_er)
            fit_params = (k_er, loc_er, scale_er)
            dist_obj = stats.erlang(a=k_er, loc=loc_er, scale=scale_er)
            num_params_est = 3
        elif dist_test == "pois":
            lam_est = float(max(np.mean(data), 0.0001))
            fit_params = (lam_est,)
            dist_obj = stats.poisson(mu=lam_est)
            num_params_est = 1
        elif dist_test == "geom":
            media_pos = max(float(np.mean(data)), 1.0001)
            p_est = float(np.clip(1.0 / media_pos, 0.0001, 1.0))
            fit_params = (p_est,)
            dist_obj = stats.geom(p=p_est)
            num_params_est = 1
        elif dist_test == "binom":
            media_b = max(float(np.mean(data)), 0.01)
            var_b = float(np.var(data, ddof=1))
            if var_b < media_b and media_b > 0:
                p_est = 1.0 - (var_b / media_b)
                n_est = max(int(round(media_b / p_est)), 1)
                p_est = float(np.clip(media_b / n_est, 0.0001, 0.9999))
            else:
                n_est = max(int(round(max(data))), 1)
                p_est = float(np.clip(media_b / n_est, 0.0001, 0.9999))
            fit_params = (n_est, p_est)
            dist_obj = stats.binom(n=n_est, p=p_est)
            num_params_est = 2
        elif dist_test == "bernoulli":
            prop_pos = np.mean(data > 0) if (np.min(data) < 0 or np.max(data) > 1) else np.mean(data)
            p_bern = float(np.clip(prop_pos, 0.0001, 0.9999))
            fit_params = (p_bern,)
            dist_obj = stats.bernoulli(p=p_bern)
            num_params_est = 1
        elif dist_test == "nbinom":
            media_nb = max(float(np.mean(np.maximum(data, 0))), 0.01)
            var_nb = float(np.var(data, ddof=1))
            if var_nb > media_nb:
                p_nb = float(np.clip(media_nb / var_nb, 0.0001, 0.9999))
                r_nb = max(int(round((media_nb ** 2) / (var_nb - media_nb))), 1)
            else:
                p_nb = 0.5
                r_nb = max(int(round(media_nb)), 1)
            fit_params = (r_nb, p_nb)
            dist_obj = stats.nbinom(n=r_nb, p=p_nb)
            num_params_est = 2
        elif dist_test == "hypergeom":
            N_hyp = max(int(round(np.max(np.abs(data)))), 10)
            M_hyp = N_hyp * 2
            media_pos = float(np.clip(np.mean(np.maximum(data, 0)), 1.0, N_hyp - 1))
            K_hyp = int(np.clip(round(media_pos * (M_hyp / N_hyp)), 1, M_hyp - 1))
            fit_params = (M_hyp, K_hyp, N_hyp)
            dist_obj = stats.hypergeom(M=M_hyp, n=K_hyp, N=N_hyp)
            num_params_est = 3
        elif dist_test == "randint":
            low_ud = int(np.floor(np.min(data)))
            high_ud = int(np.ceil(np.max(data)))
            if high_ud <= low_ud:
                high_ud = low_ud + 1
            fit_params = (low_ud, high_ud)
            dist_obj = stats.randint(low=low_ud, high=high_ud + 1)
            num_params_est = 2
        elif dist_test == "multinomial":
            k_cat = 5
            datos_cat = np.clip(np.round(data[data >= 0]), 0, k_cat - 1).astype(int) if np.any(data >= 0) else np.zeros(10, dtype=int)
            conteos = np.bincount(datos_cat, minlength=k_cat)[:k_cat] + 1e-4
            dist_obj, pk_norm = construir_multinomial_1d(1, conteos)
            fit_params = (1, k_cat, np.round(pk_norm, 4).tolist())
            num_params_est = k_cat - 1
    else:
        num_params_est = 0
        if dist_test == "t":
            dist_obj = stats.t(df=params_manuales['df_t'], loc=params_manuales['loc_t'], scale=params_manuales['scale_t'])
        elif dist_test == "norm":
            dist_obj = stats.norm(loc=params_manuales['mean'], scale=params_manuales['sd'])
        elif dist_test == "lnorm":
            dist_obj = stats.lognorm(s=params_manuales['s'], loc=params_manuales['loc'], scale=params_manuales['scale'])
        elif dist_test == "laplace":
            dist_obj = stats.laplace(loc=params_manuales['loc'], scale=params_manuales['scale'])
        elif dist_test == "exp":
            dist_obj = stats.expon(loc=params_manuales['loc'], scale=params_manuales['scale'])
        elif dist_test == "chisq":
            dist_obj = stats.chi2(df=params_manuales['df'], loc=params_manuales['loc'], scale=params_manuales['scale'])
        elif dist_test == "erlang":
            dist_obj = stats.erlang(a=int(params_manuales['k_er']), loc=params_manuales['loc_er'], scale=params_manuales['scale_er'])
        elif dist_test == "pois":
            dist_obj = stats.poisson(mu=params_manuales['lambda'])
        elif dist_test == "geom":
            dist_obj = stats.geom(p=params_manuales['p_geom'])
        elif dist_test == "binom":
            dist_obj = stats.binom(n=int(params_manuales['n_binom']), p=params_manuales['p_binom'])
        elif dist_test == "bernoulli":
            dist_obj = stats.bernoulli(p=params_manuales['p_bern'])
        elif dist_test == "nbinom":
            dist_obj = stats.nbinom(n=int(params_manuales['r_nb']), p=params_manuales['p_nb'])
        elif dist_test == "hypergeom":
            dist_obj = stats.hypergeom(M=int(params_manuales['M_hyp']), n=int(params_manuales['K_hyp']), N=int(params_manuales['N_hyp']))
        elif dist_test == "randint":
            dist_obj = stats.randint(low=int(params_manuales['low_ud']), high=int(params_manuales['high_ud']) + 1)
        elif dist_test == "multinomial":
            try:
                p_list = [float(x.strip()) for x in params_manuales['p_multi_str'].split(",") if x.strip()]
                if len(p_list) < 2:
                    p_list = [0.5, 0.5]
            except Exception:
                p_list = [0.2, 0.5, 0.3]
            dist_obj, _ = construir_multinomial_1d(int(params_manuales['n_multi']), p_list)

# ==============================================================================
# 3. RENDERIZADO DEL PANEL PRINCIPAL
# ==============================================================================
if len(data) >= 2:
    # 1. Prueba Kolmogorov-Smirnov (K-S)
    cdf_evaluar = (lambda x: dist_obj.cdf(np.floor(x))) if es_discreta else dist_obj.cdf
    ks_stat, ks_pval = stats.kstest(data, cdf_evaluar)
    
    # 2. Prueba Chi-cuadrada de Bondad de Ajuste
    chi2_stat, chi2_pval, chi2_crit, chi2_gl, chi2_k, f_obs_c, f_exp_c, labels_c, df_chi2 = calcular_bondad_chi2(
        data, cdf_evaluar, num_params_estimados=num_params_est, n_bins=n_bins_chi2, alpha=alpha
    )
    
    # 3. Prueba de Anderson-Darling (A²)
    ad_stat, ad_pval = calcular_anderson_darling(data, cdf_evaluar)
    
    # Evaluar condición de desbloqueo para Monte Carlo
    pasa_ks = ks_pval >= alpha
    pasa_chi2 = chi2_pval >= alpha
    pasa_ad = ad_pval >= alpha
    
    if criterio_desbloqueo == "Al menos una prueba (K-S, Chi² o AD)":
        modelo_validado = pasa_ks or pasa_chi2 or pasa_ad
    elif criterio_desbloqueo == "Solo Prueba K-S":
        modelo_validado = pasa_ks
    elif criterio_desbloqueo == "Solo Prueba Chi-cuadrada":
        modelo_validado = pasa_chi2
    elif criterio_desbloqueo == "Solo Prueba Anderson-Darling":
        modelo_validado = pasa_ad
    else:
        modelo_validado = pasa_ks and pasa_chi2 and pasa_ad
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Diagnóstico",
        "📜 Bondad de Ajuste (K-S, Chi², AD)",
        "📉 Precios, Volatilidad & Estadística",
        "🔄 Prueba Bootstrap",
        "🔮 Monte Carlo & GBM",
        "📈 Backtesting (Kupiec & Christoffersen)"
    ])
    
    plt.style.use('dark_background')
    plt.rcParams.update({
        'axes.facecolor': '#0A0A0A', 'figure.facecolor': '#0A0A0A', 
        'text.color': '#E2E8F0', 'axes.labelcolor': '#90CAF9',
        'xtick.color': '#64748B', 'ytick.color': '#64748B',
        'axes.edgecolor': '#1E293B', 'axes.titlesize': 13,
        'axes.titleweight': 'bold', 'axes.titlecolor': '#4FC3F7',
        'grid.color': '#1E293B', 'grid.linestyle': '--'
    })

    # --------------------------------------------------------------------------
    # TAB 1: DIAGNÓSTICO GRÁFICO
    # --------------------------------------------------------------------------
    with tab1:
        st.write("")
        c1, c2 = st.columns(2)
        x_min, x_max = min(data), max(data)
        x_grid = np.linspace(x_min - (x_max - x_min)*0.1, x_max + (x_max - x_min)*0.1, 500)
        
        with c1:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.hist(data, bins=30, density=True, color="#075985", edgecolor="#0284C7", alpha=0.7, label="Empírica")
            
            if es_discreta:
                x_disc = np.unique(np.round(x_grid)).astype(int)
                y_prob = dist_obj.pmf(x_disc)
                ax1.plot(x_disc, y_prob, color="#00E5FF", lw=2, marker="o", markersize=4, label="Teórica (PMF)")
                ax1.set_title(f"Función de Masa (PMF) - {dist_sel}")
            else:
                ax1.plot(x_grid, dist_obj.pdf(x_grid), color="#00E5FF", lw=2.5, label="Teórica (PDF)")
                ax1.set_title(f"Función de Densidad (PDF) - {dist_sel}")
                
            ax1.set_xlabel(etiqueta_var)
            ax1.legend(frameon=False)
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            
        with c2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.step(np.sort(data), np.arange(1, len(data) + 1)/len(data), color="#0284C7", lw=2, label="Fn(x) Empírica")
            ax2.plot(x_grid, cdf_evaluar(x_grid), color="#00E5FF", lw=2, linestyle="--", label="F(x) Teórica")
            ax2.set_title("Función de Distribución Acumulada (FDA)")
            ax2.set_xlabel(etiqueta_var)
            ax2.legend(frameon=False)
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            
        fig_chi, ax_chi = plt.subplots(figsize=(12, 3.2))
        idx_clases = np.arange(len(labels_c))
        ancho_barra = 0.38
        ax_chi.bar(idx_clases - ancho_barra/2, f_obs_c, width=ancho_barra, color="#0284C7", edgecolor="#4FC3F7", label="Frecuencia Observada (Oi)")
        ax_chi.bar(idx_clases + ancho_barra/2, f_exp_c, width=ancho_barra, color="#1E293B", edgecolor="#00E5FF", hatch="//", label="Frecuencia Esperada (Ei)")
        ax_chi.set_xticks(idx_clases)
        ax_chi.set_xticklabels(labels_c, rotation=20, ha="right", fontsize=9)
        ax_chi.set_title(f"Contraste de Frecuencias por Clase Agrupada (Prueba Chi-cuadrada - {chi2_k} clases)")
        ax_chi.set_ylabel("Frecuencia (Datos)")
        ax_chi.legend(frameon=False)
        ax_chi.grid(True, alpha=0.2)
        st.pyplot(fig_chi)
            
        if modo_estimacion == "Automático (MLE)":
            params_str = ""
            if dist_test == "t":
                params_str = f"<b>Grados de libertad (ν):</b> {float(fit_params[0]):.4f} &nbsp;|&nbsp; <b>Ubicación (loc):</b> {float(fit_params[1]):.4f} &nbsp;|&nbsp; <b>Escala (scale):</b> {float(fit_params[2]):.4f}"
            elif dist_test == "norm":
                params_str = f"<b>Media (μ):</b> {float(fit_params[0]):.4f} &nbsp;|&nbsp; <b>Desviación (σ):</b> {float(fit_params[1]):.4f}"
            elif dist_test == "lnorm":
                params_str = f"<b>Forma (s):</b> {float(fit_params[0]):.4f} &nbsp;|&nbsp; <b>Ubicación (loc):</b> {float(fit_params[1]):.4f} &nbsp;|&nbsp; <b>Escala (scale):</b> {float(fit_params[2]):.4f}"
            elif dist_test in ["laplace", "exp"]:
                params_str = f"<b>Ubicación (loc):</b> {float(fit_params[0]):.4f} &nbsp;|&nbsp; <b>Escala (scale):</b> {float(fit_params[1]):.4f}"
            elif dist_test == "chisq":
                params_str = f"<b>Grados de libertad (df):</b> {float(fit_params[0]):.4f} &nbsp;|&nbsp; <b>Ubicación (loc):</b> {float(fit_params[1]):.4f} &nbsp;|&nbsp; <b>Escala (scale):</b> {float(fit_params[2]):.4f}"
            elif dist_test == "erlang":
                params_str = f"<b>Forma entera (k):</b> {int(fit_params[0])} &nbsp;|&nbsp; <b>Ubicación (loc):</b> {float(fit_params[1]):.4f} &nbsp;|&nbsp; <b>Escala (scale):</b> {float(fit_params[2]):.4f}"
            elif dist_test == "pois":
                params_str = f"<b>Tasa de ocurrencia (λ):</b> {float(fit_params[0]):.4f}"
            elif dist_test in ["geom", "bernoulli"]:
                params_str = f"<b>Probabilidad de éxito (p):</b> {float(fit_params[0]):.4f}"
            elif dist_test == "binom":
                params_str = f"<b>Número de ensayos (n):</b> {int(fit_params[0])} &nbsp;|&nbsp; <b>Probabilidad de éxito (p):</b> {float(fit_params[1]):.4f}"
            elif dist_test == "nbinom":
                params_str = f"<b>Éxitos requeridos (r):</b> {int(fit_params[0])} &nbsp;|&nbsp; <b>Probabilidad de éxito (p):</b> {float(fit_params[1]):.4f}"
            elif dist_test == "hypergeom":
                params_str = f"<b>Población total (M):</b> {int(fit_params[0])} &nbsp;|&nbsp; <b>Éxitos en población (K):</b> {int(fit_params[1])} &nbsp;|&nbsp; <b>Extracciones (N):</b> {int(fit_params[2])}"
            elif dist_test == "randint":
                params_str = f"<b>Cota inferior inclusiva (a):</b> {int(fit_params[0])} &nbsp;|&nbsp; <b>Cota superior inclusiva (b):</b> {int(fit_params[1])}"
            elif dist_test == "multinomial":
                params_str = f"<b>Ensayos (n):</b> {int(fit_params[0])} &nbsp;|&nbsp; <b>Categorías (k):</b> {int(fit_params[1])} &nbsp;|&nbsp; <b>Vector de Probabilidades (p):</b> {fit_params[2]}"
            
            st.markdown(f"""<div class="well"><b>⚙️ Parámetros Optimizados ({etiqueta_var}):</b><br>{params_str}</div>""", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # TAB 2: REPORTE TRIPLE (K-S, CHI-CUADRADA Y ANDERSON-DARLING)
    # --------------------------------------------------------------------------
    with tab2:
        st.markdown("<br><h4>Dictamen Triple de Bondad de Ajuste</h4>", unsafe_allow_html=True)
        col_rep1, col_rep2, col_rep3 = st.columns(3)
        
        with col_rep1:
            decision_ks = "RECHAZAR H0" if not pasa_ks else "NO RECHAZAR H0"
            st.code(f"""
=============================================
  1. KOLMOGOROV-SMIRNOV (K-S)
=============================================
  Variable     : {etiqueta_var}
  Modelo       : {dist_sel}
  Muestra (n)  : {len(data)}
  Alpha (α)    : {alpha}

  Estadístico D: {ks_stat:.6f}
  P-Value      : {ks_pval:.6e}
  
  Veredicto    : {decision_ks}
=============================================
            """, language='text')
            if not pasa_ks:
                st.markdown(f'<div class="alert-danger"><b>❌ K-S RECHAZA H0:</b> p-value ({ks_pval:.4g}) &lt; α ({alpha}).</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-success"><b>✅ K-S VALIDA H0:</b> p-value ({ks_pval:.4g}) ≥ α ({alpha}).</div>', unsafe_allow_html=True)

        with col_rep2:
            decision_chi2 = "RECHAZAR H0" if not pasa_chi2 else "NO RECHAZAR H0"
            st.code(f"""
=============================================
  2. CHI-CUADRADA DE PEARSON (χ²)
=============================================
  Clases (k)   : {chi2_k} (Ei ≥ 5)
  Parám. (m)   : {num_params_est}
  Grados (gl)  : {chi2_gl}
  Crítico χ²   : {chi2_crit:.4f}

  Estadístico  : {chi2_stat:.4f}
  P-Value      : {chi2_pval:.6e}
  
  Veredicto    : {decision_chi2}
=============================================
            """, language='text')
            if not pasa_chi2:
                st.markdown(f'<div class="alert-danger"><b>❌ CHI² RECHAZA H0:</b> χ² ({chi2_stat:.2f}) &gt; {chi2_crit:.2f} (p={chi2_pval:.4g}).</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-success"><b>✅ CHI² VALIDA H0:</b> χ² ({chi2_stat:.2f}) ≤ {chi2_crit:.2f} (p={chi2_pval:.4g}).</div>', unsafe_allow_html=True)

        with col_rep3:
            decision_ad = "RECHAZAR H0" if not pasa_ad else "NO RECHAZAR H0"
            st.code(f"""
=============================================
  3. ANDERSON-DARLING (A²)
=============================================
  Énfasis      : Colas de la distribución
  Muestra (n)  : {len(data)}
  Alpha (α)    : {alpha}

  Estadístico A²: {ad_stat:.4f}
  P-Value       : {ad_pval:.6e}
  
  Veredicto     : {decision_ad}
=============================================
            """, language='text')
            if not pasa_ad:
                st.markdown(f'<div class="alert-danger"><b>❌ A-D RECHAZA H0:</b> Discrepancia severa en las colas (p={ad_pval:.4g}).</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-success"><b>✅ A-D VALIDA H0:</b> Excelente ajuste en colas y centro (p={ad_pval:.4g}).</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 Ver Tabla Desglosada de Clases y Frecuencias (Prueba Chi-cuadrada)", expanded=False):
            st.dataframe(df_chi2, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 3: COMPORTAMIENTO DE PRECIOS, VOLATILIDAD EWMA Y ESTADÍSTICA
    # --------------------------------------------------------------------------
    with tab3:
        st.markdown(f"<br><h4>Métricas Descriptivas ({etiqueta_var})</h4>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div style="color:#64748B;">Media Muestral (μ)</div><div style="font-size:24px; color:#4FC3F7; font-weight:bold;">{data.mean():.2f}{unidad_var}</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div style="color:#64748B;">Desviación Estándar (σ)</div><div style="font-size:24px; color:#4FC3F7; font-weight:bold;">{data.std():.2f}{unidad_var}</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div style="color:#64748B;">Asimetría (Skewness)</div><div style="font-size:24px; color:#4FC3F7; font-weight:bold;">{stats.skew(data):.2f}</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div style="color:#64748B;">Curtosis (Exceso)</div><div style="font-size:24px; color:#4FC3F7; font-weight:bold;">{stats.kurtosis(data):.2f}</div></div>', unsafe_allow_html=True)
        
        if serie_precios_hist is not None and len(serie_precios_hist) >= 5:
            st.markdown("<br><h4>Comportamiento Histórico de Precios, Bandas de Volatilidad y Drawdown</h4>", unsafe_allow_html=True)
            
            # Cálculos financieros sobre la serie histórica de precios
            sma20 = serie_precios_hist.rolling(window=20, min_periods=1).mean()
            std20 = serie_precios_hist.rolling(window=20, min_periods=1).std().fillna(0)
            bollinger_sup = sma20 + 2.0 * std20
            bollinger_inf = sma20 - 2.0 * std20
            
            cum_max = serie_precios_hist.cummax()
            drawdown_s = ((serie_precios_hist - cum_max) / cum_max) * 100.0
            max_dd = float(drawdown_s.min())
            
            rend_calc = serie_precios_hist.pct_change().dropna() * 100.0
            ewma_vol = np.sqrt((rend_calc ** 2).ewm(alpha=1.0 - 0.94, adjust=False).mean())
            sharpe_anual = float((rend_calc.mean() / rend_calc.std()) * np.sqrt(252)) if rend_calc.std() > 0 else 0.0
            retorno_acum = float(((serie_precios_hist.iloc[-1] / serie_precios_hist.iloc[0]) - 1.0) * 100.0)
            
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f'<div class="metric-card"><div style="color:#64748B;">Rendimiento Acumulado</div><div style="font-size:22px; color:#00FF66; font-weight:bold;">{retorno_acum:+.2f}%</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="metric-card"><div style="color:#64748B;">Ratio de Sharpe (Anual)</div><div style="font-size:22px; color:#4FC3F7; font-weight:bold;">{sharpe_anual:.2f}</div></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="metric-card"><div style="color:#64748B;">Maximum Drawdown</div><div style="font-size:22px; color:#EF4444; font-weight:bold;">{max_dd:.2f}%</div></div>', unsafe_allow_html=True)
            k4.markdown(f'<div class="metric-card"><div style="color:#64748B;">Volatilidad EWMA Actual</div><div style="font-size:22px; color:#90CAF9; font-weight:bold;">{ewma_vol.iloc[-1]:.2f}%</div></div>', unsafe_allow_html=True)
            
            fig_p, (ax_p1, ax_p2) = plt.subplots(2, 1, figsize=(12, 6.5), gridspec_kw={'height_ratios': [2, 1]})
            
            # Panel 1: Precio de Cierre + SMA 20 + Bandas de Bollinger
            ax_p1.plot(serie_precios_hist.index, serie_precios_hist.values, color="#00E5FF", lw=1.8, label="Precio de Cierre ($)")
            ax_p1.plot(sma20.index, sma20.values, color="#38BDF8", lw=1.2, linestyle="--", label="Media Móvil (SMA 20)")
            ax_p1.fill_between(serie_precios_hist.index, bollinger_inf.values, bollinger_sup.values, color="#0284C7", alpha=0.18, label="Bandas de Volatilidad (±2σ)")
            ax_p1.set_title("Evolución del Precio de Cierre y Bandas de Volatilidad en la Temporalidad Seleccionada")
            ax_p1.set_ylabel("Precio ($)")
            ax_p1.legend(frameon=False, loc="upper left")
            ax_p1.grid(True, alpha=0.25)
            
            # Panel 2: Volatilidad Dinámica EWMA y Drawdown
            ax_p2.plot(ewma_vol.index, ewma_vol.values, color="#4FC3F7", lw=1.5, label="Volatilidad Diaria EWMA (λ=0.94)")
            ax_p2_dd = ax_p2.twinx()
            ax_p2_dd.fill_between(drawdown_s.index, drawdown_s.values, 0, color="#EF4444", alpha=0.22, label="Drawdown (%)")
            ax_p2_dd.set_ylabel("Drawdown (%)", color="#FCA5A5")
            ax_p2_dd.tick_params(axis='y', colors='#FCA5A5')
            ax_p2.set_ylabel("Volatilidad EWMA (%)")
            ax_p2.set_title("Dinámica de Volatilidad (RiskMetrics EWMA) y Caída desde Máximos (Drawdown)")
            ax_p2.legend(frameon=False, loc="upper left")
            ax_p2.grid(True, alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig_p)
            
        st.markdown("<br><h5>Tabla de Cuantiles Muestrales</h5>", unsafe_allow_html=True)
        summary = pd.Series(data).describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_frame(etiqueta_var)
        st.dataframe(summary.T, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 4: PRUEBA BOOTSTRAP (PARAMÉTRICO Y NO PARAMÉTRICO)
    # --------------------------------------------------------------------------
    with tab4:
        st.markdown("<br><h4>Inferencia por Remuestreo Bootstrap y Corrección de Bondad de Ajuste</h4>", unsafe_allow_html=True)
        col_b_cfg1, col_b_cfg2 = st.columns([1, 2])
        with col_b_cfg1:
            n_iter_boot = st.slider("Iteraciones Bootstrap (B):", min_value=500, max_value=5000, value=1500, step=500)
        with col_b_cfg2:
            st.markdown("""
            <div class="well" style="padding:10px 15px;">
                <b>¿Por qué aplicar Bootstrap?</b> Cuando los parámetros se estiman de la propia muestra (MLE), el test K-S clásico es conservador. 
                El <b>Bootstrap Paramétrico</b> obtiene la distribución exacta del estadístico <i>D*</i>, mientras que el <b>Bootstrap No Paramétrico</b> 
                estima intervalos de confianza para el riesgo sin asumir ninguna distribución teórica.
            </div>
            """, unsafe_allow_html=True)
            
        boot_medias, boot_stds, boot_vars, boot_cvars, d_boot, pval_boot_ks, crit_boot_ks = ejecutar_bootstrap_completo(
            data, dist_obj, cdf_evaluar, ks_stat, alpha=alpha, n_boot=n_iter_boot
        )
        
        ic_low = (alpha / 2.0) * 100.0
        ic_high = (1.0 - alpha / 2.0) * 100.0
        
        b1, b2, b3, b4 = st.columns(4)
        b1.markdown(f'<div class="metric-card"><div style="color:#64748B;">P-Value Bootstrap K-S</div><div style="font-size:22px; color:{"#00FF66" if pval_boot_ks >= alpha else "#FCA5A5"}; font-weight:bold;">{pval_boot_ks:.4f}</div><div style="font-size:12px; color:#64748B;">Crítico D*: {crit_boot_ks:.4f}</div></div>', unsafe_allow_html=True)
        b2.markdown(f'<div class="metric-card"><div style="color:#64748B;">IC {(1-alpha)*100:.0f}% Media</div><div style="font-size:20px; color:#4FC3F7; font-weight:bold;">[{np.percentile(boot_medias, ic_low):.2f}, {np.percentile(boot_medias, ic_high):.2f}]{unidad_var}</div><div style="font-size:12px; color:#64748B;">Media Boot: {boot_medias.mean():.2f}{unidad_var}</div></div>', unsafe_allow_html=True)
        b3.markdown(f'<div class="metric-card"><div style="color:#64748B;">IC {(1-alpha)*100:.0f}% Cuantil α (VaR)</div><div style="font-size:20px; color:#FCA5A5; font-weight:bold;">[{np.percentile(boot_vars, ic_low):.2f}, {np.percentile(boot_vars, ic_high):.2f}]{unidad_var}</div><div style="font-size:12px; color:#64748B;">Media VaR: {boot_vars.mean():.2f}{unidad_var}</div></div>', unsafe_allow_html=True)
        b4.markdown(f'<div class="metric-card"><div style="color:#64748B;">IC {(1-alpha)*100:.0f}% Expected Shortfall</div><div style="font-size:20px; color:#EF4444; font-weight:bold;">[{np.percentile(boot_cvars, ic_low):.2f}, {np.percentile(boot_cvars, ic_high):.2f}]{unidad_var}</div><div style="font-size:12px; color:#64748B;">Media CVaR: {boot_cvars.mean():.2f}{unidad_var}</div></div>', unsafe_allow_html=True)
        
        fig_boot, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(13, 4.2))
        
        # Gráfico 1: Distribución Bootstrap del Estadístico D* (Bondad de Ajuste)
        ax_b1.hist(d_boot, bins=35, color="#075985", edgecolor="#0284C7", density=True, alpha=0.8, label="Distribución Empírica D*")
        ax_b1.axvline(ks_stat, color="#00E5FF", lw=2.5, label=f"D Observado ({ks_stat:.4f})")
        ax_b1.axvline(crit_boot_ks, color="#EF4444", lw=2, linestyle="--", label=f"Umbral Crítico α ({crit_boot_ks:.4f})")
        ax_b1.set_title("Prueba Bootstrap Paramétrica (Estadístico D de K-S)")
        ax_b1.set_xlabel("Distancia K-S Simulada (D*)")
        ax_b1.set_ylabel("Densidad")
        ax_b1.legend(frameon=False)
        ax_b1.grid(True, alpha=0.25)
        
        # Gráfico 2: Distribución Bootstrap No Paramétrica del VaR
        ax_b2.hist(boot_vars, bins=35, color="#0F172A", edgecolor="#38BDF8", density=True, alpha=0.85, label="Remuestreo VaR")
        ax_b2.axvline(boot_vars.mean(), color="#00E5FF", lw=2, label=f"Media: {boot_vars.mean():.2f}{unidad_var}")
        ax_b2.axvline(np.percentile(boot_vars, ic_low), color="#FCA5A5", lw=2, linestyle="--", label=f"IC Inf: {np.percentile(boot_vars, ic_low):.2f}{unidad_var}")
        ax_b2.axvline(np.percentile(boot_vars, ic_high), color="#00FF66", lw=2, linestyle="--", label=f"IC Sup: {np.percentile(boot_vars, ic_high):.2f}{unidad_var}")
        ax_b2.set_title(f"Distribución Bootstrap No Paramétrica del Cuantil α ({(1-alpha)*100:.0f}%)")
        ax_b2.set_xlabel(etiqueta_var)
        ax_b2.set_ylabel("Densidad")
        ax_b2.legend(frameon=False)
        ax_b2.grid(True, alpha=0.25)
        st.pyplot(fig_boot)

    # --------------------------------------------------------------------------
    # TAB 5: SIMULACIÓN MONTE CARLO Y MOVIMIENTO BROWNIANO GEOMÉTRICO (GBM)
    # --------------------------------------------------------------------------
    with tab5:
        st.markdown("<br><h4>Motor de Proyección Estocástica (Monte Carlo / GBM / Bootstrap)</h4>", unsafe_allow_html=True)
        
        if modelo_validado:
            st.success(f"Condición matemática superada ({criterio_desbloqueo}). Motor de simulación desbloqueado.")
            c_mc1, c_mc2, c_mc3, c_mc4 = st.columns(4)
            with c_mc1:
                metodo_mc = st.selectbox(
                    "Proceso Estocástico:",
                    [
                        f"Distribución Ajustada ({dist_sel})",
                        "Movimiento Browniano Geométrico (GBM)",
                        "Bootstrap Histórico (No Paramétrico)"
                    ]
                )
            with c_mc2:
                n_paths = st.number_input("Trayectorias (Escenarios):", min_value=100, max_value=50000, value=2000, step=1000)
            with c_mc3:
                n_days = st.number_input("Horizonte (Días):", min_value=10, max_value=1000, value=252)
            with c_mc4:
                p_initial = st.number_input("Precio Spot (S0):", min_value=0.1, value=p0_default, step=1.0)
                
            if st.button("🎲 Ejecutar Simulación"):
                with st.spinner("Computando trayectorias estocásticas..."):
                    rend_base = serie_rend_hist.values if (serie_rend_hist is not None and len(serie_rend_hist) >= 2) else data
                    
                    if metodo_mc == "Movimiento Browniano Geométrico (GBM)":
                        mu_d = np.mean(rend_base) / 100.0
                        sigma_d = np.std(rend_base, ddof=1) / 100.0
                        z_ shocks = np.random.standard_normal(size=(int(n_days), int(n_paths)))
                        factores_diarios = np.exp((mu_d - 0.5 * (sigma_d ** 2)) + sigma_d * z_shocks)
                        price_paths = p_initial * np.cumprod(factores_diarios, axis=0)
                    elif metodo_mc == "Bootstrap Histórico (No Paramétrico)":
                        returns_sim = np.random.choice(rend_base, size=(int(n_days), int(n_paths)), replace=True) / 100.0
                        price_paths = p_initial * np.cumprod(np.maximum(1.0 + returns_sim, 0.0), axis=0)
                    else:
                        if unidad_var == "$":
                            # Si el usuario ajustó la distribución directamente al Precio de Cierre,
                            # convertimos las variaciones relativas de la distribución simulada respecto a su media
                            sim_raw = dist_obj.rvs(size=(int(n_days), int(n_paths)))
                            sigma_rel = np.std(rend_base, ddof=1) / 100.0 if len(rend_base) >= 2 else 0.02
                            z_estandar = (sim_raw - np.mean(data)) / max(np.std(data, ddof=1), 1e-6)
                            returns_sim = (np.mean(rend_base) / 100.0) + z_estandar * sigma_rel
                        else:
                            returns_sim = dist_obj.rvs(size=(int(n_days), int(n_paths))) / 100.0
                        price_paths = p_initial * np.cumprod(np.maximum(1.0 + returns_sim, 0.0), axis=0)
                        
                    final_prices = price_paths[-1, :]
                    
                    var_95 = np.percentile(final_prices, alpha * 100)
                    cvar_95 = final_prices[final_prices <= var_95].mean()
                    mejor_escenario = np.percentile(final_prices, (1 - alpha) * 100)
                    
                    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
                    ax4a.plot(price_paths[:, :min(n_paths, 300)], color="#0284C7", alpha=0.03)
                    ax4a.plot(price_paths.mean(axis=1), color="#00E5FF", lw=2, label="S(t) Esperado")
                    ax4a.axhline(p_initial, color="#64748B", linestyle=":", label="S(0)")
                    ax4a.set_title(f"Evolución de Trayectorias ({metodo_mc})")
                    ax4a.set_xlabel("Días Proyectados")
                    ax4a.set_ylabel("Precio ($)")
                    ax4a.legend(frameon=False); ax4a.grid(True, alpha=0.2)
                    
                    ax4b.hist(final_prices, bins=50, color="#0F172A", edgecolor="#0284C7", density=True)
                    ax4b.axvline(final_prices.mean(), color="#00E5FF", lw=2, label=f"Media: ${final_prices.mean():.2f}")
                    ax4b.axvline(var_95, color="#FCA5A5", lw=2, linestyle="--", label=f"VaR ({(1-alpha)*100:.0f}%): ${var_95:.2f}")
                    ax4b.axvline(mejor_escenario, color="#00FF66", lw=2, linestyle="--", label=f"Mejor Esc. ({(1-alpha)*100:.0f}%): ${mejor_escenario:.2f}")
                    ax4b.set_title(f"Distribución Terminal (T={int(n_days)})")
                    ax4b.set_xlabel("Precio Final ($)")
                    ax4b.legend(frameon=False); ax4b.grid(True, alpha=0.2)
                    
                    st.pyplot(fig4)
                    
                    rm1, rm2, rm3, rm4 = st.columns(4)
                    rm1.markdown(f'<div class="metric-card"><div style="color:#64748B;">Valor Esperado E[S(T)]</div><div style="font-size:24px; color:#4FC3F7; font-weight:bold;">${final_prices.mean():.2f}</div></div>', unsafe_allow_html=True)
                    rm2.markdown(f'<div class="metric-card"><div style="color:#64748B;">Value at Risk</div><div style="font-size:24px; color:#FCA5A5; font-weight:bold;">${var_95:.2f}</div></div>', unsafe_allow_html=True)
                    rm3.markdown(f'<div class="metric-card"><div style="color:#64748B;">Expected Shortfall</div><div style="font-size:24px; color:#EF4444; font-weight:bold;">${cvar_95:.2f}</div></div>', unsafe_allow_html=True)
                    rm4.markdown(f'<div class="metric-card"><div style="color:#64748B;">Mejor Escenario ({(1-alpha)*100:.0f}%)</div><div style="font-size:24px; color:#00FF66; font-weight:bold;">${mejor_escenario:.2f}</div></div>', unsafe_allow_html=True)

                    csv_buffer = io.StringIO()
                    pd.DataFrame(price_paths).to_csv(csv_buffer, index=False)
                    st.download_button(label="📥 Exportar Matriz de Trayectorias (CSV)", data=csv_buffer.getvalue(), file_name="simulacion_monte_carlo.csv", mime="text/csv")
        else:
            st.markdown(f'<div class="alert-danger" style="margin-top: 0;"><b>🛑 Bloqueado por Riesgo de Modelo:</b> El modelo no cumple con el criterio seleccionado (<b>{criterio_desbloqueo}</b>) para α = {alpha}. Ajusta la distribución o los parámetros antes de simular.</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # TAB 6: BACKTESTING VAR (KUPIEC POF + CHRISTOFFERSEN + COMPARATIVA NORMAL)
    # --------------------------------------------------------------------------
    with tab6:
        st.markdown("<br><h4>Backtesting Regulatorio: Kupiec (POF) e Independencia de Christoffersen</h4>", unsafe_allow_html=True)
        
        var_teorico = float(dist_obj.ppf(alpha))
        var_normal = float(stats.norm.ppf(alpha, loc=np.mean(data), scale=np.std(data, ddof=1)))
        
        excepciones = data[data < var_teorico]
        n_obs = len(data)
        n_excepciones = len(excepciones)
        excepciones_esperadas = n_obs * alpha
        tasa_empirica = n_excepciones / n_obs
        
        try:
            p_val_kupiec = stats.binomtest(n_excepciones, n_obs, alpha, alternative='two-sided').pvalue
        except AttributeError:
            p_val_kupiec = stats.binom_test(n_excepciones, n_obs, alpha, alternative='two-sided')
            
        lr_ind, p_val_ind, lr_cc, p_val_cc, (n00, n01, n10, n11) = test_christoffersen(data, var_teorico, alpha=alpha)

        c_b1, c_b2, c_b3, c_b4 = st.columns(4)
        c_b1.markdown(f'<div class="metric-card"><div style="color:#64748B;">VaR Modelo vs Normal</div><div style="font-size:21px; color:#4FC3F7; font-weight:bold;">{var_teorico:.2f}{unidad_var} <span style="font-size:14px; color:#90CAF9;">(Norm: {var_normal:.2f}{unidad_var})</span></div></div>', unsafe_allow_html=True)
        c_b2.markdown(f'<div class="metric-card"><div style="color:#64748B;">Excepciones (Esp. vs Real)</div><div style="font-size:22px; color:{"#00FF66" if p_val_kupiec >= 0.05 else "#EF4444"}; font-weight:bold;">{n_excepciones} <span style="font-size:14px; color:#64748B;">/ {excepciones_esperadas:.1f} esp.</span></div></div>', unsafe_allow_html=True)
        c_b3.markdown(f'<div class="metric-card"><div style="color:#64748B;">P-Value Kupiec (POF)</div><div style="font-size:22px; color:{"#00FF66" if p_val_kupiec >= 0.05 else "#EF4444"}; font-weight:bold;">{p_val_kupiec:.4f}</div></div>', unsafe_allow_html=True)
        c_b4.markdown(f'<div class="metric-card"><div style="color:#64748B;">P-Value Christoffersen (Ind.)</div><div style="font-size:22px; color:{"#00FF66" if p_val_ind >= 0.05 else "#FCA5A5"}; font-weight:bold;">{p_val_ind:.4f}</div></div>', unsafe_allow_html=True)

        fig5, ax5 = plt.subplots(figsize=(12, 4))
        ax5.plot(data, color="#0284C7", alpha=0.6, label=etiqueta_var, lw=1)
        ax5.axhline(var_teorico, color="#EF4444", lw=2, linestyle="--", label=f"Barrera VaR {dist_sel} ({var_teorico:.2f}{unidad_var})")
        ax5.axhline(var_normal, color="#00E5FF", lw=1.5, linestyle=":", label=f"Barrera VaR Normal ({var_normal:.2f}{unidad_var})")
        
        indices_excepcion = np.where(data < var_teorico)[0]
        ax5.scatter(indices_excepcion, data[indices_excepcion], color="#EF4444", s=25, zorder=5, label="Excepciones (Brechas)")
        
        ax5.set_title("Evolución Histórica vs Límites VaR (Modelo Seleccionado vs Normal)")
        ax5.set_xlabel("Observaciones (Días)")
        ax5.set_ylabel(etiqueta_var)
        ax5.legend(frameon=False, loc="upper right")
        ax5.grid(True, alpha=0.2)
        st.pyplot(fig5)
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            if p_val_kupiec < 0.05:
                st.markdown(f'<div class="alert-danger"><b>🔴 KUPIEC (COBERTURA INCONDICIONAL):</b> Rechaza H0 (p={p_val_kupiec:.4f}). La proporción de fallos ({tasa_empirica*100:.2f}%) difiere significativamente del {alpha*100:.0f}% esperado.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-success"><b>🟢 KUPIEC (COBERTURA INCONDICIONAL):</b> Valida H0 (p={p_val_kupiec:.4f}). La frecuencia de excepciones ({tasa_empirica*100:.2f}%) está calibrada con el nivel de confianza.</div>', unsafe_allow_html=True)
        with col_v2:
            if p_val_ind < 0.05:
                st.markdown(f'<div class="alert-danger"><b>🔴 CHRISTOFFERSEN (INDEPENDENCIA):</b> Rechaza H0 (p={p_val_ind:.4f}). Existen rachas de pérdidas consecutivas (<i>n11={n11}</i>), indicando agrupamiento de volatilidad.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-success"><b>🟢 CHRISTOFFERSEN (INDEPENDENCIA):</b> Valida H0 (p={p_val_ind:.4f}). Las excepciones ocurren de manera independiente sin agruparse en clústeres (<i>n11={n11}</i>).</div>', unsafe_allow_html=True)

else:
    st.warning("Se requieren al menos 2 datos numéricos válidos para procesar el modelo.")
