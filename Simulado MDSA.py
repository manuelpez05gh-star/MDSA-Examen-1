import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
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
    
    # Agrupar intervalos adyacentes donde E_i < 5
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
    .stTabs [data-baseweb="tab"] { color: #64748B; font-weight: 600; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { border-bottom: 2px solid #00B4D8 !important; color: #00B4D8 !important; background-color: rgba(0, 180, 216, 0.05) !important; }
    .stButton > button { background-color: #0284C7 !important; color: #ffffff !important; border: 1px solid #0369A1; font-weight: bold; width: 100%; border-radius: 6px; transition: 0.3s; }
    .stButton > button:hover { background-color: #0369A1 !important; box-shadow: 0 0 10px rgba(2, 132, 199, 0.5); }
    .well { background-color: #111827; border-left: 4px solid #00B4D8; padding: 15px; border-radius: 4px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .metric-card { background-color: #111827; border: 1px solid #1F2937; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.4); }
    .alert-danger { background-color: rgba(153, 27, 27, 0.2); color: #FCA5A5; border: 1px solid #7F1D1D; padding: 15px; border-radius: 6px; margin-top: 20px; }
    .alert-success { background-color: rgba(6, 78, 59, 0.2); color: #6EE7B7; border: 1px solid #064E3B; padding: 15px; border-radius: 6px; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h2>Análisis Estocástico y Simulación Monte Carlo</h2>", unsafe_allow_html=True)
st.markdown("<h5>Evaluación de Riesgos y Modelado de Series Financieras</h5>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ==============================================================================
# 1. INTERFAZ DE USUARIO (BARRA LATERAL)
# ==============================================================================
df = None
p0_default = 100.0

with st.sidebar:
    st.markdown("<h4 style='color: #4FC3F7;'>1. Carga de Datos</h4>", unsafe_allow_html=True)
    origen_datos = st.radio("Fuente de información:", ["Cargar Archivo CSV", "Generar Datos Simulados"])
    
    if origen_datos == "Generar Datos Simulados":
        n_sim = st.number_input("Tamaño muestral (n):", min_value=10, max_value=5000, value=500, step=1)
        gl_sim = st.number_input("Grados de libertad (df):", min_value=1, value=7, step=1)
        st.caption("Datos generados mediante distribución Chi-cuadrada.")
    else:
        file_csv = st.file_uploader("Seleccione archivo CSV:", type=["csv"])
        sep_csv = st.selectbox("Separador de columnas:", [",", ";"])
        col_csv = None
        if file_csv is not None:
            try:
                df = pd.read_csv(file_csv, sep=sep_csv)
                default_idx = df.columns.get_loc("Change %") if "Change %" in df.columns else (6 if len(df.columns) >= 7 else 0)
                col_csv = st.selectbox("Variable a modelar (Rendimientos):", options=df.columns, index=default_idx)
                if 'Price' in df.columns:
                    p0_default = float(str(df['Price'].iloc[0]).replace(',', ''))
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
        ["Al menos una prueba (K-S o Chi²)", "Solo Prueba K-S", "Solo Prueba Chi-cuadrada", "Ambas pruebas simultáneamente"],
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
# 2. PROCESAMIENTO DE DATOS Y ESTIMACIÓN DE PARÁMETROS
# ==============================================================================
data = np.array([])
if origen_datos == "Generar Datos Simulados":
    np.random.seed(42)
    data = np.random.chisquare(df=gl_sim, size=n_sim)
elif file_csv is not None and col_csv is not None:
    col_limpia = df[col_csv].astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False)
    data = pd.to_numeric(col_limpia, errors='coerce').dropna().values

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
    
    # Evaluar condición de desbloqueo para Monte Carlo
    pasa_ks = ks_pval >= alpha
    pasa_chi2 = chi2_pval >= alpha
    if criterio_desbloqueo == "Al menos una prueba (K-S o Chi²)":
        modelo_validado = pasa_ks or pasa_chi2
    elif criterio_desbloqueo == "Solo Prueba K-S":
        modelo_validado = pasa_ks
    elif criterio_desbloqueo == "Solo Prueba Chi-cuadrada":
        modelo_validado = pasa_chi2
    else:
        modelo_validado = pasa_ks and pasa_chi2
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Diagnóstico", "📜 Reporte K-S & Chi²", "📋 Análisis Estadístico", 
        "🔮 Monte Carlo & Riesgo", "📈 Backtesting VaR"
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

    # --- TAB 1: DIAGNÓSTICO ---
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
                
            ax1.legend(frameon=False)
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            
        with c2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.step(np.sort(data), np.arange(1, len(data) + 1)/len(data), color="#0284C7", lw=2, label="Fn(x) Empírica")
            ax2.plot(x_grid, cdf_evaluar(x_grid), color="#00E5FF", lw=2, linestyle="--", label="F(x) Teórica")
            ax2.set_title("Función de Distribución Acumulada (FDA)")
            ax2.legend(frameon=False)
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            
        # Gráfica adicional: Frecuencias Observadas vs Esperadas (Chi-cuadrada)
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
            
            st.markdown(f"""<div class="well"><b>⚙️ Parámetros Optimizados:</b><br>{params_str}</div>""", unsafe_allow_html=True)

    # --- TAB 2: REPORTE K-S & CHI-CUADRADA ---
    with tab2:
        st.markdown("<br><h4>Dictamen Estadístico de Bondad de Ajuste</h4>", unsafe_allow_html=True)
        col_rep1, col_rep2 = st.columns(2)
        
        with col_rep1:
            decision_ks = "RECHAZAR H0" if not pasa_ks else "NO RECHAZAR H0"
            st.code(f"""
=========================================================
  1. PRUEBA KOLMOGOROV-SMIRNOV (K-S)
=========================================================
  Modelo evaluado      : {dist_sel}
  Método de ajuste     : {modo_estimacion}
  Muestra observada (n): {len(data)}
  Significancia (α)    : {alpha}

  Estadístico D        : {ks_stat:.6f}
  P-Value              : {ks_pval:.6e}
  
  Conclusión K-S       : {decision_ks}
=========================================================
            """, language='text')
            
            if not pasa_ks:
                st.markdown(f'<div class="alert-danger"><b>❌ K-S RECHAZA H0:</b> El p-value ({ks_pval:.4g}) es menor que α ({alpha}). La distancia vertical máxima D supera el umbral crítico.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-success"><b>✅ K-S VALIDA H0:</b> El p-value ({ks_pval:.4g}) es mayor o igual que α ({alpha}). La FDA empírica se ajusta al modelo teórico.</div>', unsafe_allow_html=True)

        with col_rep2:
            decision_chi2 = "RECHAZAR H0" if not pasa_chi2 else "NO RECHAZAR H0"
            st.code(f"""
=========================================================
  2. PRUEBA CHI-CUADRADA DE PEARSON (χ²)
=========================================================
  Clases agrupadas (k) : {chi2_k} (Frec. Esp. ≥ 5)
  Parámetros est. (m)  : {num_params_est}
  Grados libertad (gl) : {chi2_gl}
  Valor Crítico χ²(α)  : {chi2_crit:.4f}

  Estadístico χ²       : {chi2_stat:.4f}
  P-Value              : {chi2_pval:.6e}
  
  Conclusión Chi²      : {decision_chi2}
=========================================================
            """, language='text')
            
            if not pasa_chi2:
                st.markdown(f'<div class="alert-danger"><b>❌ CHI² RECHAZA H0:</b> El estadístico χ² ({chi2_stat:.4f}) supera el valor crítico ({chi2_crit:.4f}), con p-value={chi2_pval:.4g} &lt; α.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-success"><b>✅ CHI² VALIDA H0:</b> El estadístico χ² ({chi2_stat:.4f}) es menor al valor crítico ({chi2_crit:.4f}), con p-value={chi2_pval:.4g} ≥ α.</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 Ver Tabla Desglosada de Clases y Frecuencias (Prueba Chi-cuadrada)", expanded=False):
            st.dataframe(df_chi2, use_container_width=True, hide_index=True)

    # --- TAB 3: RESUMEN ESTADÍSTICO ---
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div style="color:#64748B;">Media (μ)</div><div style="font-size:24px; color:#4FC3F7; font-weight:bold;">{data.mean():.2f}%</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div style="color:#64748B;">Volatilidad (σ)</div><div style="font-size:24px; color:#4FC3F7; font-weight:bold;">{data.std():.2f}%</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div style="color:#64748B;">Asimetría (Skew)</div><div style="font-size:24px; color:#4FC3F7; font-weight:bold;">{stats.skew(data):.2f}</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div style="color:#64748B;">Curtosis (Exceso)</div><div style="font-size:24px; color:#4FC3F7; font-weight:bold;">{stats.kurtosis(data):.2f}</div></div>', unsafe_allow_html=True)
        
        summary = pd.Series(data).describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_frame("Rendimientos (%)")
        st.dataframe(summary.T, use_container_width=True)

    # --- TAB 4: MONTE CARLO ---
    with tab4:
        st.markdown("<br><h4>Motor Estocástico (Random Walk)</h4>", unsafe_allow_html=True)
        
        if modelo_validado:
            st.success(f"Condición matemática superada según el criterio seleccionado ({criterio_desbloqueo}). Motor desbloqueado.")
            c_mc1, c_mc2, c_mc3 = st.columns(3)
            with c_mc1: n_paths = st.number_input("Trayectorias (Escenarios):", min_value=100, max_value=50000, value=2000, step=1000)
            with c_mc2: n_days = st.number_input("Horizonte (Días):", min_value=10, max_value=1000, value=252)
            with c_mc3: p_initial = st.number_input("Precio Spot (S0):", min_value=0.1, value=p0_default, step=1.0)
                
            if st.button("🎲 Ejecutar Simulación"):
                with st.spinner("Computando trayectorias estocásticas..."):
                    returns_sim = dist_obj.rvs(size=(int(n_days), int(n_paths))) / 100.0
                    price_paths = p_initial * np.cumprod(np.maximum(1 + returns_sim, 0.0), axis=0)
                    final_prices = price_paths[-1, :]
                    
                    var_95 = np.percentile(final_prices, alpha * 100)
                    cvar_95 = final_prices[final_prices <= var_95].mean()
                    mejor_escenario = np.percentile(final_prices, (1 - alpha) * 100)
                    
                    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
                    ax4a.plot(price_paths[:, :min(n_paths, 300)], color="#0284C7", alpha=0.03)
                    ax4a.plot(price_paths.mean(axis=1), color="#00E5FF", lw=2, label="S(t) Esperado")
                    ax4a.axhline(p_initial, color="#64748B", linestyle=":", label="S(0)")
                    ax4a.set_title("Evolución de Trayectorias")
                    ax4a.legend(frameon=False); ax4a.grid(True, alpha=0.2)
                    
                    ax4b.hist(final_prices, bins=50, color="#0F172A", edgecolor="#0284C7", density=True)
                    ax4b.axvline(final_prices.mean(), color="#00E5FF", lw=2, label=f"Media: ${final_prices.mean():.2f}")
                    ax4b.axvline(var_95, color="#FCA5A5", lw=2, linestyle="--", label=f"VaR ({(1-alpha)*100:.0f}%): ${var_95:.2f}")
                    ax4b.axvline(mejor_escenario, color="#00FF66", lw=2, linestyle="--", label=f"Mejor Esc. ({(1-alpha)*100:.0f}%): ${mejor_escenario:.2f}")
                    ax4b.set_title(f"Distribución Terminal (T={int(n_days)})")
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

    # --- TAB 5: BACKTESTING VAR ---
    with tab5:
        st.markdown("<br><h4>Prueba de Excepciones Históricas (Kupiec POF)</h4>", unsafe_allow_html=True)
        
        var_teorico = float(dist_obj.ppf(alpha))
        
        excepciones = data[data < var_teorico]
        n_obs = len(data)
        n_excepciones = len(excepciones)
        excepciones_esperadas = n_obs * alpha
        tasa_empirica = n_excepciones / n_obs
        
        try:
            p_val_kupiec = stats.binomtest(n_excepciones, n_obs, alpha, alternative='two-sided').pvalue
        except AttributeError:
            p_val_kupiec = stats.binom_test(n_excepciones, n_obs, alpha, alternative='two-sided')

        c_b1, c_b2, c_b3, c_b4 = st.columns(4)
        c_b1.markdown(f'<div class="metric-card"><div style="color:#64748B;">Límite VaR ({(1-alpha)*100:.0f}%)</div><div style="font-size:24px; color:#4FC3F7; font-weight:bold;">{var_teorico:.2f}%</div></div>', unsafe_allow_html=True)
        c_b2.markdown(f'<div class="metric-card"><div style="color:#64748B;">Excepciones Esperadas</div><div style="font-size:24px; color:#4FC3F7; font-weight:bold;">{excepciones_esperadas:.1f}</div></div>', unsafe_allow_html=True)
        
        color_excepcion = "#00FF66" if p_val_kupiec >= 0.05 else "#EF4444"
        c_b3.markdown(f'<div class="metric-card"><div style="color:#64748B;">Excepciones Reales</div><div style="font-size:24px; color:{color_excepcion}; font-weight:bold;">{n_excepciones}</div></div>', unsafe_allow_html=True)
        c_b4.markdown(f'<div class="metric-card"><div style="color:#64748B;">Tasa de Falla Real</div><div style="font-size:24px; color:#4FC3F7; font-weight:bold;">{tasa_empirica*100:.2f}%</div></div>', unsafe_allow_html=True)

        fig5, ax5 = plt.subplots(figsize=(12, 4))
        ax5.plot(data, color="#0284C7", alpha=0.6, label="Rendimientos Diarios", lw=1)
        ax5.axhline(var_teorico, color="#EF4444", lw=2, linestyle="--", label=f"Barrera VaR ({(1-alpha)*100:.0f}%)")
        
        indices_excepcion = np.where(data < var_teorico)[0]
        ax5.scatter(indices_excepcion, data[indices_excepcion], color="#EF4444", s=25, zorder=5, label="Excepciones (Brechas)")
        
        ax5.set_title("Evolución Histórica vs Límite Teórico VaR")
        ax5.set_xlabel("Observaciones (Días)")
        ax5.set_ylabel("Rendimiento (%)")
        ax5.legend(frameon=False)
        ax5.grid(True, alpha=0.2)
        st.pyplot(fig5)
        
        st.markdown("<br><h5>Veredicto del Test de Kupiec</h5>", unsafe_allow_html=True)
        if p_val_kupiec < 0.05:
            st.markdown(f'<div class="alert-danger"><b>🔴 MODELO NO CALIBRADO:</b> El p-value ({p_val_kupiec:.4f}) rechaza la hipótesis nula. La cantidad de excepciones reales ({n_excepciones}) difiere significativamente del {alpha*100}% esperado. El modelo requiere ajuste para evitar sorpresas de riesgo de capital.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-success"><b>🟢 MODELO ROBUSTO (BACKTESTED):</b> El p-value ({p_val_kupiec:.4f}) valida estadísticamente que las excepciones reales se alinean con la precisión teórica exigida por el regulador. El VaR es confiable.</div>', unsafe_allow_html=True)

elif file_csv is not None:
    st.warning("Se requieren al menos 2 datos numéricos para procesar el modelo.")