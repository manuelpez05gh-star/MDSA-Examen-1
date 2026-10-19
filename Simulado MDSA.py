import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import io

# ==============================================================================
# 0. CONFIGURACIÓN INICIAL Y ESTILOS CSS (BLACK & BLUE THEME)
# ==============================================================================
st.set_page_config(
    page_title="Plataforma de Simulación Actuarial",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        "t-Student (Finanzas)": "t",
        "Normal (Gaussiana)": "norm",
        "Lognormal": "lnorm",
        "Laplace": "laplace",
        "Exponencial": "exp",
        "Chi-cuadrada": "chisq",
        "Poisson (Discreta)": "pois",
        "Geométrica (Discreta)": "geom",
        "Binomial (Discreta)": "binom"
    }
    dist_sel = st.selectbox("Distribución Teórica:", list(dist_map.keys()), index=0) 
    dist_test = dist_map[dist_sel]
    
    alpha = st.selectbox("Nivel de Confianza (α):", [0.10, 0.05, 0.01], index=1, format_func=lambda x: f"{(1-x)*100:.0f}% (α={x})")
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
                params_manuales['scale'] = st.number_input("Escala (scale / exp(meanlog)):", min_value=0.0001, value=1.0, step=0.1)
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
            elif dist_test == "pois":
                params_manuales['lambda'] = st.number_input("Tasa de ocurrencia (λ):", min_value=0.0001, value=4.0, step=0.5)
            elif dist_test == "geom":
                params_manuales['p_geom'] = st.number_input("Probabilidad de éxito (p):", min_value=0.0001, max_value=1.0, value=0.25, step=0.05)
            elif dist_test == "binom":
                params_manuales['n_binom'] = st.number_input("Número de ensayos (n):", min_value=1, value=10, step=1)
                params_manuales['p_binom'] = st.number_input("Probabilidad de éxito (p):", min_value=0.0001, max_value=1.0, value=0.5, step=0.05)
                
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
es_discreta = dist_test in ["pois", "geom", "binom"]

if len(data) >= 2:
    if modo_estimacion == "Automático (MLE)":
        if dist_test == "t":
            fit_params = stats.t.fit(data)
            dist_obj = stats.t(*fit_params)
        elif dist_test == "norm":
            fit_params = stats.norm.fit(data)
            dist_obj = stats.norm(*fit_params)
        elif dist_test == "lnorm":
            # Ajuste seguro desplazando si existen datos negativos
            shift = abs(min(data)) + 1e-4 if min(data) <= 0 else 0
            fit_params = stats.lognorm.fit(data + shift)
            dist_obj = stats.lognorm(*fit_params)
        elif dist_test == "laplace":
            fit_params = stats.laplace.fit(data)
            dist_obj = stats.laplace(*fit_params)
        elif dist_test == "exp":
            fit_params = stats.expon.fit(data)
            dist_obj = stats.expon(*fit_params)
        elif dist_test == "chisq":
            fit_params = stats.chi2.fit(data)
            dist_obj = stats.chi2(*fit_params)
        elif dist_test == "pois":
            # Estimador MLE para Poisson: lambda = media muestral (restringida > 0)
            lam_est = float(max(np.mean(data), 0.0001))
            fit_params = (lam_est,)
            dist_obj = stats.poisson(mu=lam_est)
        elif dist_test == "geom":
            # Estimador MLE para Geométrica en soporte {1, 2, ...}: p = 1 / media
            media_pos = max(float(np.mean(data)), 1.0001)
            p_est = float(np.clip(1.0 / media_pos, 0.0001, 1.0))
            fit_params = (p_est,)
            dist_obj = stats.geom(p=p_est)
        elif dist_test == "binom":
            # Estimación por Método de Momentos / MLE truncado para Binomial
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
    else:
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
        elif dist_test == "pois":
            dist_obj = stats.poisson(mu=params_manuales['lambda'])
        elif dist_test == "geom":
            dist_obj = stats.geom(p=params_manuales['p_geom'])
        elif dist_test == "binom":
            dist_obj = stats.binom(n=int(params_manuales['n_binom']), p=params_manuales['p_binom'])

# ==============================================================================
# 3. RENDERIZADO DEL PANEL PRINCIPAL
# ==============================================================================
if len(data) >= 2:
    ks_stat, ks_pval = stats.kstest(data, dist_obj.cdf)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Diagnóstico", "📜 Reporte K-S", "📋 Análisis Estadístico", 
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
                x_disc = np.unique(np.round(x_grid))
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
            ax2.plot(x_grid, dist_obj.cdf(x_grid), color="#00E5FF", lw=2, linestyle="--", label="F(x) Teórica")
            ax2.set_title("Función de Distribución Acumulada (FDA)")
            ax2.legend(frameon=False)
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            
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
            elif dist_test == "pois":
                params_str = f"<b>Tasa de ocurrencia (λ):</b> {float(fit_params[0]):.4f}"
            elif dist_test == "geom":
                params_str = f"<b>Probabilidad de éxito (p):</b> {float(fit_params[0]):.4f}"
            elif dist_test == "binom":
                params_str = f"<b>Número de ensayos (n):</b> {int(fit_params[0])} &nbsp;|&nbsp; <b>Probabilidad de éxito (p):</b> {float(fit_params[1]):.4f}"
            
            st.markdown(f"""<div class="well"><b>⚙️ Parámetros Optimizados:</b><br>{params_str}</div>""", unsafe_allow_html=True)

    # --- TAB 2: REPORTE ---
    with tab2:
        st.markdown("<br><h4>Dictamen Estadístico de Ajuste</h4>", unsafe_allow_html=True)
        decision = "RECHAZAR H0" if ks_pval < alpha else "NO RECHAZAR H0"
        
        st.code(f"""
        =========================================================
          TEST KOLMOGOROV-SMIRNOV - RESULTADOS ANALÍTICOS
        =========================================================
          Modelo de ajuste     : {dist_sel}
          Método de parametriz : {modo_estimacion}
          Muestra observada (n): {len(data)}
          Nivel de Alpha (α)   : {alpha}

          Estadístico D        : {ks_stat:.6f}
          P-Value              : {ks_pval:.6e}
          
          Conclusión del Test  : {decision}
        =========================================================
        """, language='text')
        
        if ks_pval < alpha:
            st.markdown(f'<div class="alert-danger"><b>❌ RECHAZO DE HIPÓTESIS:</b> La evidencia estadística (p-value={ks_pval:.4g}) indica que los datos no siguen una distribución {dist_sel}.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-success"><b>✅ VALIDACIÓN EXITOSA:</b> El p-value ({ks_pval:.4g}) es mayor que α. La distribución {dist_sel} captura adecuadamente el comportamiento probabilístico de la muestra.</div>', unsafe_allow_html=True)

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
        
        if ks_pval >= alpha:
            st.success("Condición matemática superada. Motor de simulación desbloqueado.")
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
            st.markdown('<div class="alert-danger" style="margin-top: 0;"><b>🛑 Bloqueado por Riesgo de Modelo:</b> Ajusta el modelo a una distribución válida (p-value ≥ α) antes de simular.</div>', unsafe_allow_html=True)

    # --- TAB 5: BACKTESTING VAR ---
    with tab5:
        st.markdown("<br><h4>Prueba de Excepciones Históricas (Kupiec POF)</h4>", unsafe_allow_html=True)
        
        var_teorico = dist_obj.ppf(alpha)
        
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