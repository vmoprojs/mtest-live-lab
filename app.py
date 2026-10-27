import os
for thread_setting in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(thread_setting, "1")

from dataclasses import asdict
from importlib.metadata import version
import json
import time
import platform

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.exceptions import ConvergenceWarning

from lab.data import (BootstrapConfig, DataError, SCENARIOS, csv_bytes, data_id,
                      descriptive_ols, read_csv, simulate, validate_data)
from lab.engines import EngineError, ROOT, compare_results, r_probe, run_python, run_r
from lab.ml import compare_models
from lab.runtime import prepare_runtime, analysis_slot
from lab.plots import bootstrap_plots, coefficients, correlation, scatter, style
from lab.reproduce import reproduction_zip, snippets

st.set_page_config(page_title="MTest Live Lab · R ↔ Python", page_icon="🔬", layout="wide")
st.markdown("""<style>
.block-container {padding-top:1.4rem; padding-bottom:2rem; max-width:1600px;}
[data-testid="stSidebar"] {border-right:1px solid #DDE7F2;}
h1,h2,h3 {letter-spacing:-.035em;}
.hero {background:linear-gradient(115deg,#102D50,#155BA5);border-radius:18px;
padding:24px 30px;color:white;margin-bottom:18px;display:flex;justify-content:space-between;gap:24px;}
.hero h1 {font-size:38px;color:white;margin:0;line-height:1.15;}
.hero p {font-size:17px;margin:9px 0 0;color:#D8E9FC;}
.eyebrow {font-size:12px;letter-spacing:2px;font-weight:700;color:#9BCEFF;margin-bottom:9px;}
.celebrate {min-width:155px;text-align:right;align-self:center;font-size:15px;color:#D8E9FC;}
.celebrate strong {display:block;font-size:43px;color:white;line-height:1.1;}
[data-testid="stMetric"] {background:#F1F5FA;border:1px solid #E1E9F3;padding:12px 18px;border-radius:12px;}
[data-testid="stMetricLabel"] {font-size:15px;}
[data-testid="stMetricValue"] {font-size:28px;}
button[data-baseweb="tab"] {font-size:18px;padding-left:20px;padding-right:20px;}
div[data-testid="stAlert"] p {font-size:16px;}
@media (max-width:800px) {.hero {display:block}.celebrate {display:none}.hero h1 {font-size:30px}}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparando el motor R oficial…")
def engine_status(runtime_revision):
    return prepare_runtime()


@st.cache_data(max_entries=20, show_spinner=False)
def cached_simulation(n, rho, noise, seed):
    # Synthetic data only. Uploaded data/results are never put in shared cache.
    return simulate(n, rho, noise, seed)


def reset():
    st.session_state.clear()
    # Explicit selector values also reset their browser-side state.
    st.session_state.update(source="Simulador", scenario="Alta", rho=.95)


def apply_scenario():
    st.session_state.rho = SCENARIOS[st.session_state.scenario]


def chart(fig, key):
    st.plotly_chart(fig, use_container_width=True, theme=None, key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


status = engine_status("conda-rscript-v3")
python_ok = version("mtest-py") == "0.1.4"
with st.sidebar:
    st.markdown("### MTest Live Lab")
    st.caption("MTEST-LIVE-02 · Laboratorio web")
    st.button("Reiniciar demostración", on_click=reset, width="stretch")
    st.divider()
    st.markdown("#### A · Datos compartidos")
    source = st.radio("Fuente", ["Simulador", "Cargar CSV"], horizontal=True, key="source")
    if source == "Simulador":
        st.selectbox("Escenario", ["Alta", "Baja", "Casi perfecta"], key="scenario", on_change=apply_scenario)
        st.session_state.setdefault("rho", .95)
        rho = st.slider("Correlación poblacional ρ", -.999, .999, step=.001, key="rho", format="%.3f")
        n = st.slider("Observaciones", 50, 1000, 300, 50, key="n")
        noise = st.slider("Ruido σ", 0.1, 5., 1., .1, key="noise")
        seed_data = st.number_input("Semilla de datos", 0, 2**31-1, 27, key="seed_data")
        frame = cached_simulation(n, rho, noise, seed_data)
        response, predictors = "y", ["X1", "X2"]
    else:
        st.caption("CSV UTF-8 · máximo 2 MB / 5000 filas. Solo variables numéricas.")
        uploaded = st.file_uploader("Seleccionar CSV", type="csv", key="upload")
        st.caption("Use exclusivamente datos públicos o sintéticos, sin datos personales. Se procesan en el servidor y permanecen en la memoria de su sesión. El CSV temporal de R se elimina al terminar. Reiniciar limpia los resultados de esta sesión.")
        if uploaded is None:
            st.info("Cargue un archivo o vuelva al simulador.")
            st.stop()
        try:
            frame = read_csv(uploaded.getvalue())
        except DataError as exc:
            st.error(str(exc)); st.stop()
        numeric = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]
        if len(numeric) < 3:
            st.error("Se necesitan una respuesta y al menos dos predictores numéricos."); st.stop()
        response = st.selectbox("Variable dependiente", numeric, key="response")
        choices = [c for c in numeric if c != response]
        predictors = st.multiselect("Predictores numéricos", choices, default=choices[:2], max_selections=8)
    try:
        data, notices = validate_data(frame, response, predictors)
    except DataError as exc:
        st.error(str(exc)); st.stop()
    st.divider()
    st.markdown("#### Bootstrap · ambos motores")
    n_boot = st.select_slider("Réplicas · n_boot / nboot", [100, 200, 500], value=200, key="n_boot")
    if "nsam" not in st.session_state or st.session_state.get("previous_n") != len(data):
        st.session_state.nsam = min(len(data), 2000)
        st.session_state.previous_n = len(data)
    nsam = st.number_input("Tamaño de remuestreo · nsam", 20, 2000, key="nsam", step=10)
    boot_seed = st.number_input("Semilla bootstrap", 0, 2**31-1, 2026, key="boot_seed")
    threshold = st.slider("Umbral auxiliar R² · c", .5, .99, .90, .01, key="threshold")
    st.caption(f"VIF equivalente: {1/(1-threshold):.2f} · se evalúa R² auxiliar > c")
    st.caption("Los controles no ejecutan el bootstrap. Use los botones del panel.")
    st.divider()
    st.caption(f"{status.get('R', 'R no disponible')} · MTest {status.get('MTest', 'no disponible')}\n\nPython {platform.python_version()} · mtest-py {version('mtest-py')}")

config = BootstrapConfig(n_boot, nsam, boot_seed, threshold)
try:
    config.validate(len(predictors))
except DataError as exc:
    st.error(str(exc)); st.stop()
fingerprint = data_id(data)
active = fingerprint + json.dumps(asdict(config), sort_keys=True)
if st.session_state.get("active") != active:
    for key in ("result_R", "result_Python", "ml_result"):
        st.session_state.pop(key, None)
    st.session_state.active = active

st.markdown("""<div class="hero"><div><div class="eyebrow">WEBINAR · 27 OCT 2026 · 19:00 ECUADOR</div>
<h1>MTest Live Lab <span style="color:#80C0FF">/ R ↔ Python</span></h1>
<p>Multicolinealidad: ¿sigue importando en la era del Machine Learning?</p>
</div><div class="celebrate"><strong>10K+</strong>descargas de MTest en CRAN<br>Una comunidad, dos motores.</div></div>""", unsafe_allow_html=True)

observed_r2, ols_table = descriptive_ols(data, response, predictors)
cols = st.columns(4)
cols[0].metric("Observaciones / predictores", f"{len(data)} / {len(predictors)}")
cols[1].metric("R² global · OLS descriptivo", f"{observed_r2:.3f}")
cols[2].metric("VIF máximo · observado", f"{ols_table['VIF'].max():,.2f}")
cols[3].metric("Umbral VIF equivalente", f"{1/(1-threshold):.2f}")
st.caption(f"Misma muestra en ambos motores · intercepto incluido · ID {fingerprint[:10]} · resultados se invalidan al cambiar datos o parámetros")
for notice in notices:
    st.warning(notice)


def execute(engine):
    key = "result_"+engine
    st.session_state.pop(key, None)
    try:
        with analysis_slot(), st.spinner(f"Ejecutando el paquete oficial en {engine}…"):
            st.session_state[key] = (run_r if engine == "R" else run_python)(data, response, predictors, config)
    except (EngineError, DataError) as exc:
        st.error(str(exc))


def show_result(engine):
    result = st.session_state.get("result_"+engine)
    if result is None:
        st.info("Listo para ejecutar. Los resultados aparecerán al pulsar el botón.")
        return
    if not result.clean:
        st.warning("Hay valores no finitos o réplicas degeneradas. Las ASL afectadas se ocultan; "
                   "R excluye NA y Python puede contar comparaciones con NaN como falsas. "
                   "El CSV conserva las salidas originales, marcadas por separado. No compare esas ASL.")
    a, b, c = st.columns(3)
    a.metric("R² global · paquete oficial", f"{result.r2_global:.4f}")
    b.metric("Tiempo total", f"{result.seconds:.2f} s")
    c.metric("Réplicas / tamaño", f"{config.n_boot} / {config.nsam}")
    st.caption(f"Motor {engine} · versión {result.version} · seed={config.seed} · c={config.r2_threshold:.2f}. "
               "Tiempo R incluye arranque y transferencia CSV/JSON.")
    st.dataframe(result.table().drop(columns=["Réplicas válidas VIF", "Réplicas válidas Klein"]),
                 hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4f")
                                for c in ["R² auxiliar", "VIF observado", "ASL VIF", "ASL Klein"]})
    selected = st.selectbox("Predictor para las distribuciones", predictors, key="plot_"+engine)
    left, right = st.columns(2)
    hist, klein = bootstrap_plots(result, selected)
    with left: chart(hist, "hist_"+engine)
    with right: chart(klein, "klein_"+engine)
    st.caption("ASL VIF = proporción de réplicas con R² auxiliar > c. "
               "ASL Klein = proporción con R² auxiliar > R² global en la misma réplica. "
               "Una ASL mayor indica que la regla se cumple con más frecuencia; no es un p-valor paramétrico tradicional.")
    a, b = st.columns(2)
    a.download_button(f"Descargar resultados {engine} · CSV", csv_bytes(result.export_table()),
                      f"mtest_{engine}_resultados.csv", "text/csv", key="download_"+engine)
    distributions = pd.DataFrame(result.boot_aux, columns=[f"R2_aux:{p}" for p in predictors])
    distributions.insert(0, "R2_global", result.boot_global)
    b.download_button(f"Distribuciones {engine} · CSV", csv_bytes(distributions),
                      f"mtest_{engine}_bootstrap.csv", "text/csv", key="dist_"+engine)
    with st.expander("Calidad del bootstrap y denominadores"):
        st.dataframe(result.table()[["Predictor", "Réplicas válidas VIF", "Réplicas válidas Klein"]], hide_index=True)
        st.write("Las réplicas con R² auxiliar ≥ 1 o no finito se marcan como degeneradas. "
                 "No se reemplazan ni se recalculan las ASL del paquete. Los histogramas omiten únicamente valores no finitos.")


r_tab, data_tab, py_tab, comparison_tab = st.tabs(
    ["B · MTest en R", "A · Datos", "C · MTest en Python", "D · R ↔ Python"], default="B · MTest en R")
with data_tab:
    st.subheader("La correlación cambia lo que podemos separar")
    st.write("El modelo puede predecir bien y, aun así, estimar con poca precisión el efecto individual de cada predictor.")
    if source == "Simulador":
        st.latex(r"X_2=\rho X_1+\sqrt{1-\rho^2}\,Z,\qquad Y=1+2X_1-2X_2+\sigma\varepsilon")
        st.caption("X₁, Z y ε independientes N(0,1). ρ es poblacional; la correlación muestral puede diferir. "
                   "Al variar ρ se conserva la semilla, pero también cambia la relación señal/ruido de Y.")
    left, right = st.columns([1.25, 1])
    with left: chart(scatter(data, predictors), "scatter")
    with right: chart(correlation(data, predictors), "corr")
    if len(data) > 1500: st.caption("El scatter muestra 1500 filas; los diagnósticos utilizan todas las filas.")
    left, right = st.columns([1.25, 1])
    with left: chart(coefficients(ols_table), "coef_ols")
    with right:
        st.dataframe(ols_table[["Predictor", "R² auxiliar", "VIF"]], hide_index=True, width="stretch")
        st.caption("OLS descriptivo, sin bootstrap. IC clásicos bajo homocedasticidad y supuestos del modelo; no son intervalos MTest ni prueban causalidad.")
    st.download_button("Descargar datos numéricos · CSV", csv_bytes(data), "datos_mtest.csv", "text/csv")
    with st.expander("Validación, rango y cardinalidad"):
        st.dataframe(pd.DataFrame({"Mínimo": data.min(), "Máximo": data.max(), "Valores únicos": data.nunique(),
                                   "Faltantes": data.isna().sum()}))
        st.write("Sin imputación ni eliminación automática de filas. Respuesta y predictores numéricos, "
                 "sin constantes, infinitos ni dependencia lineal exacta. No se expanden factores o fórmulas.")

r_code, py_code = snippets(response, predictors, config)
with r_tab:
    st.subheader("MTest en R · el punto de partida")
    st.write("De un VIF observado a la frecuencia con que se supera un umbral al remuestrear los datos.")
    if not status.get("available"):
        st.error("Motor R no disponible: " + status.get("error", "Revise Rscript."))
    if st.button("Ejecutar MTest en R", type="primary", disabled=not status.get("available")):
        execute("R")
    show_result("R")
    with st.expander("Código R reproducible · llamada oficial"):
        st.code(r_code, language="r")
        st.caption("Este ejemplo usa los mismos datos y argumentos; el adaptador ejecutado además valida el diseño y serializa las salidas.")
        st.download_button("Descargar adaptador R ejecutado", (ROOT/"r/run_mtest.R").read_text(), "run_mtest.R")

with py_tab:
    st.subheader("MTest en Python · la misma pregunta")
    st.write("La función oficial recibe los mismos valores, predictores y parámetros, con intercepto explícito.")
    if not python_ok: st.error("Motor Python no disponible: se requiere mtest-py 0.1.4.")
    if st.button("Ejecutar MTest en Python", type="primary", disabled=not python_ok): execute("Python")
    show_result("Python")
    with st.expander("Código Python reproducible · llamada oficial"):
        st.code(py_code, language="python")

with comparison_tab:
    st.subheader("Mismo diseño. Dos implementaciones reales.")
    st.write("Las métricas observadas deben coincidir dentro de la precisión numérica. Las proporciones bootstrap pueden diferir.")
    if st.button("Ejecutar ambos", type="primary", disabled=not (status.get("available") and python_ok)):
        execute("R"); execute("Python")
        if "result_R" in st.session_state and "result_Python" in st.session_state:
            st.rerun()
    r_result, py_result = st.session_state.get("result_R"), st.session_state.get("result_Python")
    if r_result is not None and py_result is not None:
        compatible, reasons, comparison = compare_results(r_result, py_result)
        if compatible:
            st.success("Comparables: datos idénticos verificados, mismo diseño con intercepto y métricas observadas coherentes.")
        else:
            st.error("Comparación no válida. " + " ".join(reasons))
        st.caption(f"R² global: R={r_result.r2_global:.10f} · Python={py_result.r2_global:.10f} · "
                   f"Δ={abs(r_result.r2_global-py_result.r2_global):.2e} · {config.n_boot} réplicas por motor. "
                   f"Tiempo R {r_result.seconds:.2f} s / Python {py_result.seconds:.2f} s.")
        st.dataframe(comparison[["Predictor", "VIF · R", "VIF · Py", "ASL VIF · R", "ASL VIF · Py",
                                 "ASL Klein · R", "ASL Klein · Py"]], hide_index=True, width="stretch")
        with st.expander("Diferencias absolutas y R² auxiliares"):
            st.dataframe(comparison, hide_index=True, width="stretch")
        exported_comparison = comparison.assign(n_boot=config.n_boot, nsam=config.nsam, seed=config.seed,
            r2_threshold=config.r2_threshold, version_R=r_result.version, version_Python=py_result.version,
            dataset_sha256=fingerprint, comparable=compatible)
        st.download_button("Descargar comparación · CSV", csv_bytes(exported_comparison), "comparacion_mtest.csv", "text/csv")
    else:
        st.info("Ejecute ambos motores para ver el panel comparativo.")
    st.warning("La misma semilla no genera los mismos remuestreos entre R y Python. "
               "No se exige igualdad de distribuciones ni de ASL. Aumentar las réplicas reduce la variación Monte Carlo.")

with st.expander("Econometría vs Machine Learning · predecir no equivale a identificar efectos"):
    st.write("MTest diagnostica la colinealidad del diseño en el contexto del modelo; no certifica rendimiento predictivo. "
             "Ridge y Lasso regularizan los coeficientes: no eliminan la correlación entre predictores.")
    a, b, c = st.columns(3)
    ridge_alpha = a.number_input("α Ridge", .01, 1000., 10., key="ridge_alpha")
    lasso_alpha = b.number_input("α Lasso", .001, 10., .1, format="%.3f", key="lasso_alpha")
    ml_seed = c.number_input("Semilla train/test y estabilidad", 0, 2**31-1, 27, key="ml_seed")
    ml_key = (active, ridge_alpha, lasso_alpha, ml_seed)
    if st.button("Comparar OLS, Ridge y Lasso"):
        st.session_state.pop("ml_result", None)
        try:
            with analysis_slot(), st.spinner("Ajustando pipelines y 80 remuestreos solo del entrenamiento…"):
                start = time.perf_counter()
                ml = compare_models(data, response, predictors, ml_seed, 80, ridge_alpha, lasso_alpha)
                st.session_state.ml_result = (ml_key, ml, time.perf_counter()-start)
        except (EngineError, ValueError, ConvergenceWarning, np.linalg.LinAlgError) as exc:
            st.error("No se completó el experimento ML. Revise los datos o reintente cuando el servidor esté libre.")
    stored = st.session_state.get("ml_result")
    if stored and stored[0] == ml_key:
        _, ml, elapsed = stored
        st.caption(f"Train 70 % / test 30 % · semilla {ml_seed} · 80 remuestreos de train · {elapsed:.2f} s. "
                   "Los α son ilustrativos, sin búsqueda sobre test. Coeficientes en unidades originales.")
        a, b = st.columns([1, 2])
        with a:
            st.dataframe(ml["metrics"], hide_index=True)
            st.dataframe(ml["coefficients"].pivot(index="Predictor", columns="Modelo", values="Coeficiente"))
        with b:
            chart(style(px.box(ml["stability"], x="Predictor", y="Coeficiente", color="Modelo",
                               color_discrete_sequence=["#1768CB", "#12A4A0", "#D98A26"]),
                         "Estabilidad de coeficientes · remuestreo de train"), "ml_stability")
        st.caption("Cada réplica vuelve a ajustar su StandardScaler únicamente en sus filas de entrenamiento. "
                   "El test queda fuera de ajuste y remuestreo. Una sola partición no permite declarar un ganador general; "
                   "explorar repetidamente el test deja de ser una evaluación final independiente.")

with st.expander("Para explicar en vivo · método, reproducibilidad y referencias"):
    a, b, c = st.columns(3)
    a.markdown("**Explicar**\n\nRelacionar variables bajo supuestos explícitos. La asociación no basta para afirmar causalidad.")
    b.markdown("**Predecir**\n\nEvaluar el error en observaciones fuera del entrenamiento.")
    c.markdown("**Interpretar**\n\nSeparar aportes individuales y evaluar cuánto cambian al variar la muestra.")
    st.latex(r"\mathrm{VIF}_j=\frac{1}{1-R_j^2},\quad\mathrm{ASL}_{VIF,j}=\frac{\#\{R_{j,b}^2>c\}}{B},\quad\mathrm{ASL}_{Klein,j}=\frac{\#\{R_{j,b}^2>R_{g,b}^2\}}{B}")
    st.write("Fórmulas para réplicas válidas. R² centrado = 1 − SSE/SST, con intercepto en las regresiones global y auxiliares. "
             "Bootstrap de pares: se remuestrean filas completas. Esta versión presupone observaciones independientes; "
             "series de tiempo y grupos requieren otro esquema. Cambiar nsam respecto de n cambia el experimento de remuestreo.")
    st.write("Las salidas se leen de MTest 1.0.4 y mtest-py 0.1.4. La app no reimplementa los tests ni recalcula sus ASL. "
             "El transporte CSV usa números hexadecimales para conservar los doubles exactamente; el código descargable los convierte a numéricos.")
    st.download_button("Descargar kit reproducible · R + Python + datos", reproduction_zip(data, response, predictors, config),
                       "mtest_reproducible.zip", "application/zip")
    st.markdown("[MTest en CRAN](https://cran.r-project.org/package=MTest) · "
                "[mtest-py en PyPI](https://pypi.org/project/mtest-py/) · "
                "[Artículo, Morales-Oñate y Morales-Oñate (2023)](https://doi.org/10.33333/rp.vol51n2.05) · "
                "[Repositorio R](https://github.com/vmoprojs/MTest)")
st.caption("MTest Live Lab · Webinar académico de 60 minutos · Demo sugerida: 12 minutos · Demostración académica en servidor · use solo datos públicos o sintéticos.")
