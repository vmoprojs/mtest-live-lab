import numpy as np
import plotly.express as px
import plotly.graph_objects as go

BLUE = "#1768CB"
TEAL = "#12A4A0"
ORANGE = "#D98A26"


def style(fig, title=None, height=350):
    fig.update_layout(template="plotly_white", height=height, title=title,
        font=dict(family="Arial, sans-serif", size=16, color="#193653"),
        margin=dict(l=35, r=20, t=55, b=35), paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", y=1.14, x=0),
        colorway=[BLUE, TEAL, ORANGE], hoverlabel=dict(font_size=15))
    fig.update_xaxes(showgrid=False, zerolinecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#EAF0F6", zerolinecolor="#CBD5E1")
    return fig


def scatter(frame, predictors):
    # Display at most 1500 deterministic rows; engines always use all rows.
    shown = frame if len(frame) <= 1500 else frame.sample(1500, random_state=27)
    fig = px.scatter(shown, x=predictors[0], y=predictors[1], opacity=.55)
    fig.update_traces(marker=dict(color=BLUE, size=7))
    return style(fig, "Dos predictores, información compartida")


def correlation(frame, predictors):
    corr = frame[predictors].corr()
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                    color_continuous_scale=["#123A63", "#F3F6FB", "#1768CB"], aspect="auto")
    fig.update_layout(coloraxis_showscale=False)
    return style(fig, "Correlación observada")


def coefficients(table):
    fig = go.Figure(go.Scatter(x=table["Coeficiente"], y=table["Predictor"], mode="markers",
        marker=dict(color=BLUE, size=13), error_x=dict(type="data", symmetric=False,
            array=table["IC superior"]-table["Coeficiente"],
            arrayminus=table["Coeficiente"]-table["IC inferior"], color=BLUE)))
    fig.add_vline(x=0, line_dash="dash", line_color="#8293A5")
    fig.update_xaxes(title="Coeficiente OLS · unidades originales")
    return style(fig, "Coeficientes e intervalos clásicos del 95 %", height=290)


def bootstrap_plots(result, predictor):
    j = result.predictors.index(predictor)
    aux, g = result.boot_aux[:, j], result.boot_global
    hist = go.Figure()
    for vals, label, color in [(g, "R² global", "#ADBACA"), (aux, "R² auxiliar", BLUE)]:
        hist.add_trace(go.Histogram(x=vals[np.isfinite(vals)], name=label,
            marker_color=color, opacity=.65, nbinsx=35))
    hist.update_layout(barmode="overlay")
    hist.add_vline(x=result.config.r2_threshold, line_color=ORANGE, line_dash="dash",
                   annotation_text="Umbral c", annotation_position="top left")
    hist.update_xaxes(title="R²", range=[min(0, np.nanmin(aux)), 1.01])
    hist.update_yaxes(title="Número de réplicas")
    delta = aux-g
    klein = go.Figure(go.Histogram(x=delta[np.isfinite(delta)], nbinsx=35,
                                 marker_color=TEAL, name="R² auxiliar − R² global"))
    klein.add_vline(x=0, line_color=ORANGE, line_dash="dash", annotation_text="Klein: diferencia > 0")
    klein.update_xaxes(title="R² auxiliar − R² global")
    klein.update_yaxes(title="Número de réplicas")
    return style(hist, "Regla VIF · superar c"), style(klein, "Regla de Klein · superar al modelo")
