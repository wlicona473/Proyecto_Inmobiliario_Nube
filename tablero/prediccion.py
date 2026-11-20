"""
Pestaña de predicción de RE-MONITOR.

- Formulario con las 8 variables explicativas del modelo -> valor monetario estimado.
- Explicación: aporte de cada variable a esa predicción y a todo el modelo.
- Contexto de la pregunta de negocio (72,9 % de transacciones sin valor) y calidad del modelo.

El modelo predice LN(VALOR_REAL): valor registrado ajustado por IPC a pesos constantes de
diciembre de 2018. El tablero siempre aclara esa base.

Si la variable de entorno API_URL está definida (junto con API_PORT, p. ej. 8001), todo lo del modelo se
consulta a la API. Si no, se usa el modelo local (modelo/modelo_xgboost.pkl), útil para desarrollo.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

BASE = Path(__file__).parent
def _api_base():
    """Dirección de la API. Sigue la convención del curso: API_URL (IP o host) y API_PORT (p. ej. 8001).
    También acepta una URL completa en API_URL (p. ej. http://api:8001)."""
    url = os.getenv("API_URL", "").strip().rstrip("/")
    if not url:
        return ""
    if "://" not in url:
        url = "http://" + url
    puerto = os.getenv("API_PORT", "").strip()
    if puerto and ":" not in url.split("://", 1)[1]:
        url = f"{url}:{puerto}"
    return url


API_URL = _api_base()
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE / "modelo" / "modelo_xgboost.pkl"))
CATALOGOS_PATH = BASE / "catalogos.json"        # nombres legibles (generar_artefactos.py)
CUBO_PATH = BASE / "cubo.csv"                   # datos agregados del dashboard
METRICAS_PATH = BASE / "metricas_modelo.json"   # métricas de evaluación del modelo
IPC_PATH = BASE / "ipc_factores.csv"            # opcional: YEAR,FACTOR (ver README)

VARS = [
    "YEAR_RADICA", "DIVIPOLA", "TIPO_PREDIO_ZONA", "COD_NATUJUR",
    "REGISTROS_PREVIOS", "DIAS_DESDE_ULTIMO_REGISTRO",
    "ANIOS_DESDE_PRIMER_REGISTRO", "ANIOS_CON_ACTIVIDAD_PREVIA",
]
ETIQUETAS = {
    "YEAR_RADICA": "Año de radicación",
    "DIVIPOLA": "Municipio",
    "TIPO_PREDIO_ZONA": "Tipo de predio",
    "COD_NATUJUR": "Acto jurídico",
    "REGISTROS_PREVIOS": "Registros previos de la matrícula",
    "DIAS_DESDE_ULTIMO_REGISTRO": "Días desde el último registro",
    "ANIOS_DESDE_PRIMER_REGISTRO": "Años desde el primer registro",
    "ANIOS_CON_ACTIVIDAD_PREVIA": "Años con actividad previa",
}
BASE_PESOS = "pesos constantes de diciembre de 2018 (ajustados por IPC)"

_cache = {}


# --------------------------------------------------------------------------- #
# Acceso al modelo: por API o local
# --------------------------------------------------------------------------- #
def _get(ruta, intentos=15, espera=2):
    """GET a la API con reintentos (la API puede tardar unos segundos en arrancar)."""
    import requests
    ultimo = None
    for _ in range(intentos):
        try:
            r = requests.get(f"{API_URL}{ruta}", timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            ultimo = e
            time.sleep(espera)
    raise RuntimeError(f"No se pudo contactar la API en {API_URL}: {ultimo}")


def _variable_de(nombre_columna):
    limpio = nombre_columna.split("__", 1)[1]
    return next((v for v in VARS if limpio.startswith(v)), None)


def _local():
    """Carga el modelo local (solo si no hay API)."""
    if "local" not in _cache:
        import joblib
        pipe = joblib.load(MODEL_PATH)
        pre, reg = pipe.named_steps["preprocesamiento"], pipe.named_steps["modelo"]
        ohe = pre.named_transformers_["categoricas"].named_steps["codificacion"]
        _cache["local"] = {
            "pre": pre, "reg": reg,
            "mapa": [_variable_de(n) for n in pre.get_feature_names_out()],
            "opciones": {v: [str(c) for c in cats] for v, cats in
                         zip(["DIVIPOLA", "TIPO_PREDIO_ZONA", "COD_NATUJUR"], ohe.categories_)},
        }
    return _cache["local"]


def _opciones():
    if "opciones" not in _cache:
        _cache["opciones"] = _get("/opciones") if API_URL else _local()["opciones"]
    return _cache["opciones"]


def _importancia():
    """Importancia global por variable (% de la ganancia de XGBoost), ordenada."""
    if "importancia" not in _cache:
        if API_URL:
            filas = _get("/importancia")
        else:
            m = _local()
            imp = dict.fromkeys(VARS, 0.0)
            for var, g in zip(m["mapa"], m["reg"].feature_importances_):
                if var:
                    imp[var] += float(g)
            total = sum(imp.values()) or 1.0
            filas = sorted(({"variable": v, "importancia_pct": 100 * g / total} for v, g in imp.items()),
                           key=lambda f: -f["importancia_pct"])
        d = pd.DataFrame(filas)
        d["etiqueta"] = d["variable"].map(ETIQUETAS)
        _cache["importancia"] = d
    return _cache["importancia"]


def predecir(fila: dict) -> dict:
    """Devuelve {'valor_pesos_2018', 'log_valor', 'contribuciones': DataFrame, 'avisos': [...]}."""
    if API_URL:
        import requests
        r = requests.post(f"{API_URL}/predict", json=fila, timeout=30)
        if r.status_code == 422:
            raise ValueError("Algún dato está fuera del rango permitido (por ejemplo, el año debe ser 2015–2023).")
        r.raise_for_status()
        d = r.json()
    else:
        import xgboost as xgb
        m = _local()
        Xt = m["pre"].transform(pd.DataFrame([fila], columns=VARS))
        contrib = m["reg"].get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)[0]
        base, contrib = float(contrib[-1]), contrib[:-1]
        agg = dict.fromkeys(VARS, 0.0)
        for var, c in zip(m["mapa"], contrib):
            if var:
                agg[var] += float(c)
        log_valor = base + sum(agg.values())
        d = {"log_valor": log_valor, "valor_pesos_2018": float(np.expm1(log_valor)),
             "contribuciones": [{"variable": v, "aporte_log": a} for v, a in agg.items()], "avisos": []}
    contrib_df = pd.DataFrame(d["contribuciones"])
    contrib_df["etiqueta"] = contrib_df["variable"].map(ETIQUETAS)
    return {"valor": d["valor_pesos_2018"], "log_valor": d["log_valor"],
            "contribuciones": contrib_df, "avisos": d.get("avisos", [])}


# --------------------------------------------------------------------------- #
# Catálogos, métricas, IPC y utilidades
# --------------------------------------------------------------------------- #
def _catalogos():
    """Opciones de los desplegables: valores del modelo, etiquetas de catalogos.json."""
    if "cat" in _cache:
        return _cache["cat"]
    nombres = {}
    if CATALOGOS_PATH.exists():
        nombres = json.loads(CATALOGOS_PATH.read_text(encoding="utf-8"))
    out = {}
    for var, valores in _opciones().items():
        etiq = nombres.get(var, {})

        def _nombre(v, etiq=etiq):
            # el modelo guarda algunos códigos con coma ('1,004'); el catálogo los trae sin coma
            return etiq.get(str(v)) or etiq.get(str(v).replace(",", ""))

        out[var] = [{"value": str(v), "label": f"{_nombre(v)} ({v})" if _nombre(v) else str(v)}
                    for v in valores]
    _cache["cat"] = out
    return out


def _metricas():
    if "met" not in _cache:
        try:
            _cache["met"] = json.loads(METRICAS_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            _cache["met"] = None
    return _cache["met"]


def _factor_ipc(anio):
    """Factor opcional para pasar de pesos de dic. 2018 a pesos corrientes del año (None si no hay archivo)."""
    try:
        if "ipc" not in _cache:
            d = pd.read_csv(IPC_PATH)
            _cache["ipc"] = dict(zip(d["YEAR"].astype(int), d["FACTOR"].astype(float)))
        return _cache["ipc"].get(int(anio))
    except Exception:  # noqa: BLE001
        return None


def _cop(x):
    return "$ {:,.0f}".format(x).replace(",", ".")


def _cubo():
    if "cubo" not in _cache:
        try:
            c = pd.read_csv(CUBO_PATH)
            c = c.dropna(subset=["YEAR_RADICA"]).copy()
            c["YEAR_RADICA"] = c["YEAR_RADICA"].astype(int)
        except Exception:  # noqa: BLE001
            c = None
        _cache["cubo"] = c
    return _cache["cubo"]


def _millones(x):
    return f"{x / 1e6:.2f}M"


def _kpi(titulo, valor, detalle, clase):
    return html.Div([html.P(titulo, className="rm-kpi-t"), html.H2(valor, className="rm-kpi-v"),
                     html.Small(detalle, className="rm-help")], className=f"rm-kpi {clase}")


# --------------------------------------------------------------------------- #
# Bloques de la pestaña
# --------------------------------------------------------------------------- #
def _fig_cobertura(c):
    por_anio = c.groupby("YEAR_RADICA")[["con_valor", "sin_valor"]].sum().reset_index()
    fig = go.Figure()
    fig.add_bar(x=por_anio["YEAR_RADICA"], y=por_anio["con_valor"], name="Con valor registrado",
                marker_color="#3dae7a")
    fig.add_bar(x=por_anio["YEAR_RADICA"], y=por_anio["sin_valor"], name="Sin valor (a estimar)",
                marker_color="#1f5fa8")
    fig.update_layout(barmode="stack", margin=dict(l=10, r=10, t=10, b=30), plot_bgcolor="white",
                      paper_bgcolor="white", legend=dict(orientation="h", y=1.15), yaxis_title="Transacciones")
    fig.update_xaxes(dtick=1)
    return fig


def _bloque_contexto():
    n_modelo = len(_catalogos()["DIVIPOLA"])
    intro = html.Div([
        html.H2("¿Cómo estimar el valor monetario de las transacciones inmobiliarias sin valor registrado?",
                className="rm-intro-t"),
        html.P("Bogotá D.C. y Cundinamarca, 2015–2023. El 72,9 % de las transacciones no tiene valor registrado: "
               "RE-MONITOR aprende de las que sí lo tienen y estima el valor a partir de sus características.",
               className="rm-intro-p"),
        html.P("Cómo usarlo: (1) ingrese las características de la transacción, (2) presione «Estimar valor» "
               "y (3) revise qué factores explican ese valor.", className="rm-steps-line"),
        html.Small(
            f"Alcance: el valor se expresa en {BASE_PESOS}. La estimación es aproximada, no es un avalúo comercial "
            f"ni establece causalidad, y supone que las transacciones sin valor se parecen a las que sí lo tienen. "
            f"El modelo solo reconoce {n_modelo} municipios.", className="rm-help"),
    ], className="rm-card rm-intro")

    c = _cubo()
    if c is None or not {"transacciones", "con_valor", "sin_valor"} <= set(c.columns):
        return intro, []
    total, con, sin = c["transacciones"].sum(), c["con_valor"].sum(), c["sin_valor"].sum()
    base = (con + sin) or 1
    try:
        n_muni = f"{n_modelo} de {c['MUNICIPIO'].nunique()}"
    except Exception:  # noqa: BLE001
        n_muni = str(n_modelo)
    kpis = html.Div([
        _kpi("Transacciones 2015–2023", _millones(total), "Bogotá D.C. y Cundinamarca", "rm-k1"),
        _kpi("Con valor registrado", _millones(con), f"{con / base:.1%} · base de aprendizaje del modelo", "rm-k2"),
        _kpi("Sin valor registrado", _millones(sin), f"{sin / base:.1%} · lo que el modelo permite estimar", "rm-k3"),
        _kpi("Municipios que reconoce el modelo", n_muni, "del total de municipios de la base", "rm-k4"),
    ], className="rm-kpis")
    cobertura = html.Div([
        html.H3("Magnitud del problema: transacciones con y sin valor registrado, por año"),
        dcc.Graph(figure=_fig_cobertura(c), config={"displayModeBar": False}, style={"height": "260px"}),
    ], className="rm-card")
    return intro, [kpis, cobertura]


def _bloque_metricas():
    m = _metricas()
    if not m:
        return None
    factor = float(np.exp(m["MAE"]))
    tarjetas = html.Div([
        _kpi("MAE", f"{m['MAE']:.4f}", "error absoluto medio (en logaritmo)", "rm-k1"),
        _kpi("RMSE", f"{m['RMSE']:.4f}", "raíz del error cuadrático medio", "rm-k2"),
        _kpi("R²", f"{m['R2']:.4f}", "variación explicada por el modelo", "rm-k3"),
        _kpi("Error típico", f"× ÷ {factor:.1f}", "la estimación puede desviarse por ese factor", "rm-k4"),
    ], className="rm-kpis")
    return html.Div([
        html.H3("Calidad del modelo"),
        html.P(f"{m['modelo']} · {m['conjunto']}. Escala: {m['escala']}.", className="rm-help"),
        tarjetas,
        html.Small("El modelo explica solo una parte de la variación de los valores (R² moderado): úselo como "
                   "referencia para priorizar y analizar, no como valor comercial del inmueble.",
                   className="rm-help"),
    ], className="rm-card")


def _fig_importancia():
    fig = go.Figure()
    try:
        d = _importancia().sort_values("importancia_pct")
        fig.add_bar(x=d["importancia_pct"], y=d["etiqueta"], orientation="h", marker_color="#2E5BA8",
                    text=d["importancia_pct"].round(1).astype(str) + " %", textposition="outside",
                    cliponaxis=False)
        fig.update_xaxes(title="% de la importancia total", range=[0, d["importancia_pct"].max() * 1.2])
    except Exception as e:  # noqa: BLE001
        fig.add_annotation(text=f"No se pudo calcular la importancia: {e}", showarrow=False)
    fig.update_layout(margin=dict(l=10, r=60, t=10, b=40), plot_bgcolor="white", paper_bgcolor="white")
    fig.update_yaxes(automargin=True)
    return fig


def _campo(titulo, componente, ayuda=None):
    hijos = [html.Label(titulo, className="rm-label"), componente]
    if ayuda:
        hijos.append(html.Small(ayuda, className="rm-help"))
    return html.Div(hijos, className="rm-field")


def layout_prediccion():
    cat = _catalogos()
    anios = [{"label": str(a), "value": a} for a in range(2015, 2024)]
    num = lambda id_, val: dcc.Input(id=id_, type="number", min=0, value=val, className="rm-input")

    formulario = html.Div([
        html.H3("Datos de la transacción"),
        _campo("Año de radicación", dcc.Dropdown(id="in-year", options=anios, value=2022, clearable=False)),
        _campo("Municipio", dcc.Dropdown(id="in-divipola", options=cat["DIVIPOLA"], value="11001", clearable=False)),
        _campo("Tipo de predio", dcc.Dropdown(id="in-predio", options=cat["TIPO_PREDIO_ZONA"], value="URBANO",
                                              clearable=False)),
        _campo("Acto jurídico", dcc.Dropdown(id="in-natujur", options=cat["COD_NATUJUR"], value="125",
                                             clearable=False), "Naturaleza jurídica del acto registrado."),
        html.H3("Historial de la matrícula"),
        _campo("Registros previos", num("in-regprev", 2), "Cuántas transacciones tuvo antes la matrícula."),
        _campo("Días desde el último registro", num("in-dias", 311), "0 si es el primer registro."),
        _campo("Años desde el primer registro", num("in-primer", 3)),
        _campo("Años con actividad previa", num("in-actprev", 2)),
        html.Button("Estimar valor", id="btn-predecir", n_clicks=0, className="rm-btn"),
    ], className="rm-card rm-form")

    resultado = html.Div([
        html.H3("Valor monetario estimado"),
        html.Div(id="out-valor", className="rm-kpi-big", children="—"),
        html.Small(BASE_PESOS, className="rm-help"),
        html.Div(id="out-detalle", className="rm-detalle"),
        html.P("Ingrese los datos y presione «Estimar valor».", id="out-msg", className="rm-help"),
        html.H3("¿Qué factores explican este valor?"),
        dcc.Graph(id="out-contrib", config={"displayModeBar": False}, style={"height": "340px"}),
    ], className="rm-card rm-result")

    contexto_top, contexto_resto = _bloque_contexto()
    metricas = _bloque_metricas()
    hijos = [contexto_top, html.Div([formulario, resultado], className="rm-grid-2")]
    if metricas is not None:
        hijos.append(metricas)
    hijos += contexto_resto
    hijos.append(html.Div([
        html.H3("Importancia global de las variables en el modelo"),
        html.P("Peso de cada variable en las decisiones del modelo (ganancia de XGBoost, agrupada por variable). "
               "Indica asociación predictiva, no causalidad.", className="rm-help"),
        dcc.Graph(figure=_fig_importancia(), config={"displayModeBar": False}, style={"height": "340px"}),
    ], className="rm-card"))
    return html.Div(hijos, className="rm-pred")


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
def registrar_callbacks_prediccion(app):
    @app.callback(
        Output("out-valor", "children"),
        Output("out-detalle", "children"),
        Output("out-msg", "children"),
        Output("out-contrib", "figure"),
        Input("btn-predecir", "n_clicks"),
        State("in-year", "value"), State("in-divipola", "value"),
        State("in-predio", "value"), State("in-natujur", "value"),
        State("in-regprev", "value"), State("in-dias", "value"),
        State("in-primer", "value"), State("in-actprev", "value"),
        prevent_initial_call=True,
    )
    def _estimar(_, year, divipola, predio, natujur, regprev, dias, primer, actprev):
        vals = [year, divipola, predio, natujur, regprev, dias, primer, actprev]
        if any(v is None for v in vals):
            return "—", None, "⚠️ Complete todos los campos.", go.Figure()
        fila = dict(zip(VARS, vals))
        for k in ("YEAR_RADICA",):
            fila[k] = int(fila[k])
        try:
            r = predecir(fila)
        except Exception as e:  # noqa: BLE001
            return "—", None, f"No se pudo calcular la estimación: {e}", go.Figure()

        valor = r["valor"]
        m = _metricas()
        detalle = []
        if m:
            f = float(np.exp(m["MAE"]))
            detalle.append(html.Div(f"Rango orientativo: {_cop(valor / f)} a {_cop(valor * f)}", className="rm-range"))
        fac = _factor_ipc(year)
        if fac:
            detalle.append(html.Div(f"≈ {_cop(valor * fac)} COP corrientes de {year} (aproximado)",
                                    className="rm-range"))
        if m:
            detalle.append(html.Small(
                "El rango suma y resta el error absoluto medio del modelo (en logaritmo); "
                "no es un intervalo de confianza.", className="rm-help"))

        df = r["contribuciones"]
        df = df.reindex(df["aporte_log"].abs().sort_values().index)
        x = (np.exp(df["aporte_log"]) - 1) * 100  # efecto multiplicativo aproximado en %
        fig = go.Figure(go.Bar(
            x=x, y=df["etiqueta"], orientation="h", text=[f"{v:+.1f} %" for v in x], textposition="outside",
            cliponaxis=False, marker_color=["#2E8B57" if v >= 0 else "#C0392B" for v in x]))
        lim = float(np.abs(x).max()) * 1.35 if len(x) else 1
        fig.update_layout(xaxis_title="Efecto sobre el valor (%)", margin=dict(l=10, r=70, t=10, b=40),
                          plot_bgcolor="white", paper_bgcolor="white")
        fig.update_xaxes(range=[-lim if (x < 0).any() else 0, lim if (x > 0).any() else 0])
        fig.update_yaxes(automargin=True)

        top = df.iloc[::-1].iloc[0]
        msg = (f"Factor más influyente: {top['etiqueta']}. Las barras verdes suben el valor y las rojas lo bajan "
               "respecto al valor promedio del modelo; los efectos se multiplican, no se suman.")
        if r["avisos"]:
            msg += " ⚠️ " + " ".join(r["avisos"])
        return f"{_cop(valor)} COP", detalle, msg, fig
