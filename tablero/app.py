from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, dash_table, Input, Output, State

BASE = Path(__file__).resolve().parent
COLORES = ["#1f5fa8", "#2a9d8f", "#3dae7a", "#8bbf3f", "#f5a623", "#6b7280"]
VERDE, AZUL = "#3dae7a", "#1f5fa8"

COLS_TABLA = [("MATRICULA", "Matrícula"), ("departamento", "Departamento"), ("municipio", "Municipio"),
              ("ultimo_anio", "Último año"), ("frecuencia", "Frecuencia"), ("sin_valor", "Sin valor"),
              ("risk_score", "Risk Score"), ("reason", "Reason")]


def load_data():
    """Carga los archivos del tablero y los devuelve como DataFrames / dict."""
    cubo = pd.read_csv(BASE / "cubo.csv")
    top = pd.read_csv(BASE / "top_riesgo.csv")
    kpis = pd.read_csv(BASE / "kpis.csv")

    metricas = {}
    f = BASE / "metricas.csv"
    if f.exists():
        m = pd.read_csv(f)
        metricas = dict(zip(m.iloc[:, 0].astype(str).str.strip(), m.iloc[:, 1].astype(str)))

    cubo = cubo.dropna(subset=["YEAR_RADICA"]).copy()
    cubo["YEAR_RADICA"] = cubo["YEAR_RADICA"].astype(int)
    for c in ["MUNICIPIO", "TIPO_PREDIO_ZONA", "ACTO"]:
        cubo[c] = cubo[c].fillna("SIN DATO")

    top = top.dropna(subset=["ultimo_anio"]).copy()
    top["ultimo_anio"] = top["ultimo_anio"].astype(int)
    if "reason" not in top.columns:
        top["reason"] = "—"
    top = top.sort_values("risk_score", ascending=False)
    return cubo, top, kpis, metricas


cubo, top, kpis, metricas = load_data()

DEPTOS = sorted(cubo["DEPARTAMENTO"].dropna().unique())
MUNIS = sorted(cubo["MUNICIPIO"].unique())
ZONAS = sorted(cubo["TIPO_PREDIO_ZONA"].unique())
ACTOS = sorted(cubo["ACTO"].unique())
Y_MIN, Y_MAX = int(cubo["YEAR_RADICA"].min()), int(cubo["YEAR_RADICA"].max())
MATRICULAS = int(kpis.iloc[0, 0])
q50, q80 = top["risk_score"].quantile([0.5, 0.8])


def filtrar_cubo(deptos, ciudades, zonas, actos, anios):
    d = cubo[cubo["YEAR_RADICA"].between(anios[0], anios[1])]
    if deptos:
        d = d[d["DEPARTAMENTO"].isin(deptos)]
    if ciudades:
        d = d[d["MUNICIPIO"].isin(ciudades)]
    if zonas:
        d = d[d["TIPO_PREDIO_ZONA"].isin(zonas)]
    if actos:
        d = d[d["ACTO"].isin(actos)]
    return d


def filtrar_top(deptos, ciudades, anios):
    t = top[top["ultimo_anio"].between(anios[0], anios[1])]
    if deptos:
        t = t[t["departamento"].isin(deptos)]
    if ciudades:
        t = t[t["municipio"].isin(ciudades)]
    return t


def tarjeta(titulo, valor, clase):
    return html.Div([html.P(titulo), html.H2(valor)], className="kpi " + clase)


def mini(nombre):
    return html.Div([html.Span(nombre), html.B(metricas[nombre])], className="metric")


# las métricas del modelo son opcionales: solo se muestran si existen en metricas.csv
MINIS = [mini(n) for n in ("TMR", "CNMM", "Precision@K") if n in metricas]


app = Dash(__name__)
server = app.server  # para gunicorn
app.title = "RE-MONITOR"

app.layout = html.Div([
    html.Div([
        html.H2("RE-MONITOR", className="logo"),
        html.Div([html.Button("Download", id="btn-dl", className="btn"), dcc.Download(id="dl")]),
    ], className="topbar"),

    html.Div([
        # ---------------- filtros
        html.Div([
            html.H4("Filter Panels"),
            html.Label("Región (departamento)"),
            dcc.Checklist(DEPTOS, DEPTOS, id="f-depto", className="check"),
            html.Label("Ciudad (municipio)"),
            dcc.Dropdown(MUNIS, id="f-ciudad", multi=True, placeholder="Todas"),
            html.Label("Tipo de predio"),
            dcc.Checklist(ZONAS, ZONAS, id="f-zona", className="check"),
            html.Label("Tipo de transacción"),
            dcc.Dropdown(ACTOS, id="f-acto", multi=True, placeholder="Todas"),
            html.Label("Rango de fechas (años)"),
            dcc.RangeSlider(Y_MIN, Y_MAX, step=1, value=[Y_MIN, Y_MAX], id="f-anio",
                            marks={y: str(y) for y in range(Y_MIN, Y_MAX + 1, 2)},
                            tooltip={"placement": "bottom"}),
        ], className="sidebar"),

        # ---------------- contenido
        html.Div([
            html.H3("Analytics Dashboard"),
            html.Div(id="kpis", className="kpi-row"),
            html.Div([
                html.Div(dcc.Graph(id="g-barras"), className="card"),
                html.Div(dcc.Graph(id="g-dona"), className="card"),
            ], className="row"),
            html.Div([
                html.Div([
                    html.H4("Matrículas priorizadas: top 5% en riesgo"),
                    html.P(id="tabla-info", className="nota"),
                    dash_table.DataTable(
                        id="tabla", page_size=15,
                        columns=[{"name": n, "id": c} for c, n in COLS_TABLA],
                        style_table={"overflowX": "auto"},
                        style_header={"fontWeight": "bold"},
                        style_cell={"textAlign": "left", "fontSize": "12px", "padding": "4px 6px",
                                    "whiteSpace": "normal", "height": "auto"},
                        style_cell_conditional=[
                            {"if": {"column_id": "reason"}, "minWidth": "220px", "width": "220px"},
                            {"if": {"column_id": "MATRICULA"}, "minWidth": "95px"},
                        ],
                        style_data_conditional=[
                            {"if": {"filter_query": f"{{risk_score}} >= {q80}", "column_id": "risk_score"},
                             "backgroundColor": "#e74c3c", "color": "white"},
                            {"if": {"filter_query": f"{{risk_score}} >= {q50} && {{risk_score}} < {q80}",
                                    "column_id": "risk_score"},
                             "backgroundColor": "#f5a623"},
                        ]),
                ], className="card wide"),
                html.Div([
                    html.H4("Detección de anomalías"),
                    html.Div(MINIS, className="metrics") if MINIS else None,
                    dcc.Graph(id="g-anomalias", style={"height": "300px"}),
                    dcc.Graph(id="g-atipicos", style={"height": "300px"}),
                ], className="card"),
            ], className="row"),
            html.Div(dcc.Graph(id="g-predios"), className="card"),
        ], className="main"),
    ], className="app"),
])

ENTRADAS = [Input("f-depto", "value"), Input("f-ciudad", "value"), Input("f-zona", "value"),
            Input("f-acto", "value"), Input("f-anio", "value")]


@app.callback(
    Output("kpis", "children"),
    Output("g-barras", "figure"),
    Output("g-dona", "figure"),
    Output("tabla", "data"),
    Output("tabla-info", "children"),
    Output("g-anomalias", "figure"),
    Output("g-atipicos", "figure"),
    Output("g-predios", "figure"),
    *ENTRADAS,
)
def actualizar(deptos, ciudades, zonas, actos, anios):
    d = filtrar_cubo(deptos, ciudades, zonas, actos, anios)
    t = filtrar_top(deptos, ciudades, anios)

    tarjetas = [
        tarjeta("Total transacciones", f"{d['transacciones'].sum() / 1e6:.2f}M", "k1"),
        tarjeta("Total matrículas (global)", f"{MATRICULAS / 1e6:.2f}M", "k2"),
        tarjeta("Transacciones (dinámica)", f"{d['dinamicas'].sum() / 1e6:.2f}M", "k3"),
        tarjeta("Con valor monetario", f"{d['con_valor'].sum() / 1e6:.2f}M", "k4"),
    ]
    if "Average Time to Prioritize" in metricas:
        tarjetas.append(tarjeta("Average Time to Prioritize", metricas["Average Time to Prioritize"], "k5"))

    # barras: % de transacciones por municipio (top 6)
    total = d["transacciones"].sum()
    mun = d.groupby("MUNICIPIO")["transacciones"].sum().nlargest(6).reset_index()
    mun["pct"] = mun["transacciones"] / total * 100 if total else 0
    barras = px.bar(mun, x="MUNICIPIO", y="pct", color="MUNICIPIO", color_discrete_sequence=COLORES,
                    text=mun["pct"].map(lambda v: f"{v:.2f}%"),
                    title="Transacciones por municipio (% del total)")
    barras.update_traces(textposition="outside")
    barras.update_layout(showlegend=False, yaxis_title="% del total", xaxis_title=None)
    if len(mun):
        barras.update_yaxes(range=[0, mun["pct"].max() * 1.2])

    # dona: TIENE_VALOR
    etiquetas = ["YES (con valor)", "NO (sin valor)"]
    dona = px.pie(names=etiquetas, values=[d["con_valor"].sum(), d["sin_valor"].sum()],
                  color=etiquetas, hole=0.6,
                  color_discrete_map={etiquetas[0]: VERDE, etiquetas[1]: AZUL},
                  title="Registros con valor monetario: TIENE_VALOR")

    # anomalías: transacciones sin valor por año
    por_anio = d.groupby("YEAR_RADICA")[["sin_valor", "predios_nuevos"]].sum().reset_index()
    anomalias = px.line(por_anio, x="YEAR_RADICA", y="sin_valor", markers=True,
                        title="Transacciones sin valor monetario (TIENE_VALOR = 0)")
    anomalias.update_traces(line_color=VERDE)
    anomalias.update_xaxes(dtick=1, title=None)
    anomalias.update_yaxes(title=None)

    # cambios atípicos: matrículas de alto riesgo por año de última transacción
    a = t.groupby("ultimo_anio").size().reset_index(name="matriculas")
    atipicos = px.line(a, x="ultimo_anio", y="matriculas", markers=True,
                       title="Matrículas de alto riesgo por año de su última transacción")
    atipicos.update_traces(line_color=AZUL)
    if len(a):
        atipicos.add_hline(y=a["matriculas"].mean(), line_dash="dash", line_color="#f5a623",
                           annotation_text="Promedio")
    atipicos.update_xaxes(dtick=1, title=None)
    atipicos.update_yaxes(title=None)

    # nuevos predios
    predios = px.line(por_anio, x="YEAR_RADICA", y="predios_nuevos", markers=True,
                      title="Evolución de nuevos predios")
    predios.update_traces(line_color=VERDE)
    predios.update_xaxes(dtick=1, title=None)
    predios.update_yaxes(title=None)

    info = f"{len(t):,} matrículas con los filtros actuales (se muestran las 500 de mayor riesgo)."
    datos = t[[c for c, _ in COLS_TABLA]].head(500).to_dict("records")
    return tarjetas, barras, dona, datos, info, anomalias, atipicos, predios


@app.callback(
    Output("dl", "data"),
    Input("btn-dl", "n_clicks"),
    State("f-depto", "value"), State("f-ciudad", "value"), State("f-anio", "value"),
    prevent_initial_call=True,
)
def descargar(n, deptos, ciudades, anios):
    t = filtrar_top(deptos, ciudades, anios)
    return dcc.send_data_frame(t.to_csv, "matriculas_riesgo.csv", index=False)


if __name__ == "__main__":
    runner = getattr(app, "run", None) or app.run_server
    runner(host="0.0.0.0", port=8050, debug=False)
