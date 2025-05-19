import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shiny import App, ui, render
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from matplotlib.ticker import ScalarFormatter
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator
from functools import partial

warnings.filterwarnings("ignore")

nombres_visuales = {
    "AUTOS": "Autos",
    "MOTOS": "Motos",
    "AUTOBUS DE 2 EJES": "Autobús\n(2 ejes)",
    "AUTOBUS DE 3 EJES": "Autobús\n(3 ejes)",
    "AUTOBUS DE 4 EJES": "Autobús\n(4 ejes)",
    "CAMIONES DE 2 EJES": "Camión\n(2 ejes)",
    "CAMIONES DE 3 EJES": "Camión\n(3 ejes)",
    "CAMIONES DE 4 EJES": "Camión\n(4 ejes)",
    "CAMIONES DE 5 EJES": "Camión\n(5 ejes)",
    "CAMIONES DE 6 EJES": "Camión\n(6 ejes)",
    "CAMIONES DE 7 EJES": "Camión\n(7 ejes)",
    "CAMIONES DE 8 EJES": "Camión\n(8 ejes)",
    "CAMIONES DE 9 EJES": "Camión\n(9 ejes)",
    "TRICICLOS": "Triciclos"
}

vehiculos = list(nombres_visuales.keys())

# Cargar datos
def cargar_datos():
    df = pd.read_csv("Aforos-RedPropia.csv", encoding='latin-1')
    meses_dict = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
        "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
        "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    df["MES"] = df["MES"].str.lower().map(meses_dict)
    df["AÑO"] = df["AÑO"].astype(str).str.strip()
    df["FECHA"] = pd.to_datetime(df["AÑO"] + "-" + df["MES"].astype(str) + "-01")

    columnas = df.columns.difference(["NOMBRE", "TIPO", "AÑO", "MES", "FECHA"])
    for col in columnas:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors='coerce').fillna(0)

    return df

df = cargar_datos()
anios = sorted(df["AÑO"].unique())
meses = {
    "01": "Enero", "02": "Febrero", "03": "Marzo", "04": "Abril",
    "05": "Mayo", "06": "Junio", "07": "Julio", "08": "Agosto",
    "09": "Septiembre", "10": "Octubre", "11": "Noviembre", "12": "Diciembre"
}

# Interfaz
app_ui = ui.page_fluid(

    ui.tags.head(
        ui.tags.link(
            rel="stylesheet",
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"
        )
    ),

    ui.tags.style("""
        .indicador-card {
            text-align: center;
            font-size: 1.1em;
        }
        .titulo-icono {
            font-size: 1.3em;
            font-weight: bold;
        }
    """),

    ui.div(
    ui.h3(ui.strong("Dashboard CAPUFE - Movimientos Vehiculares (2021–2025)")),
    style="text-align: center; margin-top: 20px;"
    ),
    ui.hr(),

    ui.layout_columns(
        ui.card(
            ui.div(
                [
                    ui.HTML('<i class="fa-solid fa-car fa-3x" style="color: #007bff;"></i>'),
                    ui.div(
                        [
                            ui.h5("Total Autos"),
                            ui.output_text(
                                "total_autos_valor",
                                container=partial(ui.tags.strong, style="font-size: 1.5em;")
                            )
                        ],
                        style="margin-left: 15px;"
                    )
                ],
                style="display: flex; align-items: center;"
            ),
            style="display: flex; justify-content: center; align-items: center;"
        ),
        ui.card(
            ui.div(
                [
                    ui.HTML('<i class="fa-solid fa-calendar-days fa-3x" style="color: #28a745;"></i>'),
                    ui.div(
                        [
                            ui.h5("Frecuencia"),
                            ui.output_text(
                                "frecuencia_valor",
                                container=partial(ui.tags.strong, style="font-size: 1.5em;")
                            )
                        ],
                        style="margin-left: 10px;"
                    )
                ],
                style="display: flex; align-items: center;"
            ),
            style="display: flex; justify-content: center; align-items: center;"
        ),
        ui.card(
            ui.div(
                [
                    ui.HTML('<i class="fa-solid fa-chart-line fa-3x" style="color: #dc3545;"></i>'),
                    ui.div(
                        [
                            ui.h5("Pronóstico"),
                            ui.output_text(
                                "valor_pronostico",
                                container=partial(ui.tags.strong, style="font-size: 1.5em;")
                            )
                        ],
                        style="margin-left: 10px;"
                    )
                ],
                style="display: flex; align-items: center;"
            ),
            style="display: flex; justify-content: center; align-items: center;"
        )
    ),

    ui.hr(),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4(ui.strong("Filtros")),
            ui.input_select("vehiculo", "Tipo de Vehículo", choices={k: v for k, v in nombres_visuales.items()}),
            ui.input_select("anio", "Año", choices=anios),
            ui.input_select("mes", "Mes", choices=meses),
            ui.input_checkbox_group(
                "tipos_visual",
                "Tipos de Vehículo",
                choices={k: v for k, v in nombres_visuales.items()},
                selected=vehiculos[:5]
            ),
            width=300
        ),
        ui.row(
            ui.column(6, 
                ui.card(
                    ui.h4(ui.strong("Distribución Por Año")),
                    ui.div(ui.output_plot("grafico_distribucion"), style="height: auto;"),
                    style="height: auto; min-height: 0;"
                )
            ),
            ui.column(6, 
                ui.card(
                    ui.h4(ui.strong("Pronóstico SARIMAX")),
                    ui.div(ui.output_plot("grafico_pronostico"), style="height: auto;"),
                    style="height: auto; min-height: 0;"
                )
            ),
            ui.column(6,
                ui.card(
                    ui.h4(ui.strong("Tabla Del Año Seleccionado")),
                    ui.div(
                        ui.output_table("tabla_datos"),
                        style="max-height: 300px; overflow-y: auto; text-align: center;"
                    ),
                style="height: auto; min-height: 0;" 
                )
            ),
            ui.column(6,
                ui.card(
                ui.h4(ui.strong("Estadísticas")),
                ui.div(
                    [
                        ui.div(
                            ui.output_ui("estadisticas_completas"),
                            style="flex: 1; padding-left: 10px;"
                        ),
                        ui.div(
                        [
                            ui.output_text(
                                "tipo_mayor",
                                container=lambda id, class_: ui.tags.div(
                                    [
                                        ui.tags.div("Tipo mayor:", style="font-weight: bold;"),
                                        ui.tags.div(id=id, class_=class_)
                                    ],
                                    style="margin-bottom: 10px;"
                                )
                            ),
                            ui.output_text(
                                "tipo_menor",
                                container=lambda id, class_: ui.tags.div(
                                    [
                                        ui.tags.div("Tipo menor:", style="font-weight: bold;"),
                                        ui.tags.div(id=id, class_=class_)
                                    ]
                                )
                            )
                        ],
                        style="display: flex; flex-direction: column; flex: 1; padding-right: 10px;"
                    )
                    ],
                    style="display: flex; gap: 10px;"
                ),
                style="height: auto; min-height: 0;"
                )
            )
        )
    ),
)

# Servidor
def server(input, output, session):

    @output
    @render.text
    def total_autos_valor():
        anio = input.anio()
        total = df[df["AÑO"] == anio]["AUTOS"].sum()
        return f"{int(total):,}"

    @output
    @render.text
    def frecuencia_valor():
        return "1 mes"

    @output
    @render.text
    def valor_pronostico():
        try:
            vehiculo = input.vehiculo()
            anio = input.anio()
            mes = input.mes()

            serie = df.groupby("FECHA")[vehiculo].sum()
            serie.index.freq = "MS"
            fecha_objetivo = pd.Timestamp(f"{anio}-{mes}-01")
            fecha_final = serie.index[-1]

            if fecha_objetivo in serie.index:
                valor = serie.loc[fecha_objetivo]
                return f"Valor Real: {int(valor):,}"
            else:
                # División 80/20
                train_size = int(len(serie) * 0.8)
                train = serie[:train_size]
                model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12)).fit(disp=False)

                # Calcular steps futuros desde último dato real
                steps = (fecha_objetivo.year - fecha_final.year) * 12 + (fecha_objetivo.month - fecha_final.month)
                steps = max(1, steps)

                forecast = model.get_forecast(steps=steps)
                pred = forecast.predicted_mean.clip(lower=0)

                # Alinear índice del pronóstico
                future_index = pd.date_range(start=fecha_final + pd.DateOffset(months=1), periods=steps, freq="MS")
                pred.index = future_index

                # Obtener valor predicho para la fecha objetivo
                if fecha_objetivo in pred.index:
                    return f"Pronóstico: {int(pred.loc[fecha_objetivo]):,}"
                else:
                    return "Pronóstico no disponible"

        except Exception as e:
            print("ERROR en valor_pronostico:", e)
            return "N/A"

    @output
    @render.plot
    def grafico_pronostico():
        try:
            vehiculo = input.vehiculo()
            anio = input.anio()
            mes = input.mes()

            serie = df.groupby("FECHA")[vehiculo].sum()
            serie.index.freq = "MS"
            fecha_objetivo = pd.Timestamp(f"{anio}-{mes}-01")

            # División 80/20
            train_size = int(len(serie) * 0.8)
            train = serie[:train_size]
            test = serie[train_size:]
            fecha_final = serie.index[-1]

            # Entrenar modelo con training
            model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12)).fit(disp=False)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(train, label="Entrenamiento", color="blue")
            ax.plot(test, label="Prueba", color="orange")

            if fecha_objetivo > fecha_final:
                # Calcular pasos
                steps = (fecha_objetivo.year - fecha_final.year) * 12 + (fecha_objetivo.month - fecha_final.month)
                steps = max(1, steps)

                # Pronóstico
                forecast = model.get_forecast(steps=steps)
                pred = forecast.predicted_mean.clip(lower=0)
                ci = forecast.conf_int().clip(lower=0)

                # Crear índice futuro
                future_index = pd.date_range(start=fecha_final + pd.DateOffset(months=1), periods=steps, freq="MS")
                pred.index = future_index
                ci.index = future_index

                # Dibujar pronóstico
                ax.plot(pred, label="Pronóstico", color="green")
                ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="green", alpha=0.2)

            # Línea vertical en fecha objetivo
            ax.axvline(fecha_objetivo, color="red", linestyle="--", linewidth=1.5)
            ax.text(fecha_objetivo, ax.get_ylim()[1]*0.95,
                    f"{fecha_objetivo.strftime('%Y-%b')}",
                    color="red", rotation=90, verticalalignment='top', horizontalalignment='right', fontsize=9)

            # Estilo
            ax.set_title(f"Pronóstico para {nombres_visuales.get(vehiculo, vehiculo)}")
            ax.set_ylabel("Cantidad estimada")
            ax.legend(loc="upper left", frameon=True)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y\n%b'))
            ax.tick_params(axis='x', labelsize=8)
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
            ax.ticklabel_format(useOffset=False, style='plain', axis='y')

            plt.tight_layout()
            return fig

        except Exception as e:
            print("ERROR en grafico_pronostico:", e)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.text(0.5, 0.5, "Error al generar gráfico", ha="center", va="center")
            return fig


    @output
    @render.plot
    def grafico_distribucion():
        anio = input.anio()
        tipos = list(input.tipos_visual())
        datos = df[df["AÑO"] == anio][tipos].sum()
        datos = datos.rename(index=nombres_visuales)

        colores = ["#FFCC00", "#4F81BD", "#C0504D", "#9BBB59", "#FCD116", "#8064A2",
                "#4BACC6", "#F79646", "#C00000", "#00B050", "#7030A0",
                "#FF6666", "#00B0F0"]

        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(datos.index, datos.values, color=colores[:len(datos)])

        ax.set_title(f"Distribución en {anio}")
        ax.set_ylabel("Cantidad")
        ax.grid(axis="y")
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        ax.ticklabel_format(useOffset=False, style='plain', axis='y')

        ax.set_ylim(bottom=-max(datos.values) * 0.1)
        plt.tight_layout()
        return fig

    @output
    @render.table
    def tabla_datos():
        anio = input.anio()
        tipos = list(input.tipos_visual())
        df_filtrado = df[df["AÑO"] == anio][["AÑO", "MES"] + tipos].copy()
       
        meses_nombres = {
            1: "ENERO", 2: "FEBRERO", 3: "MARZO", 4: "ABRIL",
            5: "MAYO", 6: "JUNIO", 7: "JULIO", 8: "AGOSTO",
            9: "SEPTIEMBRE", 10: "OCTUBRE", 11: "NOVIEMBRE", 12: "DICIEMBRE"
        }

        df_filtrado["MES"] = df_filtrado["MES"].map(meses_nombres)
        df_filtrado.rename(columns=nombres_visuales, inplace=True)
        return df_filtrado

    @output
    @render.text
    def tipo_mayor():
        anio = input.anio()
        suma = df[df["AÑO"] == anio][vehiculos].sum()
        mayor = suma.idxmax()
        return f"Mayor movimiento: {nombres_visuales[mayor]} ({int(suma[mayor]):,})"

    @output
    @render.text
    def tipo_menor():
        anio = input.anio()
        suma = df[df["AÑO"] == anio][vehiculos].sum()
        menor = suma.idxmin()
        return f"Menor movimiento: {nombres_visuales[menor]} ({int(suma[menor]):,})"
    
    @output
    @render.ui
    def estadisticas_completas():
        anio = input.anio()
        tipos = list(input.tipos_visual())
        datos = df[df["AÑO"] == anio][tipos].sum()
        datos = datos.rename(index=nombres_visuales)
        datos = datos.sort_values(ascending=False)

        # Asignar íconos a cada tipo
        iconos = {
            "Autos": "fa-car",
            "Motos": "fa-motorcycle",
            "Autobús\n(2 ejes)": "fa-bus",
            "Autobús\n(3 ejes)": "fa-bus",
            "Autobús\n(4 ejes)": "fa-bus",
            "Camión\n(2 ejes)": "fa-truck",
            "Camión\n(3 ejes)": "fa-truck",
            "Camión\n(4 ejes)": "fa-truck",
            "Camión\n(5 ejes)": "fa-truck",
            "Camión\n(6 ejes)": "fa-truck",
            "Camión\n(7 ejes)": "fa-truck",
            "Camión\n(8 ejes)": "fa-truck",
            "Camión\n(9 ejes)": "fa-truck",
            "Triciclos": "fa-bicycle"
        }

        html_lines = []
        for tipo, valor in datos.items():
            icono = iconos.get(tipo, "fa-chart-bar")
            line = f'<i class="fa-solid {icono} fa-2x" style="color:#007bff; margin-right: 8px;"></i> {tipo}: <strong>{int(valor):,}</strong>'
            html_lines.append(line)

        return ui.HTML("<br>".join(html_lines))

# Lanzar app
app = App(app_ui, server)
