"""
Proyecto final - Análisis Exploratorio de Bank Marketing con Streamlit.

Especialización Python for Analytics
Autor: Otto Morales Gómez
Año: 2026

La aplicación cumple los diez ítems solicitados en el caso de estudio y no
construye modelos predictivos. Todo el contenido visible y la documentación
del código se presentan en español.
"""

from __future__ import annotations

import hashlib
import html
import io
import os
import tempfile
from typing import Any, Iterable

# Matplotlib necesita una carpeta de caché escribible tanto localmente como en
# Streamlit Community Cloud. Se define antes de importar la librería.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "bankmarketing-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================

st.set_page_config(
    page_title="Bank Marketing EDA | Otto Morales",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

VERSION = "V2.3.0"

AUTOR = {
    "nombre": "Otto Morales Gómez",
    "perfil": "Ingeniero y MBA",
    "curso": "Especialización Python for Analytics",
    "año": "2026",
    "país": "Perú",
}

PALETA = {
    "fondo": "#F7F9FC",
    "blanco": "#FFFFFF",
    "navy": "#092C4D",
    "navy_medio": "#11507F",
    "turquesa": "#0AD9D8",
    "naranja": "#F17507",
    "naranja_oscuro": "#CF480E",
    "texto": "#17324D",
    "gris": "#64748B",
    "borde": "#E5EAF0",
    "verde": "#138A5B",
    "rojo": "#C2413A",
}

COLUMNAS_ESPERADAS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "y",
]

DESCRIPCION_VARIABLES = {
    "age": "Edad del cliente",
    "job": "Tipo de trabajo del cliente",
    "marital": "Estado civil",
    "education": "Nivel educativo",
    "default": "¿Tiene crédito en mora?",
    "housing": "¿Tiene crédito hipotecario?",
    "loan": "¿Tiene crédito personal?",
    "contact": "Canal de comunicación utilizado",
    "month": "Último mes de contacto",
    "day_of_week": "Día del último contacto",
    "duration": "Duración del contacto, en segundos",
    "campaign": "Contactos realizados en la campaña actual",
    "pdays": "Días desde la última gestión (999 indica sin contacto previo)",
    "previous": "Contactos previos antes de la campaña actual",
    "poutcome": "Resultado de la campaña anterior",
    "emp.var.rate": "Tasa de variación del empleo",
    "cons.price.idx": "Índice de precios al consumidor",
    "cons.conf.idx": "Índice de confianza del consumidor",
    "euribor3m": "Tasa Euribor a tres meses",
    "nr.employed": "Número de empleados",
    "y": "Resultado final: yes si aceptó y no si no aceptó",
}

MESES_ORDENADOS = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DIAS_ORDENADOS = ["mon", "tue", "wed", "thu", "fri"]

sns.set_theme(style="whitegrid", font_scale=0.95)


# =============================================================================
# CLASE PRINCIPAL: PROGRAMACIÓN ORIENTADA A OBJETOS
# =============================================================================

class DataAnalyzer:
    """Encapsula la clasificación, estadística y visualización del dataset."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        if dataframe.empty:
            raise ValueError("El dataset no contiene registros.")
        self.df = dataframe.copy()
        self.numericas, self.categoricas = self.clasificar_variables()

    def clasificar_variables(self) -> tuple[list[str], list[str]]:
        """Función personalizada que separa variables numéricas y categóricas."""
        numericas = self.df.select_dtypes(include=np.number).columns.tolist()
        categoricas = [columna for columna in self.df.columns if columna not in numericas]
        return numericas, categoricas

    def resumen_unknown(self, variables: Iterable[str]) -> pd.DataFrame:
        """Resume el total de filas y la presencia de ``unknown`` por variable."""
        total_filas = len(self.df)
        registros = []

        for variable in variables:
            valores_normalizados = self.df[variable].astype("string").str.strip().str.casefold()
            cantidad_unknown = int(valores_normalizados.eq("unknown").fillna(False).sum())
            porcentaje_unknown = (cantidad_unknown / total_filas * 100) if total_filas else 0.0
            registros.append(
                {
                    "Variable": variable,
                    "Cantidad de filas": total_filas,
                    "Valores unknown": cantidad_unknown,
                    "% unknown": round(porcentaje_unknown, 2),
                }
            )

        return (
            pd.DataFrame(
                registros,
                columns=["Variable", "Cantidad de filas", "Valores unknown", "% unknown"],
            )
            .sort_values("% unknown", ascending=False, kind="stable")
            .reset_index(drop=True)
        )

    def informacion_general(self) -> pd.DataFrame:
        """Devuelve tipo, completitud, nulos y cardinalidad por variable."""
        tabla = pd.DataFrame(
            {
                "Variable": self.df.columns,
                "Tipo de dato": self.df.dtypes.astype(str).values,
                "No nulos": self.df.notna().sum().values,
                "Nulos": self.df.isna().sum().values,
                "% completitud": (self.df.notna().mean().mul(100).round(2)).values,
                "Valores únicos": self.df.nunique(dropna=False).values,
            }
        )
        return tabla

    def captura_info(self) -> str:
        """Captura la salida de DataFrame.info() para mostrarla en la interfaz."""
        buffer = io.StringIO()
        self.df.info(buf=buffer, verbose=True, show_counts=True)
        return buffer.getvalue()

    def estadisticas_numericas(self) -> pd.DataFrame:
        """Calcula las estadísticas descriptivas equivalentes a describe()."""
        if not self.numericas:
            return pd.DataFrame()
        resultado = self.df[self.numericas].describe().T
        resultado = resultado.rename(
            columns={
                "count": "conteo",
                "mean": "media",
                "std": "desv. estándar",
                "min": "mínimo",
                "25%": "percentil 25",
                "50%": "mediana",
                "75%": "percentil 75",
                "max": "máximo",
            }
        )
        return resultado.round(2)

    def resumen_numerico(self, variable: str) -> dict[str, float]:
        """Resume media, mediana, moda, dispersión y asimetría de una variable."""
        serie = pd.to_numeric(self.df[variable], errors="coerce").dropna()
        moda = serie.mode()
        q1 = float(serie.quantile(0.25))
        q3 = float(serie.quantile(0.75))
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        mascara_atipicos = serie.lt(limite_inferior) | serie.gt(limite_superior)
        cantidad_atipicos = int(mascara_atipicos.sum())
        return {
            "media": float(serie.mean()),
            "mediana": float(serie.median()),
            "moda": float(moda.iloc[0]) if not moda.empty else np.nan,
            "desviacion": float(serie.std()),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "limite_inferior": limite_inferior,
            "limite_superior": limite_superior,
            "cantidad_atipicos": cantidad_atipicos,
            "porcentaje_atipicos": float(cantidad_atipicos / len(serie) * 100),
            "asimetria": float(serie.skew()),
        }

    def valores_faltantes(self) -> pd.DataFrame:
        """Cuenta valores nulos y su proporción por variable."""
        tabla = pd.DataFrame(
            {
                "Variable": self.df.columns,
                "Valores nulos": self.df.isna().sum().values,
                "% nulos": self.df.isna().mean().mul(100).round(2).values,
            }
        )
        return tabla.sort_values(["Valores nulos", "Variable"], ascending=[False, True])

    def valores_desconocidos(self) -> pd.DataFrame:
        """Identifica la etiqueta 'unknown', que no es un nulo técnico."""
        filas: list[dict[str, Any]] = []
        for variable in self.categoricas:
            serie = self.df[variable].astype("string").str.strip().str.lower()
            cantidad = int(serie.eq("unknown").sum())
            if cantidad:
                filas.append(
                    {
                        "Variable": variable,
                        "Valores 'unknown'": cantidad,
                        "% del dataset": round(cantidad / len(self.df) * 100, 2),
                    }
                )
        return pd.DataFrame(filas).sort_values("Valores 'unknown'", ascending=False) if filas else pd.DataFrame()

    def figura_histograma(self, variable: str, bins: int, mostrar_kde: bool) -> plt.Figure:
        """Crea histograma y boxplot para estudiar distribución y atípicos."""
        serie = pd.to_numeric(self.df[variable], errors="coerce").dropna()
        resumen = self.resumen_numerico(variable)
        figura, ejes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [4, 1.35]})
        sns.histplot(
            x=serie,
            bins=bins,
            kde=mostrar_kde,
            color=PALETA["navy_medio"],
            edgecolor="white",
            ax=ejes[0],
        )
        ejes[0].axvline(
            resumen["media"],
            color=PALETA["verde"],
            linewidth=2,
            label=f"Media: {resumen['media']:.2f}",
        )
        ejes[0].axvline(
            resumen["mediana"],
            color=PALETA["naranja"],
            linewidth=2,
            linestyle="--",
            label=f"Mediana: {resumen['mediana']:.2f}",
        )
        ejes[0].set_title(f"Distribución de {variable}", fontweight="bold", color=PALETA["navy"])
        ejes[0].set_ylabel("Frecuencia")
        ejes[0].legend()

        sns.boxplot(
            x=serie,
            color=PALETA["turquesa"],
            flierprops={
                "marker": "o",
                "markerfacecolor": PALETA["rojo"],
                "markeredgecolor": PALETA["rojo"],
                "markersize": 4,
                "alpha": 0.55,
            },
            ax=ejes[1],
        )
        referencias = [
            (resumen["limite_inferior"], "Límite inferior", PALETA["rojo"], ":"),
            (resumen["q1"], "Q1", PALETA["navy"], "--"),
            (resumen["mediana"], "Mediana", PALETA["naranja"], "-"),
            (resumen["q3"], "Q3", PALETA["navy"], "--"),
            (resumen["limite_superior"], "Límite superior", PALETA["rojo"], ":"),
        ]
        for valor, etiqueta, color, estilo in referencias:
            ejes[1].axvline(valor, color=color, linestyle=estilo, linewidth=1.5, label=etiqueta)
        ejes[1].set_xlabel(variable)
        ejes[1].set_ylabel("")
        ejes[1].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.34),
            ncol=5,
            frameon=False,
            fontsize=8,
        )
        figura.tight_layout()
        return figura

    def tabla_categorica(self, variable: str, top_n: int) -> pd.DataFrame:
        """Calcula conteos y proporciones de las categorías más frecuentes."""
        conteos = self.df[variable].fillna("(Nulo)").astype(str).value_counts(dropna=False).head(top_n)
        tabla = conteos.rename("Conteo").to_frame()
        tabla["Proporción (%)"] = (tabla["Conteo"] / len(self.df) * 100).round(2)
        tabla.index.name = variable
        return tabla.reset_index()

    def figura_categorica(self, variable: str, top_n: int) -> plt.Figure:
        """Crea un gráfico de barras horizontales para una variable categórica."""
        tabla = self.tabla_categorica(variable, top_n).sort_values("Conteo")
        figura, eje = plt.subplots(figsize=(10, max(4, len(tabla) * 0.48)))
        sns.barplot(
            data=tabla,
            x="Conteo",
            y=variable,
            hue=variable,
            palette="crest",
            legend=False,
            ax=eje,
        )
        eje.set_title(f"Categorías más frecuentes de {variable}", fontweight="bold", color=PALETA["navy"])
        eje.set_xlabel("Número de registros")
        eje.set_ylabel("")
        figura.tight_layout()
        return figura

    def resumen_por_objetivo(self, variable: str, objetivo: str = "y") -> pd.DataFrame:
        """Compara conteo, media, mediana y dispersión entre grupos del objetivo."""
        tabla = self.df.groupby(objetivo, dropna=False)[variable].agg(["count", "mean", "median", "std"])
        tabla = tabla.rename(
            columns={"count": "Conteo", "mean": "Media", "median": "Mediana", "std": "Desv. estándar"}
        )
        return tabla.round(2)

    def figura_numerica_vs_objetivo(self, variable: str, objetivo: str = "y") -> plt.Figure:
        """Compara una variable numérica entre categorías de la variable objetivo."""
        figura, eje = plt.subplots(figsize=(8, 4.8))
        sns.boxplot(
            data=self.df,
            x=objetivo,
            y=variable,
            hue=objetivo,
            palette={"no": PALETA["navy_medio"], "yes": PALETA["naranja"]},
            legend=False,
            showfliers=False,
            ax=eje,
        )
        eje.set_title(f"{variable} según el resultado {objetivo}", fontweight="bold", color=PALETA["navy"])
        eje.set_xlabel("Resultado de la campaña")
        eje.set_ylabel(variable)
        figura.tight_layout()
        return figura

    def tabla_cruzada(self, variable: str, objetivo: str = "y") -> pd.DataFrame:
        """Calcula porcentajes por fila entre dos variables categóricas."""
        tabla = pd.crosstab(self.df[variable], self.df[objetivo], normalize="index").mul(100)
        return tabla.round(2)

    def tasa_por_grupo(self, variable: str, objetivo: str = "y", top_n: int = 12) -> pd.DataFrame:
        """Calcula volumen, aceptaciones y tasa de aceptación por categoría."""
        temporal = self.df[[variable, objetivo]].copy()
        temporal[variable] = temporal[variable].fillna("(Nulo)").astype(str)
        temporal["acepto"] = temporal[objetivo].astype(str).str.lower().eq("yes")
        tabla = temporal.groupby(variable, dropna=False)["acepto"].agg(Conteo="size", Aceptaciones="sum")
        tabla["Tasa de aceptación (%)"] = (tabla["Aceptaciones"] / tabla["Conteo"] * 100).round(2)
        categorias_frecuentes = tabla.nlargest(top_n, "Conteo").index
        return tabla.loc[categorias_frecuentes].sort_values("Tasa de aceptación (%)", ascending=False).reset_index()

    def figura_categorica_vs_objetivo(
        self, variable: str, objetivo: str = "y", top_n: int = 12
    ) -> plt.Figure:
        """Visualiza la tasa de aceptación de las categorías con mayor volumen."""
        tabla = self.tasa_por_grupo(variable, objetivo, top_n).sort_values("Tasa de aceptación (%)")
        figura, eje = plt.subplots(figsize=(9, max(4, len(tabla) * 0.48)))
        sns.barplot(
            data=tabla,
            x="Tasa de aceptación (%)",
            y=variable,
            hue=variable,
            palette="mako",
            legend=False,
            ax=eje,
        )
        eje.set_title(f"Aceptación por {variable}", fontweight="bold", color=PALETA["navy"])
        eje.set_ylabel("")
        figura.tight_layout()
        return figura

    def tasa_aceptacion(self, dataframe: pd.DataFrame | None = None) -> float:
        """Calcula el porcentaje de registros con resultado yes."""
        datos = self.df if dataframe is None else dataframe
        if "y" not in datos.columns or datos.empty:
            return np.nan
        return float(datos["y"].astype(str).str.lower().eq("yes").mean() * 100)


# =============================================================================
# CARGA Y VALIDACIÓN DEL CSV
# =============================================================================

@st.cache_data(show_spinner=False)
def leer_csv(contenido: bytes) -> tuple[pd.DataFrame, str, str]:
    """Lee un CSV detectando codificación y delimitador de forma controlada."""
    errores: list[str] = []
    for codificacion in ("utf-8-sig", "utf-8", "latin-1"):
        for separador, nombre_separador in ((";", "punto y coma"), (",", "coma"), ("\t", "tabulación"), ("|", "barra vertical")):
            try:
                dataframe = pd.read_csv(io.BytesIO(contenido), sep=separador, encoding=codificacion)
                if len(dataframe.columns) > 1:
                    dataframe.columns = [str(columna).strip() for columna in dataframe.columns]
                    return dataframe, codificacion, nombre_separador
            except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
                errores.append(str(error))
    detalle = errores[-1] if errores else "No se identificó una estructura tabular válida."
    raise ValueError(f"No fue posible leer el archivo CSV. Detalle: {detalle}")


def validar_dataset(dataframe: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Devuelve las columnas obligatorias ausentes y las columnas adicionales."""
    faltantes = [columna for columna in COLUMNAS_ESPERADAS if columna not in dataframe.columns]
    adicionales = [columna for columna in dataframe.columns if columna not in COLUMNAS_ESPERADAS]
    return faltantes, adicionales


def guardar_dataset(contenido: bytes, nombre: str) -> tuple[bool, str]:
    """Carga el archivo en memoria y lo conserva durante toda la sesión."""
    huella = hashlib.sha256(contenido).hexdigest()
    if huella == st.session_state.get("dataset_hash"):
        return True, "El archivo ya se encuentra cargado en esta sesión."
    try:
        dataframe, codificacion, separador = leer_csv(contenido)
        if dataframe.empty:
            raise ValueError("El archivo no contiene filas de datos.")
        faltantes, adicionales = validar_dataset(dataframe)
        st.session_state.dataset = dataframe
        st.session_state.dataset_name = nombre
        st.session_state.dataset_hash = huella
        st.session_state.dataset_encoding = codificacion
        st.session_state.dataset_separator = separador
        st.session_state.dataset_missing_columns = faltantes
        st.session_state.dataset_extra_columns = adicionales
        limpiar_preferencias_eda()
        return True, "Dataset validado y guardado correctamente en la sesión."
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        return False, str(error)


def limpiar_dataset() -> None:
    """Retira el dataset y las preferencias asociadas de la sesión."""
    for clave in list(st.session_state.keys()):
        if clave.startswith("dataset") or clave == "archivo_csv":
            del st.session_state[clave]
    limpiar_preferencias_eda()


def limpiar_preferencias_eda() -> None:
    """Evita que los selectores de un archivo anterior contaminen uno nuevo."""
    for clave in list(st.session_state.keys()):
        if clave.startswith("eda_") or clave.startswith("_persist_eda_"):
            del st.session_state[clave]


# =============================================================================
# PERSISTENCIA DE WIDGETS ENTRE SECCIONES
# =============================================================================

def _guardar_widget(clave: str) -> None:
    """Copia el valor de un widget a una clave estable de session_state."""
    st.session_state[f"_persist_{clave}"] = st.session_state[clave]


def _preparar_widget(clave: str, valor_inicial: Any, opciones: Iterable[Any] | None = None) -> None:
    """Restaura un widget aunque su sección haya dejado de renderizarse."""
    clave_persistente = f"_persist_{clave}"
    valor = st.session_state.get(clave_persistente, valor_inicial)
    if opciones is not None:
        opciones_lista = list(opciones)
        if isinstance(valor, list):
            valor = [elemento for elemento in valor if elemento in opciones_lista]
            if not valor and isinstance(valor_inicial, list):
                valor = [elemento for elemento in valor_inicial if elemento in opciones_lista]
        elif valor not in opciones_lista:
            valor = valor_inicial if valor_inicial in opciones_lista else opciones_lista[0]
    st.session_state[clave_persistente] = valor
    st.session_state[clave] = valor


def selectbox_persistente(
    etiqueta: str,
    opciones: list[Any],
    clave: str,
    valor_inicial: Any | None = None,
    ayuda: str | None = None,
) -> Any:
    """Construye un selectbox cuyo valor sobrevive a la navegación."""
    inicial = opciones[0] if valor_inicial is None else valor_inicial
    _preparar_widget(clave, inicial, opciones)
    return st.selectbox(
        etiqueta,
        opciones,
        key=clave,
        on_change=_guardar_widget,
        args=(clave,),
        help=ayuda,
    )


def multiselect_persistente(
    etiqueta: str,
    opciones: list[Any],
    clave: str,
    valores_iniciales: list[Any],
    ayuda: str | None = None,
    max_selecciones: int | None = None,
) -> list[Any]:
    """Construye un multiselect con respaldo explícito en session_state."""
    _preparar_widget(clave, valores_iniciales, opciones)
    return st.multiselect(
        etiqueta,
        opciones,
        key=clave,
        on_change=_guardar_widget,
        args=(clave,),
        help=ayuda,
        max_selections=max_selecciones,
    )


def checkbox_persistente(etiqueta: str, clave: str, valor_inicial: bool, ayuda: str | None = None) -> bool:
    """Construye un checkbox persistente."""
    _preparar_widget(clave, valor_inicial)
    return st.checkbox(
        etiqueta,
        key=clave,
        on_change=_guardar_widget,
        args=(clave,),
        help=ayuda,
    )


def slider_persistente(
    etiqueta: str,
    minimo: int | float,
    maximo: int | float,
    clave: str,
    valor_inicial: int | float | tuple[int | float, int | float],
    paso: int | float | None = None,
    ayuda: str | None = None,
) -> Any:
    """Construye un slider persistente y normaliza valores de otro dataset."""
    clave_persistente = f"_persist_{clave}"
    valor = st.session_state.get(clave_persistente, valor_inicial)
    if isinstance(valor, tuple):
        bajo = max(minimo, min(maximo, valor[0]))
        alto = max(bajo, min(maximo, valor[1]))
        valor = (bajo, alto)
    else:
        valor = max(minimo, min(maximo, valor))
    st.session_state[clave_persistente] = valor
    st.session_state[clave] = valor
    return st.slider(
        etiqueta,
        min_value=minimo,
        max_value=maximo,
        value=None,
        step=paso,
        key=clave,
        on_change=_guardar_widget,
        args=(clave,),
        help=ayuda,
    )


# =============================================================================
# IDENTIDAD VISUAL Y COMPONENTES DE INTERFAZ
# =============================================================================

def inyectar_estilos() -> None:
    """Aplica la identidad visual heredada y refinada de la Evaluación 1."""
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {PALETA['fondo']}; color: {PALETA['texto']}; }}
        .block-container {{ max-width: 1440px; padding-top: 1.7rem; padding-bottom: 3rem; }}
        h1, h2, h3, h4 {{ color: {PALETA['navy']} !important; letter-spacing: -.02em; }}
        p, li {{ line-height: 1.58; }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {PALETA['navy']} 0%, #0D3A63 100%);
            border-right: 0;
        }}
        [data-testid="stSidebar"] * {{ color: #FFFFFF !important; }}
        [data-testid="stSidebar"] [role="radiogroup"] > label {{
            background: rgba(255,255,255,.07);
            border: 1px solid rgba(255,255,255,.13);
            border-radius: 12px;
            padding: 10px 13px;
            margin-bottom: 8px;
            width: 100%;
            transition: transform .12s ease, background .12s ease;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] > label:hover {{
            background: rgba(255,255,255,.16); transform: translateX(2px);
        }}
        [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
            background: linear-gradient(90deg, {PALETA['naranja']} 0%, {PALETA['naranja_oscuro']} 100%);
            border-color: transparent;
            box-shadow: 0 5px 14px rgba(241,117,7,.35);
        }}
        [data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {{ display: none; }}

        .marca-sidebar {{
            display: flex; align-items: center; gap: 12px;
            padding: 5px 2px 14px; margin-bottom: 12px;
            border-bottom: 1px solid rgba(255,255,255,.14);
        }}
        .marca-logo, .hero-logo {{
            display: grid; place-items: center; flex-shrink: 0;
            background: #FFFFFF; color: {PALETA['navy']} !important;
            border-radius: 13px; font-weight: 900; letter-spacing: -.06em;
        }}
        .marca-logo {{ width: 46px; height: 46px; font-size: 1.05rem; }}
        .marca-nombre {{ font-weight: 800; font-size: 1.02rem; }}
        .marca-sub {{ font-size: .75rem; opacity: .72; }}
        .version {{
            background: {PALETA['naranja']}; color: #FFFFFF !important;
            font-weight: 800; text-align: center; padding: 4px 0;
            border-radius: 8px; margin: 0 0 10px; letter-spacing: .5px;
        }}
        .estado-datos {{
            border: 1px solid rgba(255,255,255,.18); background: rgba(10,217,216,.10);
            border-radius: 12px; padding: 11px 12px; margin: 10px 0 14px;
            font-size: .82rem; line-height: 1.45;
        }}

        .hero {{
            background: linear-gradient(120deg, {PALETA['navy']} 0%, #0D3A63 55%, {PALETA['navy_medio']} 100%);
            border-radius: 20px; padding: 25px 30px; margin-bottom: 18px;
            display: flex; align-items: center; gap: 20px;
            box-shadow: 0 12px 30px rgba(9,44,77,.24); overflow: hidden;
            position: relative;
        }}
        .hero:after {{
            content: ''; position: absolute; width: 240px; height: 240px;
            border-radius: 50%; right: -80px; top: -115px;
            background: rgba(10,217,216,.09);
        }}
        .hero-logo {{ width: 60px; height: 60px; font-size: 1.25rem; z-index: 1; }}
        .hero-contenido {{ z-index: 1; }}
        .hero-chip {{
            display: inline-block; border-radius: 999px; padding: 3px 11px;
            background: rgba(10,217,216,.14); border: 1px solid rgba(10,217,216,.34);
            color: {PALETA['turquesa']} !important; text-transform: uppercase;
            letter-spacing: .06em; font-size: .72rem; font-weight: 800; margin-bottom: 7px;
        }}
        .hero-titulo {{ color: #FFFFFF !important; font-size: 1.65rem; font-weight: 850; line-height: 1.2; }}
        .hero-sub {{ color: #C6DBED !important; margin: 5px 0 0; font-size: .95rem; }}

        .tarjeta {{
            background: #FFFFFF; border: 1px solid {PALETA['borde']}; border-radius: 16px;
            padding: 21px 23px; box-shadow: 0 4px 18px rgba(9,44,77,.065);
            height: 100%;
        }}
        .tarjeta h3 {{ margin-top: 0; }}
        .banda-marca {{
            height: 6px; border-radius: 6px; margin: 6px 0 18px;
            background: linear-gradient(90deg, {PALETA['navy']} 0%, {PALETA['turquesa']} 35%, {PALETA['naranja']} 72%, {PALETA['naranja_oscuro']} 100%);
        }}
        .item-tag {{
            display: inline-block; background: #EAF4FA; color: {PALETA['navy']} !important;
            border: 1px solid #CCE1EF; border-radius: 999px; padding: 3px 11px;
            font-size: .74rem; font-weight: 800; text-transform: uppercase; letter-spacing: .04em;
        }}
        .insight {{
            border-left: 5px solid {PALETA['turquesa']}; background: #FFFFFF;
            border-radius: 0 12px 12px 0; padding: 12px 16px; margin: 10px 0;
            box-shadow: 0 2px 10px rgba(9,44,77,.055);
        }}
        .advertencia-analitica {{
            border-left: 5px solid {PALETA['naranja']}; background: #FFF8F0;
            border-radius: 0 12px 12px 0; padding: 12px 16px; margin: 10px 0;
        }}
        .paso {{
            background: #FFFFFF; border: 1px solid {PALETA['borde']}; border-radius: 14px;
            padding: 15px 17px; min-height: 115px;
        }}
        .paso-numero {{ color: {PALETA['naranja']} !important; font-weight: 900; font-size: 1.3rem; }}

        [data-testid="stMetric"] {{
            background: #FFFFFF; border: 1px solid {PALETA['borde']};
            border-left: 5px solid {PALETA['turquesa']}; border-radius: 13px;
            padding: 13px 16px; box-shadow: 0 3px 13px rgba(9,44,77,.06);
        }}
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{ color: {PALETA['navy']} !important; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 7px; }}
        .stTabs [data-baseweb="tab"] {{
            background: #FFFFFF; border: 1px solid {PALETA['borde']};
            border-radius: 10px 10px 0 0; padding: 9px 13px; font-weight: 700;
        }}
        .stTabs [aria-selected="true"] {{ color: {PALETA['naranja_oscuro']} !important; border-bottom-color: {PALETA['naranja']} !important; }}
        .stButton > button, .stDownloadButton > button {{
            background: linear-gradient(90deg, {PALETA['naranja']} 0%, {PALETA['naranja_oscuro']} 100%);
            color: #FFFFFF; border: 0; border-radius: 10px; font-weight: 750;
            box-shadow: 0 3px 10px rgba(241,117,7,.28); transition: transform .12s ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{ transform: translateY(-1px); color: #FFFFFF; }}
        [data-testid="stDataFrame"] {{ border: 1px solid {PALETA['borde']}; border-radius: 12px; overflow: hidden; }}
        .pie-pagina {{ text-align: center; color: {PALETA['gris']}; font-size: .82rem; margin-top: 35px; }}

        @media (max-width: 760px) {{
            .hero {{ padding: 20px; align-items: flex-start; }}
            .hero-logo {{ width: 48px; height: 48px; }}
            .hero-titulo {{ font-size: 1.3rem; }}
            .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def encabezado_hero(chip: str, titulo: str, subtitulo: str) -> None:
    """Muestra el encabezado principal de cada módulo."""
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-logo">OM</div>
          <div class="hero-contenido">
            <span class="hero-chip">{html.escape(chip)}</span>
            <div class="hero-titulo">{html.escape(titulo)}</div>
            <p class="hero-sub">{html.escape(subtitulo)}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def titulo_item(numero: int, titulo: str, descripcion: str) -> None:
    """Identifica claramente cada ítem exigido por la rúbrica."""
    st.markdown(f'<span class="item-tag">Ítem {numero} de 10</span>', unsafe_allow_html=True)
    st.subheader(titulo)
    st.caption(descripcion)


def mostrar_figura(figura: plt.Figure) -> None:
    """Renderiza una figura y libera su memoria después de usarla."""
    st.pyplot(figura, width="stretch")
    plt.close(figura)


def bloque_insight(texto: str, advertencia: bool = False) -> None:
    """Muestra una interpretación breve junto al resultado analítico."""
    clase = "advertencia-analitica" if advertencia else "insight"
    st.markdown(f'<div class="{clase}">{html.escape(texto)}</div>', unsafe_allow_html=True)


def formato_numero(valor: float, decimales: int = 2) -> str:
    """Formatea cifras de forma legible para métricas."""
    if pd.isna(valor):
        return "No disponible"
    return f"{valor:,.{decimales}f}"


def cambiar_pagina(etiqueta: str) -> None:
    """Callback de navegación segura desde botones internos."""
    st.session_state.navegacion = etiqueta


def pie_pagina() -> None:
    """Añade autoría y alcance al final de cada módulo."""
    st.markdown(
        f'<div class="pie-pagina">{AUTOR["nombre"]} · {AUTOR["curso"]} · {AUTOR["país"]} · {AUTOR["año"]}</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# HALLAZGOS Y CONCLUSIONES AUTOMÁTICAS
# =============================================================================

def _mediana_por_resultado(df: pd.DataFrame, variable: str, resultado: str) -> float:
    mascara = df["y"].astype(str).str.lower().eq(resultado)
    return float(pd.to_numeric(df.loc[mascara, variable], errors="coerce").median())


def generar_hallazgos(analyzer: DataAnalyzer) -> list[str]:
    """Genera hallazgos descriptivos, reproducibles y sin lenguaje causal."""
    df = analyzer.df
    hallazgos: list[str] = []
    if "y" not in df.columns:
        return ["El archivo no contiene la variable objetivo y; no es posible resumir la aceptación."]

    tasa = analyzer.tasa_aceptacion()
    aceptaciones = int(df["y"].astype(str).str.lower().eq("yes").sum())
    hallazgos.append(
        f"La aceptación observada es {tasa:.2f}% ({aceptaciones:,} de {len(df):,} registros). "
        "Esta cifra describe el archivo cargado y no reemplaza el indicador corporativo del caso si usa otra base o periodo."
    )

    if "duration" in df.columns:
        mediana_si = _mediana_por_resultado(df, "duration", "yes")
        mediana_no = _mediana_por_resultado(df, "duration", "no")
        hallazgos.append(
            f"La mediana de duración es {mediana_si:,.0f} segundos en aceptaciones y {mediana_no:,.0f} en rechazos; "
            "existe asociación descriptiva, pero duration solo se conoce después del contacto."
        )

    if "contact" in df.columns:
        tabla = analyzer.tasa_por_grupo("contact", top_n=20)
        mejor = tabla.iloc[0]
        peor = tabla.iloc[-1]
        hallazgos.append(
            f"El canal {mejor['contact']} registra {mejor['Tasa de aceptación (%)']:.2f}% de aceptación, frente a "
            f"{peor['Tasa de aceptación (%)']:.2f}% en {peor['contact']}; conviene validar esta diferencia con segmentos y periodos comparables."
        )

    if "poutcome" in df.columns:
        tabla = analyzer.tasa_por_grupo("poutcome", top_n=20)
        fila_exito = tabla[tabla["poutcome"].astype(str).str.lower().eq("success")]
        if not fila_exito.empty:
            hallazgos.append(
                f"Los clientes con resultado previo success alcanzan {fila_exito.iloc[0]['Tasa de aceptación (%)']:.2f}% de aceptación, "
                "por lo que el antecedente de campaña es un criterio útil para priorizar seguimiento."
            )

    if "campaign" in df.columns:
        media_si = float(df.loc[df["y"].astype(str).str.lower().eq("yes"), "campaign"].mean())
        media_no = float(df.loc[df["y"].astype(str).str.lower().eq("no"), "campaign"].mean())
        hallazgos.append(
            f"Las aceptaciones tuvieron en promedio {media_si:.2f} contactos en la campaña, frente a {media_no:.2f} en los rechazos; "
            "el patrón respalda revisar la presión comercial y los contactos repetidos."
        )

    desconocidos = analyzer.valores_desconocidos()
    if not desconocidos.empty:
        principal = desconocidos.iloc[0]
        hallazgos.append(
            f"No hay que confundir nulos con desconocidos: {principal['Variable']} contiene "
            f"{int(principal["Valores 'unknown'"]):,} registros unknown ({principal['% del dataset']:.2f}%), "
            "un aspecto de calidad que debe documentarse antes de segmentar."
        )
    return hallazgos


def generar_conclusiones(analyzer: DataAnalyzer) -> list[str]:
    """Produce cinco conclusiones claras y orientadas a la toma de decisiones."""
    df = analyzer.df
    hallazgos = generar_hallazgos(analyzer)
    conclusiones: list[str] = []

    if "y" in df.columns:
        tasa = analyzer.tasa_aceptacion()
        conclusiones.append(
            f"Usar {tasa:.2f}% como línea base descriptiva del archivo y contrastarla con el 8% del contexto corporativo, "
            "verificando que ambos indicadores utilicen el mismo periodo, población y definición."
        )
    if "poutcome" in df.columns:
        conclusiones.append(
            "Priorizar el seguimiento de clientes con antecedentes favorables en campañas previas, sin convertir la asociación observada en una regla automática de exclusión."
        )
    if "contact" in df.columns:
        conclusiones.append(
            "Revisar la estrategia por canal y ejecutar pruebas controladas por segmento antes de trasladar las diferencias de aceptación a una política comercial."
        )
    if "campaign" in df.columns:
        conclusiones.append(
            "Definir límites y alertas para contactos repetidos, porque los rechazos presentan una intensidad media de campaña mayor que las aceptaciones."
        )
    desconocidos = analyzer.valores_desconocidos()
    if not desconocidos.empty:
        conclusiones.append(
            "Tratar la etiqueta unknown como una categoría de calidad de datos, especialmente en default, y mejorar su captura antes de realizar segmentaciones operativas."
        )
    if "duration" in df.columns:
        conclusiones.append(
            "Usar duration para comprender la interacción y capacitar equipos después del contacto, pero no como criterio previo de selección porque todavía no existe al iniciar la llamada."
        )
    for hallazgo in hallazgos:
        if len(conclusiones) >= 5:
            break
        conclusiones.append(hallazgo)
    return conclusiones[:5]


# =============================================================================
# MÓDULOS DE LA APLICACIÓN
# =============================================================================

def mostrar_home() -> None:
    """Presenta el proyecto sin ejecutar ningún análisis."""
    encabezado_hero(
        "Proyecto final · EDA",
        "Bank Marketing: análisis exploratorio para decisiones comerciales",
        "Una aplicación interactiva en Streamlit para comprender la aceptación de campañas sin construir modelos predictivos.",
    )

    columna_autor, columna_reto = st.columns([1, 1.35], gap="large")
    with columna_autor:
        st.markdown(
            f"""
            <div class="tarjeta">
              <h3>👨‍💼 Información del estudiante</h3>
              <div class="banda-marca"></div>
              <p><b>Nombre:</b> {AUTOR['nombre']}</p>
              <p><b>Perfil:</b> {AUTOR['perfil']}</p>
              <p><b>Curso:</b> {AUTOR['curso']}</p>
              <p><b>País:</b> {AUTOR['país']} 🇵🇪</p>
              <p><b>Año:</b> {AUTOR['año']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with columna_reto:
        st.markdown(
            """
            <div class="tarjeta">
              <h3>🎯 Objetivo del análisis</h3>
              <div class="banda-marca"></div>
              <p>Explorar los factores asociados con la aceptación de la última campaña bancaria y convertir los resultados en información útil para la toma de decisiones.</p>
              <p>El caso señala una caída de efectividad del <b>12% al 8%</b> durante los últimos seis meses. La aplicación permite estudiar variables demográficas, financieras, de contacto y del entorno económico.</p>
              <p><b>Alcance:</b> estadística descriptiva, calidad de datos, análisis univariado y bivariado. No se realizan predicciones.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🧭 Recorrido de la aplicación")
    pasos = st.columns(4, gap="medium")
    contenido_pasos = [
        ("01", "Cargar", "Seleccionar y validar BankMarketing.csv."),
        ("02", "Explorar", "Revisar los diez ítems del EDA."),
        ("03", "Interpretar", "Conectar cifras, gráficos y contexto."),
        ("04", "Decidir", "Convertir hallazgos en cinco conclusiones."),
    ]
    for columna, (numero, titulo, texto) in zip(pasos, contenido_pasos):
        with columna:
            st.markdown(
                f'<div class="paso"><div class="paso-numero">{numero}</div><b>{titulo}</b><br><span>{texto}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### 🛠️ Tecnologías utilizadas")
    st.markdown(
        "**Python · Streamlit · Pandas · NumPy · Matplotlib · Seaborn · Programación Orientada a Objetos**"
    )
    st.button(
        "Comenzar con la carga del dataset →",
        on_click=cambiar_pagina,
        args=("📂  Carga del dataset",),
        type="primary",
    )
    pie_pagina()


def mostrar_carga() -> None:
    """Carga, valida y conserva el CSV antes de habilitar el análisis."""
    encabezado_hero(
        "Módulo 2 · Preparación",
        "Carga y validación del dataset",
        "El análisis solo se habilita después de cargar correctamente un archivo CSV.",
    )

    izquierda, derecha = st.columns([1, 1.2], gap="large")
    with izquierda:
        st.subheader("📤 Seleccionar archivo")
        archivo = st.file_uploader(
            "Carga BankMarketing.csv",
            type=["csv"],
            key="archivo_csv",
            help="El archivo original utiliza punto y coma como separador. La aplicación lo detecta automáticamente.",
        )
        if archivo is not None:
            correcto, mensaje = guardar_dataset(archivo.getvalue(), archivo.name)
            if correcto:
                st.success(mensaje)
            else:
                st.error(mensaje)

        st.info(
            "La información queda almacenada en `st.session_state`: puedes cambiar de módulo sin volver a cargar el archivo mientras la sesión del navegador permanezca abierta."
        )

        if st.session_state.get("dataset") is not None:
            st.button("Quitar dataset de la sesión", on_click=limpiar_dataset, type="secondary")

    with derecha:
        st.subheader("✅ Estado de validación")
        dataframe = st.session_state.get("dataset")
        if dataframe is None:
            st.warning("Aún no hay un archivo cargado. Ningún análisis será ejecutado.")
            st.markdown(
                """
                **Validaciones automáticas**

                - Archivo con extensión CSV.
                - Detección de separador y codificación.
                - Presencia de filas y más de una columna.
                - Comparación con las 21 variables esperadas.
                """
            )
        else:
            faltantes = st.session_state.get("dataset_missing_columns", [])
            adicionales = st.session_state.get("dataset_extra_columns", [])
            if faltantes:
                st.warning(f"Faltan columnas del caso: {', '.join(faltantes)}")
            else:
                st.success("El dataset contiene las 21 variables esperadas del caso Bank Marketing.")
            if adicionales:
                st.info(f"Columnas adicionales detectadas: {', '.join(adicionales)}")
            st.caption(
                f"Codificación: {st.session_state.get('dataset_encoding')} · Separador: {st.session_state.get('dataset_separator')}"
            )

    dataframe = st.session_state.get("dataset")
    if dataframe is not None:
        st.markdown("---")
        st.subheader("🔎 Vista previa del dataset")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Filas", f"{len(dataframe):,}")
        m2.metric("Columnas", f"{len(dataframe.columns):,}")
        m3.metric("Celdas", f"{dataframe.size:,}")
        m4.metric("Duplicados", f"{int(dataframe.duplicated().sum()):,}")
        filas_vista = slider_persistente("Filas de la vista previa", 5, 25, "eda_filas_preview", 10, 1)
        st.dataframe(dataframe.head(filas_vista), width="stretch", hide_index=True)

        with st.expander("Ver diccionario de variables"):
            diccionario = pd.DataFrame(
                {
                    "Variable": dataframe.columns,
                    "Descripción": [DESCRIPCION_VARIABLES.get(columna, "Variable adicional") for columna in dataframe.columns],
                }
            )
            st.dataframe(diccionario, width="stretch", hide_index=True)

        if not st.session_state.get("dataset_missing_columns", []):
            st.button(
                "Continuar al análisis exploratorio →",
                on_click=cambiar_pagina,
                args=("📊  Análisis EDA",),
                type="primary",
            )
        else:
            st.error("El análisis permanece bloqueado hasta cargar el dataset con todas las variables requeridas.")
    pie_pagina()


def mostrar_bloque_sin_datos() -> None:
    """Informa el bloqueo obligatorio cuando el CSV aún no está cargado."""
    encabezado_hero(
        "Dataset requerido",
        "Primero debes cargar el archivo CSV",
        "La aplicación protege la secuencia de trabajo y no ejecuta análisis sin datos validados.",
    )
    st.warning("Ningún análisis se ha ejecutado. Ve al módulo de carga y selecciona BankMarketing.csv.")
    st.button(
        "Ir a Carga del dataset",
        on_click=cambiar_pagina,
        args=("📂  Carga del dataset",),
        type="primary",
    )


def mostrar_eda() -> None:
    """Desarrolla los diez ítems obligatorios del análisis exploratorio."""
    dataframe = st.session_state.get("dataset")
    if dataframe is None:
        mostrar_bloque_sin_datos()
        pie_pagina()
        return

    columnas_faltantes = st.session_state.get("dataset_missing_columns", [])
    if columnas_faltantes:
        encabezado_hero(
            "Validación incompleta",
            "El archivo no contiene todas las variables requeridas",
            "Corrige el CSV antes de iniciar el análisis para evitar resultados parciales o inconsistentes.",
        )
        st.error(f"Columnas faltantes: {', '.join(columnas_faltantes)}")
        st.button(
            "Volver a Carga del dataset",
            on_click=cambiar_pagina,
            args=("📂  Carga del dataset",),
            type="primary",
        )
        pie_pagina()
        return

    analyzer = DataAnalyzer(dataframe)
    encabezado_hero(
        "Núcleo del proyecto · 10 ítems",
        "Análisis Exploratorio de Datos (EDA)",
        f"{len(dataframe):,} registros y {len(dataframe.columns)} variables · Los controles conservan su estado al navegar.",
    )

    tab_diagnostico, tab_estadistica, tab_univariado, tab_bivariado, tab_decisiones = st.tabs(
        [
            "1–2 · Diagnóstico",
            "3–4 · Estadística y calidad",
            "5–6 · Univariado",
            "7–8 · Bivariado",
            "9–10 · Interactivo y hallazgos",
        ]
    )

    with tab_diagnostico:
        titulo_item(1, "Información general del dataset", "Dimensiones, DataFrame.info(), tipos de datos y conteo de valores nulos.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas", f"{len(dataframe):,}")
        c2.metric("Columnas", len(dataframe.columns))
        c3.metric("Tipos distintos", dataframe.dtypes.astype(str).nunique())
        c4.metric("Valores nulos", f"{int(dataframe.isna().sum().sum()):,}")
        st.dataframe(analyzer.informacion_general(), width="stretch", hide_index=True)
        with st.expander("Ver salida completa de .info()"):
            st.code(analyzer.captura_info(), language="text")

        st.markdown("---")
        titulo_item(
            2,
            "Clasificación de variables",
            "Identificación y auditoría de valores unknown mediante funciones personalizadas de DataAnalyzer.",
        )
        st.metric("Variables numéricas", len(analyzer.numericas))
        st.dataframe(
            analyzer.resumen_unknown(analyzer.numericas),
            width="stretch",
            hide_index=True,
            column_config={
                "Cantidad de filas": st.column_config.NumberColumn(format="%d"),
                "Valores unknown": st.column_config.NumberColumn(format="%d"),
                "% unknown": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

        tabla_categoricas = analyzer.resumen_unknown(analyzer.categoricas)
        st.metric("Variables categóricas", len(analyzer.categoricas))

        if not tabla_categoricas.empty:
            variables_con_unknown = int(tabla_categoricas["Valores unknown"].gt(0).sum())
            mayor_unknown = tabla_categoricas.iloc[0]
            bloque_insight(
                f"Conclusión: se conservan los registros con unknown como una categoría informativa. No alteran "
                f"las estadísticas descriptivas de las variables numéricas y eliminarlos ocasionaría pérdida de "
                f"información; sí deben considerarse al interpretar las categóricas. {variables_con_unknown} de "
                f"{len(analyzer.categoricas)} variables categóricas contienen unknown y {mayor_unknown['Variable']} "
                f"presenta la mayor proporción ({mayor_unknown['% unknown']:.2f}%)."
            )

        tabla_categoricas_estilizada = tabla_categoricas.style.apply(
            lambda fila: [
                "color: #C2413A; font-weight: 600;" if fila["Valores unknown"] > 0 else ""
                for _ in fila
            ],
            axis=1,
        )
        st.dataframe(
            tabla_categoricas_estilizada,
            width="stretch",
            hide_index=True,
            column_config={
                "Cantidad de filas": st.column_config.NumberColumn(format="%d"),
                "Valores unknown": st.column_config.NumberColumn(format="%d"),
                "% unknown": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    with tab_estadistica:
        titulo_item(3, "Estadísticas descriptivas", "Uso de describe() e interpretación de media, mediana, moda y dispersión.")
        if analyzer.numericas:
            st.dataframe(analyzer.estadisticas_numericas(), width="stretch")
            variable_estadistica = selectbox_persistente(
                "Variable para interpretar",
                analyzer.numericas,
                "eda_variable_estadistica",
                "age" if "age" in analyzer.numericas else analyzer.numericas[0],
            )
            resumen = analyzer.resumen_numerico(variable_estadistica)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Media", formato_numero(resumen["media"]))
            m2.metric("Mediana", formato_numero(resumen["mediana"]))
            m3.metric("Moda", formato_numero(resumen["moda"]))
            m4.metric("Desv. estándar", formato_numero(resumen["desviacion"]))
            direccion = "hacia valores altos" if resumen["asimetria"] > 0.25 else "hacia valores bajos" if resumen["asimetria"] < -0.25 else "aproximadamente simétrica"
            bloque_insight(
                f"{variable_estadistica} presenta un rango intercuartílico de {resumen['iqr']:.2f} y asimetría de "
                f"{resumen['asimetria']:.2f}; su distribución es {direccion}. La mediana es más resistente a valores extremos que la media."
            )
        else:
            st.warning("No se detectaron variables numéricas.")

        st.markdown("---")
        titulo_item(4, "Análisis de valores faltantes", "Conteo, visualización cuando aplica y discusión de valores desconocidos.")
        faltantes = analyzer.valores_faltantes()
        st.dataframe(faltantes, width="stretch", hide_index=True)
        total_nulos = int(dataframe.isna().sum().sum())
        if total_nulos:
            tabla_grafico = faltantes[faltantes["Valores nulos"] > 0]
            figura, eje = plt.subplots(figsize=(10, 4))
            sns.barplot(data=tabla_grafico, x="Valores nulos", y="Variable", color=PALETA["naranja"], ax=eje)
            eje.set_title("Valores nulos por variable", fontweight="bold", color=PALETA["navy"])
            mostrar_figura(figura)
        else:
            st.success("El dataset no contiene valores nulos técnicos.")

        desconocidos = analyzer.valores_desconocidos()
        if not desconocidos.empty:
            st.markdown("#### Calidad adicional: etiqueta `unknown`")
            st.dataframe(desconocidos, width="stretch", hide_index=True)
            bloque_insight(
                "Aunque no existen NaN, la etiqueta unknown representa información no disponible y debe conservarse como categoría explícita o tratarse según el objetivo del análisis.",
                advertencia=True,
            )

    with tab_univariado:
        titulo_item(
            5,
            "Distribución de variables numéricas",
            "Histogramas, análisis de colas y detección de valores atípicos mediante Q1, Q3 y el rango intercuartílico.",
        )
        if analyzer.numericas:
            controles_1, controles_2, controles_3 = st.columns([1.4, 1, 1])
            with controles_1:
                variable_numerica = selectbox_persistente(
                    "Variable numérica",
                    analyzer.numericas,
                    "eda_distribucion_numerica",
                    "age" if "age" in analyzer.numericas else analyzer.numericas[0],
                )
            with controles_2:
                numero_bins = slider_persistente("Número de intervalos", 10, 80, "eda_numero_bins", 30, 5)
            with controles_3:
                mostrar_kde = checkbox_persistente("Mostrar curva KDE", "eda_mostrar_kde", True)
            mostrar_figura(analyzer.figura_histograma(variable_numerica, numero_bins, mostrar_kde))
            resumen = analyzer.resumen_numerico(variable_numerica)

            q1_metrica, mediana_metrica, q3_metrica, iqr_metrica = st.columns(4)
            q1_metrica.metric("Q1 · percentil 25", formato_numero(resumen["q1"]))
            mediana_metrica.metric("Mediana · percentil 50", formato_numero(resumen["mediana"]))
            q3_metrica.metric("Q3 · percentil 75", formato_numero(resumen["q3"]))
            iqr_metrica.metric("RIC · Q3 − Q1", formato_numero(resumen["iqr"]))

            limite_inf_metrica, limite_sup_metrica, atipicos_metrica = st.columns(3)
            limite_inf_metrica.metric(
                "Límite inferior · Q1 − 1.5×RIC",
                formato_numero(resumen["limite_inferior"]),
            )
            limite_sup_metrica.metric(
                "Límite superior · Q3 + 1.5×RIC",
                formato_numero(resumen["limite_superior"]),
            )
            atipicos_metrica.metric(
                "Valores potencialmente atípicos",
                f"{int(resumen['cantidad_atipicos']):,}",
                f"{resumen['porcentaje_atipicos']:.2f}% del total",
                delta_color="off",
            )

            asimetria = resumen["asimetria"]
            magnitud_asimetria = abs(asimetria)
            if magnitud_asimetria <= 0.25:
                lectura_cola = "una forma aproximadamente simétrica, sin una cola claramente dominante"
                relacion_centro = "La cercanía entre media y mediana respalda esta lectura"
            else:
                intensidad = "leve" if magnitud_asimetria < 0.75 else "moderada" if magnitud_asimetria < 1.50 else "marcada"
                if asimetria > 0:
                    lectura_cola = f"asimetría positiva {intensidad}: la cola derecha se prolonga por valores altos menos frecuentes"
                    relacion_centro = "La media queda desplazada hacia arriba respecto de la mediana"
                else:
                    lectura_cola = f"asimetría negativa {intensidad}: la cola izquierda se prolonga por valores bajos menos frecuentes"
                    relacion_centro = "La media queda desplazada hacia abajo respecto de la mediana"

            bloque_insight(
                f"Lectura del histograma: {variable_numerica} presenta {lectura_cola}. La asimetría es "
                f"{asimetria:.2f}, la media {resumen['media']:.2f} y la mediana {resumen['mediana']:.2f}. "
                f"{relacion_centro}."
            )
            bloque_insight(
                f"Lectura del boxplot: el 50% central de los datos está entre Q1 = {resumen['q1']:.2f} y "
                f"Q3 = {resumen['q3']:.2f}, con un RIC de {resumen['iqr']:.2f}. La regla 1.5×RIC identifica "
                f"{int(resumen['cantidad_atipicos']):,} observaciones ({resumen['porcentaje_atipicos']:.2f}%) "
                f"fuera de [{resumen['limite_inferior']:.2f}, {resumen['limite_superior']:.2f}]. Son valores "
                "potencialmente atípicos, no errores automáticos; deben revisarse en su contexto antes de decidir su tratamiento.",
                advertencia=resumen["cantidad_atipicos"] > 0,
            )

        st.markdown("---")
        titulo_item(6, "Análisis de variables categóricas", "Conteos, gráficos de barras y proporciones.")
        if analyzer.categoricas:
            col_a, col_b = st.columns([1.5, 1])
            with col_a:
                variable_categorica = selectbox_persistente(
                    "Variable categórica",
                    analyzer.categoricas,
                    "eda_distribucion_categorica",
                    "job" if "job" in analyzer.categoricas else analyzer.categoricas[0],
                )
            max_categorias = max(2, min(20, int(dataframe[variable_categorica].nunique(dropna=False))))
            with col_b:
                top_n = slider_persistente(
                    "Categorías a mostrar", 2, max_categorias, f"eda_top_{variable_categorica}", min(10, max_categorias), 1
                )
            grafico, tabla = st.columns([1.45, 1], gap="large")
            with grafico:
                mostrar_figura(analyzer.figura_categorica(variable_categorica, top_n))
            with tabla:
                st.dataframe(analyzer.tabla_categorica(variable_categorica, top_n), width="stretch", hide_index=True)
            principal = analyzer.tabla_categorica(variable_categorica, top_n).iloc[0]
            bloque_insight(
                f"La categoría más frecuente de {variable_categorica} es {principal[variable_categorica]} con "
                f"{int(principal['Conteo']):,} registros ({principal['Proporción (%)']:.2f}%)."
            )

    with tab_bivariado:
        titulo_item(7, "Análisis bivariado: numérico vs categórico", "Comparación de variables numéricas entre los resultados yes y no.")
        if "y" in dataframe.columns and analyzer.numericas:
            predeterminadas = [variable for variable in ("age", "duration") if variable in analyzer.numericas]
            variables_num_biv = multiselect_persistente(
                "Variables numéricas a comparar con y",
                analyzer.numericas,
                "eda_numericas_bivariadas",
                predeterminadas or analyzer.numericas[:1],
                max_selecciones=4,
            )
            if not variables_num_biv:
                st.info("Selecciona al menos una variable numérica.")
            for variable in variables_num_biv:
                st.markdown(f"#### {variable} vs y")
                grafico, resumen_grupos = st.columns([1.5, 1], gap="large")
                with grafico:
                    mostrar_figura(analyzer.figura_numerica_vs_objetivo(variable))
                with resumen_grupos:
                    st.dataframe(analyzer.resumen_por_objetivo(variable), width="stretch")
                tabla = analyzer.resumen_por_objetivo(variable)
                if "yes" in tabla.index and "no" in tabla.index:
                    bloque_insight(
                        f"La mediana de {variable} es {tabla.loc['yes', 'Mediana']:.2f} para yes y "
                        f"{tabla.loc['no', 'Mediana']:.2f} para no. La comparación describe asociación, no causalidad."
                    )
        else:
            st.warning("Se requiere la variable objetivo y y al menos una variable numérica.")

        st.markdown("---")
        titulo_item(8, "Análisis bivariado: categórico vs categórico", "Tablas cruzadas, proporciones y tasas de aceptación por categoría.")
        categoricas_sin_objetivo = [variable for variable in analyzer.categoricas if variable != "y"]
        if "y" in dataframe.columns and categoricas_sin_objetivo:
            predeterminadas_cat = [variable for variable in ("education", "contact") if variable in categoricas_sin_objetivo]
            variables_cat_biv = multiselect_persistente(
                "Variables categóricas a comparar con y",
                categoricas_sin_objetivo,
                "eda_categoricas_bivariadas",
                predeterminadas_cat or categoricas_sin_objetivo[:1],
                max_selecciones=4,
            )
            top_biv = slider_persistente("Categorías de mayor volumen", 3, 15, "eda_top_bivariado", 10, 1)
            if not variables_cat_biv:
                st.info("Selecciona al menos una variable categórica.")
            for variable in variables_cat_biv:
                st.markdown(f"#### {variable} vs y")
                grafico, tabla_cruzada = st.columns([1.45, 1], gap="large")
                with grafico:
                    mostrar_figura(analyzer.figura_categorica_vs_objetivo(variable, top_n=top_biv))
                with tabla_cruzada:
                    st.markdown("**Distribución porcentual por fila**")
                    st.dataframe(analyzer.tabla_cruzada(variable), width="stretch")
                    st.markdown("**Tasa y volumen**")
                    st.dataframe(analyzer.tasa_por_grupo(variable, top_n=top_biv), width="stretch", hide_index=True)
        else:
            st.warning("Se requieren variables categóricas y la variable objetivo y.")

    with tab_decisiones:
        titulo_item(9, "Análisis basado en parámetros seleccionados", "Filtros y análisis dinámico mediante selectbox, multiselect, slider y checkbox.")
        datos_filtrados = dataframe.copy()
        filtro_1, filtro_2 = st.columns(2, gap="large")

        with filtro_1:
            st.markdown("#### Filtro numérico")
            variable_filtro_num = selectbox_persistente(
                "Variable para rango",
                analyzer.numericas,
                "eda_filtro_numerico",
                "age" if "age" in analyzer.numericas else analyzer.numericas[0],
            )
            serie_num = pd.to_numeric(dataframe[variable_filtro_num], errors="coerce").dropna()
            minimo_original = float(serie_num.min())
            maximo_original = float(serie_num.max())
            es_entero = pd.api.types.is_integer_dtype(dataframe[variable_filtro_num])
            minimo_slider = int(minimo_original) if es_entero else float(minimo_original)
            maximo_slider = int(maximo_original) if es_entero else float(maximo_original)
            paso = 1 if es_entero else max((maximo_slider - minimo_slider) / 100, 0.01)
            rango = slider_persistente(
                f"Rango de {variable_filtro_num}",
                minimo_slider,
                maximo_slider,
                f"eda_rango_{variable_filtro_num}",
                (minimo_slider, maximo_slider),
                paso,
            )
            datos_filtrados = datos_filtrados[
                pd.to_numeric(datos_filtrados[variable_filtro_num], errors="coerce").between(rango[0], rango[1])
            ]

        with filtro_2:
            st.markdown("#### Filtro categórico")
            categorias_filtrables = [variable for variable in analyzer.categoricas if variable != "y"]
            variable_filtro_cat = selectbox_persistente(
                "Variable para categorías",
                categorias_filtrables,
                "eda_filtro_categorico",
                "job" if "job" in categorias_filtrables else categorias_filtrables[0],
            )
            opciones_categoria = sorted(dataframe[variable_filtro_cat].dropna().astype(str).unique().tolist())
            categorias_elegidas = multiselect_persistente(
                f"Valores de {variable_filtro_cat}",
                opciones_categoria,
                f"eda_valores_{variable_filtro_cat}",
                opciones_categoria,
                ayuda="Deja seleccionadas las categorías que deseas conservar.",
            )
            if categorias_elegidas:
                datos_filtrados = datos_filtrados[
                    datos_filtrados[variable_filtro_cat].astype(str).isin(categorias_elegidas)
                ]
            else:
                datos_filtrados = datos_filtrados.iloc[0:0]

        excluir_unknown = checkbox_persistente(
            "Excluir registros con unknown en cualquier variable categórica",
            "eda_excluir_unknown",
            False,
            "Permite medir el efecto de trabajar solo con información categórica conocida.",
        )
        if excluir_unknown and not datos_filtrados.empty:
            mascara_conocidos = pd.Series(True, index=datos_filtrados.index)
            for variable in analyzer.categoricas:
                mascara_conocidos &= ~datos_filtrados[variable].astype(str).str.lower().eq("unknown")
            datos_filtrados = datos_filtrados[mascara_conocidos]

        k1, k2, k3 = st.columns(3)
        k1.metric("Registros filtrados", f"{len(datos_filtrados):,}")
        k2.metric("Retención de la base", f"{len(datos_filtrados) / len(dataframe) * 100:.1f}%")
        tasa_filtrada = analyzer.tasa_aceptacion(datos_filtrados)
        k3.metric("Aceptación filtrada", "No disponible" if pd.isna(tasa_filtrada) else f"{tasa_filtrada:.2f}%")

        variables_resumen = multiselect_persistente(
            "Medidas numéricas del resumen dinámico",
            analyzer.numericas,
            "eda_medidas_dinamicas",
            [variable for variable in ("age", "duration", "campaign") if variable in analyzer.numericas],
            max_selecciones=5,
        )
        if not datos_filtrados.empty and variables_resumen:
            resumen_dinamico = datos_filtrados[variables_resumen].agg(["count", "mean", "median", "std", "min", "max"]).T
            st.dataframe(resumen_dinamico.round(2), width="stretch")
        mostrar_registros = checkbox_persistente("Mostrar registros filtrados", "eda_mostrar_filtrados", False)
        if mostrar_registros:
            st.dataframe(datos_filtrados.head(200), width="stretch", hide_index=True)

        st.markdown("---")
        titulo_item(10, "Hallazgos clave", "Visualización resumen e insights principales derivados del EDA.")
        if "y" in dataframe.columns:
            conteo_resultado = dataframe["y"].astype(str).str.lower().value_counts().rename_axis("Resultado").reset_index(name="Conteo")
            tasas_contacto = analyzer.tasa_por_grupo("contact", top_n=10) if "contact" in dataframe.columns else pd.DataFrame()
            grafico_resultado, grafico_contacto = st.columns(2, gap="large")
            with grafico_resultado:
                figura, eje = plt.subplots(figsize=(7, 4.5))
                sns.barplot(
                    data=conteo_resultado,
                    x="Resultado",
                    y="Conteo",
                    hue="Resultado",
                    palette={"no": PALETA["navy_medio"], "yes": PALETA["naranja"]},
                    legend=False,
                    ax=eje,
                )
                eje.set_title("Resultado general de la campaña", fontweight="bold", color=PALETA["navy"])
                mostrar_figura(figura)
            with grafico_contacto:
                if not tasas_contacto.empty:
                    figura, eje = plt.subplots(figsize=(7, 4.5))
                    sns.barplot(
                        data=tasas_contacto,
                        x="contact",
                        y="Tasa de aceptación (%)",
                        hue="contact",
                        palette="crest",
                        legend=False,
                        ax=eje,
                    )
                    eje.set_title("Aceptación por canal", fontweight="bold", color=PALETA["navy"])
                    eje.set_xlabel("Canal")
                    mostrar_figura(figura)
        for indice, hallazgo in enumerate(generar_hallazgos(analyzer), start=1):
            st.markdown(f"**{indice}.** {hallazgo}")

    pie_pagina()


def mostrar_conclusiones() -> None:
    """Presenta cinco conclusiones finales y limitaciones del análisis."""
    dataframe = st.session_state.get("dataset")
    if dataframe is None:
        mostrar_bloque_sin_datos()
        pie_pagina()
        return

    columnas_faltantes = st.session_state.get("dataset_missing_columns", [])
    if columnas_faltantes:
        encabezado_hero(
            "Validación incompleta",
            "No se pueden generar conclusiones",
            "El dataset debe contener todas las variables del caso para producir un cierre confiable.",
        )
        st.error(f"Columnas faltantes: {', '.join(columnas_faltantes)}")
        pie_pagina()
        return

    analyzer = DataAnalyzer(dataframe)
    encabezado_hero(
        "Cierre · Toma de decisiones",
        "Conclusiones finales",
        "Cinco conclusiones técnicas, comprensibles y basadas en el análisis descriptivo realizado.",
    )

    conclusiones = generar_conclusiones(analyzer)
    for indice, conclusion in enumerate(conclusiones, start=1):
        st.markdown(
            f"""
            <div class="tarjeta" style="margin-bottom:12px;height:auto;">
              <span class="item-tag">Conclusión {indice}</span>
              <p style="margin:10px 0 0;">{html.escape(conclusion)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### ⚖️ Alcance y limitaciones")
    st.markdown(
        """
        - El análisis identifica **asociaciones descriptivas**, no relaciones causales.
        - No se entrenaron ni evaluaron modelos predictivos, conforme al alcance del caso.
        - `duration` se conoce al finalizar el contacto; no debe utilizarse para decidir previamente a quién llamar.
        - `pdays = 999` funciona como código de ausencia de contacto previo y no como un número ordinario de días.
        - Las categorías `unknown` son información desconocida, aunque no estén registradas como valores nulos.
        """
    )

    texto_descarga = "CONCLUSIONES - BANK MARKETING\n\n" + "\n\n".join(
        f"{indice}. {conclusion}" for indice, conclusion in enumerate(conclusiones, start=1)
    )
    st.download_button(
        "Descargar conclusiones en TXT",
        data=texto_descarga.encode("utf-8"),
        file_name="conclusiones_bank_marketing.txt",
        mime="text/plain",
    )
    pie_pagina()


# =============================================================================
# INICIALIZACIÓN, MENÚ LATERAL Y ENRUTAMIENTO
# =============================================================================

for clave, valor in {
    "dataset": None,
    "dataset_name": None,
    "dataset_hash": None,
}.items():
    st.session_state.setdefault(clave, valor)

inyectar_estilos()

OPCIONES_MENU = [
    "🏠  Home",
    "📂  Carga del dataset",
    "📊  Análisis EDA",
    "✅  Conclusiones",
]

with st.sidebar:
    st.markdown(
        """
        <div class="marca-sidebar">
          <div class="marca-logo">OM</div>
          <div>
            <div class="marca-nombre">Bank Marketing EDA</div>
            <div class="marca-sub">Python for Analytics</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="version">{VERSION}</div>', unsafe_allow_html=True)
    st.markdown("##### Navegación")
    opcion = st.radio(
        "Selecciona una sección",
        OPCIONES_MENU,
        key="navegacion",
        label_visibility="collapsed",
    )

    dataframe_sesion = st.session_state.get("dataset")
    if dataframe_sesion is None:
        st.markdown(
            '<div class="estado-datos">⚪ <b>Sin dataset</b><br>Los módulos analíticos están bloqueados.</div>',
            unsafe_allow_html=True,
        )
    else:
        nombre_seguro = html.escape(str(st.session_state.get("dataset_name") or "Dataset cargado"))
        st.markdown(
            f'<div class="estado-datos">🟢 <b>Dataset activo</b><br>{nombre_seguro}<br>{len(dataframe_sesion):,} filas · {len(dataframe_sesion.columns)} columnas</div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.caption(f"{AUTOR['nombre']} · {AUTOR['país']} 🇵🇪 · {AUTOR['año']}")

if opcion == "🏠  Home":
    mostrar_home()
elif opcion == "📂  Carga del dataset":
    mostrar_carga()
elif opcion == "📊  Análisis EDA":
    mostrar_eda()
else:
    mostrar_conclusiones()
