import streamlit as st
import polars as pl
import altair as alt
import country_converter as cc
st.write("""
# Consumo di energia in Europa
""")

#url_2 = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_cb_em?format=TSV&compressed=true"
url_2 = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_cb_e?format=TSV&compressed=true"
st.write(f"""
     La seguente tabella mostra i dati sulla produzione energetica dal [sito Eurostat]({url_2}).
 """)

@st.cache_data
def get_data_pem(url):
    data = (
        pl.read_csv(
            url,
            separator="\t",
            null_values=["", ":", ": "],
            )
        .select(
            pl.col("freq,nrg_bal,siec,unit,geo\\TIME_PERIOD")
            .str.split(",")
            .list.to_struct(fields=["freq","nrg_bal","siec", "unit", "state"])
            #.list.to_struct(fields=["freq","siec", "unit", "unique_id"])
            .alias("combined_info"),
            pl.col("*").exclude("freq,nrg_bal,siec,unit,geo\\TIME_PERIOD")
        )
        .unnest("combined_info")
        .unpivot(
            index=["freq","nrg_bal","siec", "unit", "state"],
            value_name="energy_cons",
            variable_name="date"
        ).with_columns(
            date=pl.col("date")
            .str.replace(" ", "")
            .str.to_date(format='%Y'),
            energy_cons=pl.col("energy_cons")
            .str.replace(" ", "")
            .str.replace("p", "")
            .str.replace("u", "")
            .str.replace("e", "")
            .str.replace("n", "")
            .str.replace("d", "")
            #.str.replace(":", "")
            .str.replace(":", "123456789")
            .cast(pl.Float64),
            unique_id=pl.col("state")+";"+pl.col("nrg_bal")
        )
        .filter(pl.col("energy_cons").is_not_null())
        .filter(pl.col("energy_cons") != 123456789)
        .drop("freq",
              "unit",
              "siec")
        .sort("state", "date") 
    )
    return data

df_momentaneo = get_data_pem(url_2)

# list_consuption = ["FC" , "FC_IND_E" , "FC_TRA_E", "FC_OTH_CP_E", "FC_OTH_HH_E", "FC_OTH_AF_E"]
list_consuption = ["FC_IND_E" , "FC_TRA_E", "FC_OTH_CP_E", "FC_OTH_HH_E", "FC_OTH_AF_E"]

df = pl.DataFrame()
for consuption in list_consuption:
    df_2 = df_momentaneo.filter(pl.col("nrg_bal") == consuption)
    df = pl.concat([df, df_2])

st.write(df)
list = df.select("nrg_bal").unique()
st.write(list)

# from webapp import pred_state
# from webapp import Arima

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.models import SeasonalNaive

@st.cache_data
def Arima(state):
    ts = df.filter(
        pl.col("unique_id") == state
        ).select(
            pl.col("date").alias("ds"),
            pl.col("energy_cons").alias("y"),
            pl.col("unique_id")
            )
    sf = StatsForecast(
        models = [AutoARIMA(season_length = 12)],
        freq = '1y',
        n_jobs=-1,
        fallback_model=SeasonalNaive(season_length=12)
        )
    ts_pred = sf.forecast(df=ts, h=4, level=[90])
    ts_pred = ts_pred\
        .rename({"AutoARIMA": "energy_cons"})\
        .rename({"AutoARIMA-lo-90": "AutoARIMA_low90"})\
        .rename({"AutoARIMA-hi-90": "AutoARIMA_hi90"})
    return ts_pred

@st.cache_data
def pred_cons(filtro):
    df2 = df.filter(pl.col("nrg_bal") == filtro).with_columns(
        AutoARIMA_low90 = pl.lit(0),
        AutoARIMA_hi90 = pl.lit(0), 
        predicted = pl.lit(False)
        ).sort("AutoARIMA_hi90", "AutoARIMA_low90", "date", "energy_cons", "nrg_bal","predicted", "state","unique_id")  
    countries_2 = df2.select("unique_id").unique().sort("unique_id").to_series()
    pred = pl.DataFrame()
    filtered_countries = countries_2
    for state in filtered_countries:
        pred = pl.concat([pred, Arima(state)])
        
    pred = pred.with_columns(
        predicted = pl.lit(True),
        ).with_columns(
            pl.col("unique_id")
                .str.split(";")
                .list.to_struct(fields=["state", "nrg_bal"])
                .alias("combined_info"),
            date = pl.col("ds")
        ).unnest("combined_info").drop("ds")

    sorted_columns = sorted(df2.columns)
    df2 = df2.select(sorted_columns)
    pred = pred.select(sorted(pred.columns))

    df_combined = pl.concat([df2, pred], how= "vertical_relaxed")
    return df_combined

df_combined = pl.DataFrame()
for consuption in list_consuption:
    df_3 = pred_cons(consuption)
    df_combined = pl.concat([df_combined, df_3])#.filter(pl.col("nrg_bal") != "E7000")
st.write(df_combined)

list = df_combined.select("nrg_bal").unique()
st.write(list)

##########################################################################################################

def select_state():
    state = st.selectbox(
        "Seleziona uno stato",
        df["state"].unique().sort(),
        index=13
    )
    return state

selected_state = select_state()

stati = df_combined.filter(pl.col("state") == selected_state).filter(pl.col("date") > pl.datetime(2010, 1, 1))


# Create a selection that chooses the nearest point & selects based on x-value
nearest = alt.selection_point(nearest=True, on="pointerover",
                              fields=["date"], empty=False)

highlight = alt.selection_point(on='pointerover', fields=['energy_cons'], nearest=True)

# The basic line
line = alt.Chart(stati).mark_line(interpolate="basis").encode(
    x="date",
    y="energy_cons:Q",
    color="nrg_bal:N"
)
when_near = alt.when(nearest)

conf_int = stati.filter(pl.col("predicted") == True)
st.write(conf_int)
band = alt.Chart(conf_int).mark_errorband(extent='ci').encode(
    x="date",
    y=alt.Y("AutoARIMA_low90:Q", title="Energy Consumption"),
    y2="AutoARIMA_hi90:Q",
    color="nrg_bal:N"
)

# Draw points on the line, and highlight based on selection
points = line.mark_point().encode(
    opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
)

# Draw a rule at the location of the selection
rules = alt.Chart(stati).transform_pivot(
    "nrg_bal",
    value="energy_cons",
    groupby=["date"]
).mark_rule(color="gray").encode(
    x="date",
    opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
    tooltip=[alt.Tooltip(c, type="quantitative") for c in list_consuption],
).add_params(nearest)

# Put the five layers into a chart and bind the data
prova = alt.layer(
    band, line, points, rules
).properties(
    width=800, height=300
).resolve_scale(
    y="shared"
)
prova

highlight = alt.selection_point(on='pointerover', fields=['nrg_bal'], nearest=True)

# base = alt.Chart(stati).encode(
#     x='date:T',
#     y='price:Q',
#     color='symbol:N'
# )

base = alt.Chart(stati).encode(
    x="date",
    y="energy_cons:Q",
    color="nrg_bal:N"
)

points_2 = base.mark_circle().encode(
    opacity=alt.value(0)
).add_params(
    highlight
).properties(
    width=600
)

lines_2 = base.mark_line().encode(
    size=alt.when(~highlight).then(alt.value(1)).otherwise(alt.value(3))
)

prova_2 = alt.layer(
    points_2, band, lines_2
).properties(
    width=800, height=300
).resolve_scale(
    y="shared"
)
prova_2