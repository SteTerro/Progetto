import streamlit as st
import polars as pl
import pandas as pd
import altair as alt

st.write("""
# Produzione energetica in Europa
""")

url = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_cb_pem?format=TSV&compressed=true"
st.write(f"""
     La seguente tabella mostra i dati sulla produzione energetica dal [sito Eurostat]({url}).
 """)

df = (
    pl.read_csv(
        url,
        separator="\t",
        null_values=["", ":", ": "],
        )
    .select(
        pl.col("freq,siec,unit,geo\\TIME_PERIOD")
        .str.split(",")
        #.list.to_struct(fields=["freq","siec", "unit", "state"])
        .list.to_struct(fields=["freq","siec", "unit", "unique_id"])
        .alias("combined_info"),
        pl.col("*").exclude("freq,siec,unit,geo\\TIME_PERIOD")
    )
    .unnest("combined_info")
    .unpivot(
        #index=["freq","siec", "unit", "state"],
        index=["freq","siec", "unit", "unique_id"],
        #value_name="energy_productivity",
        value_name="y",
        #variable_name="date"
        variable_name="ds"
    )
    #.with_columns(# Add a new column 'age_in_5_years' based on the 'age' column
    #    (pl.col("ds") + "-01"))

    #    pl.lit("-01", dtype="str").alias("day"))
    .with_columns(
        #date=pl.col("date")
        ds=pl.col("ds")
        .str.replace(" ", "")
        .str.to_date(format='%Y-%m'),
        #.str.strptime(pl.Date, format='%Y-%m'),
        #.str.strptime(pl.Date, format='%Y-%m'),
        #energy_productivity=pl.col("energy_productivity")
        y=pl.col("y")
        .str.replace(" ", "")
        .str.replace("p", "")
        .str.replace("u", "")
        .str.replace("e", "")
        .str.replace("n", "")
        .str.replace("d", "")
        .str.replace(":c", "123456789")
        .cast(pl.Float64),
    )
    #.filter(pl.col("energy_productivity").is_not_null())
    #.filter(pl.col("energy_productivity") != 123456789)
    .filter(pl.col("y").is_not_null())
    .filter(pl.col("y") != 123456789)
    .filter(pl.col("siec") == "TOTAL")
    .filter(pl.col("unit") == "GWH")
    #.filter(pl.col("state") == "IT")
    #.sort("state", "date")
    #.filter(pl.col("unique_id") == "IT")
    #.sort("state", "date")
    .sort("unique_id", "ds")
)

st.write("""
## Qual e' la produzione di energia di uno stato europeo?
""", df)

countries = df.select("unique_id").unique().sort("unique_id")


selected_country = st.multiselect(
    "Seleziona uno stato",
    countries,
    default="EU27_2020"
)

st.line_chart(
    df.filter(pl.col("unique_id").is_in(selected_country)),
    x="ds",
    y="y",
    color="unique_id"
)

st.write("""
## Quali sono le previsioni per la produzione di energia?
""")

#countries2 = df.select("unique_id").unique().sort("unique_id")
selected_country2 = st.multiselect(
    "Seleziona uno stato",
    countries,
    default="EU27_2020",
    key="unique_key_country2"
)

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.models import SeasonalNaive

def Arima(state):
    ts = df.filter(
        pl.col("unique_id") == state
        ).select(
            pl.col("*")
            .exclude("freq")
            .exclude("siec")
            .exclude("unit")
            )#.to_pandas()
    #ts['ds'] = pd.to_datetime(ts['ds'], format='%Y-%m')
    #ts = ts.set_index('unique_id')
    #ts["ds"].dt.offset_by("1mo")
    sf = StatsForecast(
        models = [AutoARIMA(season_length = 12)],
        #ME: monthly end, MS: monthly start, M: monthly non riconosceva chiedeva offset
        freq = '1mo',
        n_jobs=-1,
        fallback_model=SeasonalNaive(season_length=12)
        )
    #sf.fit(ts)
    #ts_pred = sf.predict(h=12, level=[95])
    ts_pred = sf.forecast(df=ts, h=48, level=[90]).drop("AutoARIMA-lo-90").drop("AutoARIMA-hi-90")
    ts_pred = ts_pred.rename({"AutoARIMA": "y"})
    #print(ts_pred.head(5))
    ris = pl.concat([ts, ts_pred])
    #ris["ds"].dt.offset_by("1mo")
    #ris = pl.DataFrame(ris)
    #ris = ris.with_columns(
    #    pl.lit(state).alias("state")
    #)
    return ris

x = pl.DataFrame()
for state in selected_country2:
    x = pl.concat([x, Arima(state)])


st.write(x)

st.line_chart(
    #data=x,
    x.filter(pl.col("unique_id").is_in(selected_country2)),
    x="ds",
    #y=["AutoARIMA", "y"],
    y = "y",
    color="unique_id"
)

from vega_datasets import data
import pycountry

source = alt.topo_feature(data.world_110m.url, 'countries')
param_projection = alt.param(value="equalEarth")

prova1 = alt.Chart(source, width=500, height=300).mark_geoshape(
    fill='lightgray',
    stroke='gray'
).project(
    type=alt.expr(param_projection.name)
).add_params(param_projection)

lista = [43, 32, 359, 385, 357, 420, 45, 372, 358, 33, 49, 30, 36, 353, 39, 371, 370, 352, 356, 31, 48, 315, 40, 421, 386, 34, 46]

prova2 = alt.Chart(source).mark_geoshape(
    fill='#666666',
    stroke='white'
).project(
    type= 'mercator',
    scale= 350,                          # Magnify
    center= [20,50],                     # [lon, lat]
    clipExtent= [[0, 0], [400, 300]],    # [[left, top], [right, bottom]]
).properties(
    title='Europe (Mercator)',
    width=400, height=300
)

def convert_iso_to_numeric(iso_code):
    try:
        return pycountry.countries.lookup(iso_code).numeric
    except LookupError:
        return None

st.write(df)
countries = alt.topo_feature(data.world_110m.url, 'countries')
source = df.filter(pl.col("ds") == pl.lit("2024-01-01").str.to_date())

min_value = df["y"].min()
max_value = df["y"].max()

prova3 = alt.Chart(source).mark_geoshape(
    stroke='gray'
).project(
    type= 'mercator',
).properties(
    title='Europe (Mercator)',
).encode(
    color=alt.Color('y:Q', sort="descending", 
                    scale=alt.Scale(
                        scheme='inferno', 
                        domain=(min_value,max_value)), 
                        legend=alt.Legend(title="", tickCount=6))
).transform_lookup(
    lookup='unique_id',
    from_=alt.LookupData(source, 'unique_id', ['y'])
)


prova1
prova2
prova3 