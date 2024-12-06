import streamlit as st
import polars as pl
import altair as alt
import country_converter as cc
#.venv\Scripts\activate
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
    #.filter(pl.col("unique_id") != "EU27_2020")
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

annual_production = (
    df.with_columns(year=pl.col("ds").dt.year())
    .group_by(["unique_id", "year"])
    .agg(y=pl.sum("y"))
    .sort(["unique_id", "year"])
)

st.write("""
## Produzione totale annuale per ogni nazione
""", annual_production)

selected_year = st.selectbox(
    "Seleziona un anno",
    annual_production["year"].unique(),
    #default = 2024
    index=0
)
annual_production = annual_production.filter(
    pl.col("year") == selected_year)
annual_production = annual_production.with_columns(
    pl.col("unique_id").alias("ISO")
)

annual_production = annual_production.filter(pl.col("unique_id") != "EU27_2020")
#converted_countries = cc.convert(names=annual_production["unique_id"], to='ISOnumeric')
converted_countries = cc.convert(names=annual_production["unique_id"], to='ISOnumeric')

annual_production = annual_production.with_columns(
    pl.Series("ISO", converted_countries)
)


#def get_country_name(countries):
#    for country in countries:
#        country_name = country["unique_id"]
#        country = cc.convert(names=country_name, to='IS03')
#        st.write(country)

from vega_datasets import data
import json

countries_map = alt.topo_feature(data.world_110m.url, 'countries')
countries_map2 = f"https://r2.datahub.io/clvyjaryy0000la0cxieg4o8o/main/raw/data/countries.geojson"
countries_map3 = alt.topo_feature(f"https://cdn.jsdelivr.net/npm/world-atlas@2/countries-50m.json", 'countries')
#with open('land-50m.json', "r", encoding="utf-8") as f:
#    map_json = json.load(f)
#countries_map3 = alt.topo_feature(map_json, 'countries')
#https://r2.datahub.io/clvyjaryy0000la0cxieg4o8o/main/raw/data/countries.geojson

source = annual_production
min_value = source['y'].min()
max_value = source['y'].max()

# source = source.with_columns(
#    pl.col("ISO").cast(pl.Utf8).alias("state2")
# )

# source = source.with_columns(
#     pl.when(pl.col("unique_id") == "BE").then(pl.lit("040"))
#     .when(pl.col("unique_id") == "AT").then(pl.lit("056"))
#     .otherwise(pl.col("ISO"))
#     .alias("state2")
# )

source = source.with_columns(
    pl.col("ISO").cast(pl.Utf8)).with_columns(
    pl.when(pl.col("ISO").str.len_chars() < 3)
    .then(pl.concat_str([pl.lit("0"), pl.col("ISO")]))
    .otherwise(pl.col("ISO")).alias("ISO_str")
    )

st.write(source)

map4 = alt.Chart(countries_map3).mark_geoshape(
    fill='#666666',
    stroke='white'
).project(
    type= 'mercator',
    scale= 350,                          # Magnify
    center= [20,50],                     # [lon, lat]
    clipExtent= [[0, 0], [800, 400]],    # [[left, top], [right, bottom]]
).properties(
    title='Europe (Mercator)',
    width=800, height=500
)

map5 = alt.Chart(countries_map3).mark_geoshape(
    stroke='black'
).project(
    type= 'mercator',
    scale= 350,                          # Magnify
    center= [20,50],                     # [lon, lat]
    clipExtent= [[000, 000], [800, 400]],    # [[left, top], [right, bottom]]
).properties(
    title='Europe (Mercator)',
    width=800, height=500
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(source, 'ISO_str', ['y', 'unique_id']),
).encode(
    color=alt.Color('y:Q', sort="descending", scale=alt.Scale(
        scheme='inferno', domain=(min_value,max_value)), legend=alt.Legend(title="", tickCount=6))
)

map4 + map5

selected_country = st.multiselect(
    "Seleziona uno stato",
    countries,
    default="EU27_2020"
)

stati = df.filter(pl.col("unique_id").is_in(selected_country))

# st.line_chart(
#     df.filter(pl.col("unique_id").is_in(selected_country)),
#     x="ds",
#     y="y",
#     color="unique_id"
# )

nearest = alt.selection_point(nearest=True, on="pointerover",
                              fields=["ds"], empty=False)

# The basic line
line = alt.Chart(stati).mark_line(interpolate="basis").encode(
    x="ds",
    y="y",
    color="unique_id"
)

# Transparent selectors across the chart. This is what tells us
# the x-value of the cursor
selectors = alt.Chart(stati).mark_point().encode(
    x="ds",
    opacity=alt.value(0),
).add_params(
    nearest
)
when_near = alt.when(nearest)

# Draw points on the line, and highlight based on selection
points = line.mark_point().encode(
    opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
)

# Draw text labels near the points, and highlight based on selection
text = line.mark_text(align="left", dx=5, dy=-5).encode(
    text=when_near.then("y:Q").otherwise(alt.value(" "))
)

# Draw a rule at the location of the selection
rules = alt.Chart(stati).mark_rule(color="gray").encode(
    x="ds",
).transform_filter(
    nearest
)

# Put the five layers into a chart and bind the data
prova = alt.layer(
    line, selectors, points, rules, text
).properties(
    width=600, height=300
)
prova
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
    #    pl.lit(state).alias("ISO")
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