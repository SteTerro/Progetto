import streamlit as st
import polars as pl
import altair as alt
import country_converter as cc
#.venv\Scripts\activate.ps1
st.write("""
# Produzione energetica in Europa
""")

url_1 = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_cb_pem?format=TSV&compressed=true"
st.write(f"""
     La seguente tabella mostra i dati sulla produzione energetica dal [sito Eurostat]({url_1}).
 """)

#Sostituire Unique_id con state
#Sostituire Code con Unique_id

@st.cache_data
def get_data(url):
    data = (
        pl.read_csv(
            url_1,
            separator="\t",
            null_values=["", ":", ": "],
            )
        .select(
            pl.col("freq,siec,unit,geo\\TIME_PERIOD")
            .str.split(",")
            .list.to_struct(fields=["freq","siec", "unit", "state"])
            #.list.to_struct(fields=["freq","siec", "unit", "unique_id"])
            .alias("combined_info"),
            pl.col("*").exclude("freq,siec,unit,geo\\TIME_PERIOD")
        )
        .unnest("combined_info")
        .unpivot(
            index=["freq","siec", "unit", "state"],
            #index=["freq","siec", "unit", "unique_id"],
            value_name="energy_prod",
            #value_name="y",
            #variable_name="date"
            variable_name="date"
        )
        #.with_columns(# Add a new column 'age_in_5_years' based on the 'age' column
        #    (pl.col("date") + "-01"))

        #    pl.lit("-01", dtype="str").alias("day"))
        .with_columns(
            #date=pl.col("date")
            date=pl.col("date")
            .str.replace(" ", "")
            .str.to_date(format='%Y-%m'),
            #.str.strptime(pl.date, format='%Y-%m'),
            #.str.strptime(pl.date, format='%Y-%m'),
            #energy_productivity=pl.col("energy_productivity")
            energy_prod=pl.col("energy_prod")
            .str.replace(" ", "")
            .str.replace("p", "")
            .str.replace("u", "")
            .str.replace("e", "")
            .str.replace("n", "")
            .str.replace("d", "")
            .str.replace(":c", "123456789")
            .cast(pl.Float64),
            unique_id=pl.col("state")+";"+pl.col("siec")
        )
        #.filter(pl.col("energy_productivity").is_not_null())
        #.filter(pl.col("energy_productivity") != 123456789)
        .filter(pl.col("energy_prod").is_not_null())
        .filter(pl.col("energy_prod") != 123456789)
        .filter(pl.col("siec") == "TOTAL")
        .filter(pl.col("unit") == "GWH")
        .drop("freq")
        .drop("unit")
        #.filter(pl.col("unique_id") != "EU27_2020")
        #.sort("state", "date")
        #.filter(pl.col("state") == "IT")
        #.sort("state", "date")
        .sort("state", "date") 
    )
    return data

df = get_data(url_1)
#df = data

st.write("""
## Qual e' la produzione di energia di uno stato europeo?
""")#, df)

countries = df.select("unique_id").unique().sort("unique_id").to_series()
#countries = df.select(["unique_id", "siec"]).unique().sort("unique_id")

#.filter(pl.col("unique_id") != "EU27_2020")
#st.write(countries)

################################################################################
st.write("""
## Quali sono le previsioni per la produzione di energia?
""")

# countries2 = df.select("unique_id").unique().sort("unique_id")
# selected_country2 = st.multiselect(
#     "Seleziona uno stato",
#     countries,
#     #default="EU27_2020",
#     key="unique_key_country2"
# )

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.models import SeasonalNaive

#per recuperare la vecchia funzione vedere commit precedenti
@st.cache_data
def Arima(state):
    ts = df.filter(
        pl.col("unique_id") == state
        ).select(
            pl.col("date").alias("ds"),
            pl.col("energy_prod").alias("y"),
            pl.col("unique_id")
            )
    sf = StatsForecast(
        models = [AutoARIMA(season_length = 12)],
        freq = '1mo',
        n_jobs=-1,
        fallback_model=SeasonalNaive(season_length=12)
        )
    ts_pred = sf.forecast(df=ts, h=48, level=[90])
    ts_pred = ts_pred\
        .rename({"AutoARIMA": "energy_prod"})\
        .rename({"AutoARIMA-lo-90": "AutoARIMA_low90"})\
        .rename({"AutoARIMA-hi-90": "AutoARIMA_hi90"})
    return ts_pred

# x = pl.DataFrame()
# st.write(x)
# for state in selected_country2:
#     if x.is_empty():
#         x = Arima(state)
#     elif state in x["unique_id"]:
#         pass
#     else: 
#         x = pl.concat([x, Arima(state)])

df = df.with_columns(
        AutoARIMA_low90 = pl.lit(0),
        AutoARIMA_hi90 = pl.lit(0), 
        predicted = pl.lit(False)
    )

pred = pl.DataFrame()
for state in countries:
    pred = pl.concat([pred, Arima(state)])
    
pred = pred.with_columns(
    predicted = pl.lit(True),
    #unit = pl.lit("GWH"),
    ).with_columns(
        pl.col("unique_id")
            .str.split(";")
            .list.to_struct(fields=["state", "siec"])
            .alias("combined_info"),
        date = pl.col("ds")
    ).unnest("combined_info").drop("ds")

df = df.select(sorted(df.columns))
pred = pred.select(sorted(pred.columns))

    # st.write(df)
    # st.write(pred)

df_combined = pl.concat([df, pred], how= "vertical_relaxed")

# st.line_chart(
#     #data=x,
#     #pred.filter(pl.col("unique_id").is_in(selected_country)),
#     x="date",
#     #y=["AutoARIMA", "y"],
#     y = "energy_prod",
#     color="unique_id"
# )
# ################################################################################
# Creazione Mappa

# annual_production = (
#     df.with_columns(year=pl.col("date").dt.year())
#     .group_by(["unique_id", "year"])
#     .agg(y=pl.sum("energy_prod"))
#     .sort(["unique_id", "year"])
#     .filter(pl.col("state") != "EU27_2020")
# )

annual_production = (
    df.with_columns(year=pl.col("date").dt.year())
    .group_by(["state", "year", "unique_id", "siec"])
    .agg(y=pl.sum("energy_prod"))
    .sort(["state", "year"])
    .filter(pl.col("state") != "EU27_2020")
)

st.write(annual_production)
st.write("""
## Produzione totale annuale per ogni nazione
""")

selected_siec = st.selectbox(
    "Seleziona un tipo di energia",
    annual_production["siec"].unique(),
    index=0
)

selected_year = st.select_slider(
        "Seleziona un anno",
        annual_production["year"].unique(),
        value = 2024
        #index=0
    )

annual_production = annual_production.filter(
    pl.col("year") == selected_year)

# converted_countries = cc.convert(names=annual_production["unique_id"], to='ISOnumeric')
converted_countries = cc.convert(names=annual_production["state"], to='ISOnumeric')

annual_production = annual_production.with_columns(
    pl.Series("state", converted_countries).alias("ISO")
).filter(pl.col("siec") == selected_siec)

countries_map = alt.topo_feature(f"https://cdn.jsdelivr.net/npm/world-atlas@2/countries-50m.json", 'countries')
#countries_map = alt.topo_feature(f"https://cdn.jsdelivr.net/npm/world-atlas@2/countries-10m.json", 'countries')

source = annual_production
min_value = source['y'].min()
max_value = source['y'].max()

source = source.with_columns(
    pl.col("ISO").cast(pl.Utf8)).with_columns(
    pl.when(pl.col("ISO").str.len_chars() < 2)
    .then(pl.concat_str([pl.lit("00"), pl.col("ISO")]))
    .when(pl.col("ISO").str.len_chars() < 3)
    .then(pl.concat_str([pl.lit("0"), pl.col("ISO")]))
    .otherwise(pl.col("ISO")).alias("ISO_str")
    )

background = alt.Chart(countries_map).mark_geoshape(
    fill='#666666',
    stroke='white'
).project(
    type= 'mercator',
    scale= 350,                          # Magnify
    center= [20,50],                     # [lon, lat]
    clipExtent= [[0, 0], [800, 400]],    # [[left, top], [right, bottom]]
).properties(
    title='Produzione energetica annuale in Europa',
    width=800, height=500
).encode(tooltip=alt.value(None))

map = alt.Chart(countries_map).mark_geoshape(
    stroke='black'
).project(
    type= 'mercator',
    scale= 350,                          # Magnify
    center= [20,50],                     # [lon, lat]
    clipExtent= [[000, 000], [800, 400]],    # [[left, top], [right, bottom]]
).properties(
    width=800, height=500
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(source, 'ISO_str', ['state', 'y']),
).encode(
    color=alt.Color('y:Q', sort="descending", scale=alt.Scale(
        scheme='inferno', domain=(min_value,max_value)), legend=alt.Legend(title="", tickCount=6)),
    tooltip=['unique_id:N','y:Q']
)

background + map
# ################################################################################


# selected_siec = st.selectbox(
#     "Seleziona un tipo di energia",
#     annual_production["siec"].unique(),
#     default="TOTAL"
#     #index=0
# )

countries_name = df.select("state").unique().sort("state")
selected_country = st.multiselect(
    "Seleziona uno stato",
    countries_name,
    default="IT"
)

stati = df.filter(pl.col("state").is_in(selected_country))

# st.line_chart(
#     df.filter(pl.col("unique_id").is_in(selected_country)),
#     x="date",
#     y="energy_prod",
#     color="unique_id"
# )

nearest = alt.selection_point(nearest=True, on="pointerover",
                              fields=["date"], empty=False)

# The basic line
line = alt.Chart(stati).mark_line(interpolate="basis").encode(
    x="date",
    y="energy_prod",
    #color="unique_id"
    color="state"
)

# Transparent selectors across the chart. This is what tells us
# the x-value of the cursor
selectors = alt.Chart(stati).mark_point().encode(
    x="date",
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
    text=when_near.then("energy_prod:Q").otherwise(alt.value(" "))
)

# Draw a rule at the location of the selection
rules = alt.Chart(stati).mark_rule(color="gray").encode(
    x="date",
).transform_filter(
    nearest
)


# Creazione Line Chart
Europe_tot = df.filter(pl.col("state") == "EU27_2020")
min_y = Europe_tot.select(pl.col("energy_prod")).min()
Europe_tot = Europe_tot.with_columns(
    y2=pl.col("energy_prod")/27 # - min_y
)
base = alt.Chart(Europe_tot).encode(
    alt.X('date').title(None)
)

line_EU = base.mark_line(opacity=0.5, stroke='#FF0000', interpolate='monotone', strokeDash=[2,2]).encode(
    alt.Y('y2').axis(title='Produzione Totale di Energia in Europa')
)

# Put the five layers into a chart and bind the data
prova = alt.layer(
    line_EU, line, selectors, points, rules
).properties(
    width=800, height=300
).resolve_scale(
    y="independent"
)
prova 
# ################################################################################

# nearest2 = alt.selection_point(nearest=True, on="pointerover",
#                               fields=["date"], empty=False)

# # The basic line
# line2 = alt.Chart(x).mark_line(interpolate="basis").encode(
#     x="date",
#     y="energy_prod",
#     color="unique_id"
# )
# # Put the five layers into a chart and bind the data
# prova2 = alt.layer(
#     line2
# ).properties(
#     width=800, height=300
# ).resolve_scale(
#     y="independent"
# ).encode(
#     strokeDash="pred:N"
# )

# prova2

# annual_production2 = (
#     x.with_columns(year=pl.col("date").dt.year())
#     .group_by(["unique_id", "year"])
#     .agg(y=pl.sum("energy_prod"))
#     .sort(["unique_id", "year"])
#     #.filter(pl.col("unique_id") != "EU27_2020")
# )

# st.write("""
# ## Produzione stimata totale annuale per ogni nazione
# """)

# selected_year2 = st.select_slider(
#     "Seleziona un anno",
#     annual_production2["year"].unique(),
#     value = 2024
# )

# annual_production2 = annual_production2.filter(
#     pl.col("year") == selected_year)


# select = alt.selection_point(name="select", on="click")
# highlight = alt.selection_point(name="highlight", on="pointerover", empty=False)

# stroke_width = (
#     alt.when(select).then(alt.value(2, empty=False))
#     .when(highlight).then(alt.value(1))
#     .otherwise(alt.value(0))
# )

# prova3 = alt.Chart(x, height=200).mark_bar(
#     fill="#4C78A8", stroke="black", cursor="pointer"
# ).encode(
#     x="unique_id",
#     y="energy_prod",
#     fillOpacity=alt.when(select).then(alt.value(1)).otherwise(alt.value(0.3)),
#     strokeWidth=stroke_width,
# ).configure_scale(bandPaddingInner=0.2).add_params(select, highlight)

# prova3