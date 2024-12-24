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
def get_data_pem(url):
    data = (
        pl.read_csv(
            url,
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
        #.filter(pl.col("siec") == "TOTAL")
        .filter(pl.col("unit") == "GWH")
        .drop("freq")
        .drop("unit")
        .filter(pl.col("siec") != "RA110",
            pl.col("siec") != "RA120",
            pl.col("siec") != "RA130",
            pl.col("siec") != "RA310",
            pl.col("siec") != "RA320",
            pl.col("siec") != "RA410",
            pl.col("siec") != "RA420",
            pl.col("siec") != "FE"
            )
        #.filter(pl.col("unique_id") != "EU27_2020")
        #.sort("state", "date")
        #.filter(pl.col("state") == "IT")
        #.sort("state", "date")
        .sort("state", "date") 
    )
    return data

df = get_data_pem(url_1)
#df = data

################################################################################################################
url_2 = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_cb_em?format=TSV&compressed=true"
################################################################################################################

countries = df.select("unique_id").unique().sort("unique_id").to_series()
#countries = df.select(["unique_id", "siec"]).unique().sort("unique_id")

#.filter(pl.col("unique_id") != "EU27_2020")
#st.write(countries)

################################################################################

# countries2 = df.select("unique_id").unique().sort("unique_id")
# selected_country2 = st.multiselect(
#     "Seleziona uno stato",
#     countries,
#     #default="EU27_2020",
#     key="unique_key_country2"
# )

#@st.cache_data
def select_siec():
    siec = st.selectbox(
        "Seleziona un tipo di energia",
        df["siec"].unique().sort(),
        index=13
    )
    return siec

#@st.cache_data
def select_year(y_range):
    year = st.select_slider(
            "Seleziona un anno",
            #annual_production["year"].unique(),
            y_range.unique(),
            value = 2024
            #index=0
        )
    return year

def select_country():
    country = st.multiselect(
        "Seleziona uno stato",
        df.select("state").unique().sort("state"),
        default="IT"
        )
    return country

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

@st.cache_data
def pred_siec(filtro):#df2: pl.DataFrame, siec):
    #df2 = df2
    df2 = df.filter(pl.col("siec") == filtro).with_columns(
        AutoARIMA_low90 = pl.lit(0),
        AutoARIMA_hi90 = pl.lit(0), 
        predicted = pl.lit(False)
        ).sort("AutoARIMA_hi90", "AutoARIMA_low90", "date", "energy_prod", "predicted", "siec", "state","unique_id")  
    st.write(df2)
    countries_2 = df2.select("unique_id").unique().sort("unique_id").to_series()
    pred = pl.DataFrame()
    filtered_countries = countries_2#countries.filter(pl.col("siec") == siec)
    for state in filtered_countries:
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

    sorted_columns = sorted(df2.columns)
    df2 = df2.select(sorted_columns)
    pred = pred.select(sorted(pred.columns))

    df_combined = pl.concat([df2, pred], how= "vertical_relaxed")
    return df_combined


# st.line_chart(
#     #data=x,
#     #pred.filter(pl.col("unique_id").is_in(selected_country)),
#     x="date",
#     #y=["AutoARIMA", "y"],
#     y = "energy_prod",
#     color="unique_id"
# )
# ################################################################################
### Creazione Mappa

selected_siec = select_siec()
df_combined = pred_siec(selected_siec)

annual_production = (
    df_combined.with_columns(year=pl.col("date").dt.year())
    .group_by(["state", "year", "unique_id", "siec"])
    .agg(y=pl.sum("energy_prod"))
    .sort(["state", "year"])
    .filter(pl.col("state") != "EU27_2020")
)
st.write(annual_production)
st.write(df_combined)

st.write("""
## Produzione totale annuale per ogni nazione
""") #, annual_production)

selected_year = select_year(annual_production["year"])

annual_production = annual_production.filter(
    pl.col("year") == selected_year)

# converted_countries = cc.convert(names=annual_production["unique_id"], to='ISOnumeric')
converted_countries = cc.convert(names=annual_production["state"], to='ISOnumeric')

annual_production = annual_production.with_columns(
    pl.Series("state", converted_countries).alias("ISO")
).filter(pl.col("siec") == selected_siec).sort("y")

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
    #title='Produzione energetica annuale in Europa',
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
    tooltip=['state:N','y:Q']
)

background + map
#################################################################################

@st.cache_data
def pred_state(filtro):#df2: pl.DataFrame, siec):
    #df2 = df2
    for state in filtro:
        df2 = df.filter(pl.col("state") == state).with_columns(
            AutoARIMA_low90 = pl.lit(0),
            AutoARIMA_hi90 = pl.lit(0), 
            predicted = pl.lit(False)
            ).sort("AutoARIMA_hi90", "AutoARIMA_low90", "date", "energy_prod", "predicted", "siec", "state","unique_id")  
        st.write(df2)
        countries_2 = df2.select("unique_id").unique().sort("unique_id").to_series()
        pred = pl.DataFrame()
        filtered_countries = countries_2#countries.filter(pl.col("siec") == siec)
        for state in filtered_countries:
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

        sorted_columns = sorted(df2.columns)
        df2 = df2.select(sorted_columns)
        pred = pred.select(sorted(pred.columns))

        df_combined = pl.concat([df2, pred], how= "vertical_relaxed")
        return df_combined

selected_country_2 = select_country()
df_combined_2 = pred_state(selected_country_2)
df_combined_2 = df_combined_2.filter(
    pl.col("siec") != "TOTAL",
    #pl.col("siec") != "CF",
    #pl.col("siec") != "RA000",
    pl.col("siec") != "CF_R",
    pl.col("siec") != "C0000",
    pl.col("siec") != "O4000XBIO",
    pl.col("siec") != "G3000",
    #pl.col("siec") != "N9000",
    pl.col("siec") != "CF_NR",
    pl.col("siec") != "FE",
    pl.col("siec") != "RA100",
    pl.col("siec") != "RA200",
    pl.col("siec") != "RA300",
    pl.col("siec") != "RA400",
    pl.col("siec") != "RA500_5160",
    )

color_palette = alt.Scale(
    domain=['RA000', 'CF', 'X9900', 'N9000'],
    range=['#00b25d', '#b51d14', '#cacaca', '#ddb310']
)

area = alt.Chart(df_combined_2).mark_area().encode(
    x="date:T",
    y=alt.Y("energy_prod:Q").stack("normalize"),
    color=alt.Color("siec:N", scale = color_palette)
).properties(
    width=800, height=500
)
area

#################################################################################
import math
# st.write(annual_production)

# annual_production_selected = annual_production.sort("y", descending=True).head(10)

# polar_bars = alt.Chart(annual_production_selected).mark_arc(stroke='white', tooltip=True).encode(
#     theta=alt.Theta("state:N"),
#     radius=alt.Radius('y:Q').scale(type='linear'),
#     radius2=alt.datum(1),
# )
# p = int(max_value/6)
# # Create the circular axis lines for the number of observations
# axis_rings = alt.Chart(pl.DataFrame({"ring": range(p, int(max_value),p)})).mark_arc(stroke='lightgrey', fill=None).encode(
#     theta=alt.value(2 * math.pi),
#     radius=alt.Radius('ring').stack(False)
# )
# axis_rings_labels = axis_rings.mark_text(color='grey', radiusOffset=5, align='left').encode(
#     text="ring",
#     theta=alt.value(math.pi / 4)
# )

# # Create the straight axis lines for the time of the day
# axis_lines = alt.Chart(pl.DataFrame({
#     "radius": max_value,
#     "theta": math.pi / 2,
#     'classifica': ['1', '2', '3', '4']
# })).mark_arc(stroke='lightgrey', fill=None).encode(
#     theta=alt.Theta('theta').stack(True),
#     radius=alt.Radius('radius'),
#     radius2=alt.datum(1),
# )
# axis_lines_labels = axis_lines.mark_text(
#         color='grey',
#         radiusOffset=5,
#         thetaOffset=-math.pi / 4,
#         # These adjustments could be left out with a larger radius offset, but they make the label positioning a bit clearner
#         #align=alt.expr('datum.classifica == "2" ? "right" : datum.classifica == "3" ? "left" : "center"'),
#         #baseline=alt.expr('datum.classifica == "4" ? "bottom" : datum.classifica == "1" ? "top" : "middle"'),
#     )#.encode(text="classifica")

# polar_bar = alt.layer(
#     axis_rings,
#     polar_bars,
#     axis_rings_labels,
#     axis_lines,
#     axis_lines_labels,
#     title=['Classifica', '']
# ).properties(
#     width=800, height=500
# )
# polar_bar
#################################################################################

siec = st.selectbox(
        "Seleziona un tipo di energia",
        df_combined_2["siec"].unique().sort(),
    )
selected_year_months = df_combined_2.with_columns(
    year = pl.col("date").dt.year(),
    month = pl.col("date").dt.month()
).filter(
    pl.col("year") == selected_year,
).filter(pl.col("siec") == siec)

st.write(selected_year_months)

polar_bars_2 = alt.Chart(selected_year_months).mark_arc(stroke='white', tooltip=True).encode(
    theta=alt.Theta("month:N"),
    radius=alt.Radius('energy_prod:Q').scale(type='linear'),
    radius2=alt.datum(1),
)

max_value_m = selected_year_months['energy_prod'].max()
p2 = int(max_value_m/3)
# Create the circular axis lines for the number of observations
axis_rings_2 = alt.Chart(pl.DataFrame({"ring": range(p2, int(max_value_m),p2)})).mark_arc(stroke='lightgrey', fill=None).encode(
    theta=alt.value(2 * math.pi),
    radius=alt.Radius('ring').stack(False)
)
axis_rings_labels_2 = axis_rings_2.mark_text(color='grey', radiusOffset=5, align='left').encode(
    text="ring",
    theta=alt.value(math.pi / 4)
)

# Create the straight axis lines for the time of the day
axis_lines_2 = alt.Chart(pl.DataFrame({
    "radius": max_value_m,
    "theta": math.pi / 2,
    'mese': ['Gennaio', 'Aprile', 'Luglio', 'Ottobre']
})).mark_arc(stroke='lightgrey', fill=None).encode(
    theta=alt.Theta('theta').stack(True),
    radius=alt.Radius('radius'),
    radius2=alt.datum(1),
)
axis_lines_labels_2 = axis_lines_2.mark_text(
        color='grey',
        radiusOffset=5,
        thetaOffset=-math.pi / 4,
        # These adjustments could be left out with a larger radius offset, but they make the label positioning a bit clearner
        # align=alt.expr('datum.mese == "Aprile" ? "right" : datum.mese == "Ottobre" ? "left" : "center"'),
        # baseline=alt.expr('datum.mese == "Luglio" ? "bottom" : datum.mese == "Gennaio" ? "top" : "middle"'),
    )#.encode(text="mese")

polar_bar_2 = alt.layer(
    axis_rings_2,
    polar_bars_2,
    axis_rings_labels_2,
    axis_lines_2,
    axis_lines_labels_2,
    title=['Produzione mensile']
).properties(
    width=800, height=500
)
polar_bar_2
#################################################################################

st.write("""
## Qual è la produzione di energia di uno stato europeo?
""")#, df)


stati = df_combined.filter(pl.col("state").is_in(selected_country_2)).filter(pl.col("siec") == selected_siec)
st.write(stati)

nearest = alt.selection_point(nearest=True, on="pointerover",
                              fields=["date"], empty=False)

min_value_2 = stati['energy_prod'].min() - 0.2*stati['energy_prod'].min()
max_value_2 = stati['energy_prod'].max() + 0.2*stati['energy_prod'].max()

# The basic line
line = alt.Chart(stati).mark_line(interpolate="basis").encode(
    x="date",
    #y="energy_prod",
    y=alt.Y('energy_prod:Q', scale=alt.Scale(domain=[min_value_2, max_value_2])).axis(title='ciao'),
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
Europe_tot = df_combined.filter(pl.col("state")== "EU27_2020").filter(pl.col("siec") == selected_siec
    ).with_columns(
        y2=pl.col("energy_prod")/27 # - min_y
)
min_value_EU = Europe_tot['y2'].min() - 0.2*Europe_tot['y2'].min()
max_value_EU = Europe_tot['y2'].max() + 0.2*Europe_tot['y2'].max()

base = alt.Chart(Europe_tot).encode(
    alt.X('date').title(None)
)

line_EU = base.mark_line(opacity=0.5, stroke='#FF0000', interpolate='monotone', strokeDash=[2,2]).encode(
    #alt.Y('y2', scale=alt.Scale(domain=[4000, 12000])).axis(title='Produzione Totale di Energia in Europa')
    alt.Y('y2', scale=alt.Scale(domain=[min_value_EU, max_value_EU]))#.axis(title='Produzione Media di Energia in Europa')
)

# Put the five layers into a chart and bind the data
prova = alt.layer(
    line_EU, line, selectors, points, rules, text
).properties(
    width=800, height=300
).resolve_scale(
    y="independent"
)
prova 

# Draw a rule at the location of the selection
rules2 = alt.Chart(stati).transform_fold(
    ["state"],
    as_=["key", "value"]
).mark_rule(color="gray").encode(
    x="date:T",
    opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
    tooltip=[alt.Tooltip("value:Q", title="energy_prod")],
).add_params(nearest)


# Put the five layers into a chart and bind the data
prova2 = alt.layer(
    line_EU, line, selectors, points, rules2, text
).properties(
    width=800, height=300
).resolve_scale(
    y="independent"
)
prova2
#################################################################################


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