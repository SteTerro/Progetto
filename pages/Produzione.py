import streamlit as st
import polars as pl
import altair as alt
import country_converter as cc

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.models import SeasonalNaive

url_1 = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_cb_pem?format=TSV&compressed=true"

st.write("""
# Produzione energetica in Europa
""")

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
            .alias("combined_info"),
            pl.col("*").exclude("freq,siec,unit,geo\\TIME_PERIOD")
        )
        .unnest("combined_info")
        .unpivot(
            index=["freq","siec", "unit", "state"],
            value_name="energy_prod",
            variable_name="date"
        )
        .with_columns(
            date=pl.col("date")
            .str.replace(" ", "")
            .str.to_date(format='%Y-%m'),
            energy_prod=pl.col("energy_prod")
            .str.replace(" ", "")
            .str.replace("p", "")
            .str.replace("u", "")
            .str.replace("e", "")
            .str.replace("n", "")
            .str.replace("d", "")
            .str.replace(":c", "123456789")
            .cast(pl.Float64),
            unique_id=pl.col("state")+";"+pl.col("siec"),
            state = pl.col("state")
            .str.replace("EL", "GR")
        )
        .filter(
            pl.col("energy_prod") > 0,
            pl.col("energy_prod").is_not_null(),
            pl.col("energy_prod") != 123456789,
            pl.col("unit") == "GWH")
        .drop("freq")
        .drop("unit")
        .filter(
            pl.col("siec") != "RA110",
            pl.col("siec") != "RA120",
            pl.col("siec") != "RA130",
            pl.col("siec") != "RA310",
            pl.col("siec") != "RA320",
            pl.col("siec") != "RA410",
            pl.col("siec") != "RA420",
            pl.col("siec") != "FE"
            )
        .sort("state", "date") 
    )
    return data

df = get_data_pem(url_1)
st.write(df.select("siec").unique().sort("siec"))
cc_2 = cc.CountryConverter()
country_list = pl.from_pandas(cc_2.EU27as('ISO2'))
country_list = country_list.select(pl.col("ISO2")).to_series()

EU27_2020_mom = df.filter(pl.col("state") == "EU27_2020")
df = df.filter(pl.col("state").is_in(country_list))
df = pl.concat([df, EU27_2020_mom])

countries = df.select("unique_id").unique().sort("unique_id").to_series()
################################################################################
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
            y_range.unique(),
            value = 2024
        )
    return year

def select_country():
    country = st.multiselect(
        "Seleziona uno stato",
        df.select("state").unique().sort("state"),
        default="IT"
        )
    return country

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

@st.cache_data
def pred_siec(filtro):
    df2 = df.filter(pl.col("siec") == filtro).with_columns(
        AutoARIMA_low90 = pl.lit(0),
        AutoARIMA_hi90 = pl.lit(0), 
        predicted = pl.lit(False)
        ).sort("AutoARIMA_hi90", "AutoARIMA_low90", "date", "energy_prod", "predicted", "siec", "state","unique_id")  
    st.write(df2)
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
                .list.to_struct(fields=["state", "siec"])
                .alias("combined_info"),
            date = pl.col("ds")
        ).unnest("combined_info").drop("ds")

    sorted_columns = sorted(df2.columns)
    df2 = df2.select(sorted_columns)
    pred = pred.select(sorted(pred.columns))

    df_combined = pl.concat([df2, pred], how= "vertical_relaxed")
    return df_combined

selected_siec = select_siec()
df_combined = pred_siec(selected_siec)

annual_production = (
    df_combined.with_columns(year=pl.col("date").dt.year())
    .group_by(["state", "year", "unique_id", "siec"])
    .agg(y=pl.sum("energy_prod"))
    .sort(["state", "year"])
    .filter(pl.col("state") != "EU27_2020")
)

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

min_value = annual_production['y'].min()
max_value = annual_production['y'].max()

source = annual_production.with_columns(
    pl.col("ISO").cast(pl.Utf8)).with_columns(
    pl.when(pl.col("ISO").str.len_chars() < 2)
    .then(pl.concat_str([pl.lit("00"), pl.col("ISO")]))
    .when(pl.col("ISO").str.len_chars() < 3)
    .then(pl.concat_str([pl.lit("0"), pl.col("ISO")]))
    .otherwise(pl.col("ISO")).alias("ISO_str")
    )
st.write(annual_production)
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
################################################################################
country_default = df.filter(pl.col("siec") == selected_siec).group_by("state").agg(
    pl.sum("energy_prod").alias("total_energy")
).sort("total_energy", descending=True).head(5).select("state").to_series()


stati_line = df_combined.filter(pl.col("state").is_in(country_default)).filter(
    pl.col("siec") == selected_siec,
    pl.col("state") != "EU27_2020")

nearest = alt.selection_point(nearest=True, on="pointerover",
                              fields=["date"], empty=False)

highlight = alt.selection_point(on='pointerover', fields=['energy_cons'], nearest=True)

# The basic line
line = alt.Chart(stati_line).mark_line(interpolate="basis").encode(
    x="date",
    y="energy_prod:Q",
    color="state"
)
when_near = alt.when(nearest)

conf_int = stati_line.filter(pl.col("predicted") == True)
band = alt.Chart(conf_int).mark_errorband(extent='ci').encode(
    x="date",
    y=alt.Y("AutoARIMA_low90:Q", title="Energy Consumption"),
    y2="AutoARIMA_hi90:Q",
    color="state:N"
)

# Draw points on the line, and highlight based on selection
points = line.mark_point().encode(
    opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
)

Europe_tot = df_combined.filter(pl.col("state")== "EU27_2020").filter(pl.col("siec") == selected_siec
    ).with_columns(
        y2=pl.col("energy_prod")/27 # - min_y
)
min_value_EU = Europe_tot['y2'].min() - 0.4*Europe_tot['y2'].min()
max_value_EU = Europe_tot['y2'].max() + 0.4*Europe_tot['y2'].max()
base = alt.Chart(Europe_tot).encode(
    alt.X('date').title(None)
)

line_EU = base.mark_line(opacity=0.5, stroke='#FF0000', interpolate='monotone', strokeDash=[2,2]).encode(
    alt.Y('y2', scale=alt.Scale(domain=[min_value_EU, max_value_EU]))#.axis(title='Produzione Media di Energia in Europa')
).interactive()
# Draw a rule at the location of the selection
rules = alt.Chart(stati_line).transform_pivot(
    "nrg_bal",
    value="energy_prod",
    groupby=["date"]
).mark_rule(color="gray").encode(
    x="date",
    opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
    tooltip=[alt.Tooltip(c, type="quantitative") for c in stati_line["state"]],
).add_params(nearest)

# Put the five layers into a chart and bind the data
line_chart = alt.layer(
    line_EU , line, points, rules
).properties(
    width=800, height=300
).resolve_scale(
    y="independent"
)
line_chart

################################################################################
@st.cache_data
def pred_state(filtro):#df2: pl.DataFrame, siec):
    #df2 = df2
    for state in filtro:
        df2 = df.filter(pl.col("state") == state).with_columns(
            AutoARIMA_low90 = pl.lit(0),
            AutoARIMA_hi90 = pl.lit(0), 
            predicted = pl.lit(False)
            ).sort("AutoARIMA_hi90", "AutoARIMA_low90", "date", "energy_prod", "predicted", "siec", "state","unique_id")  
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
    pl.col("siec") != "CF_R",
    pl.col("siec") != "C0000",
    pl.col("siec") != "O4000XBIO",
    pl.col("siec") != "G3000",
    pl.col("siec") != "CF_NR",
    pl.col("siec") != "FE",
    pl.col("siec") != "RA100",
    pl.col("siec") != "RA200",
    pl.col("siec") != "RA300",
    pl.col("siec") != "RA400",
    pl.col("siec") != "RA500_5160",
    )

nearest = alt.selection_point(nearest=True, on="pointerover",
                              fields=["date"], empty=False)

selectors = alt.Chart(df_combined_2.filter(pl.col("date") > pl.datetime(2017, 1, 1))).mark_point().encode(
    x="date",
    opacity=alt.value(0),
).add_params(
    nearest
)
when_near = alt.when(nearest)

color_palette = alt.Scale(
    domain=['RA000', 'CF', 'X9900', 'N9000'],
    range=['#00b25d', '#b51d14', '#cacaca', '#ddb310']
)

area = alt.Chart(df_combined_2.filter(pl.col("date") > pl.datetime(2017, 1, 1))
    ).mark_area(
        opacity=0.5,
        interpolate='step-after',
        line=True
    ).encode(
        x="date:T",
        y=alt.Y("energy_prod:Q").stack(True),
        color=alt.Color("siec:N", scale = color_palette)
)

# Draw points on the line, and highlight based on selection
points = area.mark_point().encode(
    opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
)

# Draw text labels near the points, and highlight based on selection
text = area.mark_text(align="left", dx=5, dy=-5).encode(
    text=when_near.then("energy_prod:Q").otherwise(alt.value(" "))
)

# Draw a rule at the location of the selection
rules = alt.Chart(df_combined_2.filter(pl.col("date") > pl.datetime(2017, 1, 1))).mark_rule(color="gray").encode(
    x="date",
).transform_filter(
    nearest
)

area_chart = alt.layer(
    area, selectors, points, rules, text
).properties(
    width=800, height=500
).resolve_scale(
    y="shared"
)
area_chart