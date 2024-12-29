import streamlit as st
import polars as pl
import altair as alt
import country_converter as cc

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.models import SeasonalNaive

url_2 = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_cb_e?format=TSV&compressed=true"

st.write(f"""
# Consumo di energia in Europa
In questa pagina vengono analizzati i dati sul consumo di energia in Europa. I dati sono presi dal [sito Eurostat]({url_2}).
Di default l'analisi è svolta per solo gli stati membri dell'UE. Se si desidera estendere l'analisi anche ai paesi non membri dell'UE, è possibile farlo selezionando l'adeguata opzione.

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
        .filter(
            pl.col("energy_cons").is_not_null(),
            pl.col("energy_cons") != 123456789,
            pl.col("state")!="EA20",
            pl.col("nrg_bal").is_in(["FC", "FC_IND_E" , "FC_TRA_E", "FC_OTH_CP_E", "FC_OTH_HH_E", "FC_OTH_AF_E"]),
        )
        .drop("freq",
              "unit",
              "siec")
        .sort("state", "date") 
    )
    return data

list_consuption = ["FC", "FC_IND_E" , "FC_TRA_E", "FC_OTH_CP_E", "FC_OTH_HH_E", "FC_OTH_AF_E"]

# which_country = st.checkbox("Includi i paesi non membri dell'UE", value=False)


df = get_data_pem(url_2)

cc_2 = cc.CountryConverter()
country_list = pl.from_pandas(cc_2.EU27as('ISO2'))
country_list = country_list.select(pl.col("ISO2")).to_series()

EU27_2020_mom = df.filter(pl.col("state") == "EU27_2020")
df = df.filter(pl.col("state").is_in(country_list))
df = pl.concat([df, EU27_2020_mom])
# uso = df_2.filter(pl.col("state").is_in(country_list).not_())

# if which_country_radio == "Solo EU":
#     st.write("Solo EU")
#     EU27_2020_mom = df_2.filter(pl.col("state") == "EU27_2020")
#     df = df_2.filter(pl.col("state").is_in(country_list))
#     df = pl.concat([df, EU27_2020_mom])
# else:
#     df = pl.concat([df_2, EU27_2020_mom])



country_default = df.filter(pl.col("nrg_bal") == "FC").group_by("state").agg(
    pl.sum("energy_cons").alias("total_energy")
).sort("total_energy", descending=True).head(5).select("state").to_series()

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

##########################################################################################################
st.write(f"""
    ### Confronto del consumo di energia di vari stati europei.
 """)

year = st.select_slider(
            "Seleziona un anno",
            df_combined["date"].unique(),
        )

stati_map = df_combined.filter(
    pl.col("date") == year,
    pl.col("state") != "EU27_2020"
    )

converted_countries = cc.convert(names=stati_map["state"], to='ISOnumeric')

selected_bal = st.selectbox(
        "Seleziona un tipo di consumo di energia",
        df["nrg_bal"].unique().sort(),
    )

stati_map = stati_map.with_columns(
    pl.Series("state", converted_countries).alias("ISO")
).filter(pl.col("nrg_bal") == selected_bal).sort("energy_cons", "state")

countries_map = alt.topo_feature(f"https://cdn.jsdelivr.net/npm/world-atlas@2/countries-50m.json", 'countries')
#countries_map = alt.topo_feature(f"https://cdn.jsdelivr.net/npm/world-atlas@2/countries-10m.json", 'countries')

min_value = stati_map['energy_cons'].min()
max_value = stati_map['energy_cons'].max()

source = stati_map.with_columns(
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
    from_=alt.LookupData(source, 'ISO_str', ['state', 'energy_cons']),
).encode(
    color=alt.Color('energy_cons:Q', sort="descending", scale=alt.Scale(
        scheme='inferno', domain=(min_value,max_value)), legend=alt.Legend(title="", tickCount=6)),
    tooltip=['state:N','energy_cons:Q']
)

background + map

##########################################################################################################

st.write(f"""
    ### Consumo di energia nei vari settori per un singolo stato europeo.
 """)

def select_state():
    state = st.selectbox(
        "Seleziona uno stato",
        df["state"].unique().sort(),
        index=12
    )
    return state

selected_single_state = select_state()

stati_line = df_combined.filter(
    pl.col("state") == selected_single_state,
    pl.col("date") > pl.datetime(2010, 1, 1),
    pl.col("nrg_bal") != "FC")


# Create a selection that chooses the nearest point & selects based on x-value
nearest = alt.selection_point(nearest=True, on="pointerover",
                              fields=["date"], empty=False)

highlight = alt.selection_point(on='pointerover', fields=['energy_cons'], nearest=True)

# The basic line
line = alt.Chart(stati_line).mark_line(interpolate="basis").encode(
    x="date",
    y="energy_cons:Q",
    color="nrg_bal:N"
)
when_near = alt.when(nearest)

conf_int = stati_line.filter(pl.col("predicted") == True)
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
rules = alt.Chart(stati_line).transform_pivot(
    "nrg_bal",
    value="energy_cons",
    groupby=["date"]
).mark_rule(color="gray").encode(
    x="date",
    opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
    tooltip=[alt.Tooltip(c, type="quantitative") for c in stati_line["nrg_bal"].unique()],
).add_params(nearest)

# Put the five layers into a chart and bind the data
line_chart = alt.layer(
    band, line, points, rules
).properties(
    width=800, height=300
).resolve_scale(
    y="shared"
)
line_chart

##########################################################################################################
selected_multi_state = st.multiselect(
    "Seleziona uno stato",
    df_combined.select("state").unique().sort("state"),
    default=country_default
)
# for country in selected_multi_state:
#     stati_sel = df_combined.filter(pl.col("state") == country)#.filter(pl.col("date") > pl.datetime(2010, 1, 1))
#     stati_2 = pl.concat([stati_2, stati_sel])

stati_bar = df_combined.filter(
    pl.col("state").is_in(selected_multi_state),
    pl.col("date") == year,
    pl.col("nrg_bal") != "FC"
    )

bars = alt.Chart(stati_bar).mark_bar().encode(
    x=alt.X('energy_cons:Q').stack("normalize"),#.stack('zero'), 
    y=alt.Y('state:N'),
    color=alt.Color('nrg_bal:N')
)

text = alt.Chart(stati_bar).mark_text(dx=-15, dy=3, color='white').encode(
    x=alt.X('energy_cons:Q').stack('normalize'),
    y=alt.Y('state:N'),
    detail='nrg_bal:N',
    text=alt.Text('energy_cons:Q', format='.1f')
)

bar_plot = alt.layer(
    bars#, text
).properties(
    width=800, height=300
)
bar_plot

##########################################################################################################

#source = df_momentaneo.group_by([pl.group_by(key="date", freq="6MS"),"symbol"]).mean().reset_index()
# source = df_momentaneo.group_by("date").agg(
#     pl.mean("energy_cons").alias("mean_energy_cons")
# )

# country_default = df_momentaneo.filter(pl.col("nrg_bal") == "FC").group_by("state").agg(
#     pl.sum("energy_cons").alias("total_energy")
# ).sort("total_energy", descending=True).head(6).select("state").to_series()
stati_rank = pl.DataFrame()
for time in df_combined["date"].unique():
    stati_sel = df_combined.filter(
        pl.col("date") == time,
        pl.col("state") != "EU27_2020",
        pl.col("nrg_bal") == "FC").sort("energy_cons", descending=True).head(5)
    stati_rank = pl.concat([stati_rank, stati_sel])

rules = alt.Chart(stati_rank).transform_pivot(
    "rank:O",
    value="rank",
    groupby=["date"]
).mark_rule(color="gray").encode(
    x="date",
    opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
    tooltip=[alt.Tooltip(c, type="ordinal") for c in stati_sel],
).add_params(nearest)

ranking_plot = alt.Chart(stati_rank).mark_line(point=True).encode(
    x=alt.X("date").timeUnit("year").title("date"),
    y="rank:O",
    color=alt.Color("state:N")
).transform_window(
    rank="rank()",
    sort=[alt.SortField("energy_cons", order="descending")],
    groupby=["date"]
).properties(
    title="Bump Chart for Stock Prices",
    width=800,
    height=300,
)
ranking_plot


