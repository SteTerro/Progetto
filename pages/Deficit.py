import streamlit as st
import polars as pl
import altair as alt
import country_converter as cc
import numpy as np

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS

url_consumption = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_cb_e?format=TSV&compressed=true"
url_productivity = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_cb_pem?format=TSV&compressed=true"

st.write(f"""
# Consumo di energia in Europa""")

@st.cache_data
def get_data_consumption(url):
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
            .str.to_date(format='%Y', strict=True),
            energy_cons=pl.col("energy_cons")
            .str.replace(" ", "")
            .str.replace("p", "")
            .str.replace("u", "")
            .str.replace("e", "")
            .str.replace("n", "")
            .str.replace("d", "")
            .str.replace(":", "123456789")
            .cast(pl.Float64),
            unique_id=pl.col("state")+";"+pl.col("nrg_bal"),
            state = pl.col("state")
            .str.replace("EL", "GR")
        )
        .filter(
            pl.col("energy_cons").is_not_null(),
            pl.col("energy_cons") != 123456789,
            pl.col("state")!="EA20",
            pl.col("nrg_bal").is_in(["FC", "FC_IND_E" , "FC_TRA_E", "FC_OTH_CP_E", "FC_OTH_HH_E", "FC_OTH_AF_E"]),
        )
        .drop("freq",
              "unit")
        .sort("state", "date") 
    )
    return data

@st.cache_data
def get_data_productivity(url):
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
            pl.col("unit") == "GWH",
            pl.col("siec").is_in(["TOTAL","X9900","RA000","N9000","CF","CF_R","RA100","RA200","RA300","RA400","RA500_5160","C0000","CF_NR","G3000","O4000XBIO"]),
            )
        .drop("freq",
              "unit")
        .sort("state", "date") 
    )
    return data

df_prod = get_data_productivity(url_productivity)
df_cons = get_data_consumption(url_consumption)

# df_prod_A = (
#     df_prod.with_columns(
#         year=pl.col("date").dt.year())
#     .group_by(["state", "year", "unique_id", "siec"])
#     .agg(energy_prod=pl.sum("energy_prod"))
#     .sort(["state", "year"])
# ).with_columns(date=pl.col("year").cast(pl.Utf8).str.to_date(format='%Y')).drop("year")

cc_2 = cc.CountryConverter()
country_list = pl.from_pandas(cc_2.EU27as('ISO2'))
country_list = country_list.select(pl.col("ISO2")).to_series()

list_consuption = ["FC", "FC_IND_E" , "FC_TRA_E", "FC_OTH_CP_E", "FC_OTH_HH_E", "FC_OTH_AF_E"]
list_productivity = ["TOTAL","X9900","RA000","N9000","CF","CF_R","RA100","RA200","RA300","RA400","RA500_5160","C0000","CF_NR","G3000","O4000XBIO"]
Tot_list = ["TOTAL","X9900","RA000","N9000","CF"]
RA_list = ["CF_R","RA100","RA200","RA300","RA400","RA500_5160"]
CF_list = ["C0000","CF_NR","G3000","O4000XBIO"]

# df_comb = df_prod_A.join(df_cons, on = ["state", "date"], how = "inner")

# EU27_2020_mom = df_comb.filter(pl.col("state") == "EU27_2020")
# df = df_comb.filter(pl.col("state").is_in(country_list))
# df = pl.concat([df, EU27_2020_mom])

EU27_2020_filter = df_prod.filter(pl.col("state") == "EU27_2020")
df_prod = df_prod.filter(pl.col("state").is_in(country_list))
df_prod = pl.concat([df_prod, EU27_2020_filter])

EU27_2020_filter = df_cons.filter(pl.col("state") == "EU27_2020")
df_cons = df_cons.filter(pl.col("state").is_in(country_list))
df_cons = pl.concat([df_cons, EU27_2020_filter])

@st.cache_data
def Arima_prod(state):
    ts = df_prod.filter(
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
        fallback_model=AutoETS(season_length=12)
        )

    ts_pred = sf.forecast(df=ts, h=48, level=[95]) 
    ts_pred = ts_pred\
        .rename({"AutoARIMA": "energy_prod"})\
        .rename({"AutoARIMA-lo-95": "AutoARIMA_low90"})\
        .rename({"AutoARIMA-hi-95": "AutoARIMA_hi90"})
    
    return ts_pred

@st.cache_data
def pred_siec(filtro):
    df_funz = df_prod.filter(pl.col("siec") == filtro).with_columns(
        AutoARIMA_low90 = pl.lit(0),
        AutoARIMA_hi90 = pl.lit(0), 
        predicted = pl.lit(False)
        ).sort("AutoARIMA_hi90", "AutoARIMA_low90", "date", "energy_prod", "predicted", "siec", "state","unique_id")  
    
    countries = df_funz.select("unique_id").unique().sort("unique_id").to_series()
    pred = pl.DataFrame()
    for state in countries:
        state_pred = Arima_prod(state)
        pred = pl.concat([pred, state_pred])

    pred = pred.with_columns(
        predicted = pl.lit(True),
        ).with_columns(
            pl.col("unique_id")
                .str.split(";")
                .list.to_struct(fields=["state", "siec"])
                .alias("combined_info"),
            date = pl.col("ds")
        ).unnest("combined_info").drop("ds")

    df_funz = df_funz.select(sorted(df_funz.columns))
    pred = pred.select(sorted(pred.columns))

    df_combined = pl.concat([df_funz, pred], how= "vertical_relaxed")
    return df_combined

@st.cache_data
def Arima_cons(state):
    ts = df_cons.filter(
        pl.col("unique_id") == state
        ).select(
            pl.col("date").alias("ds"),
            pl.col("energy_cons").alias("y"),
            pl.col("unique_id")
            )

    sf = StatsForecast(
        models = [AutoARIMA(season_length = 6 )],#season_length = 6
        freq = '2mo',
        n_jobs=-1,
        fallback_model=AutoETS(season_length=6),
        )

    ts_upscale = (
        ts.upsample(time_column="ds", every="2mo")
        .interpolate()
        .fill_null(strategy="forward")
    )
    
    ts_pred = sf.forecast(df=ts_upscale, h=24, level=[95])
    ts_pred = ts_pred\
        .rename({"AutoARIMA": "energy_cons"})\
        .rename({"AutoARIMA-lo-95": "AutoARIMA_low90"})\
        .rename({"AutoARIMA-hi-95": "AutoARIMA_hi90"})
    return ts_pred 

@st.cache_data
def pred_cons(filtro):
    df_funz = df_cons.filter(pl.col("nrg_bal") == filtro).with_columns(
        AutoARIMA_low90 = pl.lit(0),
        AutoARIMA_hi90 = pl.lit(0), 
        predicted = pl.lit(False)
        ).sort("AutoARIMA_hi90", "AutoARIMA_low90", "date","energy_cons","nrg_bal","predicted", "state","unique_id")  
    
    countries = df_funz.select("unique_id").unique().sort("unique_id").to_series()
    pred = pl.DataFrame()
    for state in countries:
        pred_state = Arima_cons(state)
        pred = pl.concat([pred, pred_state])

    pred = pred.with_columns(
        predicted = pl.lit(True),
        ).with_columns(
            pl.col("unique_id")
                .str.split(";")
                .list.to_struct(fields=["state", "nrg_bal"])
                .alias("combined_info"),
            date = pl.col("ds")
        ).unnest("combined_info").drop("ds")

    df_funz = df_funz.select(sorted(df_funz.columns)).drop("siec")
    pred = pred.select(sorted(pred.columns))
    df_combined = pl.concat([df_funz, pred], how= "vertical_relaxed")

    return df_combined

# df_A = df_A.filter(pl.col("def_id") == "TOTAL;FC").drop("energy_prod","siec","nrg_bal","energy_cons","unique_id")

df_prod_pred = pred_siec("TOTAL")
df_cons_pred = pred_cons("FC")

df_prod_pred_A = (
    df_prod_pred.with_columns(
        year=pl.col("date").dt.year())
    .group_by(["state", "year", "unique_id", "siec"])
    .agg(energy_prod=pl.sum("energy_prod"))
    .sort(["state", "year"])
).with_columns(date=pl.col("year").cast(pl.Utf8).str.to_date(format='%Y')).drop("year")

df_cons_pred_A = (
    df_cons_pred.with_columns(
        year=pl.col("date").dt.year())
    .group_by(["state", "year", "unique_id", "nrg_bal"])
    .agg(energy_cons=pl.mean("energy_cons"))
    .sort(["state", "year"])
).with_columns(date=pl.col("year").cast(pl.Utf8).str.to_date(format='%Y')).drop("year")

df_comb = df_prod_pred_A.join(df_cons_pred_A, on = ["state", "date"], how = "inner")

# df_comb = df_prod_pred_A.join(df_cons, on = ["state", "date"], how = "inner")

df_A = df_comb.with_columns(
    deficit = 
    (pl.when(pl.col("siec") == "TOTAL")
     .then(pl.col("energy_prod"))
     .otherwise(pl.lit(0))
     - pl.when(pl.col("nrg_bal") == "FC")
     .then(pl.col("energy_cons"))
     .otherwise(pl.lit(0))),
     def_id=pl.col("siec")+";"+pl.col("nrg_bal")
)
country = st.selectbox(
        "Seleziona uno stato",
        df_A.select("state").unique().sort("state")
        )
###############################################################################################################
st.write(f"""
    ### Confronto del consumo di energia di vari stati europei.
 """)

def mappa(df_input):
    if df_input.columns == df_prod_pred_A.columns:
        x = "energy_prod"
        type = "siec"
    if df_input.columns == df_cons_pred_A.columns:
        x = "energy_cons"
        type = "nrg_bal"
    if df_input.columns == df_A.columns:
        x = "deficit"
        type = "def_id"

    year = st.select_slider(
                "Seleziona un anno",
                df_input["date"].unique(),
            )

    stati_map = df_input.filter(
        pl.col("date") == year,
        pl.col("state") != "EU27_2020"
        )

    converted_countries = cc.convert(names=stati_map["state"], to='ISOnumeric')

    selected_bal = st.selectbox(
            "Seleziona un tipo di consumo di energia",
            df_input[type].unique().sort(),
        )

    stati_map = stati_map.with_columns(
        pl.Series("state", converted_countries).alias("ISO")
    ).filter(pl.col(type) == selected_bal).sort(x, "state")

    countries_map = alt.topo_feature(f"https://cdn.jsdelivr.net/npm/world-atlas@2/countries-50m.json", 'countries')
    #countries_map = alt.topo_feature(f"https://cdn.jsdelivr.net/npm/world-atlas@2/countries-10m.json", 'countries')

    min_value = stati_map[x].min()
    max_value = stati_map[x].max()

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
        from_=alt.LookupData(source, 'ISO_str', ['state', x]),
    ).encode(
        color=alt.Color(x+":Q", sort="descending", scale=alt.Scale(
            scheme='inferno', domain=(min_value,max_value)), legend=alt.Legend(title="", tickCount=6)),
        tooltip=['state:N',x+":Q"]
    )

    background + map
###############################################################################################################
st.write(f"""
    ### Consumo di energia nei vari settori per un singolo stato europeo.
 """)

for consuption in list_consuption:
    df_cons_pred = pl.concat([df_cons_pred, pred_cons(consuption)])

def select_state():
    state = st.selectbox(
        "Seleziona uno stato",
        df_cons["state"].unique().sort(),
        index=12
    )
    return state

selected_single_state = select_state()
st.write(df_cons_pred)
stati_line = df_cons_pred.filter(
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

###############################################################################################################
predicate = alt.datum.deficit > 0
color = alt.when(predicate).then(alt.value("green")).otherwise(alt.value("red"))
select = alt.selection_point(name="select", on="click")
highlight = alt.selection_point(name="highlight", on="pointerover", empty=False)

stroke_width = (
    alt.when(select).then(alt.value(2, empty=False))
    .when(highlight).then(alt.value(1))
    .otherwise(alt.value(0))
)

base = alt.Chart(df_A.filter(pl.col("date") == pl.datetime(2018,1,1), pl.col("def_id") == "TOTAL;FC")
).mark_bar(stroke="black", cursor="pointer"
).encode(
    y=alt.Y("state", sort="-x"),
    x="deficit:Q",
    color=color,
    fillOpacity=alt.when(select).then(alt.value(1)).otherwise(alt.value(0.3)),
    strokeWidth=stroke_width,
).add_params(select, highlight).properties(width=600)

text_conditioned = base.mark_text(
    align="left",
    baseline="middle",
    dx=alt.expr(alt.expr.if_(alt.datum.deficit >= 0, 5, -55))
).encode(text="deficit:Q")

prova = alt.layer(base, text_conditioned).configure_scale(bandPaddingInner=0.2)
prova


















# source = df_A.filter(pl.col("state") == country, pl.col("def_id") == "TOTAL;FC").sort("date").with_columns(
#     amount = pl.col("deficit"),#pl.col("deficit") - pl.col("deficit").shift(1),
#     label = pl.col("date")
# )
# st.write(source)
# amount = alt.datum.amount
# label = alt.datum.label
# window_lead_label = alt.datum.window_lead_label
# window_sum_amount = alt.datum.window_sum_amount

# # Define frequently referenced/long expressions
# calc_prev_sum = alt.expr.if_(label == "End", 0, window_sum_amount - amount)
# calc_amount = alt.expr.if_(label == "End", window_sum_amount, amount)
# calc_text_amount = (
#     alt.expr.if_((label != "Begin") & (label != "End") & calc_amount > 0, "+", "")
#     + calc_amount
# )

# # The "base_chart" defines the transform_window, transform_calculate, and X axis
# base_chart = alt.Chart(source).transform_window(
#     window_sum_amount="sum(amount)",
#     window_lead_label="lead(label)",
# ).transform_calculate(
#     calc_lead=alt.expr.if_((window_lead_label == None), label, window_lead_label),
#     calc_prev_sum=calc_prev_sum,
#     calc_amount=calc_amount,
#     calc_text_amount=calc_text_amount,
#     calc_center=(window_sum_amount + calc_prev_sum) / 2,
#     calc_sum_dec=alt.expr.if_(window_sum_amount < calc_prev_sum, window_sum_amount, ""),
#     calc_sum_inc=alt.expr.if_(window_sum_amount > calc_prev_sum, window_sum_amount, ""),
# ).encode(
#     x=alt.X("label:O", axis=alt.Axis(title="Months", labelAngle=0), sort=None)
# )

# color_coding = (
#     alt.when((label == "Begin") | (label == "End"))
#     .then(alt.value("#878d96"))
#     .when(calc_amount < 0)
#     .then(alt.value("#24a148"))
#     .otherwise(alt.value("#fa4d56"))
# )

# bar = base_chart.mark_bar(size=45).encode(
#     y=alt.Y("calc_prev_sum:Q", title="Amount"),
#     y2=alt.Y2("window_sum_amount:Q"),
#     color=color_coding,
# )

# # The "rule" chart is for the horizontal lines that connect the bars
# rule = base_chart.mark_rule(xOffset=-22.5, x2Offset=22.5).encode(
#     y="window_sum_amount:Q",
#     x2="calc_lead",
# )

# # Add values as text
# text_pos_values_top_of_bar = base_chart.mark_text(baseline="bottom", dy=-4).encode(
#     text=alt.Text("calc_sum_inc:N"),
#     y="calc_sum_inc:Q",
# )
# text_neg_values_bot_of_bar = base_chart.mark_text(baseline="top", dy=4).encode(
#     text=alt.Text("calc_sum_dec:N"),
#     y="calc_sum_dec:Q",
# )
# text_bar_values_mid_of_bar = base_chart.mark_text(baseline="middle").encode(
#     text=alt.Text("calc_text_amount:N"),
#     y="calc_center:Q",
#     color=alt.value("white"),
# )

# prova = alt.layer(
#     bar,
#     rule,
#     text_pos_values_top_of_bar,
#     text_neg_values_bot_of_bar,
#     text_bar_values_mid_of_bar
# ).properties(
#     width=800,
#     height=450
# )
# prova