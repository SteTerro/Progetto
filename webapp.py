import streamlit as st
import polars as pl
import altair as alt
import country_converter as cc
import datetime as dt

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS

url_consumption = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_cb_e?format=TSV&compressed=true"
url_productivity = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_cb_pem?format=TSV&compressed=true"

st.write(f"""
# Consumo di energia in Europa""")

## Funzioni lettura DataFrame #################################################################################
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
            .str.strptime(pl.Date, '%Y'),#, strict=False).cast(pl.Date),
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
            .str.to_date(format='%Y-%m', strict=False),#, strict=True),
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

def EU_filter(df_input):
    cc_2 = cc.CountryConverter()
    country_list = pl.from_pandas(cc_2.EU27as('ISO2'))
    country_list = country_list.select(pl.col("ISO2")).to_series()

    EU27_2020_filter = df_input.filter(pl.col("state") == "EU27_2020")
    df_input = df_input.filter(pl.col("state").is_in(country_list))
    df_input = pl.concat([df_input, EU27_2020_filter])
    return df_input
## Utils ######################################################################################################
def select_state():
    state = st.selectbox(
        "Seleziona uno stato",
        df_prod["state"].unique().sort(),
    )
    return state

def select_multi_state(df_input):
    selected_multi_state = st.multiselect(
        "Seleziona uno stato",
        df_input.select("state").unique().sort("state"),
        default=top4_EU
    )
    return selected_multi_state

def select_type(df_input):
    if df_input.columns == df_prod.columns:
        type = st.selectbox(
            "Seleziona un tipo di energia",
            df_input["siec"].unique().sort(),
            index=13
        )
    elif df_input.columns == df_cons.columns:
        type = st.selectbox(
            "Seleziona un tipo di consumo di energia",
            df_input["nrg_bal"].unique().sort(),
            index=0
        )
    return type

def select_country():
    country = st.selectbox(
        "Seleziona uno stato",
        df_comb.select("state").unique().sort("state"),
        default="IT"
        )
    return country

def select_year(df_input):
    first_year = df_input["date"].min().year
    last_year = df_input["date"].max().year
    year = st.select_slider(
                "Seleziona un anno",
                range(first_year, last_year)
            )
    year = pl.datetime(year, 1, 1)
    return year

def get_first_5_countries(df_input, filter):
    if df_input.columns == df_prod.columns:
        x = "energy_prod"
        type = "siec"
    elif df_input.columns == df_cons.columns:
        x = "energy_cons"
        type = "nrg_bal"
    if df_input.columns == df_prod_pred.columns:
        x = "energy_prod"
        type = "siec"
    elif df_input.columns == df_cons_pred_A.columns:
        x = "energy_cons"
        type = "nrg_bal"
    elif df_input.columns == df_A.columns:
        x = "deficit"
        type = "def_id"

    country_default = df_input.filter(pl.col(type) == filter).group_by("state").agg(
        pl.sum(x).alias("total_energy")
    ).sort("total_energy", descending=True).head(5).select("state").to_series()
    return country_default

def df_from_M_to_A(df_input):
    if df_input.columns == df_prod_pred.columns:
        x = "energy_prod"
        type = "siec"
        agg = energy_prod=pl.sum("energy_prod")
    elif df_input.columns == df_cons_pred.columns:
        x = "energy_cons"
        type = "nrg_bal"
        agg = energy_cons=pl.mean("energy_cons")
    
    df_A = (
        df_input.with_columns(
            year=pl.col("date").dt.year())
        .group_by(["state", "year", "unique_id", type] + ["predicted"]) #[col for col in df_input.columns if col not in ["state", "year", "unique_id", type, "date", x]])
        .agg(agg)
        .sort(["state", "year"])
    ).with_columns(
        date=pl.col("year").cast(pl.Utf8).str.to_date(format='%Y'),
        ).drop("year")#.rename({"col": x})
    if x == "energy_prod":
        df_A = df_A.with_columns(
            energy_prod=pl.sum("energy_prod").over("unique_id", "date")
        )
    elif x == "energy_cons":
        df_A = df_A.with_columns(
            energy_cons=pl.mean("energy_cons").over("unique_id", "date")
        )
    return df_A

## Predizione ##################################################################################################
@st.cache_data
def Arima_prod(state):
    ts = df_prod.filter(
        pl.col("unique_id") == state
        ).select(
            pl.col("date").alias("ds"),
            pl.col("energy_prod").alias("y"),
            pl.col("unique_id")
            )
    if len(ts) == 0:
        st.warning("Non ci sono dati per questo stato")
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
def pred_siec(filtro, state = None):
    if state is None:
        x = "siec"
    else:
        x = "state"
    df_funz = df_prod.filter(pl.col(x) == filtro).with_columns(
        AutoARIMA_low90 = pl.lit(0),
        AutoARIMA_hi90 = pl.lit(0), 
        predicted = pl.lit(False)
        ).sort("AutoARIMA_hi90", "AutoARIMA_low90", "date", "energy_prod", "predicted", "siec", "state","unique_id")  
    if state is None:
        countries = df_funz.select("unique_id").unique().sort("unique_id").to_series()
    else:
        countries = [state + ";" + filtro]
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
def pred_state(filtro):
    for state in filtro:
        df_funz = df_prod.filter(pl.col("state") == filtro).with_columns(
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
def pred_cons(filtro, state = None):
    df_funz = df_cons.filter(pl.col("nrg_bal") == filtro).with_columns(
        AutoARIMA_low90 = pl.lit(0),
        AutoARIMA_hi90 = pl.lit(0), 
        predicted = pl.lit(False)
        ).sort("AutoARIMA_hi90", "AutoARIMA_low90", "date","energy_cons","nrg_bal","predicted", "state","unique_id")  
    if state is None:
        countries = df_funz.select("unique_id").unique().sort("unique_id").to_series()
    else:
        countries = [state + ";" + filtro]
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
## Creazione Dataframe ########################################################################################
df_prod = get_data_productivity(url_productivity)
df_cons = get_data_consumption(url_consumption)

top4_EU = ["DE", "FR", "IT", "ES"]
list_consuption = ["FC", "FC_IND_E" , "FC_TRA_E", "FC_OTH_CP_E", "FC_OTH_HH_E", "FC_OTH_AF_E"]
list_productivity = ["TOTAL","X9900","RA000","N9000","CF","CF_R","RA100","RA200","RA300","RA400","RA500_5160","C0000","CF_NR","G3000","O4000XBIO"]
prod_list = ["X9900","RA000","N9000","CF"]
RA_list = ["CF_R","RA100","RA200","RA300","RA400","RA500_5160"]
CF_list = ["C0000","CF_NR","G3000","O4000XBIO"]

df_prod = EU_filter(df_prod)
df_cons = EU_filter(df_cons)

## Mappa ######################################################################################################
def mappa(df_input, year, selected_bal):
    stati_map = df_input.filter(
        pl.col("date") == year,
        pl.col("state") != "EU27_2020"
        )

    if df_input.columns == df_prod_pred_A.columns:
        x = "energy_prod"
        type = "siec"
    if df_input.columns == df_cons_pred_A.columns:
        x = "energy_cons"
        type = "nrg_bal"
    if df_input.columns == df_A.columns:
        x = "deficit"
        type = "def_id"
        
    converted_countries = cc.convert(names=stati_map["state"], to='ISOnumeric')

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
        width=800, height=400
    ).encode(tooltip=alt.value(None))

    map = alt.Chart(countries_map).mark_geoshape(
        stroke='black'
    ).project(
        type= 'mercator',
        scale= 350,                          # Magnify
        center= [20,50],                     # [lon, lat]
        clipExtent= [[000, 000], [800, 400]],    # [[left, top], [right, bottom]]
    ).properties(
        width=800, height=400
    ).transform_lookup(
        lookup='id',
        from_=alt.LookupData(source, 'ISO_str', ['state', x]),
    ).encode(
        color=alt.Color(x+":Q", sort="descending", scale=alt.Scale(
            scheme='inferno', domain=(min_value,max_value)), legend=alt.Legend(title="", tickCount=6)),
        tooltip=['state:N',x+":Q"]
    )

    background + map

## Line Chart #################################################################################################
## Utils per Line Chart #######################################################################################



## Line Chart con Doppio Asse Y ###############################################################################
def line_chart_prod(df_input, countries, siec):
    df_input = df_input.with_columns(
        # predicted=pl.col("predicted").fill_null(False),
        #predicted = pl.when(pl.col("predicted") == True).then(pl.lit(1)).otherwise(pl.lit(0))
    )
    stati_line = df_input.filter(pl.col("state").is_in(countries)).filter(
        pl.col("siec") == siec,
        pl.col("state") != "EU27_2020",
    )
    
    nearest = alt.selection_point(nearest=True, on="pointerover",
                                fields=["date"], empty=False)

    # highlight = alt.selection_point(on='pointerover', fields=['energy_cons'], nearest=True)
    # The basic line
    line = alt.Chart(stati_line).mark_line(interpolate="basis").encode(
        x="date",
        y="energy_prod:Q",
        color="state",
        strokeDash="predicted:N"
    )
    when_near = alt.when(nearest)

    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
    )

    Europe_mean = df_input.filter(pl.col("state")== "EU27_2020"
        ).filter(
            pl.col("siec") == siec,
        ).with_columns(
            y2=pl.col("energy_prod")/27 # - min_y
    )
    min_value_EU = Europe_mean['y2'].min() - 0.3*Europe_mean['y2'].min()
    max_value_EU = Europe_mean['y2'].max() + 0.3*Europe_mean['y2'].max()
    
    line_EU = alt.Chart(Europe_mean).mark_line(
        opacity=0.2, color='#lightgrey'#, interpolate='monotone', strokeDash=[2,2]
        ).encode(
            alt.X('date').title(None),
            alt.Y('y2', scale=alt.Scale(domain=[min_value_EU, max_value_EU])).axis(title='Produzione Media di Energia in Europa'),
            strokeDash="predicted:N"
    )
    
    # Draw a rule at the location of the selection
    rules = alt.Chart(stati_line).transform_pivot(
        "state",
        value="energy_prod",
        groupby=["date"]
    ).mark_rule(color="gray").encode(
        x="date",
        opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
        tooltip=[alt.Tooltip(c, type="quantitative") for c in stati_line["state"].unique()],
    ).add_params(nearest)

    source2 = [
        {"start": "2024-10", "end": "2029"},
    ]
    source2_df = pl.DataFrame(source2)

    xrule = (
        alt.Chart(source2_df)
        .mark_rule(color="grey", strokeDash=[12, 6], size=2, opacity=0.4)
        .encode(x="start:T")
    )

    text_left = alt.Chart(source2_df).mark_text(
        align="left", dx=-55, dy=-165, color="grey"
    ).encode(
        x="start:T",
        text=alt.value("Valori Reali")
    )

    text_right = alt.Chart(source2_df).mark_text(
        align="left", dx=5, dy=-165, color="grey"
    ).encode(
        x="start:T",
        text=alt.value("Forecast")
    )
    
    rect = alt.Chart(source2_df).mark_rect(color="lightgrey", opacity=0.4).encode(
        x="start:T",
        x2="end:T",
    )
    last_price = alt.Chart(stati_line.filter(pl.col("predicted")==True)).mark_circle().encode(
        alt.X("last_date['date']:T"),
        alt.Y("last_date['energy_prod']:Q").axis(title='Produzione di Energia'),
        color="state:N"
    ).transform_aggregate(
        last_date="argmax(date)",
        groupby=["state"]
    )
    company_name = last_price.mark_text(align="left", dx=4).encode(text="state", color="state:N")
    
    first_layer = alt.layer(
        rect, xrule, line, points, text_left, text_right,rules, last_price, company_name
    ).properties(
        width=800, height=400)

    line_chart = alt.layer(
        line_EU , first_layer
    ).resolve_scale(
        y="independent"
    )
    line_chart 

## Line Chart con Banda dell'intervallo #######################################################################
def line_chart_with_IC(df_cons_pred, selected_single_state):
    for consuption in list_consuption:
        df_cons_pred = pl.concat([df_cons_pred, pred_cons(consuption, selected_single_state)])
    df_cons_pred = cast_int_cons(df_cons_pred)

    stati_line = df_cons_pred.filter(
        pl.col("state") == selected_single_state,
        pl.col("nrg_bal") != "FC",
        pl.col("date") > pl.datetime(2010, 1, 1)
        )

    nearest = alt.selection_point(nearest=True, on="pointerover",
                                fields=["date"], empty=False)

    # The basic line
    line = alt.Chart(stati_line).mark_line(interpolate="basis").encode(
        x="date",
        y="energy_cons:Q",
        color="nrg_bal:N",
        strokeDash="predicted:N"

    )
    when_near = alt.when(nearest)

    conf_int = stati_line.filter(pl.col("predicted") == True)
    band = alt.Chart(conf_int).mark_errorband(extent='ci').encode(
        x="date",
        y=alt.Y("AutoARIMA_low90:Q", title=None),
        y2="AutoARIMA_hi90:Q",
        color="nrg_bal:N"
    )

    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
    )
    source2 = [
        {"start": "2023", "end": "2027"},
    ]
    source2_df = pl.DataFrame(source2)

    text_left = alt.Chart(source2_df).mark_text(
        align="left", dx=-55, dy=-145, color="grey"
    ).encode(
        x="start:T",
        text=alt.value("Valori Reali")
    )

    text_right = alt.Chart(source2_df).mark_text(
        align="left", dx=5, dy=-145, color="grey"
    ).encode(
        x="start:T",
        text=alt.value("Forecast")
    )

    last_price = alt.Chart(stati_line.filter(pl.col("predicted")==True)).mark_circle().encode(
        alt.X("last_date['date']:T"),
        alt.Y("last_date['energy_cons']:Q"),
        color="nrg_bal:N"
    ).transform_aggregate(
        last_date="argmax(date)",
        groupby=["nrg_bal"],
    )
    company_name = last_price.mark_text(align="left", dx=4).encode(text="nrg_bal", color="nrg_bal:N")

    xrule = (
        alt.Chart(source2_df)
        .mark_rule(color="grey", strokeDash=[12, 6], size=2, opacity=0.4)
        #.encode(x=alt.datum(alt.DateTime(year=2023)))
        .encode(x="start:T")
    )
    
    rect = alt.Chart(source2_df).mark_rect(color="lightgrey", opacity=0.4).encode(
        x="start:T",
        x2="end:T",
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
        xrule, rect,text_right, text_left, band, line, points, rules, last_price, company_name
    ).properties(
        width=800, height=400
    ).resolve_scale(
        y="shared"
    )
    line_chart

## line_chart Deficit ############################################################################################
def line_chart_deficit(df_input):
    # per etichetta nome: line + last_price + company_name
    # per linea verticale: x rules
    # per quadrato grigio: rect
    selected_multi_state = select_multi_state(df_input)
    
    base = alt.Chart(df_input.filter(pl.col("state").is_in(selected_multi_state))#, pl.col("date")<=pl.datetime(2023,1,1))
    ).encode(
        alt.Color("state").legend(None)
    ).transform_filter(
        "datum.state !== 'EU27_2020'"
    ).properties(
        width=800, height=400
    )
    line = base.mark_line().encode(
        x="date",
        y="deficit",
        strokeDash="predicted:N"
        )

    last_price = base.mark_circle().encode(
        alt.X("last_date['date']:T"),
        alt.Y("last_date['deficit']:Q")
    ).transform_aggregate(
        last_date="argmax(date)",
        groupby=["state"]
    )
    source2 = [
        {"start": "2022", "end": "2027"},
    ]
    source2_df = pl.DataFrame(source2)

    text_left = alt.Chart(source2_df).mark_text(
        align="left", dx=-55, dy=-145, color="grey"
    ).encode(
        x="start:T",
        text=alt.value("Valori Reali")
    )

    text_right = alt.Chart(source2_df).mark_text(
        align="left", dx=5, dy=-145, color="grey"
    ).encode(
        x="start:T",
        text=alt.value("Forecast")
    )

    xrule = (
        alt.Chart(source2_df)
        .mark_rule(color="grey", strokeDash=[12, 6], size=2, opacity=0.4)
        #.encode(x=alt.datum(alt.DateTime(year=2023)))
        .encode(x="start:T")
    )
    
    rect = alt.Chart(source2_df).mark_rect(color="lightgrey", opacity=0.4).encode(
        x="start:T",
        x2="end:T",
    )

    company_name = last_price.mark_text(align="left", dx=4).encode(text="state")

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    nearest = alt.selection_point(nearest=True, on="pointerover",
                              fields=["date"], empty=False)

    selectors = alt.Chart(df_input).mark_point().encode(
        x="date:T",
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
    text = line.mark_text(align="left", dx=5, dy=-10).encode(
        text=when_near.then("deficit:Q").otherwise(alt.value(" "))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(df_input).mark_rule(color="lightgray", opacity=0.7).encode(
        x="date:T",
    ).transform_filter(
        nearest
    )

    chart = alt.layer( rect ,text_left, text_right, xrule , line,  last_price , selectors , points ,rules, text , company_name).encode(
        x=alt.X().title("date"),
        y=alt.Y().title("deficit")
    ).properties(
        width=800, height=400
    )
    # ).resolve_scale(
    #     x="shared"
    # )

    chart# + line_2

## Area Chart ##################################################################################################
def area_chart(df_prod_pred, selected_single_state, prod_list):
    df_prod_pred = pl.concat([df_prod_pred, pred_state(selected_single_state)], how="vertical_relaxed")
    df_prod_pred = cast_int_prod(df_prod_pred)

    stati_line = df_prod_pred.filter(
        pl.col("state") == selected_single_state,
        pl.col("siec").is_in(prod_list),
        pl.col("date") > pl.datetime(2017, 1, 1))

    nearest = alt.selection_point(nearest=True, on="pointerover",
                              fields=["date"], empty=False)

    selectors = alt.Chart(stati_line).mark_point().encode(
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
    
    area = alt.Chart(stati_line
        ).mark_area(
            opacity=0.5,
            interpolate='step-after',
            line=True
        ).encode(
            x="date:T",
            y=alt.Y("energy_prod:Q").stack(True),
            color=alt.Color("siec:N", scale = color_palette)
    )
    last_price = alt.Chart(stati_line.filter(pl.col("predicted")==True)).mark_circle().encode(
        alt.X("last_date['date']:T"),
        alt.Y("last_date['energy_prod']:Q").axis(title='Produzione di Energia'),
        color="state:N"
    ).transform_aggregate(
        last_date="argmax(date)",
        groupby=["state"]
    )
    company_name = last_price.mark_text(align="left", dx=60).encode(text="state", color="state:N")

    # Draw points on the line, and highlight based on selection
    points = area.mark_point().encode(
        opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    text = area.mark_text(align="left", dx=5, dy=-5).encode(
        text=when_near.then("energy_prod:Q").otherwise(alt.value(" "))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(stati_line).mark_rule(color="gray").encode(
        x="date",
    ).transform_filter(
        nearest
    )

    area_chart = alt.layer(
        area, last_price, company_name, selectors, points, rules, text
    ).properties(
        width=800, height=400
    ).resolve_scale(
        y="shared"
    )
    area_chart

## Barchart consumption ########################################################################################
def barchart_cons(df_input, df_cons_pred, year, list_consuption):
    selected_multi_state = select_multi_state(df_input)
    for state in selected_multi_state:
        for consuption in list_consuption:
            df_cons_pred = pl.concat([df_cons_pred, pred_cons(consuption, state)], how="vertical_relaxed")
    df_input = df_from_M_to_A(df_cons_pred)
    df_input = cast_int_cons(df_input)

    stati_bar = df_input.filter(
        pl.col("state").is_in(selected_multi_state),
        pl.col("date") == year,
        # pl.col("nrg_bal") != "FC"
        )

    stati_bar = stati_bar.with_columns(
        percentage = (pl.col("energy_cons") / pl.col("energy_cons").filter(pl.col("nrg_bal") == "FC").sum()) * 100
    ).filter(pl.col("nrg_bal") != "FC")

    bars = alt.Chart(stati_bar).mark_bar().encode(
        x=alt.X('energy_cons:Q').stack("normalize"),#.stack('zero'), 
        y=alt.Y('state:N'),
        color=alt.Color('nrg_bal:N')
    )
    
    text = alt.Chart(stati_bar).mark_text(dx=-6, dy=3, color='white').encode(
        x=alt.X('energy_cons:Q').stack('normalize'),
        y=alt.Y('state:N'),
        detail='nrg_bal:N',
        text=alt.Text('percentage:Q', format='.1%')
    )

    bar_plot = alt.layer(
        bars, text
    ).properties(
        width=800, height=400
    )
    bar_plot

## Barchart Classifica Deficit ################################################################################
def barchart_classifica(df_input, year):
    df_input = df_input.filter(
        pl.col("date") == year,
        pl.col("def_id") == "TOTAL;FC"
        )
    df_EU = df_input.filter(pl.col("state") == "EU27_2020"
        ).with_columns(
            deficit=pl.col("deficit") // 27)
    df_input = df_input.filter(pl.col("state") != "EU27_2020")
    df_input = pl.concat([df_input, df_EU])

    predicate = alt.datum.deficit > 0
    color = alt.when(predicate).then(alt.value("green")).otherwise(alt.value("red"))
    color_EU = alt.when(state = "EU27_2020").then(alt.value("blue")).otherwise(color)
    select = alt.selection_point(name="select", on="click")
    highlight = alt.selection_point(name="highlight", on="pointerover", empty=False)

    stroke_width = (
        alt.when(select).then(alt.value(2, empty=False))
        .when(highlight).then(alt.value(1))
        .otherwise(alt.value(0))
    )

    base = alt.Chart(df_input).mark_bar(
        stroke="black", cursor="pointer"
    ).encode(
        y=alt.Y("state", sort="-x"),
        x=alt.X("deficit:Q").axis(labels=False),
        color=color,
        fillOpacity=alt.when(select).then(alt.value(1)).otherwise(alt.value(0.3)),
        strokeWidth=stroke_width,
    ).add_params(select, highlight).properties(width=800)

    base_2 = alt.Chart(df_EU).mark_bar(
        stroke="black", cursor="pointer"
    ).encode(
        y=alt.Y("state"),
        x=alt.X("deficit:Q").axis(labels=False),
        color=color_EU,
        fillOpacity=alt.when(select).then(alt.value(1)).otherwise(alt.value(0.3)),
        strokeWidth=stroke_width,
    ).add_params(select, highlight).properties(width=800)

    text_conditioned = base.mark_text(
        align="left",
        baseline="middle",
        dx=alt.expr(alt.expr.if_(alt.datum.deficit >= 0, 5, -35))
    ).encode(text="deficit:Q")

    barchart_classifica = alt.layer(base,  text_conditioned).configure_scale(bandPaddingInner=0.2)
    barchart_classifica

## Implementazione Pagine ######################################################################################
def last_date(df_input, state):
    last_date = df_input.filter(pl.col("state")==state).agg(pl.col("date").max().alias("last_date"))
    return last_date
df_prod_pred = pred_siec("TOTAL")
df_cons_pred = pred_cons("FC")

df_prod_pred_A = df_from_M_to_A(df_prod_pred)
df_cons_pred_A = df_from_M_to_A(df_cons_pred)

df_comb = df_prod_pred_A.join(df_cons_pred_A, on = ["state", "date"], how = "inner")
df_A = df_comb.with_columns(
    deficit = 
    (pl.when(pl.col("siec") == "TOTAL")
     .then(pl.col("energy_prod"))
     .otherwise(pl.lit(0))
     - pl.when(pl.col("nrg_bal") == "FC")
     .then(pl.col("energy_cons"))
     .otherwise(pl.lit(0))),
     def_id=pl.col("siec")+";"+pl.col("nrg_bal"),
     predicted = pl.when(pl.col("predicted") | pl.col("predicted_right")).then(pl.lit(True)).otherwise(pl.lit(False))
).drop("predicted_right")

def cast_int_deficit(df_input):
    df_input = df_input.with_columns(
            deficit = pl.col("deficit").cast(pl.Int64)
        )
    return df_input

def cast_int_prod(df_input):
    df_input = df_input.with_columns(
            energy_prod = pl.col("energy_prod").cast(pl.Int64)
        )
    return df_input

def cast_int_cons(df_input):
    df_input = df_input.with_columns(
            energy_cons = pl.col("energy_cons").cast(pl.Int64)
        )
    return df_input

def page_deficit():
    st.title("Pagina Deficit/Surplus")
    df_A = cast_int_deficit(df_A)
    year = select_year(df_A)
    
    mappa(df_A, year, "TOTAL;FC")
    
    barchart_classifica(df_A, year)
    
    line_chart_deficit(df_A)

def page_production():
    st.title("Pagina Produzione")

    selected_siec = select_type(df_prod)

    df_prod_pred = pred_siec(selected_siec)
    df_prod_pred_A = df_from_M_to_A(df_prod_pred)
    # df_prod_pred_A = cast_int_prod(df_prod_pred_A)
    # df_prod_pred = cast_int_prod(df_prod_pred)

    year = select_year(df_prod_pred_A)
    mappa(df_prod_pred_A, year, selected_siec)

    selected_multi_state = select_multi_state(df_prod_pred_A)
    
    line_chart_prod(df_prod_pred, selected_multi_state, selected_siec)
    
    selected_single_state = select_state()

    area_chart(df_prod_pred, selected_single_state, prod_list)

def page_consumption():
    st.title("Pagina Consumo")

    selected_nrg_bal = select_type(df_cons)

    df_cons_pred = pred_cons(selected_nrg_bal)
    df_cons_pred_A = df_from_M_to_A(df_cons_pred)
    # df_cons_pred_A = cast_int_cons(df_cons_pred_A)
    # df_cons_pred = cast_int_cons(df_cons_pred)

    year = select_year(df_cons_pred_A)
    mappa(df_cons_pred_A, year, selected_nrg_bal)

    selected_single_state = select_state()
    
    barchart_cons(df_cons_pred_A, df_cons_pred, year, list_consuption)
    
    st.write(f"""
    ### Consumo di energia nei vari settori per un singolo stato europeo.
     """)
    line_chart_with_IC(df_cons_pred, selected_single_state)


pg = st.navigation([
    st.Page(page_deficit, title="First page"),
    st.Page(page_production, title="Second page"),
    st.Page(page_consumption, title="Third page"),
])
pg.run()