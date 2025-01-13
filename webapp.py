import streamlit as st
import polars as pl
import altair as alt
import country_converter as cc
import countryinfo as ci

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS

url_consumption = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_cb_e?format=TSV&compressed=true"
url_productivity = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_cb_pem?format=TSV&compressed=true"
url_population = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/demo_gind?format=TSV&compressed=true"
url_pop_pred = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/proj_stp24?format=TSV&compressed=true"
st.write(f"""
# Consumo di energia in Europa""")

## Funzioni lettura DataFrame #################################################################################
@st.cache_data
def get_data_population(url):
    data = (
        pl.read_csv(
            url,
            separator="\t",
            null_values=["", ":", ": "],
            )
        .select(
            # pl.col("freq,nrg_bal,siec,unit,geo\\TIME_PERIOD")
            pl.col("freq,indic_de,geo\\TIME_PERIOD")
            .str.split(",")
            .list.to_struct(fields=["freq","indic_de","state"])
            .alias("combined_info"),
            pl.col("*").exclude("freq,indic_de,geo\\TIME_PERIOD")
        )
        .unnest("combined_info")
        .unpivot(
            index=["freq","indic_de","state"],
            value_name="population",
            variable_name="date"
        ).with_columns(
            date=pl.col("date")
            .str.replace(" ", "")
            .str.strptime(pl.Date, '%Y'),#, strict=False).cast(pl.Date),
            population=pl.col("population")
            .str.replace(" ", "")
            .str.replace("p", "")
            .str.replace("b", "")
            .str.replace("e", "")
            .str.replace("ep", "")
            .str.replace("be", "")
            .str.replace("bep", "")
            .str.replace(":", "123456789")
            .cast(pl.Float64),
            state = pl.col("state")
            .str.replace("EL", "GR"),
        )
        .filter(
            pl.col("population").is_not_null(),
            pl.col("population") != 123456789,
            pl.col("indic_de") == "AVG"
        )
        .drop("freq",
              "indic_de")
        .sort("state", "date") 
    )
    return data

@st.cache_data
def get_data_pop_pred(url):
    data = (
        pl.read_csv(
            url,
            separator="\t",
            null_values=["", ":", ": "],
            )
        .select(
            pl.col("freq,indic_de,projection,geo\\TIME_PERIOD")
            .str.split(",")
            .list.to_struct(fields=["freq","indic_de","projection","state"])
            .alias("combined_info"),
            pl.col("*").exclude("freq,indic_de,projection,geo\\TIME_PERIOD")
        )
        .unnest("combined_info")
        .unpivot(
            index=["freq","indic_de","projection","state"],
            value_name="population",
            variable_name="date"
        ).with_columns(
            date=pl.col("date")
            .str.replace(" ", "")
            .str.strptime(pl.Date, '%Y'),#, strict=False).cast(pl.Date),
            population=pl.col("population")
            .str.replace(" ", "")
            .str.replace("p", "")
            .str.replace("b", "")
            .str.replace("e", "")
            .str.replace("ep", "")
            .str.replace("be", "")
            .str.replace("bep", "")
            .str.replace(":", "123456789")
            .cast(pl.Float64),
            state = pl.col("state")
            .str.replace("EL", "GR"),
        )
        .filter(
            pl.col("population").is_not_null(),
            pl.col("population") != 123456789,
            pl.col("indic_de") == "JAN",
            pl.col("projection") == "BSL",
        )
        .drop("freq",
              "indic_de",
              "projection")
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
              "unit",
              "siec")
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

def get_df_A(df_prod_pred_A, df_cons_pred_A):
    df_A = df_prod_pred_A.join(df_cons_pred_A, on = ["state", "date"], how = "inner")
    df_A = df_A.with_columns(
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
    return df_A

## Utils ######################################################################################################
def select_state():
    state = st.selectbox(
        "Seleziona uno stato",
        df_prod["state"].unique().sort(),
        index = 9
    )
    return state

def select_multi_state(df_input, EU = False):
    if EU == False:
        df_input = df_input.filter(pl.col("state") != "EU27_2020")

    selected_multi_state = st.multiselect(
        "Seleziona uno o più stati",
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

def select_year(df_input):
    # if df_input.columns == df_prod_pred_A.columns:
    #     first_year = 2017
    # else: first_year = df_input["date"].min().year
    first_year = df_input["date"].min().year
    last_year = df_input["date"].max().year
    year = st.select_slider(
                "Seleziona un anno",
                range(first_year, last_year)
            )
    year_datetime = pl.datetime(year, 1, 1)
    return year, year_datetime

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

def last_date(df_input, state):
    last_date = df_input.filter(pl.col("state")==state).agg(pl.col("date").max().alias("last_date"))
    return last_date

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
        .rename({"AutoARIMA-lo-95": "AutoARIMA_low95"})\
        .rename({"AutoARIMA-hi-95": "AutoARIMA_hi95"})
    
    return ts_pred

@st.cache_data
def pred_siec(filtro, state = None):
    if state is None:
        x = "siec"
    else:
        x = "state"
    df_funz = df_prod.filter(pl.col(x) == filtro).with_columns(
        AutoARIMA_low95 = pl.lit(0),
        AutoARIMA_hi95 = pl.lit(0), 
        predicted = pl.lit(False)
        ).sort("AutoARIMA_hi95", "AutoARIMA_low95", "date", "energy_prod", "predicted", "siec", "state","unique_id")  
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

    pred = adjust_pred_prod(pred)

    df_funz = df_funz.select(sorted(df_funz.columns))
    pred = pred.select(sorted(pred.columns))

    df_Ained = pl.concat([df_funz, pred], how= "vertical_relaxed")
    return df_Ained

@st.cache_data
def pred_state(filtro):
    for state in filtro:
        df_funz = df_prod.filter(pl.col("state") == filtro).with_columns(
            AutoARIMA_low95 = pl.lit(0),
            AutoARIMA_hi95 = pl.lit(0), 
            predicted = pl.lit(False)
            ).sort("AutoARIMA_hi95", "AutoARIMA_low95", "date", "energy_prod", "predicted", "siec", "state","unique_id")  
    
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

        pred = adjust_pred_prod(pred)

        df_funz = df_funz.select(sorted(df_funz.columns))
        pred = pred.select(sorted(pred.columns))

        df_Ained = pl.concat([df_funz, pred], how= "vertical_relaxed")
        return df_Ained

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
        .rename({"AutoARIMA-lo-95": "AutoARIMA_low95"})\
        .rename({"AutoARIMA-hi-95": "AutoARIMA_hi95"})
    return ts_pred 

@st.cache_data
def pred_cons(filtro, state = None):
    df_funz = df_cons.filter(pl.col("nrg_bal") == filtro).with_columns(
        AutoARIMA_low95 = pl.lit(0),
        AutoARIMA_hi95 = pl.lit(0), 
        predicted = pl.lit(False)
        ).sort("AutoARIMA_hi95", "AutoARIMA_low95", "date","energy_cons","nrg_bal","predicted", "state","unique_id")  
    if state is None:
        countries = df_funz.select("unique_id").unique().sort("unique_id").to_series()
    else:
        countries = [state + ";" + filtro]
    pred = pl.DataFrame()
    for state in countries:
        pred_state = Arima_cons(state)
        pred = pl.concat([pred, pred_state])

    pred = adjust_pred_cons(pred)

    pred = pred.with_columns(
        predicted = pl.lit(True),
        ).with_columns(
            pl.col("unique_id")
                .str.split(";")
                .list.to_struct(fields=["state", "nrg_bal"])
                .alias("combined_info"),
            date = pl.col("ds")
        ).unnest("combined_info").drop("ds")

    df_funz = df_funz.select(sorted(df_funz.columns))
    pred = pred.select(sorted(pred.columns))

    df_Ained = pl.concat([df_funz, pred], how= "vertical_relaxed")

    return df_Ained

def adjust_pred_prod(pred):
    pred = pred.with_columns(
            energy_prod=pl.when(pl.col("energy_prod") < 0).then(0).otherwise(pl.col("energy_prod")),
            AutoARIMA_low95=pl.when(pl.col("AutoARIMA_low95") < 0).then(0).otherwise(pl.col("AutoARIMA_low95")),
            AutoARIMA_hi95=pl.when(pl.col("AutoARIMA_hi95") < 0).then(0).otherwise(pl.col("AutoARIMA_hi95"))
        )
    return pred

def adjust_pred_cons(pred):
    pred = pred.with_columns(
            energy_cons=pl.when(pl.col("energy_cons") < 0).then(0).otherwise(pl.col("energy_cons")),
            AutoARIMA_low95=pl.when(pl.col("AutoARIMA_low95") < 0).then(0).otherwise(pl.col("AutoARIMA_low95")),
            AutoARIMA_hi95=pl.when(pl.col("AutoARIMA_hi95") < 0).then(0).otherwise(pl.col("AutoARIMA_hi95"))
        )
    return pred
## Creazione Dataframe ########################################################################################
df_prod = get_data_productivity(url_productivity)
df_cons = get_data_consumption(url_consumption)
pop = get_data_population(url_population)
pop_pred = get_data_pop_pred(url_pop_pred)

pop = pl.concat([pop, pop_pred], how="vertical_relaxed").group_by(["state", "date"]).agg(
    pl.col("population").mean().alias("population")
)

top4_EU = ["DE", "FR", "IT", "ES"]
list_consuption = ["FC", "FC_IND_E" , "FC_TRA_E", "FC_OTH_CP_E", "FC_OTH_HH_E", "FC_OTH_AF_E"]
list_productivity = ["TOTAL","X9900","RA000","N9000","CF","CF_R","RA100","RA200","RA300","RA400","RA500_5160","C0000","CF_NR","G3000","O4000XBIO"]
prod_list_2 = ["X9900","N9000","CF_R","RA100","RA200","RA300","RA400","RA500_5160","C0000","CF_NR","G3000","O4000XBIO"]
prod_list = ["RA000", "CF", "N9000", "X9900"]
RA_list = ["CF_R","RA100","RA200","RA300","RA400","RA500_5160"]
CF_list = ["C0000","CF_NR","G3000","O4000XBIO"]

df_prod = EU_filter(df_prod)
df_cons = EU_filter(df_cons)

df_prod_pred = pred_siec("TOTAL")
df_cons_pred = pred_cons("FC")

df_prod_pred_A = df_from_M_to_A(df_prod_pred)
df_cons_pred_A = df_from_M_to_A(df_cons_pred)

df_A = get_df_A(df_prod_pred_A, df_cons_pred_A)


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
## Line Chart con Doppio Asse Y ###############################################################################
def line_chart_prod(df_input, countries, siec):
    
    # Creazione del DataFrame che verrà utilizzato per il grafico
    stati_line = df_input.filter(
        pl.col("state").is_in(countries)).filter(
        pl.col("siec") == siec,
        pl.col("state") != "EU27_2020",
    )

    # Grafico di base
    line = alt.Chart(stati_line).mark_line(interpolate="basis").encode(
        x="date",
        y="energy_prod:Q",
        color="state",
        strokeDash="predicted:N"
    )

    # Dataframe contente inizio e fine del pannello del forecast
    source_date = [
        {"start": "2024-10", "end": "2029"},
    ]
    source_date_df = pl.DataFrame(source_date)

    # Creazione del rettangolo grigio
    rect = alt.Chart(source_date_df).mark_rect(color="lightgrey", opacity=0.4).encode(
        x="start:T",
        x2="end:T",
    )
    
    # Linea verticale tratteggiata per indicare l'inizio del pannello del forecast
    xrule = (
        alt.Chart(source_date_df)
        .mark_rule(color="grey", strokeDash=[12, 6], size=2, opacity=0.4)
        .encode(x="start:T")
    )

    # Creazione del testo per indicare "Valori Reali" e "Forecast"
    text_left = alt.Chart(source_date_df).mark_text(
        align="left", dx=-55, dy=-165, color="grey"
    ).encode(
        x="start:T",
        text=alt.value("Valori Reali")
    )
    text_right = alt.Chart(source_date_df).mark_text(
        align="left", dx=5, dy=-165, color="grey"
    ).encode(
        x="start:T",
        text=alt.value("Forecast")
    )

    # Creazione del cerchio vicino all'etichetta
    lable_circle = alt.Chart(stati_line.filter(pl.col("predicted")==True)).mark_circle().encode(
        alt.X("last_date['date']:T"),
        alt.Y("last_date['energy_prod']:Q").axis(title='Produzione di Energia'),
        color="state:N"
    ).transform_aggregate(
        last_date="argmax(date)",
        groupby=["state"]
    )
    # Creazione dell'etichetta
    lable_name = lable_circle.mark_text(align="left", dx=4).encode(text="state", color="state:N")
    
    # Selettori trasparenti attraverso il grafico. Questo è ciò che ci dice il valore x del cursore
    nearest = alt.selection_point(nearest=True, on="pointerover",
                                fields=["date"], empty=False)
    when_near = alt.when(nearest)

    # Disegna punti sulla linea e evidenzia in base alla selezione
    points = line.mark_point().encode(
        opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
    )

    # Disegna una rules nella posizione della selezione
    # Questa rules contiene le misura di produzione di energia di ogni stato selezionato
    rules = alt.Chart(stati_line).transform_pivot(
        "state",
        value="energy_prod",
        groupby=["date"]
    ).mark_rule(color="gray").encode(
        x="date",
        opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
        tooltip=[alt.Tooltip(c, type="quantitative") for c in stati_line["state"].unique()],
    ).add_params(nearest)

    # Preparazione secondo DataFrame che verrà visualizzato nel secondo asse.
    # Il DataFrame contiene la media di produzione di energia in Europa
    Europe_mean = df_input.filter(pl.col("state")== "EU27_2020"
        ).filter(
            pl.col("siec") == siec,
        ).with_columns(
            y2=pl.col("energy_prod")/27 
    )
    # Prendo il valore massimo e minimo e aggiungiamo il 30% per avere un margine di visualizzazione,
    # evitiamo così di avere punti  che siano troppo vicini ai bordi del grafico
    min_value_EU = Europe_mean['y2'].min() - 0.3*Europe_mean['y2'].min()
    max_value_EU = Europe_mean['y2'].max() + 0.3*Europe_mean['y2'].max()
    
    # Creazione della linea per la media di produzione di energia in Europa
    # Reppresenterà il secondo layer del grafico
    line_EU = alt.Chart(Europe_mean).mark_line(
        opacity=0.2, color='#lightgrey'#, interpolate='monotone', strokeDash=[2,2]
        ).encode(
            alt.X('date').title(None),
            alt.Y('y2', scale=alt.Scale(domain=[min_value_EU, max_value_EU])).axis(title='Produzione Media di Energia in Europa'),
            strokeDash="predicted:N"
    )

    # Costituzione del primo layer del grafico
    first_layer = alt.layer(
        rect, xrule, text_left, text_right, lable_circle, lable_name, line, points, rules, 
    )

    # Creazione del grafico
    line_chart = alt.layer(
        line_EU , first_layer
    ).resolve_scale(
        # Asse y indipendente così da avere due scale diverse
        y="independent"
    ).properties(
        width=800, height=400
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

    # Grafico di base
    line = alt.Chart(stati_line).mark_line(interpolate="basis").encode(
        x="date",
        y="energy_cons:Q",
        color="nrg_bal:N",
        strokeDash="predicted:N"
    )

    # Dataframe contente inizio e fine del pannello del forecast
    source_date = [
        {"start": "2023", "end": "2027"},
    ]
    source_date_df = pl.DataFrame(source_date)

    # Creazione del rettangolo grigio
    rect = alt.Chart(source_date_df).mark_rect(color="lightgrey", opacity=0.4).encode(
        x="start:T",
        x2="end:T",
    )

    # Linea verticale tratteggiata per indicare l'inizio del pannello del forecast
    xrule = (
        alt.Chart(source_date_df)
        .mark_rule(color="grey", strokeDash=[12, 6], size=2, opacity=0.4)
        #.encode(x=alt.datum(alt.DateTime(year=2023)))
        .encode(x="start:T")
    )

    # Creazione del testo per indicare "Valori Reali" e "Forecast"
    text_left = alt.Chart(source_date_df).mark_text(
        align="left", dx=-55, dy=-145, color="grey"
    ).encode(
        x="start:T",
        text=alt.value("Valori Reali")
    )
    text_right = alt.Chart(source_date_df).mark_text(
        align="left", dx=5, dy=-145, color="grey"
    ).encode(
        x="start:T",
        text=alt.value("Forecast")
    )

    # Creazione del cerchio vicino all'etichetta
    lable_circle = alt.Chart(stati_line.filter(pl.col("predicted")==True)).mark_circle().encode(
        alt.X("last_date['date']:T"),
        alt.Y("last_date['energy_cons']:Q"),
        color="nrg_bal:N"
    ).transform_aggregate(
        last_date="argmax(date)",
        groupby=["nrg_bal"],
    )
    # Creazione dell'etichetta
    lable_name = lable_circle.mark_text(align="left", dx=4).encode(text="nrg_bal", color="nrg_bal:N")

    # Creazione del selettore per il punto più vicino
    nearest = alt.selection_point(nearest=True, on="pointerover",
                                fields=["date"], empty=False)
    when_near = alt.when(nearest)

    # Disegna punti sulla linea e evidenzia in base alla selezione
    points = line.mark_point().encode(
        opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
    )

    # Disegna una rules nella posizione della selezione
    # Questa rules contiene le misura di produzione di energia di ogni stato selezionato
    rules = alt.Chart(stati_line).transform_pivot(
        "nrg_bal",
        value="energy_cons",
        groupby=["date"]
    ).mark_rule(color="gray").encode(
        x="date",
        opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
        tooltip=[alt.Tooltip(c, type="quantitative") for c in stati_line["nrg_bal"].unique()],
    ).add_params(nearest)

    # Creazione del layer per la banda dell'intervallo di confidenza
    # Selezioniamo solo i valori predetti
    conf_int = stati_line.filter(pl.col("predicted") == True)
    # Creazione della banda dell'intervallo di confidenza
    band = alt.Chart(conf_int).mark_errorband(extent='ci').encode(
        x="date",
        y=alt.Y("AutoARIMA_low95:Q", title=None),
        y2="AutoARIMA_hi95:Q",
        color="nrg_bal:N"
    )

    # Put the five layers into a chart and bind the data
    line_chart = alt.layer(
        rect, xrule, text_left, text_right, lable_circle, lable_name, line, points, rules, band
    ).resolve_scale(
        y="shared"
    ).properties(
        width=800, height=400
    )
    line_chart

## line_chart Deficit ##########################################################################################
def line_chart_deficit(df_input):
    selected_multi_state = select_multi_state(df_input, False)
    
    stati_line = df_input.filter(
        pl.col("state").is_in(selected_multi_state),
        pl.col("state") != "EU27_2020",
    )

    line = alt.Chart(stati_line
    ).mark_line(interpolate="basis"
    ).encode(
        x="date",
        y="deficit:Q",
        color="state",
        strokeDash="predicted:N"
    )

    source_date = [
        {"start": "2022", "end": "2027"},
    ]
    source_date_df = pl.DataFrame(source_date)

    rect = alt.Chart(source_date_df).mark_rect(color="lightgrey", opacity=0.4).encode(
        x="start:T",
        x2="end:T",
    )

    xrule = (
        alt.Chart(source_date_df)
        .mark_rule(color="grey", strokeDash=[12, 6], size=2, opacity=0.4)
        .encode(x="start:T")
    )

    text_left = alt.Chart(source_date_df).mark_text(
        align="left", dx=-55, dy=-145, color="grey"
    ).encode(
        x="start:T",
        text=alt.value("Valori Reali")
    )
    text_right = alt.Chart(source_date_df).mark_text(
        align="left", dx=5, dy=-145, color="grey"
    ).encode(
        x="start:T",
        text=alt.value("Forecast")
    )

    lable_circle = alt.Chart(stati_line.filter(pl.col("predicted")==True)).mark_circle().encode(
        alt.X("last_date['date']:T"),
        alt.Y("last_date['deficit']:Q"),
        color="state:N"
    ).transform_aggregate(
        last_date="argmax(date)",
        groupby=["state"]
    )
    # Creazione dell'etichetta
    lable_name = lable_circle.mark_text(align="left", dx=4).encode(text="state", color="state:N")

    # Selettori trasparenti attraverso il grafico. Questo è ciò che ci dice il valore x del cursore
    nearest = alt.selection_point(nearest=True, on="pointerover",
                              fields=["date"], empty=False)
    when_near = alt.when(nearest)

    points = line.mark_point().encode(
        opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
    )

    selectors = alt.Chart(df_input).mark_point().encode(
        x="date:T",
        opacity=alt.value(0),
    ).add_params(
        nearest
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

    line_chart = alt.layer( 
        # rect ,text_left, text_right, xrule , line,  lable_circle , selectors , points ,rules, text , lable_name
         rect, xrule, text_left, text_right, lable_circle, lable_name, line, selectors, points, rules, text
    ).encode(
            x=alt.X().title("date"),
            y=alt.Y().title("deficit")
    ).properties(
        width=800, height=400
    )
    line_chart

## Grafici "Caratteristici" ####################################################################################
## Area Chart ##################################################################################################
def area_chart(df_prod_pred, selected_single_state, prod_list):
    df_prod_pred_updated = pl.concat([df_prod_pred, pred_state(selected_single_state)], how="vertical_relaxed")
    df_prod_pred_updated = cast_int_prod(df_prod_pred_updated)
    # Creazione del DataFrame che verrà utilizzato per il grafico
    # Filtro per data, seleziono solo i dati a partire dal 2017 perchè prima di quella data non veniva segnata la produzione di energia rinnovabile 
    stati_line = df_prod_pred_updated.filter(
        pl.col("state") == selected_single_state,
        pl.col("siec").is_in(prod_list),
        pl.col("date") > pl.datetime(2017, 1, 1)
    )

    color_palette = alt.Scale(
        ## 12 energie analizzate. Divise in 3 gruppi di 4:
        # Rinnovabili classiche: RA100, RA200, RA300, RA400 -> Scala di azzurro
        # Rinnovabili non classiche o fonti alternative: RA500_5160, CF_R, X9900, N9000 -> Scala di orancione
        # Combustibili fossili: C0000, CF_NR, G3000, O4000XBIO -> Scala di grigio
        domain=["RA000", "CF", "N9000", "X9900"],
        range=["#009E73", "#000000", "#F0E442", "#0072B2"]
    )

    # Grafico di base
    area = alt.Chart(stati_line
        ).mark_area(
            opacity=0.5,
            interpolate='step-after',
            line=True,
        ).encode(
            x="date:T",
            y=alt.Y("energy_prod:Q").stack(True),
            color=alt.Color("siec:N", scale = color_palette),
            #color = "siec:N",
    )

    source_date = [
        {"start": "2024-11", "end": "2028-10"},
    ]
    source_date_df = pl.DataFrame(source_date)

    # Creazione del rettangolo grigio
    # rect = alt.Chart(source_date_df).mark_rect(color="lightgrey", opacity=0.4).encode(
    #     x="start:T",
    #     x2="end:T",
    # )
# .filter(pl.col("predicted")==True)
    rect = alt.Chart(stati_line.filter(pl.col("predicted") == True)).mark_rect(color="lightgrey", opacity=0.4).encode(
        x="date:T",
        # x2="end:T",
    )
    
    # Linea verticale tratteggiata per indicare l'inizio del pannello del forecast
    xrule = (
        alt.Chart(source_date_df)
        .mark_rule(color="grey", strokeDash=[12, 6], size=2, opacity=0.4)
        .encode(x="start:T")
    )

    # Creazione del testo per indicare "Valori Reali" e "Forecast"
    text_left = alt.Chart(source_date_df).mark_text(
        align="left", dx=-55, dy=-150, color="grey"
    ).encode(
        x="start:T",
        text=alt.value("Valori Reali")
    )
    text_right = alt.Chart(source_date_df).mark_text(
        align="left", dx=5, dy=-150, color="grey"
    ).encode(
        x="start:T",
        text=alt.value("Forecast")
    )

    lable_circle = alt.Chart(stati_line.filter(pl.col("predicted")==True)).mark_circle().encode(
        alt.X("last_date['date']:T"),
        alt.Y("last_date['energy_prod']:Q").axis(title='Produzione di Energia'),
        color="state:N"
    ).transform_aggregate(
        last_date="argmax(date)",
        groupby=["state"]
    )
    lable_name = lable_circle.mark_text(align="left", dx=60).encode(text="state", color="state:N")


    nearest = alt.selection_point(nearest=True, on="pointerover",
                              fields=["date"], empty=False)
    when_near = alt.when(nearest)

    # Draw points on the line, and highlight based on selection
    points = area.mark_point().encode(
        opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
    )
    
    rules = alt.Chart(stati_line).transform_pivot(
        "siec",
        value="energy_prod",
        groupby=["date"]
    ).mark_rule(color="gray").encode(
        x="date",
        opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
        tooltip=[alt.Tooltip(c, type="quantitative") for c in stati_line["siec"].unique()],
    ).add_params(nearest)

    # rect, xrule, text_left, text_right,
    area_chart = alt.layer(
        rect, xrule, text_left, text_right,  lable_circle, lable_name, area, points, rules
    ).properties(
        width=800, height=400
    ).resolve_scale(
        x="shared"
    )
    area_chart
    return df_prod_pred_updated

## Barchart Production #########################################################################################
def bar_chart(df_prod_pred_updated, selected_single_state, prod_list_2, year):
    # df_prod_pred = pl.concat([df_prod_pred, pred_state(selected_single_state)], how="vertical_relaxed")
    df_prod_pred_A_updated = df_from_M_to_A(df_prod_pred_updated)
    # df_prod_pred_A = cast_int_prod(df_prod_pred_A)
    
    stati_line = df_prod_pred_A_updated.filter(
        pl.col("state") == selected_single_state,
        pl.col("siec").is_in(prod_list_2),
        pl.col("date") == year
    ).with_columns(
        percentage= ((pl.col("energy_prod") / pl.col("energy_prod").sum()) * 100).round(3),   
    )

    bars = alt.Chart(stati_line).mark_bar().encode(
        x=alt.X('state:N').axis(None),
        y=alt.Y('sum(energy_prod):Q').stack('zero').axis(None),
        color=alt.Color('siec', scale=alt.Scale(scheme='tableau10')),
    )

    highlight = alt.selection_point(on='pointerover', fields=['siec'], nearest=True)

    points = bars.mark_bar().encode(
        opacity=alt.value(0)
    ).add_params(
        highlight
    )
# 249711
    bar_highlated = bars.mark_bar().encode(
        opacity=alt.condition(highlight, alt.value(1), alt.value(0.6))
    )

    bar = alt.layer(points, bar_highlated).properties(
        width=100, height=450
    )

    # Crea l'unique_id per filtrare il dataframe
    filter = selected_single_state + ";" + "TOTAL"

    # Valore di confontro per vedere se mancano dati
    # Il valore preso è la produzione totale originale, è stato utilizzato il dataset originale, usato anche per la mappa
    total_prod_og = df_prod_pred_A.filter(
        pl.col("unique_id") == filter, 
        pl.col("date")==year
    ).select(pl.col("energy_prod").cast(pl.Int32)
    ).unique().item()

    # Calcola la percentuale totale
    total_percentage = stati_line["percentage"].sum()
    #Calcola la produzione totale
    total_prod = stati_line["energy_prod"].sum()
    st.write(f"Total percentage: {total_percentage}%")
    st.write(f"Total production: {total_prod}")
    st.write(f"Total production original: {total_prod_og}")

    
    # Se la differenza tra la produzione totale e la produzione totale originale è maggiore del 10% (soglia scelta arbitrariamente da me) allora ci sono dati mancanti
    if total_prod > (total_prod_og + 0.05*total_prod_og):
        extra_percentage = ((total_prod - total_prod_og) / total_prod_og) * 100
        st.write(f"Ops! Qualcosa è andato storto! Percentuale di dati extra rispetto a quelli attesi: {extra_percentage:.2f}%")
        st.warning("Ops! Qualcosa è andato storto, ")
    elif total_prod < (total_prod_og - 0.05*total_prod_og):
        missing_percentage = ((total_prod_og - total_prod) / total_prod_og) * 100
        st.warning(f"Presenti dati mancanti! Percentuale dei dati mancanti: {missing_percentage:.2f}%")


    percentage_col_name = "percentage_" + selected_single_state
    energy_prod_col_name = "energy_prod_" + selected_single_state
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(
            stati_line.sort("siec").pivot(values=["percentage", "energy_prod"], columns="state", index="siec"),
            column_config={
                "siec": st.column_config.TextColumn(
                    "Fonte",
                    help="Fonte di energia"
                ),
                percentage_col_name: st.column_config.NumberColumn(
                    "Percentuale",
                    format="%f",
                ),
                energy_prod_col_name: st.column_config.NumberColumn(
                    "Energia prodotta",
                    format="%d"
                ),
            },
            hide_index=True,
        )
    with col2:
        st.altair_chart(bar, use_container_width=True)

def get_population(state):
    country = ci(state)
    country.population()

## Barchart consumption ########################################################################################
def barchart_cons(df_input, df_cons_pred, year, list_consuption, selected_single_state):
    selected_multi_state = select_multi_state(df_input, True)
    for state in selected_multi_state:
        for consuption in list_consuption:
            df_cons_pred = pl.concat([df_cons_pred, pred_cons(consuption, state)], how="vertical_relaxed")
    df_input = df_from_M_to_A(df_cons_pred)
    df_input = cast_int_cons(df_input)
    
    df_input = df_input.join(pop, on = ["state", "date"], how = "inner"
    ).with_columns(
        energy_cons_per_capita = (pl.col("energy_cons") / pl.col("population")).round(5),
    )

    stati_bar = df_input.filter(
        pl.col("state").is_in(selected_multi_state),
        pl.col("date") == year,
        pl.col("nrg_bal") != "FC"
        )

    EU_mean = df_input.filter(pl.col("state") == "EU27_2020", pl.col("date") == year).rename({"energy_cons_per_capita": "europe_mean"}).drop("state", "energy_cons", "population", "predicted", "unique_id")#.item()
    
    stati_bar = stati_bar.join(EU_mean, on = ["date", "nrg_bal"], how = "inner")

    bar = alt.Chart(stati_bar).mark_bar(color="lightgray").encode(
        x=alt.X('energy_cons_per_capita:Q', sort='ascending'),
        y=alt.Y('state').axis(None),
    )

    highlight = bar.mark_bar(color="#FF0000").encode(
        x2=alt.X2("europe_mean:Q")
    ).transform_filter(
        alt.datum.energy_cons_per_capita > alt.datum.europe_mean
    )

    xrule = (
        alt.Chart(stati_bar)
        .mark_rule(color="black", strokeDash=[1, 1], size=1, opacity=0.4)
        .encode(
            x="europe_mean:Q"
        )
    ).transform_filter(
        alt.datum.energy_cons_per_capita > alt.datum.europe_mean
    )

    text = bar.mark_text(align='left', dx=30).encode(
        text='energy_cons_per_capita:Q',
        color = "state:N"
    )
    text_2 = bar.mark_text(align='left', dx=12).encode(
        text='state:N',
        color = "state:N"
    )
    layered_chart = alt.layer(bar, text, text_2, highlight, xrule
    ).properties(
        width=600, height=80)

    faceted_chart = layered_chart.facet(
        row='nrg_bal:N'
    )
    faceted_chart 
    return df_input
    #############################################################################################################
    
## Barchart Consumption 2 ######################################################################################
def barchart_cons_2(df_input, year, selected_single_state):
    stati_bar_2 = df_input.filter(
        pl.col("date") == year,
        pl.col("state") == selected_single_state,
        pl.col("nrg_bal") != "FC"
    ).with_columns(
        percentage = ((pl.col("energy_cons_per_capita") / pl.col("energy_cons_per_capita").sum()) * 100).round(2)#.cast(pl.Int32),
    )
    st.write(stati_bar_2.select(pl.col(["percentage", "energy_cons_per_capita", "nrg_bal"])))

    bar_2 = alt.Chart(stati_bar_2).mark_bar(color="lightgray").encode(
        x=alt.X('percentage:Q'),
        y=alt.Y('nrg_bal', sort = 'x', axis = None),
        # color="nrg_bal:N",
    )
    bar_3 = alt.Chart(stati_bar_2).mark_bar().encode(
        x=alt.X('percentage:Q'),
        y=alt.Y('nrg_bal', sort = '-x', axis = None),
        color="nrg_bal:N",
    ).properties(
        width=800, height=400
    )

    rect = bar_3.mark_bar(color="lightgray", opacity=0.4).encode(
        x2 = alt.X2('percentage_max:Q'),
        y=alt.Y('nrg_bal', sort = '-x', axis = None),
    )
    rect
    bar_3 + rect

## Barchart Omino #############################################################################################
# def barchart_omino(df_input, year, selected_single_state):
    


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
    color = alt.when(predicate).then(alt.value("#00FF00")).otherwise(alt.value("#D81B60"))
    select = alt.selection_point(name="select", on="click")
    highlight = alt.selection_point(name="highlight", on="pointerover", empty=False)

    stroke_width = (
        alt.when(select).then(alt.value(2, empty=False))
        .when(highlight).then(alt.value(1))
        .otherwise(alt.value(0))
    )

    # rect = alt.Chart(df_input).mark_rect(color="lightgrey", opacity=0.4).encode(
    #     x=alt.datum(0),
    #     x2=alt.datum(df_input.filter(pl.col("deficit") > 0)["deficit"].max() * 1.1),
    # )

    # xrule = (
    #     alt.Chart(df_input)
    #     .mark_rule(color="grey", strokeDash=[12, 6], size=2, opacity=0.4)
    #     .encode(x=alt.datum(0))
    # )

    base = alt.Chart(df_input).mark_bar(
        stroke="black", cursor="pointer"
    ).encode(
        y=alt.Y("state", sort="-x", axis=None),#.axis(labels=False, ),
        x=alt.X("deficit:Q").axis(labels=False, ),
        color=color,
        fillOpacity=alt.when(select).then(alt.value(1)).otherwise(alt.value(0.3)),
        strokeWidth=stroke_width,
    ).add_params(select, highlight).properties(width=800)

    text_deficit = base.mark_text(
        align="left",
        baseline="middle",
        dx=alt.expr(alt.expr.if_(alt.datum.deficit >= 0, 25, -35))
    ).encode(text="deficit:Q")

    text_state = base.mark_text(
        align="left",
        baseline="middle",
        dx=alt.expr(alt.expr.if_(alt.datum.deficit >= 0, 5, -55))
    ).encode(text="state:N")
    # rect, xrule,
    barchart_classifica = alt.layer( base, text_deficit, text_state).configure_scale(bandPaddingInner=0.2)
    barchart_classifica

## Implementazione Pagine ######################################################################################

#Rect deve avere come X date a partire dal databse filtrato

def page_deficit():
    st.title("Pagina Deficit/Surplus")
    st.write(f"""    
    In questa pagina vengono analizzati i dati sul deficit/surplus di elettricità in Europa.
    Si vuole evidenziare come i vari stati producono e consumano elettricità.
    L'analisi è stata effettuata prendendo i dati di produzione e consumo di elettricità dal sito Eurostat. 
    """)    
    df_comb = cast_int_deficit(df_A)
    year_int, year = select_year(df_comb)
    st.write(f"""### Mappa del deficit/surplus di elettricità in Europa nel {year_int}.""")
    st.write(f"""La mappa va dai colori più scuri degli stati che hanno un surplus di elettricità,
    (ovvero stati che producono più elettricità di quanta ne consumano) ai colori più 
    chiari, ovvero gli stati in deficit (stati che consumano più elettricità di quanta
    ne producono).
             """)
    mappa(df_comb, year, "TOTAL;FC")

    st.write(f"""### Evoluzione del deficit/surplus di elettricità di vari stati europei.""")
    st.write(f"""Si passa ora ad osservare come varia la situazione nel tempo. 
    Si può visualizzare un singolo stato o si possono selezionare più stati contemporaneamente, 
    così da poter confrontare la situazione tra i vari stati.""")
    line_chart_deficit(df_comb)

    st.write(f"""### Classifica del deficit/surplus di elettricità in Europa nel {year_int}.""")
    st.write(f"""infine la classifica del deficit/surplus di elettricità che mostra quale interamente 
    quali stati dell'Unione Europea sono in postivo e quali in negativo. 
    Viene anche mostarto il valore medio per l'intera UE.""")
    barchart_classifica(df_comb, year)
    
my_expander = st.expander(label='Filtro')

def page_production():
    st.title("Pagina Produzione")
    
    with my_expander:
        selected_siec = select_type(df_prod)

    df_prod_pred = pred_siec(selected_siec).filter(pl.col("date") >= pl.datetime(2017, 1, 1))
    df_prod_pred_A = df_from_M_to_A(df_prod_pred).filter(pl.col("date") >= pl.datetime(2017, 1, 1))

    year_int, year = select_year(df_prod_pred_A)

    # with my_expander:
    
    st.write(f"""### Analisi della produzione energetica annuale in Europa.
    Analisi della produzione tra vari paesi dell'unione
    """)

    mappa(df_prod_pred_A, year, selected_siec)

    selected_multi_state = select_multi_state(df_prod_pred_A, False)

    line_chart_prod(df_prod_pred, selected_multi_state, selected_siec)
    
    st.write(f"""### Analisi della produzione in un singolo stato.
    """)

    selected_single_state = select_state()

    df_prod_pred_updated = area_chart(df_prod_pred, selected_single_state, prod_list)
    bar_chart(df_prod_pred_updated, selected_single_state, prod_list_2, year)

def page_consumption():
    st.title("Pagina Consumo")
    st.write(f"""### Analisi del consumo di elettricità annuale in Europa.
    """)
    selected_nrg_bal = select_type(df_cons)

    df_cons_pred = pred_cons(selected_nrg_bal)
    df_cons_pred_A = df_from_M_to_A(df_cons_pred)

    year_int, year = select_year(df_cons_pred_A)
    mappa(df_cons_pred_A, year, selected_nrg_bal)    
    
    selected_single_state = select_state()

    df_bar_2 = barchart_cons(df_cons_pred_A, df_cons_pred, year, list_consuption, selected_single_state) #DA rimuovere 
    barchart_cons_2(df_bar_2, year, selected_single_state)
    st.write(f"""
    ### Consumo di energia nei vari settori per un singolo stato europeo.
     """)
    line_chart_with_IC(df_cons_pred, selected_single_state)
    # barchart_omino(df_cons_pred, year, selected_single_state)

pg = st.navigation([
    st.Page(page_deficit, title="Deficit/Surplus"),
    st.Page(page_production, title="Produzione"),
    st.Page(page_consumption, title="Consumo"),
])
pg.run()