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
              "unit",
              "siec")
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
    df_prod_pred = pl.concat([df_prod_pred, pred_state(selected_single_state)], how="vertical_relaxed")
    df_prod_pred = cast_int_prod(df_prod_pred)

    # Creazione del DataFrame che verrà utilizzato per il grafico
    # Filtro per data, seleziono solo i dati a partire dal 2017 perchè prima di quella data non veniva segnata la produzione di energia rinnovabile 
    stati_line = df_prod_pred.filter(
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

    # selectors = alt.Chart(stati_line).mark_point().encode(
    #     x="date",
    #     opacity=alt.value(0),
    # ).add_params(
    #     nearest
    # )
    
    # # Draw text labels near the points, and highlight based on selection
    # text = area.mark_text(align="left", dx=5, dy=-5).encode(
    #     text=when_near.then("energy_prod:Q").otherwise(alt.value(" "))
    # )

    # # Draw a rule at the location of the selection
    # rules = alt.Chart(stati_line).mark_rule(color="gray").encode(
    #     x="date",
    # ).transform_filter(
    #     nearest
    # )
    
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
    )
    area_chart

## Barchart Production #########################################################################################
def bar_chart(df_prod_pred, selected_single_state, prod_list_2, year):
    st.write(f"Produzione energetica annuale in {selected_single_state} nel {year.cast(pl.Int32)}")

    df_prod_pred = pl.concat([df_prod_pred, pred_state(selected_single_state)], how="vertical_relaxed")
    df_prod_pred = cast_int_prod(df_prod_pred)

    stati_line = df_prod_pred.filter(
        pl.col("state") == selected_single_state,
        pl.col("siec").is_in(prod_list_2),
        pl.col("date") == year
    )

    bars = alt.Chart(stati_line).mark_bar().encode(
        x=alt.X('sum(energy_prod):Q').stack('zero').axis(None),
        y=alt.Y('state:N'),
        color=alt.Color('siec', scale=alt.Scale(scheme='tableau10')),
    )
    
    # text = alt.Chart(stati_line).mark_text(dx=-150, dy=30, color='white').encode(
    #     x=alt.X('sum(energy_prod):Q').stack('zero'),
    #     y=alt.Y('state:N'),
    #     detail='siec:N',
    #     text=alt.Text('sum(energy_prod):Q', format='.1f'),
    # )

    predicate = alt.datum.energy_prod > 300
    text = bars.mark_text(baseline="middle",dx = -160
    ).encode(
        text="energy_prod:Q",
        color=alt.condition(predicate, alt.value("white"), alt.value(""))
    )


    bar_prod = alt.layer(bars).properties(
        width=800, height=150)

    bar_prod

## Barchart consumption ########################################################################################
def barchart_cons(df_input, df_cons_pred, year, list_consuption):
    selected_multi_state = select_multi_state(df_input, True)
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
    
    
def page_production():
    st.title("Pagina Produzione")
    
    selected_siec = select_type(df_prod)

    df_prod_pred = pred_siec(selected_siec).filter(pl.col("date") >= pl.datetime(2017, 1, 1))
    df_prod_pred_A = df_from_M_to_A(df_prod_pred).filter(pl.col("date") >= pl.datetime(2017, 1, 1))

    st.write(f"""### Analisi della produzione energetica annuale in Europa.
    Analisi della produzione tra vari paesi dell'unione
    """)

    year_int, year = select_year(df_prod_pred_A)
    mappa(df_prod_pred_A, year, selected_siec)

    selected_multi_state = select_multi_state(df_prod_pred_A, False)

    line_chart_prod(df_prod_pred, selected_multi_state, selected_siec)
    
    st.write(f"""### Analisi della produzione in un singolo stato.
    """)

    selected_single_state = select_state()

    area_chart(df_prod_pred, selected_single_state, prod_list)
    bar_chart(df_prod_pred, selected_single_state, prod_list_2, year)

def page_consumption():
    st.title("Pagina Consumo")
    st.write(f"""### Analisi del consumo di elettricità annuale in Europa.
    """)
    selected_nrg_bal = select_type(df_cons)

    df_cons_pred = pred_cons(selected_nrg_bal)
    df_cons_pred_A = df_from_M_to_A(df_cons_pred)

    year_int, year = select_year(df_cons_pred_A)
    mappa(df_cons_pred_A, year, selected_nrg_bal)    
    
    barchart_cons(df_cons_pred_A, df_cons_pred, year, list_consuption)
    
    st.write(f"""
    ### Consumo di energia nei vari settori per un singolo stato europeo.
     """)
    selected_single_state = select_state()
    line_chart_with_IC(df_cons_pred, selected_single_state)

pg = st.navigation([
    st.Page(page_deficit, title="Deficit/Surplus"),
    st.Page(page_production, title="Produzione"),
    st.Page(page_consumption, title="Consumo"),
])
pg.run()