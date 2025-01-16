import streamlit as st
import polars as pl
import altair as alt
import country_converter as cc

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS
from streamlit_pdf_viewer import pdf_viewer

url_consumption = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_cb_e?format=TSV&compressed=true"
url_productivity = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_cb_pem?format=TSV&compressed=true"
url_population = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/demo_gind?format=TSV&compressed=true"
url_pop_pred = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/proj_stp24?format=TSV&compressed=true"

## Funzioni lettura DataFrame #################################################################################
# Lettura DataFrame sulla popolazione degli stati europei
@st.cache_data
def get_data_population(url):
    data = (
        pl.read_csv(
            url,
            separator="\t",
            null_values=["", ":", ": "],
            )
        .select(
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
            .str.strptime(pl.Date, '%Y'),
            population=pl.col("population")
            .str.replace(" ", "")
            .str.replace("p", "")
            .str.replace("b", "")
            .str.replace("e", "")
            .str.replace("ep", "")
            .str.replace("be", "")
            .str.replace("bep", "")
            # Per un problema di lettura del dato, sostituisco i valori nulli con un valore impossibile
            .str.replace(":", "123456789")
            .cast(pl.Float64),
            state = pl.col("state")
            # Cambio la sigla della Grecia da EL (ISO 3166-1 alpha-2 di Eurostat) in GR (teoricamente più comune)
            .str.replace("EL", "GR"),
        )
        # Elimino valori nulli
        .filter(
            pl.col("population").is_not_null(),
            pl.col("population") != 123456789,
            # Seleziono solo i valori medi annuali
            pl.col("indic_de") == "AVG"
        )
        .drop("freq",
              "indic_de")
        .sort("state", "date") 
    )
    return data

# Lettura dati di popolazione predetta
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
            # Per un problema di lettura del dato, sostituisco i valori nulli con un valore impossibile
            .str.replace(":", "123456789")
            .cast(pl.Float64),
            state = pl.col("state")
            # Cambio la sigla della Grecia da EL (ISO 3166-1 alpha-2 di Eurostat) in GR (teoricamente più comune)
            .str.replace("EL", "GR"),
        )
        .filter(
            # Elimino valori nulli
            pl.col("population").is_not_null(),
            pl.col("population") != 123456789,
            # Seleziono solo i valori al primo del mese e un solo tipo di previsione
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
# Lettura dati produzione elettrica
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
            # Per un problema di lettura del dato, sostituisco i valori nulli con un valore impossibile
            .str.replace(":c", "123456789")
            .str.replace(":", "123456789")
            .cast(pl.Float64),
            state = pl.col("state")
            # Cambio la sigla della Grecia da EL (ISO 3166-1 alpha-2 di Eurostat) in GR (teoricamente più comune)
            .str.replace("EL", "GR"),
            unique_id=(pl.col("state")+";"+pl.col("siec"))
            .str.replace("EL", "GR"),
        )
        .filter(
            # Seleziono solo i valori maggiori di 0 (non si può produrre energia negativa)
            pl.col("energy_prod") > 0,
            # Elimino valori nulli
            pl.col("energy_prod").is_not_null(),
            pl.col("energy_prod") != 123456789,
            # Seleziono solo un tipo di unità di misura, GWH (Gigawattora)
            pl.col("unit") == "GWH",
            # Filtro solo i tipi di energia che mi interessano, ignoro tutti quelli più specifici (es. Eolico offshore e onshore, Pannello fotovaltaico e solare, ecc.)
            pl.col("siec").is_in(["TOTAL","X9900","RA000","N9000","CF","CF_R","RA100","RA200","RA300","RA400","RA500_5160","C0000","CF_NR","G3000","O4000XBIO"]),
            )
        .drop("freq",
              "unit")
        .sort("state", "date") 
    )
    return data

@st.cache_data
# Lettura dati consumo elettrico
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
            # Per un problema di lettura del dato, sostituisco i valori nulli con un valore impossibile
            .str.replace(":", "123456789")
            .cast(pl.Float64),
            state = pl.col("state")
            # Cambio la sigla della Grecia da EL (ISO 3166-1 alpha-2 di Eurostat) in GR (teoricamente più comune)
            .str.replace("EL", "GR"),
            unique_id=(pl.col("state")+";"+pl.col("nrg_bal"))
            .str.replace("EL", "GR"),
        )
        .filter(
            # Elimino valori nulli
            pl.col("energy_cons").is_not_null(),
            pl.col("energy_cons") != 123456789,
            # Seleziono solo un tipo di unità di misura, GWH (Gigawattora), elimino valori europei lasciando solo quello dei 27 stati membri
            pl.col("state")!="EA20",
            # Filtro i  tipi di consumi che saranno utilizzati
            pl.col("nrg_bal").is_in(["FC", "FC_IND_E" , "FC_TRA_E", 
            "FC_OTH_CP_E", "FC_OTH_HH_E", "FC_OTH_AF_E"]),
        )
        .drop("freq",
              "unit",
              "siec")
        .sort("state", "date") 
    )
    return data

# Lettura DataFrame sul deficit/Surplus
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

## Utils: Modifica dei DataFrame #####################################################################################################
# Funzione che filtra i dati per gli stati membri dell'Unione Europea
def EU_filter(df_input):
    cc_2 = cc.CountryConverter()
    country_list = pl.from_pandas(cc_2.EU27as('ISO2'))
    country_list = country_list.select(pl.col("ISO2")).to_series()

    EU27_2020_filter = df_input.filter(pl.col("state") == "EU27_2020")
    df_input = df_input.filter(pl.col("state").is_in(country_list))
    df_input = pl.concat([df_input, EU27_2020_filter])
    return df_input

# Funzione che trasforma il DataFrame da frequenza mensile a frequenza annuale
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

# Funzione che fa il cast ad intero della colonna deficit
def cast_int_deficit(df_input):
    df_input = df_input.with_columns(
            deficit = pl.col("deficit").cast(pl.Int64)
        )
    return df_input

# Funzione che fa il cast ad intero della colonna produzione
def cast_int_prod(df_input):
    df_input = df_input.with_columns(
            energy_prod = pl.col("energy_prod").cast(pl.Int64)
        )
    return df_input

# Funzione che fa il cast ad intero della colonna consumo
def cast_int_cons(df_input):
    df_input = df_input.with_columns(
            energy_cons = pl.col("energy_cons").cast(pl.Int64)
        )
    return df_input

## Utils: Selezionatori ######################################################################################################
# Funzione che permette di selezionare uno stato
def select_state():
    state = st.selectbox(
        "Seleziona uno stato",
        df_prod["state"].unique().sort(),
        index = 9
    )
    return state

# Funzione che permette di selezionare uno o più stati
# Possibile anche filtrare la presenza o meno del dato su tutta l'unione europea, default è False, ovvero non presente
def select_multi_state(df_input, filter = None, EU = False):
    if EU == False:
        df_input = df_input.filter(pl.col("state") != "EU27_2020")

    selected_multi_state = st.multiselect(
        "Seleziona uno o più stati",
        df_input.select("state").unique().sort("state"),
        default=get_first_4_countries(df_input, filter=filter)
    )
    return selected_multi_state

# Funzione che permette di selezionare un tipo di energia o un tipo di consumo di energia
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

# Funzione che permette di selezionare un anno
def select_year(df_input):
    first_year = df_input["date"].min().year
    last_year = df_input["date"].max().year
    year = st.select_slider(
                "Seleziona un anno",
                range(first_year, last_year),
                value=2023
            )
    year_datetime = pl.datetime(year, 1, 1)
    return year, year_datetime

# Funzione che permette di ritornare una lista formata dai 5 maggiori stati in base al filtro

def get_first_4_countries(df_input, filter):

    # Se non do nessuna nazione in input ritorno le 4 nazioni principale
    if filter is None:
        return top4_EU

    # Sennò filtro per le colonne
    if df_input.columns == df_prod.columns:
        x = "energy_prod"
        type = "siec"
    elif df_input.columns == df_cons.columns:
        x = "energy_cons"
        type = "nrg_bal"
    if df_input.columns == df_prod_pred_A.columns:
        x = "energy_prod"
        type = "siec"
    elif df_input.columns == df_cons_pred_A.columns:
        x = "energy_cons"
        type = "nrg_bal"
    elif df_input.columns == df_A.columns:
        x = "deficit"
        type = "def_id"

    # e poi faccio la somma di tutta l'energia prodotta/consumata e ordino
    country_default = df_input.filter(pl.col(type) == filter).group_by("state").agg(
        pl.sum(x).alias("total_energy")
    ).sort("total_energy", descending=True).head(4).select("state").to_series()
    return country_default

# Funzione che ritorna se sono presenti valori simulati
def last_date(df_input, date):
    # Pendo filtro per la data in input
    check =  df_input.filter(pl.col("date") == date, pl.col("predicted") == True)

    # Ritona un warning se c'è almeno un valore calcoalto tramite ARIMA
    if len(check) != 0: 
        st.warning("Sono presenti valori simulati!")

## Predizione ##################################################################################################
# Funzione che permette di fare la predizione della produzione di energia
# Prende in input l'unique_id dello stato, ovvero lo stato e il tipo di energia prodotta
@st.cache_data
def Arima_prod(state):
    # Preparo la serie temporale
    ts = df_prod.filter(
        # Seleziono solo i dati relativi allo stato
        pl.col("unique_id") == state
        # Rinomino le colonne per poterle utilizzare con la libreria statsforecast
        ).select(
            pl.col("date").alias("ds"),
            pl.col("energy_prod").alias("y"),
            pl.col("unique_id")
            )
    # Verifico che ci siano dati per lo stato selezionato
    if len(ts) == 0:
        st.warning("Non ci sono dati per questo stato")
    
    # Creo il modello di previsione
    sf = StatsForecast(
        # Il modello scelto è un ARIMA
        models = [AutoARIMA(season_length = 12)],
        freq = '1mo',
        n_jobs=-1,
        # Se il modello ARIMA non è in grado di fare la previsione, allora uso un modello ETS
        fallback_model=AutoETS(season_length=12)
        )
    # Faccio la previsione, orientativamente 4 anni
    ts_pred = sf.forecast(df=ts, h=48, level=[95])

    # Rinomino le colonne così da poter fare il merge con il DataFrame originale
    ts_pred = ts_pred\
        .rename({"AutoARIMA": "energy_prod"})\
        .rename({"AutoARIMA-lo-95": "AutoARIMA_low95"})\
        .rename({"AutoARIMA-hi-95": "AutoARIMA_hi95"})
    
    return ts_pred

# Funzione di supporto ad Arima_prod, si occupa di filtrare i dati e di sistemare il dataframe
# Prende in input un tipo di produzione di energia e (potenzialmente) uno stato, anche se al momento non è utilizzato
@st.cache_data
def pred_siec(filtro, state = None):
    if state is None:
        x = "siec"
    else:
        x = "state"
    # Preparo il DataFrame
    df_funz = df_prod.filter(pl.col(x) == filtro).with_columns(
        AutoARIMA_low95 = pl.lit(0),
        AutoARIMA_hi95 = pl.lit(0),
        # Indico che i dati non sono previsioni 
        predicted = pl.lit(False)
        ).sort("AutoARIMA_hi95", "AutoARIMA_low95", "date", "energy_prod", "predicted", "siec", "state","unique_id")  
    # Se non è specificato uno stato, allora prendo tutti gli stati
    if state is None:
        countries = df_funz.select("unique_id").unique().sort("unique_id").to_series()
    else:
        countries = [state + ";" + filtro]
    # Creo un DataFrame vuoto che conterrà le previsioni
    pred = pl.DataFrame()
    # Per ogni unique_id, faccio la previsione
    for state in countries:
        state_pred = Arima_prod(state)
        pred = pl.concat([pred, state_pred])
    # Sistemo il nuovo dataFrame, inoltre rinomino le colonne così da poter fare il merge con il DataFrame originale
    pred = pred.with_columns(
        # Indico che i dati sono previsioni
        predicted = pl.lit(True),
        ).with_columns(
            # Dall'unique_id ricavo lo stato e il tipo di energia prodotta
            pl.col("unique_id")
                .str.split(";")
                .list.to_struct(fields=["state", "siec"])
                .alias("combined_info"),
            date = pl.col("ds")
        ).unnest("combined_info").drop("ds")

    # Sistemo i dati, se ci sono valori negativi li porto a 0
    pred = adjust_pred_prod(pred)

    # Ordino i DataFrame così da poter fare il merge
    df_funz = df_funz.select(sorted(df_funz.columns))
    pred = pred.select(sorted(pred.columns))

    # Faccio il merge dei due DataFrame e ritorno il risultato
    df_Ained = pl.concat([df_funz, pred], how= "vertical_relaxed")
    return df_Ained

# Funzione di supporto ad Arima_prod, si occupa di filtrare i dati e di sistemare il dataframe
# Prende in input uno stato e fa la previsione del consumo di energia per vari tipi di consumo
# Al momento sostituisce parzialmente la funzione pred_siec
@st.cache_data
def pred_state(filtro):
    # Preparo il DataFrame
    for state in filtro:
        df_funz = df_prod.filter(pl.col("state") == filtro).with_columns(
            AutoARIMA_low95 = pl.lit(0),
            AutoARIMA_hi95 = pl.lit(0), 
            # Indico che i dati non sono previsioni 
            predicted = pl.lit(False)
            ).sort("AutoARIMA_hi95", "AutoARIMA_low95", "date", "energy_prod", "predicted", "siec", "state","unique_id")  

        # Seleziono tutti i tipi di energia prodotta
        countries = df_funz.select("unique_id").unique().sort("unique_id").to_series()
        # Creo un DataFrame vuoto che conterrà le previsioni
        pred = pl.DataFrame()
        # Per ogni unique_id, faccio la previsione
        for state in countries:
            state_pred = Arima_prod(state)
            pred = pl.concat([pred, state_pred])
        # Sistemo il nuovo dataFrame, inoltre rinomino le colonne così da poter fare il merge con il DataFrame originale
        pred = pred.with_columns(
            # Indico che i dati sono previsioni
            predicted = pl.lit(True),
            ).with_columns(
                # Dall'unique_id ricavo lo stato e il tipo di energia prodotta
                pl.col("unique_id")
                    .str.split(";")
                    .list.to_struct(fields=["state", "siec"])
                    .alias("combined_info"),
                date = pl.col("ds")
            ).unnest("combined_info").drop("ds")
        
        # Sistemo i dati, se ci sono valori negativi li porto a 0
        pred = adjust_pred_prod(pred)

        # Ordino i DataFrame così da poter fare il merge
        df_funz = df_funz.select(sorted(df_funz.columns))
        pred = pred.select(sorted(pred.columns))

        # Faccio il merge dei due DataFrame e ritorno il risultato
        df_Ained = pl.concat([df_funz, pred], how= "vertical_relaxed")
        return df_Ained

# Funzione che permette di fare la predizione del consumo di eletricità
# Prende in input l'unique_id dello stato, ovvero lo stato e il tipo di consumo di eletricità 
# Funzionamento analogo alla funzione Arima_prod
@st.cache_data
def Arima_cons(state):
    # Preparo la serie temporale
    ts = df_cons.filter(
        # Seleziono solo i dati relativi allo stato
        pl.col("unique_id") == state
        ).select(
            # Rinomino le colonne per poterle utilizzare con la libreria statsforecast
            pl.col("date").alias("ds"),
            pl.col("energy_cons").alias("y"),
            pl.col("unique_id")
            )

    # Creo il modello di previsione
    sf = StatsForecast(
        # Il modello scelto è un ARIMA
        models = [AutoARIMA(season_length = 6 )],#season_length = 6
        freq = '2mo',
        n_jobs=-1,
        # Se il modello ARIMA non è in grado di fare la previsione, allora uso un modello ETS
        fallback_model=AutoETS(season_length=6),
        )

    # Essendo una serie di frequenza annuale, e avendo a disposizione solo 26 anni di dati, provo ad effettuare un upscaling
    # La "nuova" serie temporale avrà una frequenza bimestrale
    ts_upscale = (
        ts.upsample(time_column="ds", every="2mo")
        .interpolate()
        .fill_null(strategy="forward")
    )
    # Forecast, orizzonte temporale di 4 anni
    ts_pred = sf.forecast(df=ts_upscale, h=24, level=[95])

    # Rinomino le colonne così da poter fare il merge con il DataFrame originale
    ts_pred = ts_pred\
        .rename({"AutoARIMA": "energy_cons"})\
        .rename({"AutoARIMA-lo-95": "AutoARIMA_low95"})\
        .rename({"AutoARIMA-hi-95": "AutoARIMA_hi95"})
    
    return ts_pred 

# Funzione di supporto ad Arima_cons, si occupa di filtrare i dati e di sistemare il dataframe
# Prende in input un tipo di consumo di energia e (opzionalmente) uno stato
@st.cache_data
def pred_cons(filtro, state = None):
    # Preparo il DataFrame
    df_funz = df_cons.filter(pl.col("nrg_bal") == filtro).with_columns(
        AutoARIMA_low95 = pl.lit(0),
        AutoARIMA_hi95 = pl.lit(0), 
        # Indico che i dati non sono previsioni
        predicted = pl.lit(False)
        ).sort("AutoARIMA_hi95", "AutoARIMA_low95", "date","energy_cons","nrg_bal","predicted", "state","unique_id")  
    # Se non è specificato uno stato, allora prendo tutti gli stati
    if state is None:
        countries = df_funz.select("unique_id").unique().sort("unique_id").to_series()
    # Altrimenti prendo solo lo stato specificato
    else:
        countries = [state + ";" + filtro]
    # Creo un DataFrame vuoto che conterrà le previsioni
    pred = pl.DataFrame()
    # Per ogni unique_id, faccio la previsione
    for state in countries:
        pred_state = Arima_cons(state)
        pred = pl.concat([pred, pred_state])

    # Sistemo i dati, se ci sono valori negativi li porto a 0
    pred = adjust_pred_cons(pred)

    # Sistemo il nuovo dataFrame, inoltre rinomino le colonne così da poter fare il merge con il DataFrame originale
    pred = pred.with_columns(
        predicted = pl.lit(True),
        ).with_columns(
            pl.col("unique_id")
                .str.split(";")
                .list.to_struct(fields=["state", "nrg_bal"])
                .alias("combined_info"),
            date = pl.col("ds")
        ).unnest("combined_info").drop("ds")

    # Ordino i DataFrame così da poter fare il merge
    df_funz = df_funz.select(sorted(df_funz.columns))
    pred = pred.select(sorted(pred.columns))

    # Faccio il merge dei due DataFrame e ritorno il risultato
    df_Ained = pl.concat([df_funz, pred], how= "vertical_relaxed")
    return df_Ained

# Funzione che dato in input un DataFrame, porta a 0 i valori negativi
def adjust_pred_prod(pred):
    pred = pred.with_columns(
            energy_prod=pl.when(pl.col("energy_prod") < 0).then(0).otherwise(pl.col("energy_prod")),
            AutoARIMA_low95=pl.when(pl.col("AutoARIMA_low95") < 0).then(0).otherwise(pl.col("AutoARIMA_low95")),
            AutoARIMA_hi95=pl.when(pl.col("AutoARIMA_hi95") < 0).then(0).otherwise(pl.col("AutoARIMA_hi95"))
        )
    return pred

# Funzione che dato in input un DataFrame, porta a 0 i valori negativi
def adjust_pred_cons(pred):
    pred = pred.with_columns(
            energy_cons=pl.when(pl.col("energy_cons") < 0).then(0).otherwise(pl.col("energy_cons")),
            AutoARIMA_low95=pl.when(pl.col("AutoARIMA_low95") < 0).then(0).otherwise(pl.col("AutoARIMA_low95")),
            AutoARIMA_hi95=pl.when(pl.col("AutoARIMA_hi95") < 0).then(0).otherwise(pl.col("AutoARIMA_hi95"))
        )
    return pred

## Creazione Dataframe ########################################################################################
# Creo i DataFrame
df_prod = get_data_productivity(url_productivity)
df_cons = get_data_consumption(url_consumption)
pop = get_data_population(url_population)
pop_pred = get_data_pop_pred(url_pop_pred)

# Unisco i DataFrame sulla popolazione
pop = pl.concat([pop, pop_pred], how="vertical_relaxed").group_by(["state", "date"]).agg(
    pl.col("population").mean().alias("population")
)

# Lista di variabili utili
top4_EU = ["DE", "FR", "IT", "ES"] # Lista degli stati "più importanti"
list_consuption = ["FC", "FC_IND_E" , "FC_TRA_E", "FC_OTH_CP_E", "FC_OTH_HH_E", "FC_OTH_AF_E"] # Lista di tutti i tipi di consumo
list_productivity = ["TOTAL","X9900","RA000","N9000","CF","CF_R","RA100","RA200","RA300","RA400","RA500_5160","C0000","CF_NR","G3000","O4000XBIO"] # Lista di tutti i tipi di produzione
prod_list_2 = ["X9900","N9000","CF_R","RA100","RA200","RA300","RA400","RA500_5160","C0000","CF_NR","G3000","O4000XBIO"] # Lista senza il totale della produzione, il totale delle rinnovabili e il totale del fossile
prod_list = ["RA000", "CF", "N9000", "X9900"] # Lista con solo le fonti considerate come "macrocategorie"

# Filtri per i DataFrame, seleziono solo i dati relativi all'Unione Europea
df_prod = EU_filter(df_prod)
df_cons = EU_filter(df_cons)

# Creo i DataFrame per le previsioni, di default seleziono solo i dati relativi alla produzione e al consumo totale
df_prod_pred = pred_siec("TOTAL")
df_cons_pred = pred_cons("FC")

# Trasformo i DataFrame da frequenza mensile a frequenza annuale
df_prod_pred_A = df_from_M_to_A(df_prod_pred)
df_cons_pred_A = df_from_M_to_A(df_cons_pred)

# Creo il DataFrame per il deficit/Surplus
df_A = get_df_A(df_prod_pred_A, df_cons_pred_A)

### Grafici ###################################################################################################
## utils ######################################################################################################

#Funzione che implementa il rettangolo grigio con la rispettiva linea
# x_left, x_righe e y servono a cambiare la posizione delle scritte
def rect_and_label(df_input, y, x_left=-55, x_right=5, year_start = None, year_end=None):
    # Filtro il database per pendere solo i valori predetti
    if year_start == None: 
        df_input = df_input.filter(pl.col("predicted") == True)
        # Prendo la prima e l'ultima data
        first_year = df_input["date"].min()
    else: first_year = year_start
    last_year = df_input["date"].max()
    # Dataframe contente inizio e fine del pannello del forecast
    source_date = [
        {"start": first_year, "end": last_year},
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
        align="left", dx=x_left, dy=y, color="grey"
    ).encode(
        x="start:T",
        text=alt.value("Valori Reali")
    )
    text_right = alt.Chart(source_date_df).mark_text(
        align="left", dx=x_right, dy=y, color="grey"
    ).encode(
        x="start:T",
        text=alt.value("Forecast")
    )
    return rect, xrule, text_left, text_right

## Mappa ######################################################################################################

# Funzione che crea una mappa 
# Prende in input il DataFrame, l'anno e il tipo di bilancio energetico
def mappa(df_input, year, selected_bal):

    # Preparo il DataFrame filtrando per anno, inoltre rimuovo il dato relativo all'Unione Europea
    stati_map = df_input.filter(
        pl.col("date") == year,
        pl.col("state") != "EU27_2020"
        )

    # In base al deficit in input, seleziono le colonne corrette
    if df_input.columns == df_prod_pred_A.columns:
        x = "energy_prod"
        type = "siec"
    if df_input.columns == df_cons_pred_A.columns:
        x = "energy_cons"
        type = "nrg_bal"
    if df_input.columns == df_A.columns:
        x = "deficit"
        type = "def_id"
        
    # Converto i nomi degli stati in ISO, questo mi permette di utilizzare la mappa che prende in input i codici ISO numerici
    converted_countries = cc.convert(names=stati_map["state"], to='ISOnumeric')

    # Creo il DataFrame con i codici ISO numerici 
    stati_map = stati_map.with_columns(
        pl.Series("state", converted_countries).alias("ISO")
    ).filter(pl.col(type) == selected_bal).sort(x, "state")

    # Creo la mappa, uso una mappa con risoluzione 50m
    countries_map = alt.topo_feature(f"https://cdn.jsdelivr.net/npm/world-atlas@2/countries-50m.json", 'countries')
    #countries_map = alt.topo_feature(f"https://cdn.jsdelivr.net/npm/world-atlas@2/countries-10m.json", 'countries')

    # Definisco i valori minimo e massimo per la scala dei colori
    min_value = stati_map[x].min()
    max_value = stati_map[x].max()

    # Verifico che i codici siano del fromato corretto
    # I codici ISO di country_converter sono di tipo intero, mentre i codic salvati della mappa sono stringhe.
    # Questo fa sì che i codici ISO di alcuni stati siano di lunghezza inferiore a 3
    # Per risolvere questo problema, aggiungo degli zeri davanti ai codici ISO di lunghezza inferiore a 3
    source = stati_map.with_columns(
        pl.col("ISO").cast(pl.Utf8)).with_columns(
        pl.when(pl.col("ISO").str.len_chars() < 2)
        .then(pl.concat_str([pl.lit("00"), pl.col("ISO")]))
        .when(pl.col("ISO").str.len_chars() < 3)
        .then(pl.concat_str([pl.lit("0"), pl.col("ISO")]))
        .otherwise(pl.col("ISO")).alias("ISO_str")
        )

    # Creo il background la mappa
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
        width=790, height=400
    ).encode(tooltip=alt.value(None))

    # Creo la mappa con i dati
    map = alt.Chart(countries_map).mark_geoshape(
        stroke='black'
    ).project(
        type= 'mercator',
        scale= 350,                          # Magnify
        center= [20,50],                     # [lon, lat]
        clipExtent= [[000, 000], [800, 400]],    # [[left, top], [right, bottom]]
    ).properties(
        width=790, height=400
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
        x=alt.X("date"),
        y="energy_prod:Q",
        color = alt.Color("state:N", scale=alt.Scale(scheme="tableau10")).legend(None),
        # color="state",
        strokeDash=alt.StrokeDash("predicted:N").legend(None)
    ).properties(
        title = alt.Title("Produzione energetica in Europa", anchor='middle')
    )
    
    # Implemento le funzioni grafiche di base
    rect, xrule, text_left, text_right = rect_and_label(stati_line, x_left=-55, x_right=5, y = -145)

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
    lable_name = lable_circle.mark_text(align="left", dx=4, fontSize=14).encode(text="state", color="state:N")
    
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
        width=750, height=400
    )
    line_chart 

## Line Chart con Banda dell'intervallo #######################################################################
def line_chart_with_IC(df_cons_pred, selected_single_state):
  
    #Faccio la previsione per i vari tipi di consumo di energia
    for consuption in list_consuption:
        df_cons_pred = pl.concat([df_cons_pred, pred_cons(consuption, selected_single_state)])
    # Cast ad intero della colonna consumo di energia, così da poter visualizzare meglio i dati
    df_cons_pred = cast_int_cons(df_cons_pred)

    # Creazione del DataFrame che verrà utilizzato per il grafico
    stati_line = df_cons_pred.filter(
        pl.col("state") == selected_single_state,
        pl.col("nrg_bal") != "FC",
        # pl.col("date") > pl.datetime(2010, 1, 1)
        )

    # Grafico di base
    line = alt.Chart(stati_line).mark_line(interpolate="basis").encode(
        x=alt.X("date", title = None),
        y=alt.Y("energy_cons:Q"),
        # color="nrg_bal:N",
        color = alt.Color("nrg_bal:N", scale=alt.Scale(scheme="category10"), legend=alt.Legend(
            title="Settori di consumo:",
            orient = "top",
            labelExpr="{'FC_IND_E': 'Industriale', 'FC_TRA_E': 'Trasorti', 'FC_OTH_CP_E': 'Commercio e Servizi', 'FC_OTH_HH_E': 'Familiare', 'FC_OTH_AF_E': 'Agricolo e Forestale'}[datum.label] || datum.label"
        )),
        strokeDash=alt.StrokeDash("predicted:N").legend(None)
    ).interactive()

    # Implemento le funzioni grafiche di base
    rect, xrule, text_left, text_right = rect_and_label(stati_line, x_left=-55, x_right=5, y = -120)

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
    lable_name = lable_circle.mark_text(align="left", dx=4, fontSize=14).encode(text="nrg_bal", color="nrg_bal:N")

    # Creazione del selettore per il punto più vicino
    nearest = alt.selection_point(nearest=True, on="pointerover",
                                fields=["date"], empty=False)
    when_near = alt.when(nearest)

    # Disegna punti sulla linea e evidenzia in base alla selezione
    points = line.mark_point().encode(
        opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
    )

    # Disegna una rules nella posizione della selezione
    # Questa rules contiene le misura del consumo di energia di ogni stato selezionato
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
        y=alt.Y("AutoARIMA_low95:Q", title="Consumo di Energia in GWH"),
        y2="AutoARIMA_hi95:Q",
        color="nrg_bal:N"
    )

    # Metti i vari livelli in un grafico e gli associo i dati
    line_chart = alt.layer(
        rect, xrule, text_left, text_right, lable_circle, lable_name, line, points, rules, band
    ).resolve_scale(
        y="shared"
    ).properties(
        width=720, height=400
    )
    line_chart

    return df_cons_pred

## line_chart Deficit ##########################################################################################
def line_chart_deficit(df_input, ):
    # Seleziono solo gli stati che mi interessano
    selected_multi_state = select_multi_state(df_input, filter="TOTAL;FC", EU=False)
    
    # Creazione del DataFrame che verrà utilizzato per il grafico
    stati_line = df_input.filter(
        pl.col("state").is_in(selected_multi_state),
        pl.col("state") != "EU27_2020",
    )

    # Grafico di base
    line = alt.Chart(stati_line).mark_line(interpolate="basis").encode(
        x=alt.X("date",title=None),
        y=alt.Y("deficit:Q", title="Deficit/Surplus in GWH"),
        # color="state",
        color = alt.Color("state:N", scale=alt.Scale(scheme="tableau10")).legend(orient="top"),
        strokeDash=alt.StrokeDash("predicted:N").legend(orient="top", title="Valore predetto")
    )

    # Implemento le funzioni grafiche di base
    rect, xrule, text_left, text_right = rect_and_label(stati_line, x_left=-55, x_right=5, y = -145)

    # Disegno la linea sullo zero
    yrule = (
        alt.Chart(stati_line)
        .mark_rule(color="grey", strokeDash=[4, 4], size=2, opacity=0.4)
        .encode(y = alt.datum(0))
    )
    # Creazione del cerchio vicino all'etichetta
    lable_circle = alt.Chart(stati_line.filter(pl.col("predicted")==True)).mark_circle().encode(
        alt.X("last_date['date']:T"),
        alt.Y("last_date['deficit']:Q"),
        color="state:N"
    ).transform_aggregate(
        last_date="argmax(date)",
        groupby=["state"]
    )
    # Creazione dell'etichetta
    lable_name = lable_circle.mark_text(align="left", dx=5, fontSize=14).encode(text="state", color="state:N")

    # Selettori trasparenti attraverso il grafico. Ci diranno il valore x del cursore
    nearest = alt.selection_point(nearest=True, on="pointerover",
                              fields=["date"], empty=False)
    when_near = alt.when(nearest)

    # Disegna punti sulla linea e evidenzia in base alla selezione
    points = line.mark_point().encode(
        opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
    )

    selectors = alt.Chart(df_input).mark_point().encode(
        x="date:T",
        opacity=alt.value(0),
    ).add_params(
        nearest
    )    
    # Disegna una etichette di testo vicino ai punti e evidenzia in base alla selezione
    text = line.mark_text(align="left", dx=5, dy=-10, fontSize=12).encode(
        text=when_near.then("deficit:Q").otherwise(alt.value(" "))
    )
    # Disegna una linea nella posizione della selezione
    rules = alt.Chart(df_input).mark_rule(color="lightgray", opacity=0.7).encode(
        x="date:T",
    ).transform_filter(
        nearest
    )

    # Creazione del grafico
    line_chart = alt.layer( 
         rect, xrule, yrule, text_left, text_right, lable_circle, lable_name, line, selectors, points, rules, text
    ).properties(
        width=720, height=400
    )
    line_chart

## Grafici "Caratteristici" ####################################################################################
## Area Chart ##################################################################################################
def area_chart(df_prod_pred, df_prod_pred_total, selected_single_state, prod_list):

    # Faccio la previsione per i vari tipi di produzione di energia
    df_prod_pred_updated = pl.concat([df_prod_pred, pred_state(selected_single_state)], how="vertical_relaxed")
    df_prod_pred_updated = cast_int_prod(df_prod_pred_updated)

    # Creazione del DataFrame che verrà utilizzato per il grafico
    # Filtro per data, seleziono solo i dati a partire dal 2017 perchè prima di quella data non veniva segnata la produzione di energia rinnovabile 
    stati_area = df_prod_pred_updated.filter(
        pl.col("state") == selected_single_state,
        pl.col("siec").is_in(prod_list),
        pl.col("date") > pl.datetime(2017, 1, 1)
    ).unique()

    color_palette = alt.Scale(
        ## Seleziono le 4 Fonti di energia più generali:
        # RA000: Energia Rinnovabile
        # CF: Carburanti Fossili
        # N9000: Nucleare 
        # X9900: Altre Rinnovabili
        domain=["RA000", "CF", "N9000", "X9900"],
        range=["#009E73", "#000000", "#F0E442", "#0072B2"]
    )

    # Grafico di base
    area = alt.Chart(stati_area
        ).mark_area(
            opacity=0.5,
            interpolate='step-after',
            line=True,
        ).encode(
            x="date:T",
            y=alt.Y("energy_prod:Q").stack(True),
            color=alt.Color("siec:N", scale=color_palette, title="Tipi di Fonti", legend=alt.Legend(
                orient="bottom",
                columns = 2,
                # Uso il metodo labelExpr per rinominare tutte le label della legenda
                labelExpr="{'RA000': 'RA000 Energia Rinnovabili', 'CF': 'CF Carburanti Fossili', 'N9000': 'N9000 Nucleare', 'X9900': 'X9900 Altre Rinnovabili'}[datum.label] || datum.label"
            )),
    )
    
    # Implemento le funzioni grafiche di base
    rect, xrule, text_left, text_right = rect_and_label(stati_area, x_left=-55, x_right=5, y = -145)

    # Creazione del cerchio vicino all'etichetta
    lable_circle = alt.Chart(stati_area.filter(pl.col("predicted")==True)).mark_circle().encode(
        alt.X("last_date['date']:T").title(None),
        alt.Y("last_date['energy_prod']:Q").axis(title='Produzione di Energia'),
        color="state:N"
    ).transform_aggregate(
        last_date="argmax(date)",
        groupby=["state"]
    )
    # Creazione dell'etichetta
    lable_name = lable_circle.mark_text(align="left", dx=60,fontSize=14).encode(text="state", color="state:N")

    # Creazione del selettore per il punto più vicino
    nearest = alt.selection_point(nearest=True, on="pointerover",
                              fields=["date"], empty=False)
    when_near = alt.when(nearest)

    # Disegna punti sulla linea e evidenzia in base alla selezione
    points = area.mark_point().encode(
        opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
    )
    
    filter = selected_single_state + ";" + "TOTAL"
    total_prod_og = df_prod_pred_total.filter(
        pl.col("unique_id") == filter, 
    ).select(pl.col("energy_prod").cast(pl.Int32)
    ).unique().sum().item()

    total_prod = stati_area["energy_prod"].sum()
    # st.write(f"Total production: {total_prod}")
    # st.write(f"Total production original: {total_prod_og}")

    # Se la differenza tra la produzione totale e la produzione totale originale è minore del 0.05% (soglia scelta arbitrariamente da me) mandiamo un messaggio di warning
    # Purtroppo non sembra funzionare benissimo, alcuni stati come la finlandia non ritornano il messaggio
    if total_prod < (total_prod_og - 0.05*total_prod_og):
        st.warning(f"Presenti dati mancanti!")

    # Disegna una rules nella posizione della selezione
    # Questa rules contiene le misura della produzione di elettricita' di ogni stato selezionato
    rules = alt.Chart(stati_area).transform_pivot(
        "siec",
        value="energy_prod",
        groupby=["date"]
    ).mark_rule(color="gray").encode(
        x="date",
        opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
        tooltip=[alt.Tooltip(c, type="quantitative") for c in stati_area["siec"].unique()],
    ).add_params(nearest)

    area_chart = alt.layer(
        rect, xrule, text_left, text_right,  lable_circle, lable_name, area, points, rules
    ).properties(
        width=750, height=450
    ).resolve_scale(
        x="shared"
    )
    area_chart
    return df_prod_pred_updated

def bump_chart(df_input):
    
    df_input = df_input.filter(
        pl.col("state") != "EU27_2020"
    ).group_by(["state", "date"]).agg(
        pl.col("deficit").mean().alias("deficit")
    )
    # highlight = alt.selection_point(on='pointerover', fields=['state'], nearest=True)
    nearest = alt.selection_point(nearest=True, on="pointerover",
                              fields=["date"], empty=False)

    stati_rank = pl.DataFrame()
    for time in df_input["date"].unique():
        stati_sel = df_input.filter(
            pl.col("date") == time,
            ).sort("deficit", descending=True).head(5)
        stati_rank = pl.concat([stati_rank, stati_sel])
    
    # Implemento le funzioni grafiche di base
    rect, xrule, text_left, text_right = rect_and_label(stati_rank, x_left=-55, x_right=5, y = -135, year_start="2023")

    # Grafico di base
    ranking_plot = alt.Chart(stati_rank).mark_line(point=True, strokeDash=[4,1]).encode(
        x=alt.X("date", title=None).timeUnit("year"),
        y=alt.Y("rank:O", title="Classifica"),
        color = alt.Color("state:N", scale=alt.Scale(scheme="tableau10")).legend(orient="top"),
    ).transform_window(
        rank="rank()",
        sort=[alt.SortField("deficit", order="descending")],
        groupby=["date"])

    # Selettore verticale
    selectors = alt.Chart(df_input).mark_point().encode(
        x="date:T",
        opacity=alt.value(0),
    ).add_params(
        nearest
    )
    when_near = alt.when(nearest)

    # Disegna i punti sulla linea
    points = ranking_plot.mark_point().encode(
        opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
    )

    # Disegna le label vicino ai punti
    text = ranking_plot.mark_text(align="left", dx=5, dy=-7, fontSize=14).encode(
        text=when_near.then("state:N").otherwise(alt.value(" "))
    )

    # Selettore verticale
    # Dovrebbe fare la stessa cosa di rules, ma per adesso non lo tocco
    rules = alt.Chart(df_input).mark_rule(color="gray").encode(
        x="date:T",
    ).transform_filter(
        nearest
    )   

    # Unione dei vari layer
    classifica = alt.layer(
        rect, xrule, ranking_plot, selectors, points, rules, text, text_left, text_right
    ).properties(
        width=690, height=400
    )
    classifica

def pie_chart(df_input, selected_single_state, year):
    
    # Creo il DataFrame per il grafico
    stati_pie = df_input.filter(
        pl.col("date") == year,
        pl.col("state") == selected_single_state,
        pl.col("nrg_bal") != "FC",
    ).group_by("nrg_bal", "state").agg(
            pl.col("energy_cons").mean().alias("energy_cons")
    ).with_columns(
        # Calcola la percentuale di consumo di energia per ogni tipo di bilancio energetico
        percentage = ((pl.col("energy_cons") / pl.col("energy_cons").sum()) * 100).round(2),
        # Rinomino nrg_bal con etichette più comprensibili
        settore = pl.col("nrg_bal")
        .str.replace("FC_IND_E","Industria")
        .str.replace("FC_TRA_E","Trasporti")
        .str.replace("FC_OTH_CP_E","Servizi")
        .str.replace("FC_OTH_HH_E","Familiare")
        .str.replace("FC_OTH_AF_E","Agricolo")
    )
    # Creazione del grafico di base
    base = alt.Chart(stati_pie).mark_arc().encode(
        theta = alt.Theta("percentage:Q").stack(True),
        color = alt.Color("nrg_bal:N", scale=alt.Scale(scheme="category10"), legend=None),
    )
    # Creo il "foro" del donut chart
    pie = base.mark_arc(outerRadius=120, innerRadius=50)
    # Aggiungo il testo per la percentuale
    text_percentage = base.mark_text(radius=140, size=15).encode(text=("percentage:Q"))

    # Creo il grafico finale
    pie_chart = alt.layer(
        pie + text_percentage 
    ).properties(
        width=600, height=600
    )

    # Preparo la visualizzazione su due colonne
    # Preparo i nomi delle colonne
    percentage_col_name = "percentage_" + selected_single_state
    energy_prod_col_name = "energy_cons_" + selected_single_state
    label_name = "settore_" + selected_single_state
    # Scelgo il layout a due colonne
    col1, col2 = st.columns(2)
    # Inserisco i dati nel primo pannello
    with col1:
        st.dataframe(
            stati_pie.sort("nrg_bal").pivot(values=["settore","percentage", "energy_cons"], columns="state", index="nrg_bal"),
            column_config={
                label_name: st.column_config.TextColumn(
                    "Settore di consumo",
                ),
                "nrg_bal": st.column_config.TextColumn(
                    "Codice",
                    help="Codice identificativo dello settore di consumo"
                ),
                percentage_col_name: st.column_config.NumberColumn(
                    "Percentuale",
                    format="%f",
                ),
                energy_prod_col_name: st.column_config.NumberColumn(
                    "Energia consumata",
                    format="%d"
                ),
            },
            hide_index=True,
        )
    # Inserisco il grafico nel secondo pannello
    with col2:
        st.altair_chart(pie_chart, use_container_width=True)

## Grafico a Barre ############################################################################################
## Barchart Production #########################################################################################
def bar_chart_with_db(df_prod_pred_updated, df_prod_pred_total, selected_single_state, prod_list_2, year):
    
    # Trasformo il DataFrame da frequenza mensile a frequenza annuale
    df_prod_pred_A_updated = df_from_M_to_A(df_prod_pred_updated)
    
    # Creazione del DataFrame che verrà utilizzato per il grafico
    stati_bar = df_prod_pred_A_updated.filter(
        pl.col("state") == selected_single_state,
        pl.col("siec").is_in(prod_list_2),
        pl.col("date") == year
    ).group_by("siec", "state").agg(
            pl.col("energy_prod").mean().alias("energy_prod")
    ).with_columns(
        # Calcola la percentuale di produzione di energia
        percentage= ((pl.col("energy_prod") / pl.col("energy_prod").sum()) * 100).round(3),   
    )

    # Creazione del grafico di base
    bars = alt.Chart(stati_bar).mark_bar().encode(
        x=alt.X('state:N').axis(None),
        y=alt.Y('sum(energy_prod):Q').stack('zero').axis(None),
        color=alt.Color('siec', scale=alt.Scale(scheme='category10'), legend=None),
    )

    # Creazione del selettore per il punto più vicino
    highlight = alt.selection_point(on='pointerover', fields=['siec'], nearest=True)

    points = bars.mark_bar().encode(
        opacity=alt.value(0)
    ).add_params(
        highlight
    )

    # Definisce le colonne che si ingrandiscnon quando si passa sopra con il mouse
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
    total_prod_og = df_prod_pred_total.filter(
        pl.col("unique_id") == filter, 
        pl.col("date")==year
    ).select(pl.col("energy_prod").cast(pl.Int32)
    ).unique().item()

    # Calcola la percentuale totale
    # total_percentage = stati_line["percentage"].sum()
    # Calcola la produzione totale
    total_prod = stati_bar["energy_prod"].sum()
    # st.write(f"Total percentage: {total_percentage}%")
    # st.write(f"Total production: {total_prod}")
    # st.write(f"Total production original: {total_prod_og}")

    # Se la differenza tra la produzione totale e la produzione totale originale è maggiore o minore del 0.05% (soglia scelta arbitrariamente da me) mandiamo un messaggio di warning
    if total_prod > (total_prod_og + 0.05*total_prod_og):
        extra_percentage = ((total_prod - total_prod_og) / total_prod_og) * 100
        st.write(f"Ops! Qualcosa è andato storto! Percentuale di dati extra rispetto a quelli attesi: {extra_percentage:.2f}%")
        st.warning("Ops! Qualcosa è andato storto, ")
    elif total_prod < (total_prod_og - 0.05*total_prod_og):
        missing_percentage = ((total_prod_og - total_prod) / total_prod_og) * 100
        st.warning(f"Presenti dati mancanti! Percentuale dei dati mancanti: {missing_percentage:.2f}%")

    # Creo una nuova colonna e rinomino le colonne così da poter capire più facilmente di quale fonte di energia si tratta
    stati_label = stati_bar.with_columns(
        Fonte = pl.col("siec")
        .str.replace("TOTAL","Totale")
        .str.replace("X9900","Altri combustibili")
        .str.replace("RA000","Rinnovabili")
        .str.replace("N9000","Nucleare")
        .str.replace("CF","Carburante Fossile")
        .str.replace("CF_R","Fossile Rinnovabile")
        .str.replace("RA100","Idroelettrico")
        .str.replace("RA200","Geotermico")
        .str.replace("RA300","Eolico")
        .str.replace("RA400","Solare")
        .str.replace("RA500_5160","Altre Rinnovabili")
        .str.replace("C0000","Carbone")
        .str.replace("CF_NR","Fossile non Rinnovabile")
        .str.replace("G3000","Gas Naturale")
        .str.replace("O4000XBIO","Petrolio e Derivati"),
    )

    # Preparo la visualizzazione su due colonne
    # Preparo i nomi delle colonne
    percentage_col_name = "percentage_" + selected_single_state
    energy_prod_col_name = "energy_prod_" + selected_single_state
    fonte_col_name = "Fonte_" + selected_single_state
    # Scelgo il layout a due colonne
    col1, col2 = st.columns([0.7, 0.3])
    # Inserisco i dati nel primo pannello
    with col1:
        st.dataframe(
            stati_label.sort("siec").pivot(values=["Fonte","percentage", "energy_prod"], on="state", index="siec"),
            column_config={
                "siec": st.column_config.TextColumn(
                    "Siec",
                    help="Standard International Energy Product Classification"
                ),
                fonte_col_name: st.column_config.TextColumn(
                    "Fonte",
                    help="Fonte di Energia"
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
    # Inserisco il grafico nel secondo pannello
    with col2:
        st.altair_chart(bar.properties(width=300), use_container_width=True)

## Barchart consumption ########################################################################################
##### Devo sistemare il problema del 2022 
def bar_chart_cons(df_input, df_cons_pred, year, list_consuption):

    # Seleziono solo gli stati che mi interessano
    selected_multi_state = select_multi_state(df_input, filter=None, EU=False)

    # Faccio la previsione per i vari tipi di consumo di energia
    for state in selected_multi_state:
        for consuption in list_consuption:
            df_cons_pred = pl.concat([df_cons_pred, pred_cons(consuption, state)], how="vertical_relaxed")
    # Trasformo il DataFrame da frequenza mensile a frequenza annuale e faccio il cast ad intero della colonna consumo di energia
    for consuption in list_consuption:
        df_cons_pred = pl.concat([df_cons_pred, pred_cons(consuption, "EU27_2020")], how="vertical_relaxed")
    df_input = df_from_M_to_A(df_cons_pred)
    df_input = cast_int_cons(df_input)
    # Facendo il join con la popolazione, posso calcolare il consumo di energia procapite
    df_input = df_input.join(pop, on = ["state", "date"], how = "inner"
    ).with_columns(
        energy_cons_per_capita = (pl.col("energy_cons") / pl.col("population")).round(5),
    )
    # Creazione del DataFrame che verrà utilizzato per il grafico
    stati_bar = df_input.filter(
        pl.col("state").is_in(selected_multi_state),
        pl.col("date") == year,
        pl.col("nrg_bal") != "FC"
        )
    # Creazione del DataFrame per la media di consumo di energia in Europa

    EU_mean = df_input.filter(pl.col("state") == "EU27_2020", pl.col("date") == year).rename({"energy_cons_per_capita": "europe_mean"}).drop("state", "energy_cons", "population", "predicted", "unique_id")#.item()

    # Join tra i due DataFrame
    # Faccio la media dei valori che hanno la stessa data e unique_id così da eliminare doppioni
    stati_bar = stati_bar.join(EU_mean, on = ["date", "nrg_bal"], how = "inner"
            ).group_by("unique_id", "state", "nrg_bal", "europe_mean").agg(
            pl.col("energy_cons_per_capita").mean().alias("energy_cons_per_capita")
        ).with_columns(
        nrg_bal = pl.col("nrg_bal")
        .str.replace("FC_IND_E","Industria")
        .str.replace("FC_TRA_E","Trasporti")
        .str.replace("FC_OTH_CP_E","Servizi")
        .str.replace("FC_OTH_HH_E","Familiare")
        .str.replace("FC_OTH_AF_E","Agricolo")
    )

    # Creazione del grafico
    bar = alt.Chart(stati_bar).mark_bar(color="lightgray").encode(
        x=alt.X('energy_cons_per_capita:Q', title="Consumo pro-capite di energia"),
        y=alt.Y('state:N', sort="-x", axis=None),
    )

    # Se il valroe del consumo di energia procapite è maggiore della media europea, coloro la barra di rosso
    highlight = bar.mark_bar(color="#FF0000").encode(
        x2=alt.X2("europe_mean:Q")
    ).transform_filter(
        alt.datum.energy_cons_per_capita > alt.datum.europe_mean
    )

    # Creo una linea verticale per la media europea
    xrule_EU = (
        alt.Chart(stati_bar)
        .mark_rule(color="black", strokeDash=[1, 1], size=1, opacity=0.4)
        .encode(
            x=alt.X("europe_mean:Q", title="in verticale tratteggiata la media europea")
        )
    ).transform_filter(
        alt.datum.energy_cons_per_capita > alt.datum.europe_mean
    )

    # Creo le etichette per i valori e per i nomi degli stati
    text_energy_cons = bar.mark_text(align='left', dx=30, fontSize=14).encode(
        text='energy_cons_per_capita:Q',
        # color = "state:N"
        color = alt.Color("state:N", scale=alt.Scale(scheme="tableau10")),
    )
    text_state = bar.mark_text(align='left', dx=12, fontSize=14).encode(
        text='state:N',
        # color = "state:N"
        color = alt.Color("state:N", scale=alt.Scale(scheme="tableau10")).legend(orient="top"),
    )
    # Creo il grafico
    layered_chart = alt.layer(bar, text_energy_cons, text_state, highlight, xrule_EU
    ).properties(
        width=650, height=80)



    # Creo il grafico facendo il facet per i vari tipi di consumo di energia
    faceted_chart = layered_chart.facet(
        row=alt.Row('nrg_bal:N', title="Tipo di Settore di consumo")
    )
    faceted_chart 
    
    # Ritorno il DataFrame con i nuovi dati predetti del consumo 
    # return df_input    
  
## Barchart Classifica Deficit ################################################################################
def barchart_classifica(df_input, year):
    # Creazione del DataFrame che verrà utilizzato per il grafico
    df_input = df_input.filter(
        pl.col("date") == year,
        pl.col("def_id") == "TOTAL;FC"
        # Elimino valori doppioni
        ).group_by("state").agg(
            pl.col("deficit").mean().alias("deficit")
        )
    # Creazione del DataFrame per la media di consumo medio di energia in Europa
    df_EU = df_input.filter(pl.col("state") == "EU27_2020"
        ).with_columns(
            deficit= pl.col("deficit") // 27,
            deficit_status= pl.lit("EU")
        )

    # Elimino i dati relativi all'Unione Europea
    # Aggiungo una nuova colonna chiamata deficit status che mi servirà per colorare il grafico
    df_input = df_input.filter(pl.col("state") != "EU27_2020").with_columns(
        pl.when(pl.col("deficit") > 0)
        .then(pl.lit("Maggiore"))
        .otherwise(pl.lit("Minore"))
        .alias("deficit_status"))
    
    # Unisco i due DataFrame
    df_input = pl.concat([df_input, df_EU]).with_columns(
            state = pl.col("state")
            .str.replace("EU27_2020","EU")
        )
    # Definizione di var passaggi intermedi
    # Definisco la palette dei colori
    color_palette = alt.Scale(
        domain=["Maggiore", "EU", "Minore"],
        range=["#117733", "#003399", "#5F021F"]
    )
    # predicate = alt.datum.deficit > 0
    # color = alt.when(predicate).then(alt.value("#117733")).otherwise(alt.value("#5F021F")) # In alternativa: #00FF00 e #D81B60

    select = alt.selection_point(name="select", on="click")
    highlight = alt.selection_point(name="highlight", on="pointerover", empty=False)

    # Definizione dello stroke width, quando selezionato lo stato avrà uno stroke più spesso
    stroke_width = (
        alt.when(select).then(alt.value(2, empty=False))
        .when(highlight).then(alt.value(1))
        .otherwise(alt.value(0))
    )

    # Grafico di base
    base = alt.Chart(df_input).mark_bar(
        stroke="black", cursor="pointer"
    ).encode(
        y=alt.Y("state", sort="-x", axis=None),#.axis(labels=False, ),
        x=alt.X("deficit:Q", title=None).axis(labels=False),
        color=alt.Color("deficit_status:N", scale=color_palette,legend=None),
        fillOpacity=alt.when(select).then(alt.value(1)).otherwise(alt.value(0.3)),
        strokeWidth=stroke_width,
    ).add_params(select, highlight).properties(width=720)

    # Etichette che indicano lo stato e il deficit
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

# Definisco le tre pagine
def page_deficit():
    st.title("Evoluzione del Deficit/Surplus di energia elettrica in Europa")
    st.write(f"""    
    In questa pagina vengono analizzati i dati sul deficit/surplus di elettricità in Europa.
    Si vuole evidenziare come i vari stati producono e consumano elettricità.
    L'analisi è stata effettuata prendendo i dati di [produzione](https://ec.europa.eu/eurostat/databrowser/product/view/nrg_cb_pem?category=nrg.nrg_quant.nrg_quantm.nrg_cb_m) 
                                                    e [consumo](https://ec.europa.eu/eurostat/databrowser/view/nrg_cb_e__custom_15038169/default/table?lang=en) di elettricità dal sito Eurostat. 
    """)    
    df_comb = cast_int_deficit(df_A)
    st.divider()
    year_int, year = select_year(df_comb)
    
    st.divider()
    st.write(f"""### Mappa del deficit/surplus di elettricità in Europa nel {year_int}.""")
    st.write(f"""La mappa va dai colori più scuri degli stati che hanno un surplus di elettricità,
    (ovvero stati che producono più elettricità di quanta ne consumano) ai colori più 
    chiari, ovvero gli stati in deficit (stati che consumano più elettricità di quanta
    ne producono). Si può notare come alcuni paesi (Francia, Svezia e Spagna su tutti) siano quasi sempre quelli con il surplus maggiore. 
    Al contrario, stabile nelle nazioni con un grosso deficit c'è l'Italia. 
    Però la nazione più interessante è sicuramente la Germania. Andando avanti nel tempo si può 
    osservare il veloce declino del suo surplus energetico accentuato probabilmente dal conflitto russo-ucraino.
                      """)
    last_date(df_comb, year)
    mappa(df_comb, year, "TOTAL;FC")

    st.write(f"""### Evoluzione del deficit/surplus di elettricità di vari stati europei.""")
    st.write(f"""Si passa ora ad osservare come varia la situazione nel tempo. 
    Si può visualizzare un singolo stato o si possono selezionare più stati contemporaneamente, 
    così da poter confrontare la situazione tra i vari stati. Il tooltip verticale permette una veloce lettura dei dati. 
    """)
    line_chart_deficit(df_comb)

    st.write(f"""### Classifica del deficit/surplus di elettricità in Europa nel {year_int}.""")
    st.write(f"""Classifica del deficit/surplus di elettricità che mostra
    quali stati dell'Unione Europea sono in positivo e quali in negativo. 
    Viene anche mostrato il valore medio per l'intera UE. È possibile selezionare uno stato in particolare cliccandoci sopra.""")
    last_date(df_cons_pred_A, year)
    barchart_classifica(df_comb, year)
    st.write(f"""### Evolzuione dei primi 5 stati per surplus di elettricità in Europa.""")
    bump_chart(df_comb)
    st.write(f"""Nell'ultimo grafico le nazioni con il surplus più grande vengono visualizzate in classifica per ogni anno. 
             Vengono confermate le osservazioni della mappa. Stati come Francia, Svezia e Spagna occupano stabilmente le prime posizioni.
             Interessante notare come queste tre nazioni molto diverse tra di loro occupino i primi posti. Da un lato abbiamo un paese come la Francia che ha dalla sua 
             un'immensa forza lavoro e una produzione di energia elettrica molto elevata dovuta soprattutto al nucleare. Poi c'è una nazione come la Spagna, 
             una nazione grande, ma mai considerata come una grande potenza economica che però ha dalla sua un territorio molto vasto e un entroterra, tolte un paio di città, 
             quasi totalmente disabitato. Infine abbiamo la Svezia. Una nazione relativamente vasta, ma forte di un'importante produzione di idroelettrico,
              che le permette agevolmente di superare il proprio fabbisogno.  
             """)
    
    
def page_production():
    st.title("Produzione di energia elettrica in Europa")
    st.write(f"""In questa pagina è possibile analizzare la produzione di energia in Europa. Le fonti di energia disponibili sono 15 tra fonti rinnovabili, non rinnovabili, nucleare e altre fonti.
    Nella prima parte della pagina sarà possibile studiare l'evoluzione della produzione di energia elettrica per tutti gli stati membri dell'unione. Nella seconda parte ci si potrà concentrare su un singolo stato.
    È possibile filtrare l'analisi selezionando un tipo di fonte di energia che si vuole analizzare nello specifico o si può lasciare l'opzione di default (produzione totale). Si può anche selezionare un anno specifico. 
    """)
    st.divider() 
    selected_siec = select_type(df_prod)
    df_prod_pred = pred_siec(selected_siec).filter(pl.col("date") >= pl.datetime(2017, 1, 1))
    df_prod_pred_A = df_from_M_to_A(df_prod_pred).filter(pl.col("date") >= pl.datetime(2017, 1, 1))
    df_prod_pred_total = df_A.filter(pl.col("siec") == "TOTAL")
    year_int, year = select_year(df_prod_pred_A)
    st.divider() 
    # with my_expander:
    
    st.write(f"""### Analisi della produzione energetica in Europa nell'anno {year_int}.""")
    st.write(f"""In questa prima parte si può osservare tramite una mappa la quantità di energia elettrica prodotta da ogni stato europeo.
    È possibile cambiare la fonte tramite lo slide ad inizio pagina. Da questa mappa si possono già fare delle osservazioni sulla produzione di energia. Ad esempio, la dipendenza di alcuni stati dalle fonti non rinnovabili e, nel dettaglio, si può notare il divario di produzione tra i paesi dell'Est e i paesi dell'Ovest in merito alla produzione di energia rinnovabile. 
    """)
    last_date(df_prod_pred_A, year)
    mappa(df_prod_pred_A, year, selected_siec)
    st.write(f"""### Evoluzione della produzione energetica da {selected_siec} annuale in Europa.""")
    st.write(f"""Dopo aver osservato quanto produce uno stato in un determinato anno, è possibile osservare l'evolversi di tale produzione in Unione Europea a partire dal 2017.
    Il grafico permette anche la selezione più stati europei in contemporanea così da potermettere un confronto tra di loro e con la media europea, presente in grigio nel grafico (attenzione alle due scale diverse). 
    L'energia visualizzata è quella selezionata nello slide ad inizio pagina.     
    """)

    selected_multi_state = select_multi_state(df_prod_pred_A, filter=selected_siec,EU= False)
    line_chart_prod(df_prod_pred, selected_multi_state, selected_siec)
    
    st.write(f"""
    Prima di spostarci al grafico successivo, vorrei analizzare velocemente alcuni picchi di produzione di energia.
    Interessante è il confronto tra gli stati che producono energia solo da fossili e rinnovabili (come Italia e Spagna) e stati che usano anche fonti alternative come il nucleare (ad esempio Francia e Svezia).
    In questi ultimi notiamo che il picco di produzione ha una cadenza annuale e si concentra nei mesi invernali (dove certamente si ha bisogno di più energia). Al contrario in stati come Spagna e Italia questi picchi sembrano essere più semestrali, con il picco più alto che si raggiunge nei mesi estivi (dove il clima è più favorevole alla produzione di energia rinnovabile). 
    Le cause di questo trend sono sicuramente numerose (come ad esempio il diverso clima), però sicuramente la capacità di produrre energia senza dipendere da eventi meteorologici permette ad alcuni stati di concentrare la loro produzione nei mesi in cui se ne ha più bisogno. 
    
    """)
    st.divider() 
    st.write(f"""## Analisi della produzione in un singolo stato.""")
    st.write(f"""Nella seconda parte della pagina è possibile concentrarsi sulla produzione energetica di un singolos tato europeo. 
             In questa parte verranno visualizzati due grafici che ci auteranno a comprendere come varia nel corso degli anni la produzione 
             e quanto ogni singola fonte inficia nel totale della produzione di uno stato.    
    """)
    st.divider()
    selected_single_state = select_state()
    st.divider()
    st.write(f"""### Evoluzione della produzione di energia elettrica in {selected_single_state}""")
    st.write(f"""Ora possiamo osservare come è variata, e come si preveda che vari, la produzione di energia elettrica in un singolo stato europeo.
             Per permettere una più facile lettura dei dati il grafico mostra solo le macrocategorie di fonti energetiche (rinnovabili, non rinnovabili, nucleare e altro). Inoltre è implementato un tooltip (attivabile con il passaggio del mouse sul grafico) che permette la lettura immediata della quantità di energia prodotta misurata in GWH.
    """)

    df_prod_pred_updated = area_chart(df_prod_pred, df_prod_pred_total, selected_single_state, prod_list)
    
    st.write(f"""Nonostante l'intervallo temporale sia piccolo, questo grafico è particolarmente interessante perché ci permette di vedere come, in questo periodo di mutamenti, ogni stato stia cambiando il proprio modo di produrre energia.
             Ad esempio, possiamo vedere come la Germania si è adattata a produrre energia dopo la chiusura dei propri reattori nucleari. 
             O, al contrario, si può osservare la Francia che, grazie proprio al nucleare e stando alle previsioni dell'ARIMA, nei prossimi anni sembrerebbe diventare quasi totalmente indipendente dal fossile. 
             Infine, si può osservare come un paese dell'est Europa come la Polonia, fortemente dipendente dal carbone, stia piano piano cambiando il proprio modo di produrre energia, aumentando la sua quota di energia prodotta da fonti rinnovabili.    
    """)

    st.write(f"""### Analisi della produzione di energia elettrica in {selected_single_state} nell'anno {year_int}""")
    st.write(f"""Ora possiamo osservare come è variata, e come si prevede che vari, la produzione di energia elettrica in un singolo stato europeo.
             Per permettere una più facile lettura dei dati il grafico mostra solo le macrocategorie di fonti energetiche (rinnovabili, non rinnovabili, nucleare e altro). Inoltre è implementato un tooltip (attivabile con il passaggio del mouse sul grafico) che permette la lettura immediata della quantità di energia prodotta misurata in GWH.
    """)
    bar_chart_with_db(df_prod_pred_updated, df_prod_pred_total,selected_single_state, prod_list_2, year)


def page_consumption():
    st.title("Consumo di energia elettrica in Europa")
    st.write(f"""In questa pagina è possibile analizzare il consumo di energia in Europa. I tipi di consumo di energia elettrica analizzati sono 5 e rappresentano il settore industriale, dei trasporti, agricolo e forestale, dei servizi commerciali e pubblici e il consumo domestico.
    Nella prima parte della pagina sarà possibile studiare l'evoluzione di tale consumo con la possibilità di concentrarsi prima sul singolo consumo e poi su tutti con la possibilità di fare anche un confronto. 
    Nella seconda parte ci si potrà concentrare invece su un singolo stato.
    È possibile filtrare l'analisi selezionando un tipo di consumo di energia che si vuole analizzare nello specifico o si può lasciare l'opzione di default (consumo totale). Si può anche selezionare un anno specifico. 
    """)

    st.divider() 
    selected_nrg_bal = select_type(df_cons)
    df_cons_pred = pred_cons(selected_nrg_bal)
    df_cons_pred_A = df_from_M_to_A(df_cons_pred)
    year_int, year = select_year(df_cons_pred_A)
    st.divider() 
    st.write(f"""### Analisi del consumo energetico in Europa nell'anno {year_int}.""")
    st.write(f"""In questa prima parte si può osservare tramite una mappa la quantità di energia elettrica consumata da ogni stato europeo in un determinato anno.
    È possibile cambiare il tipo di settore di consumo tramite lo slide ad inizio pagina.
    """)
    last_date(df_cons_pred_A, year)
    mappa(df_cons_pred_A, year, selected_nrg_bal)    
    st.write(f"""
    La mappa mostra quello che ci si potrebbe aspettare: i paesi più grandi e popolosi come Italia, Francia, Germania e Spagna sono quelli che consumano più energia.
    Un dato interessante lo si osserva se ci si sposta nel consumo relativo al settore agricolo. 
    Si nota subito infatti l'alto consumo dei Paesi Bassi che in alcuni anni riesce ad imporsi come lo stato che consuma più energia elettrica per il settore agricolo.
    """)
    st.write(f"""### Evoluzione del consumo energetico pro-capite in Europa nell'anno {year_int}.""")
    st.write(f"""Dopo aver osservato quanto consuma in assoluto uno stato in un determinato anno, ci concentriamo sul consumo pro-capite. 
    Il consumo pro-capite ci permette anche di fare un confronto tra i vari stati europei in quanto ci permette di staccarci dal valore assoluto 
    e di vedere quanto inficiano i settori di consumo analizzati sul singolo cittadino. In particolare ci interessa osservare i valori sopra la media europea. 
    Per fare ciò nel seguente grafico tutti i valori sopra la media dell'UE (linea tratteggiata), sono evidenziati in rosso. 
    Accanto ad ogni barra è inoltre presente l'etichetta dello stato e il valore di consumo pro-capite.
    """)

    bar_chart_cons(df_cons_pred_A, df_cons_pred, year, list_consuption) 
    
    st.write(f"""
    Sfogliando tra gli stati si possono osservare alcuni valori nella norma e altri un po' sorprendenti. 
    Ad esempio, come ci si poteva aspettare, la Germania sfora la media in ben 4 categorie su 5, 
    mentre la Francia, nonostante un importante apparato industriale e agricolo, sfora solo nei settori legati alla vita del cittadino, come ad esempio i servizi commerciali e pubblici e nel dato sulle famiglie.
    Non sorprendono neanche i valori elevati nel settore industriale di Belgio e Paesi Bassi che sono paesi piccoli ma densamente popolati.
    I dati forse più sorprendenti arrivano dai paesi nordici: a seconda dell'anno che si osserva, mediamente Svezia e Finlandia superano la media europea 4 volte su 5 o addirittura in tutte le categorie. 
    In particolare, se si va a riprendere il dato sul deficit/surplus, si può notare come la Svezia sia stabilmente in surplus (e anche di molto). 
    Quindi si può dire che, tenendo conto di una popolazione non troppo numerosa e di un territorio comunque vasto e ricco di risorse, ogni cittadino svedese consuma molto di più di un altro cittadino europeo, ma allo stesso tempo riesce anche a produrre abbastanza energia (soprattutto in gran parte da rinnovabili), da rimediare a questo consumo così elevato. 
    """)         
    st.divider() 
    st.write(f"""## Analisi del consumo in un singolo stato.""")
    st.write(f"""Nella seconda parte della pagina è possibile concentrarsi sul consumo energetico di un singolo stato europeo. 
             In questa parte verranno visualizzati due grafici che ci aiuteranno a comprendere come varia nel corso degli anni il consumo, 
             quanto una nazione consuma in ogni settore e in che percentuale si divide il consumo in ogni settore rispetto al consumo totale.    
    """)
    st.divider()
    selected_single_state = select_state()
    st.divider()

    st.write(f"""### Evoluzione del consumo di energia elettrica in {selected_single_state}""")
    st.write(f"""Ora possiamo osservare l'andamento passato e futuro del consumo di energia elettrica in un singolo stato europeo.
            Il consumo è stato diviso nelle 5 categorie in analisi e, per comprendere meglio l'incertezza che ci può essere dietro all'analisi, 
             sono stati aggiunti gli intervalli di confidenza per ogni previsione sul consumo. 
             Il grafico è stato reso anche interattivo così da permettere di evidenziare un periodo o alcuni settori nello specifico. 
                 """)

    df_cons_pred_updated = line_chart_with_IC(df_cons_pred, selected_single_state)
    
    st.write(f"""In questo grafico le conclusioni possono essere varie, alla fine ogni stato ha la sua storia. Certamente, come ci si può aspettare essendo 
             l'UE una unione di paesi "avanzati", la maggior parte del consumo è divisa nel settore secondario e terziario come servizi, industria, commercio, ecc..., 
             mentre altri settori come quello agricolo sono molto ridotti anche in paesi che comunque mantengono un forte settore (come Francia e Italia).
              """)
    st.write(f"""### Distribuzione del consumo energetico pro-capite in {selected_single_state} nell'anno {year_int}.""")
    st.write(f"""
        Infine viene valutata la fetta che ogni settore di consumo mostrato nel grafico sovrastante occupa sul consumo totale. 
        Per fare ciò è stato implementato il seguente grafico a torta corredato da una tabella con la produzione assoluta e la percentuale che rappresenta sul totale.
    """)
    selected_nrg_bal = "FC"
    pie_chart(df_cons_pred_updated, selected_single_state, year)

# Visualizzo il pdf, purtroppo la libreria è ancora nuova e la visualizzazione non è perfetta, ma almeno è qualcosa
# Non volevo usare un markdown perchè sennò sarebbero scomparse le immagini.
def page_doc():
    # st.title("Documentazione")
    pdf_viewer("documentazione.pdf")

pg = st.navigation([
    st.Page(page_deficit, title="Deficit/Surplus"),
    st.Page(page_production, title="Produzione"),
    st.Page(page_consumption, title="Consumo"),
    st.Page(page_doc, title = "Documentazione" )
])
pg.run()