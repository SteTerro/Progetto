import streamlit as st
import polars as pl
import altair as alt
import country_converter as cc

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, AutoCES
from statsforecast.models import SeasonalNaive
from sklearn.utils import resample
import numpy as np

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
            unique_id=pl.col("state")+";"+pl.col("siec")
        )
        .filter(
            pl.col("energy_prod") > 0,
            pl.col("energy_prod").is_not_null(),
            pl.col("energy_prod") != 123456789,
            pl.col("unit") == "GWH")
        .drop("freq")
        .drop("unit")
        .filter(
            pl.col("siec").is_in(["TOTAL","X9900","RA000","N9000","CF","CF_R","RA100","RA200","RA300","RA400","RA500_5160","C0000","CF_NR","G3000","O4000XBIO"]),
            )
        .sort("state", "date") 
    )
    return data


# which_country = st.checkbox("Includi i paesi non membri dell'UE", value=False)
df_prod = get_data_productivity(url_productivity)
df_cons = get_data_consumption(url_consumption)
# df_comb = pl.concat([df_prod, df_cons], how = "align")

# df_prod_A = (
#     df_prod.group_by(["unique_id", "date"]).agg(
#         pl.col("energy_prod").sum().alias("energy_prod")
#     ).with_columns(
#         pl.col("unique_id")
#             .str.split(";")
#             .list.to_struct(fields=["state", "siec"])
#             .alias("combined_info"),
#         ).unnest("combined_info").drop("unique_id")
#     )

df_prod_A = (
    df_prod.with_columns(
        year=pl.col("date").dt.year())
    .group_by(["state", "year", "unique_id", "siec"])
    .agg(energy_prod=pl.sum("energy_prod"))
    .sort(["state", "year"])
).with_columns(date=pl.col("year").cast(pl.Utf8).str.to_date(format='%Y')).drop("year")


df_comb = df_prod_A.join(df_cons, on = ["state", "date"], how = "inner")

cc_2 = cc.CountryConverter()
country_list = pl.from_pandas(cc_2.EU27as('ISO2'))
country_list = country_list.select(pl.col("ISO2")).to_series()

list_consuption = ["FC", "FC_IND_E" , "FC_TRA_E", "FC_OTH_CP_E", "FC_OTH_HH_E", "FC_OTH_AF_E"]

list_productivity = ["TOTAL","X9900","RA000","N9000","CF","CF_R","RA100","RA200","RA300","RA400","RA500_5160","C0000","CF_NR","G3000","O4000XBIO"]
Tot_list = ["TOTAL","X9900","RA000","N9000","CF"]
RA_list = ["CF_R","RA100","RA200","RA300","RA400","RA500_5160"]
CF_list = ["C0000","CF_NR","G3000","O4000XBIO"]

EU27_2020_mom = df_comb.filter(pl.col("state") == "EU27_2020")
df = df_comb.filter(pl.col("state").is_in(country_list))
df = pl.concat([df, EU27_2020_mom])

EU27_2020_mom_2 = df_prod.filter(pl.col("state") == "EU27_2020")
df_prod = df_prod.filter(pl.col("state").is_in(country_list))
df_prod = pl.concat([df_prod, EU27_2020_mom_2])

EU27_2020_mom_3 = df_cons.filter(pl.col("state") == "EU27_2020")
df_cons = df_cons.filter(pl.col("state").is_in(country_list))
df_cons = pl.concat([df_cons, EU27_2020_mom_3])

df_A = df.with_columns(
    deficit = 
    (pl.when(pl.col("siec") == "TOTAL")
     .then(pl.col("energy_prod"))
     .otherwise(pl.lit(0))
     - pl.when(pl.col("nrg_bal") == "FC")
     .then(pl.col("energy_cons"))
     .otherwise(pl.lit(0))),
     def_id=pl.col("siec")+";"+pl.col("nrg_bal")
)

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
        fallback_model=SeasonalNaive(season_length=12)
        )
    ts_pred = sf.forecast(df=ts, h=48, level=[95]) #h = 48
    # sf.fit(df=Y_train_df)
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

    season_length = 1
    models = [
        AutoARIMA(season_length=season_length), # ARIMA model with automatic order selection and seasonal component
        AutoETS(season_length=season_length), # ETS model with automatic error, trend, and seasonal component
        AutoTheta(season_length=season_length), # Theta model with automatic seasonality detection
        AutoCES(season_length=season_length), # CES model with automatic seasonality detection
    ]
    
    sf = StatsForecast(
        #models = [AutoARIMA(season_length = 1)],
        models=models,
        freq = '1y',
        n_jobs=-1,
        fallback_model=SeasonalNaive(season_length=1)
        )
    ts_pred = sf.forecast(df=ts, h=5, level=[95])
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
        ).sort("AutoARIMA_hi90", "AutoARIMA_low90", "date", "energy_cons", "nrg_bal","predicted", "state","unique_id")  
    
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

    df_funz = df_funz.select(sorted(df_funz.columns))
    pred = pred.select(sorted(pred.columns))

    df_combined = pl.concat([df_funz, pred], how= "vertical_relaxed")
    return df_combined

df_A = df_A.filter(pl.col("def_id") == "TOTAL;FC").drop("energy_prod","siec","nrg_bal","energy_cons","unique_id")
st.write(df_A)

# df_1 = pred_siec("TOTAL")
df_2 = pred_cons("FC")

st.write(df_2)

# df_1_2 = (
#     df_1.with_columns(
#         year=pl.col("date").dt.year())
#     .group_by(["state", "year", "unique_id", "siec"])
#     .agg(energy_prod=pl.sum("energy_prod"))
#     .sort(["state", "year"])
# ).with_columns(date=pl.col("year").cast(pl.Utf8).str.to_date(format='%Y')).drop("year")

# df_comb_2 = df_1_2.join(df_2, on = ["state", "date"], how = "inner")

# df_A_2 = df_comb_2.with_columns(
#     deficit = 
#     (pl.when(pl.col("siec") == "TOTAL")
#      .then(pl.col("energy_prod"))
#      #.otherwise(pl.lit(0))
#      - pl.when(pl.col("nrg_bal") == "FC")
#      .then(pl.col("energy_cons"))),
#      #.otherwise(pl.lit(0),
#      def_id=pl.col("siec")+";"+pl.col("nrg_bal"))

country = st.selectbox(
        "Seleziona uno stato",
        df.select("state").unique().sort("state")
        )

# adattamento_mape = adattamento.select(["state", "siec", pl.col("metric")=="mape", "AutoARIMA"]).filter(pl.col("metric") == True)

# csv_file_path = "C:/Users/Stefa/Downloads/adattamento_mape.csv"
# adattamento_mape.write_csv(csv_file_path)

# st.write(f"File CSV creato: {csv_file_path}")
# Y_test_df = Y_test_df.with_columns(
#             pl.col("unique_id")
#                 .str.split(";")
#                 .list.to_struct(fields=["state", "nrg_bal"])
#                 .alias("combined_info"),
#         ).unnest("combined_info")

# st.write(df_A_2)
# st.write(df_A_2.filter(pl.col("state") == country))
# line = alt.Chart(df_A_2.filter(pl.col("state") == country, pl.col("def_id") == "TOTAL;FC")
# ).mark_line(color="red").encode(
#     x='date',
#     y='deficit',
# ).properties(
#     width=300, height=300)

line_4 = alt.Chart(df_2.filter(pl.col("state") == country, pl.col("nrg_bal") == "FC", pl.col("predicted") == False)
).mark_line(color="green").encode(
    x='date',
    y='energy_cons'
).properties(
    width=800, height=400)

line_6 = alt.Chart(df_2.filter(pl.col("state") == country, pl.col("predicted") == True)
).mark_line(color="red").encode(
    x='date',
    y='energy_cons'
).properties(
    width=800, height=400)

# st.write("""A sinistra prima calcolati le predizioni e poi fatta la differenza, a destra forecast di deficit""")
# st.altair_chart(line | line_2)
# # line_3
# line + line_2
line_4 + line_6