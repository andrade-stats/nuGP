
import pandas as pd
import numpy as np
import commonSettings

BASE_FOLDER = "openDatasets_raw/"

# NMES1988 data from R package "AER"
# used here as in also used in "Regression Models for Count Data in R", 
# after preprocessing, X contains 6 features (hosp (number of hospital stays), health (self-perceived health status), numchron (number of chronic conditions), gender, school (number of years of education), and privins (privateinsurance indicator) 
# y = number of physician oﬃce visits (ofp)
def prepare_NMES_data():
    
    filename_raw = BASE_FOLDER + "NMES.csv"
    
    df_all = pd.read_csv(filename_raw)
    
    # remove date, year, and casual and registered (summing both equals total count "cnt")
    df_all = df_all.drop(columns = ["Unnamed: 0", "nvisits", "ovisits", "novisits", "emergency", "adl", "region", "age", "afam", "married", "income", "employed", "medicaid"])

    df_all = df_all.replace({'health': {'poor': -1, 'average': 0, "excellent":1}})
    df_all = df_all.replace({'gender': {'male': 0, 'female': 1}})
    df_all = df_all.replace({'insurance': {'no': 0, 'yes': 1}})

    assert(np.all(df_all.notna().to_numpy()))
    
    # get counts as y and rest as X
    y = df_all["visits"].to_numpy()
    df_all = df_all.drop(columns = "visits")

    X = df_all.to_numpy()

    print("X.shape = ", X.shape)
    save_original(X, y, datasetName = "NMES")
    return


# bike sharing data from https://archive.ics.uci.edu/ml/index.php
# also used in "Informative Bayesian Neural Network Priors for Weak Signals", Bayesian Analysis, 2022 
# after preprocessing, X contains 12 features
# y = count of total rental bikes including both casual and registered
def prepare_bike_sharing_data(unit):
    assert(unit == "hour" or unit == "day")

    filename_raw = BASE_FOLDER + f"bike+sharing+dataset/{unit}.csv"
    
    df_all = pd.read_csv(filename_raw)

    # add time (here hours after first measurement)
    df_all = df_all.rename(columns={"instant":"t"})

    
    # remove date, year, and casual and registered (summing both equals total count "cnt")
    df_all = df_all.drop(columns = ["dteday", "yr", "casual", "registered"])

    df_all["previous_hr_cnt"]=np.nan
    print("df_all = ")
    print(df_all.head(100).to_string(index=False))
    
    for t in df_all["t"]:
        current_row = df_all[df_all["t"] == t]
        if current_row["hr"].item() > 0:
            previous_value = ((df_all[df_all["t"] == t-1])["cnt"]).item()
            df_all.loc[df_all["t"] == t, "previous_hr_cnt"] = previous_value
            # print("currnet = ", current_row["hr"].item() - 1)
            # print("prev = ", ((df_all[df_all["t"] == t-1])["hr"]).item())
            # assert(((df_all[df_all["t"] == t-1])["hr"]).item() == current_row["hr"].item() - 1)

    print("df_all = ")
    print(df_all.head(100).to_string(index=False))
    
    df_all = df_all.dropna()

    print("df_all = ")
    print(df_all.head(100).to_string(index=False))
    
    df_all = df_all.drop(columns = ["t", "season", "mnth"])

    print("df_all = ")
    print(df_all.head(100).to_string(index=False))
    # assert(False)

    # get counts as y and rest as X
    y = df_all["cnt"].to_numpy()
    df_all = df_all.drop(columns = "cnt")

    X = df_all.to_numpy()

    print("X.shape = ", X.shape)
    
    save_original(X, y, datasetName = "bike_sharing_" + unit)
    
    return


# short description:
# after preprocessing, X contains 15 features (including t = day)
# t  Sunday  Monday  CosAnnual     SinAnnual        H7    NO2max  T1.1990  T2.1990  T1.1991  T2.1991  T1.1992  T2.1992  T1.1993  T2.1993
# and y = Daily counts of asthma at Campbelltown Hospital.
# data extracted from library(glarma) R-package
def prepare_asthma_data():
    
    filename_raw = BASE_FOLDER + "asthma.csv"
    
    df_all = pd.read_csv(filename_raw)

    # add time (here day of year, see Fig 2 in "Observation-driven models for Poisson counts")
    df_all = df_all.rename(columns={"Unnamed: 0":"t"})

    # remove intercept
    df_all = df_all.drop(columns = "Intercept")

    # get counts as y and rest as X
    y = df_all["Count"].to_numpy()
    df_all = df_all.drop(columns = "Count")

    X = df_all.to_numpy()
    
    print(df_all)

    save_original(X, y, datasetName = "asthma")
    return

# description see 
# https://r-packages.io/datasets/bioChemists
# X.shape =  (n = 915, d = 5)
def prepare_bioChemists_data():
    
    filename_raw = BASE_FOLDER + "bioChemists.csv"
    
    df_all = pd.read_csv(filename_raw)

    df_all = df_all.drop(columns = "Unnamed: 0")

    # get counts as y and rest as X
    y = df_all["art"].to_numpy()
    df_all = df_all.drop(columns = "art")

    df_all = df_all.replace({'fem': {'Men': 0, 'Women': 1}})
    df_all = df_all.replace({'mar': {'Married': 0, 'Single': 1}})

    df_all['fem'] = df_all['fem'].astype('int64')
    df_all['mar'] = df_all['mar'].astype('int64')

    X = df_all.to_numpy()
    
    save_original(X, y, datasetName = "bioChemists")
    return


# see description in folder below (Full details of this project can be found at <https://predict.cdc.gov> and <http://dengueforecasting.noaa.gov>.)
def prepare_dengue_data(location):
    assert(location == "san_juan" or location == "iquitos")

    DENGUE_FOLDER = "dengue-forecasting-project-2015-master/Dengue_data/"

    TRAINING_DATA_FILENAME = BASE_FOLDER + DENGUE_FOLDER + location + "_training_data.csv"
    TEST_DATA_FILENAME = BASE_FOLDER + DENGUE_FOLDER + location + "_testing_data.csv"

    df_training = pd.read_csv(TRAINING_DATA_FILENAME)
    df_all = pd.read_csv(TEST_DATA_FILENAME)

    df_all.insert(0, "sine_wave", 0.0)
    df_all.insert(0, "starting_level", 0)

    print(df_all)

    # ****************** add Sine wave ******************
    for i in range(len(df_all)):
        season_week = df_all.loc[i, "season_week"] - 1
        assert(season_week <= 51)
        df_all.loc[i,"sine_wave"] = np.sin(np.pi * (season_week / 52))

    # ****************** square root transformation for total_cases ******************
    # for i in range(len(df_all)):
    #    df_all.loc[i, "total_cases"] = np.sqrt(df_all.loc[i, "total_cases"])

    # ****************** add Starting level ******************
    for i in range(1, len(df_all)):
        if i == 1 or df_all.loc[i, "season_week"] == 1:
            cases_previous_season = df_all.loc[i - 1, "total_cases"]
            print("cases_previous_season = ", cases_previous_season)
        df_all.loc[i,"starting_level"] = cases_previous_season

    # ****************** remove first row (since we have no starting_level for that one) ******************
    df_all = df_all.drop(0)


    # ****************** split into train and test data
    df_train = df_all.loc[df_all["week_start_date"].isin(df_training["week_start_date"])]
    df_test = df_all.copy()

    for date in df_training["week_start_date"]:
        df_test.drop(df_test[df_test["week_start_date"] == date].index, inplace=True)
        
    # print("df_train = ", df_train)
    # print("df_test = ", df_test)


    # ****************** get numpy arrays and scale X and y ****************
    def getNumpyArrays(df):
        X = df[["season_week", "sine_wave", "starting_level"]].to_numpy()
        y = df["total_cases"].to_numpy()
        return X, y

    X_train, y_train = getNumpyArrays(df_train)
    X_test, y_test = getNumpyArrays(df_test)

    print(X_train.shape)
    print(y_train.shape)

    X = np.concatenate((X_train, X_test), axis = 0)
    y = np.concatenate((y_train, y_test), axis = 0)
    
    # ****************** save  ****************

    datasetName = "dengue" + "_" + location
    save_original(X, y, datasetName)
    return

def save_original(X, y, datasetName):
    print("X = ", X.shape)
    print("y = ", y.shape)

    allData_original = {}
    allData_original["X_original"] = X
    allData_original["y_original"] = y

    np.save(commonSettings.PREPARED_DATA_FOLDER + datasetName,  allData_original)


np.random.seed(3523421)

# prepare_NMES_data()
# prepare_bike_sharing_data(unit = "hour")
# prepare_asthma_data()
# prepare_dengue_data(location = "iquitos")
# prepare_bioChemists_data()

