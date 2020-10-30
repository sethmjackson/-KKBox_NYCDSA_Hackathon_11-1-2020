import pandas as pd
import Seth.Util as ut
from statsmodels.stats.outliers_influence import variance_inflation_factor

datasetDir = 'Seth/Dataset/'

def readDF(fileName: str):
    data = pd.read_csv(datasetDir + fileName)
    return data

def writeDF(df: pd.DataFrame, fileName: str):
    df.to_csv(datasetDir + fileName, index=False)
    return df

def mergeDFs():

    churn = pd.read_csv(datasetDir + 'trunc_churn.csv')
    members = pd.read_csv(datasetDir + 'trunc_members.csv')
    transactions = pd.read_csv(datasetDir + 'trunc_transaction.csv')
    users = pd.read_csv(datasetDir + 'trunc_users.csv')

    users = users.drop(columns=['date']).groupby('msno').mean()
    data = churn.merge(members, on='msno').merge(transactions, on='msno').merge(users, on='msno')
    writeDF(data, 'Data.csv')
    return data

def processDF(df: pd.DataFrame):
    ## deal with null values
    #print(ut.getNullPercents(df))

    ## remove unnecessary columns
    # msno registration_init_time transaction_date membership_expire_date date gender
    ut.dropIfExists(df, columns=['msno', 'registration_init_time', 'transaction_date',
                     'membership_expire_date', 'date', 'gender'], inplace=True)
    #print(ut.getNullPercents(df))

    ## alter columns
    removeColumnsByVif(df, 10)
    print('DF Processed')

    return df

def imputeDF(df: pd.DataFrame):
    print('Start Ages == 0: ' + str(ut.getRowsNum(df[df['bd'] == 0])))
    ages = df[df['bd'] > 0]['bd']
    noAgeIndexes = df[df['bd'] == 0].index
    noAgeNum = len(noAgeIndexes)
    i=0

    for index in noAgeIndexes:
        age = ages.sample(1)
        df.loc[index, 'bd'] = age.iloc[0]
        if i%1000 == 0:
            print(str(i))
        i+=1

    print('End Ages == 0: ' + str(ut.getRowsNum(df[df['bd'] == 0])))
    writeDF(df, 'Data (Imputed Age).csv')
    return df

def addColumns(df: pd.DataFrame):
    df['pop'] = pd.cut(df['city'], bins=[0, 1, 22], labels=['high', 'low'])

    def generation(x):
        if x < 21:
            return 'genz'
        elif x < 38:
            return 'millennial'
        elif x < 53:
            return 'genx'
        else:
            return 'boomer'

    df['gen'] = df['bd'].apply(generation)

    df = pd.get_dummies(df, columns=['gen', 'pop'], drop_first=True)
    return df



def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

def removeColumnsByVif(df: pd.DataFrame, vifCutoff = 5):
    #ut.printNulls(modDF)
    #nulls = ut.getNulls(modDF)
    vif = calc_vif(df)
    mcColumns = vif[vif['VIF'] > vifCutoff]['variables']
    df.drop(columns=mcColumns, inplace=True)
    vif = vif[vif['VIF'] <= vifCutoff]
    return vif
