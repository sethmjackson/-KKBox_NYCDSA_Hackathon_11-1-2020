import pandas as pd
import Seth.Util as ut

datasetDir = 'Seth/Dataset/'

def readDF():
    data = pd.read_csv(datasetDir + 'Data (Imputed Age).csv')
    return data

def writeDF(df: pd.DataFrame):
    df.to_csv(datasetDir + 'Data Imputed.csv', index=False)
    return df

def mergeDFs():

    churn = pd.read_csv(datasetDir + 'trunc_churn.csv')
    members = pd.read_csv(datasetDir + 'trunc_members.csv')
    transactions = pd.read_csv(datasetDir + 'trunc_transaction.csv')
    users = pd.read_csv(datasetDir + 'trunc_users.csv')

    data = churn.merge(members, on='msno').merge(transactions, on='msno').merge(users, on='msno')
    data.to_csv(datasetDir + 'Data.csv', index=False)

def processDF(df: pd.DataFrame):
    ## deal with null values
    #print(ut.getNullPercents(df))

    ## remove unnecessary columns
    # msno registration_init_time transaction_date membership_expire_date date gender
    df.drop(columns=['msno', 'registration_init_time', 'transaction_date',
                     'membership_expire_date', 'date', 'gender', 'payment_plan_days'], inplace=True)
    #print(ut.getNullPercents(df))

    ## alter columns

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

    writeDF(df)
    print('End Ages == 0: ' + str(ut.getRowsNum(df[df['bd'] == 0])))
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




