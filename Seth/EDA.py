import pandas as pd

datasetDir = 'Dataset/'
data = pd.read_csv(datasetDir + 'Data.csv')
def mergeDFs():
    churn = pd.read_csv(datasetDir + 'trunc_churn.csv')
    members = pd.read_csv(datasetDir + 'trunc_members.csv')
    transactions = pd.read_csv(datasetDir + 'trunc_transaction.csv')
    users = pd.read_csv(datasetDir + 'trunc_users.csv')

    data = churn.merge(members, on='msno').merge(transactions, on='msno').merge(users, on='msno')
    data.to_csv(datasetDir + 'Data.csv', index=False)

print('finished')