import pandas as pd
from Seth.DfManip import *
from Seth.Regressions import *
from Seth.Plots import *
data = readDF('Data (Imputed Age).csv')
#data = mergeDFs()
#imputeDF(data)

processDF(data)


data = addColumns(data)
outputColumn = 'is_churn'

continuousColumns = [lambda x: x for x in ['plan_list_price', 'actual_amount_paid', 'total_secs'] if x in data.columns]
data.rename(columns={"bd": "age"}, inplace=True)
models = performRegressions(data, continuousColumns, outputColumn)
plotRegressionData(data, models, outputColumn)




plotChurn(data, 'num_25')
plotChurn(data, 'num_985')
plotChurn(data, 'num_75')
plotChurn(data, 'num_50')

plotChurn(data, 'age')
plotChurn(data, 'city')

print('finished')