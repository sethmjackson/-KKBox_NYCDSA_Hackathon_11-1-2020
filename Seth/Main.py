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
models = performRegressions(data, continuousColumns, outputColumn)
plotRegressionData(data, models, outputColumn)
#plotEDA(data)
#plotFeatures(data)
print('finished')