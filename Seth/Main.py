import pandas as pd
from Seth.DfManip import *
from Seth.Regressions import *
from Seth.Plots import *
data = readDF()
#age = imputeDF(data)

processDF(data)

continuousColumns = ['plan_list_price', 'actual_amount_paid', 'total_secs']

outputColumn = 'is_churn'
# models = performRegressions(data, continuousColumns, outputColumn)
# plotRegressionData(data, models, outputColumn)
plotEDA(data)
print('finished')