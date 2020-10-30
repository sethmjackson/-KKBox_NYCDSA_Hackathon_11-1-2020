import pandas as pd
import Seth.Util as ut
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
histDir = 'Seth/Output/Histograms/'
scatterDir = 'Seth/Output/Scatterplots/'
barDir = 'Seth/Output/Barplots/'

histParams = {'kind': 'hist', 'legend': False}
barParams = {'kind': 'bar', 'legend': False}
figParams= {'x': 8, 'y': 8}



plt.rc('font', size=35)
plt.rc('axes', labelsize=50)
plt.rc('axes', titlesize=50)

xTickMult = lambda: ut.multiplyRange(plt.xticks()[0], 0.5)
xTickMultLS = lambda: ut.multiplyLinSpace(plt.xticks()[0], 2)
yTickFormat = lambda : plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
xTickFormatPercent = lambda: plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
xTickFormatCommas = lambda: plt.gca().xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
xTickFormatDollars = lambda x=0:  plt.gca().xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.'+str(x)+'f}'))
#setTickIn = lambda: plt.gca().tick_params(axis='x', direction='in')
trimTicks = lambda: plt.xticks()[0:-1]
nullsDir = 'Seth/Visualizations/Nulls/'
histParams = {'kind': 'hist', 'legend': False, 'bins': 100}


def plotRegressionData(df: pd.DataFrame, models, outputColumn):
    columns = df.drop(columns=[outputColumn]).columns
    gbm = models['Gradient Boost']
    coefs = gbm.modelCV.feature_importances_

    gbmCoefs = pd.DataFrame({'Variable': columns, 'Coefficient': coefs}).sort_values(by='Coefficient', ascending=True)
    gbmCoefs = gbmCoefs[gbmCoefs['Coefficient'] > 0.001]

    ut.plotDF(gbmCoefs, {'kind': 'barh', 'x': 'Variable', 'y': 'Coefficient', 'legend': False},
              {
                  'xticks': xTickMultLS,
                  'grid': None,
                  'xlabel': 'Gradient Boost Coefficient',
                  'ylabel': 'Variable Name',
                  'title': 'Bar Plot of Gradient Boost Feature Selection ',
                  'savefig': barDir + 'Gradient Boost Feature Selection.png'}, removeOutliersBeforePlotting=False)

def plotEDA(df: pd.DataFrame):
    churnPPD = df[df['is_churn'] == 1]
    noChurnPPD = df[df['is_churn'] == 0]

    print(churnPPD['payment_plan_days'].value_counts())

    ut.plotDF(df[['payment_plan_days']], histParams,
           {
            #yTickFormatPercent: '',
            'grid': None,
            'xlabel': 'Days on Payment Plan',
            'title': 'Histogram of Days customers are on payment plans (Total)',
            'savefig': histDir + 'Days on Payment Plan (Total).png'})

    ut.plotDF(churnPPD[['payment_plan_days']], histParams,
              {
                  # yTickFormatPercent: '',
                  'grid': None,
                  'xlabel': 'Days on Payment Plan',
                  'title': 'Histogram of Days customers are on payment plans (Churn)',
                  'savefig': histDir + 'Days on Payment Plan (Churn).png'})

    ut.plotDF(noChurnPPD[['payment_plan_days']], histParams,
              {
                  # yTickFormatPercent: '',
                  'grid': None,
                  'xlabel': 'Days on Payment Plan',
                  'title': 'Histogram of Days customers are on payment plans (No Churn)',
                  'savefig': histDir + 'Days on Payment Plan (No Churn).png'})
