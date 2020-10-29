import pandas as pd
import Seth.Util as ut
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
histDir = 'Output/Histograms/'
scatterDir = 'Output/Scatterplots/'
barDir = 'Output/Barplots/'

histParams = {'kind': 'hist', 'legend': False, 'bins': 50}
barParams = {'kind': 'bar', 'legend': False}
figParams= {'x': 7, 'y': 7}



plt.rc('font', size=35)
plt.rc('axes', labelsize=60)
plt.rc('axes', titlesize=60)

xTickMult = lambda: ut.multiplyRange(plt.xticks()[0], 0.5)
xTickMultLS = lambda: ut.multiplyLinSpace(plt.xticks()[0], 2)
yTickFormat = lambda : plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
xTickFormatPercent = lambda: plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
xTickFormatCommas = lambda: plt.gca().xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
xTickFormatDollars = lambda x=0:  plt.gca().xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.'+str(x)+'f}'))
#setTickIn = lambda: plt.gca().tick_params(axis='x', direction='in')
trimTicks = lambda: plt.xticks()[0:-1]
nullsDir = 'Visualizations/Nulls/'
histParams = {'kind': 'hist', 'legend': False, 'bins': 100}


def plotData(df: pd.DataFrame, models, outputColumn):
    columns = df.drop(columns=[outputColumn]).columns
    gbm = models['Gradient Boost']
    coefs = gbm.model.feature_importances_

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