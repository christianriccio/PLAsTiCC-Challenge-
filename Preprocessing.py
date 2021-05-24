import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pandas as pd
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import statsmodels.api as sm
import seaborn as sns
import math
import gc
import sys
import copy
import time
import keras.backend as k
import tensorflow_addons as tfa
import tensorflow as tf
from scipy import stats
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import RepeatVector 
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from keras.layers.convolutional import ZeroPadding1D, Conv1D
from keras.layers.pooling import AveragePooling1D, MaxPooling1D
from keras.layers import Flatten
from keras.utils import plot_model
from xgboost import XGBClassifier, plot_importance, plot_tree

pd.options.mode.chained_assignment = None
sns.set(style='darkgrid', color_codes=True)
#config = tf.compat.v1.ConfigProto(
#                                    intra_op_parallelism_threads=4,
#                                    inter_op_parallelism_threads=4,
#                                    allow_soft_placement=True,
#                                    device_count = {'CPU' : 1,
#                                                    'GPU' : 0}
#                                   )
#session = tf.compat.v1.Session(config=config)

class my_data():
    """Data structure of either simulated or real datasets ready to feed the B-spline method."""
    
    def __init__(self, x=None, y=None):
        """
        Initialize a dataset.
        
        Attributes:
        - x (ndarray)  Explanatory variable distribution
        - y (ndarray)  Response variable distribution
        """
        if x is None or y is None:
            return
        self.init_point = min(x)
        self.final_point = max(x)
        self.minStep = np.min(x[1:] - x[:-1])
        self.signal = None
        self.x = x
        self.y = y
        
        
    def simulateData(self, n_points=1500, init_point=0, final_point=1, signal=None):
        """
        Simulates a dataset.
        
        Attributes:
        - init_point (float : default=0)   Start of the observational window
        - final_point (float : default=1)  End of the observational window
        - signal (func : default=None)     Function representing the main signal
        - n_points (int : default=1500)    Number of point to generate
        """
        x = np.random.rand(n_points) * (final_point - init_point) + init_point
        x = np.linspace(init_point, final_point, n_points)
        self.x = np.sort(x, axis=None)
        self.minStep = np.min(self.x[1:] - self.x[:-1])
        
        self.init_point = init_point
        self.final_point = final_point
        
        self.signal = signal
        if signal is None:
            signal = lambda x: 0
        self.signal = np.array(list(map(signal, self.x)))
        self.y = self.signal.copy()
        
    
    def addNoise(self, distribution='', devstd=0):
        """
        Add some noise to the response variable.
        
        Attributes:
        - distribution (str : default='gaussian')  distribution of the noise
        - devstd (float : default=0)               standard deviation (not always applicable)
        """
        distribution = distribution.lower().replace(" ", "")
        if distribution == '':
            distribution = 'gaussian'
        if devstd < 0:
            devstd = 0
        
        if distribution == 'gaussian':
            self.y = np.random.normal(self.signal, size = len(self.x), scale=devstd)
        elif distribution == 'poisson':
            self.y = np.random.poisson(self.signal, size = len(self.x))
            if min(self.y) < 0:
                self.y += min(self.y)
        else:
            raise Exception('my_data Noise -> Invalid distribution type.')

    
    def crossVal(self, fold):
        """
        Returns a list of the subset (in my_data form) on the purpose of a cross Validation.
        
        Attributes:
        - fold (int)     Number of subset in output
        """
        index = np.array((range(len(self.x))))
        np.random.shuffle(index)
        segments = np.array(np.linspace(0, len(self.x)-1, fold+1), dtype=int)
        
        dataList = []
        for i in range(1, len(segments)):
            x = np.delete(self.x, index[segments[i-1]:segments[i]])
            y = np.delete(self.y, index[segments[i-1]:segments[i]])
            dataList.append(my_data(x,y))
        return dataList

    
    def plot(self, ax=None):
        if ax is None:
            ax = plt.axes()
            ax.set_title("Data")
        ax.scatter(self.x, self.y, label="data", marker='.', c='gray')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if self.signal is not None:
            ax.plot(self.x, self.signal, label="signal", linestyle='-', c='orange', linewidth=3.0)
        return ax


class B_spline():
    
    def __init__(self, dataset, K=-1, totalKnots=-1, df=-1, order=-1, K_optimization=False):
        """
        Initializes a class B-spline.
        
        Attributes:
        - dataset (my_data)                           Collection of data in my_data form
        - K (int)                                     Number of internal knots ξ
        - order (int)                                 Order of spline (ie: 3 for a cubic)
        - df (int)                                    Total degree of freedom
        - totalKnots (int)                            Total number of knots ξ
        - K_optimization (boolean : default=False)    Automatically adjusts the location of the knots
        """
        self.dataset = dataset
        
        K = int(K)
        totalKnots = int(totalKnots)
        df = int(df)
        order = int(order)
        
        if K < 0: K = totalKnots - 2
        if (K <= 0 and df <= 0) or (K <= 0 and df > 0 and order < 0):
            raise Exception('BSpline init -> Invalid number of knots.')
        if K <= 0:
            K = df - order - 1
        if K < 0:
            raise Exception('BSpline init -> Invalid inputs.')
        
        self.K = K
        self.totalKnots = self.K + 2
            
        self.init_point = dataset.init_point - dataset.minStep
        self.final_point = dataset.final_point + dataset.minStep
        if self.final_point - self.init_point <= 0: 
            raise Exception('BSpline init -> Invalid initial/final points.')
        
        if df > self.K:
            order = df - self.K - 1
        elif order >= 0:
            df = order + self.K + 1
        else:
            raise Exception('BSpline init -> Invalid order/df value.')
        self.order = order
        self.df = df
        self.opt = K_optimization
        
        self.generateKnots()
        if K_optimization:
            try:
                self.base =  np.array([self.generate_basis(x, self.order + 1) for x in self.dataset.x])
            except:
                self.opt = False
                self.generateKnots()
                self.base =  np.array([self.generate_basis(x, self.order + 1) for x in self.dataset.x])
        else:
            self.base =  np.array([self.generate_basis(x, self.order + 1) for x in self.dataset.x])


    def generateKnots(self):
        """
        Generate an array of values representing the knots positions depending on the activation
        or not of the optimization.
        """
        if not self.opt:
            # Uniform knots position
            self.knots = np.linspace(self.init_point, self.final_point, self.totalKnots)
        else:
            # Variation-dependent knots position
            data_x = self.dataset.x
            try:
                rough_spline = B_spline(self.dataset, K=30, order=1, K_optimization=False)
                rough_spline.fitGLM(response='gaussian', verbose=False, show=False)
                if rough_spline.result is None:
                    self.knots = np.linspace(self.init_point, self.final_point, self.totalKnots)
                    return
                pred = rough_spline.predicted
                derivate = pred[1:] - pred[:-1]
                derivate = np.concatenate((derivate, np.array([derivate[-1]])))
                cumulative = np.cumsum(np.fabs(derivate))
                cumulative = cumulative / np.max(cumulative)
                y_tick = np.linspace(0, 1, self.totalKnots)
                y_index = np.linspace(0, 1, len(data_x))
                x_tick = [0]

                for tick in y_tick:
                    for i in range(1, len(cumulative)-1):
                        if cumulative[i] >= tick and cumulative[i-1] < tick:
                            x_tick.append(i-1)
                            break

                x_tick.append(len(data_x)-1)
                self.knots = data_x[x_tick]
                self.knots[0] -= self.dataset.minStep
                self.knots[-1] += self.dataset.minStep
            except:
                self.knots = np.linspace(self.init_point, self.final_point, self.totalKnots)
    
    
    def generate_augmKnots(self, m):
        """
        Generates an array of Knots with repeated Knots on boundaries.
        
        Attribute:
        - m (int)    Degree of spline 
        """
        init_boundaries = self.init_point * np.ones(m-1)
        final_boundaries = self.final_point * np.ones(m-1)
        augmKnots = np.concatenate((init_boundaries, self.knots, final_boundaries), axis = None)
        return augmKnots
    
    
    def generate_basis(self, x, m):
        """
        Generates basis spline of the m th order.
        Attributes:
        
        - x (ndarray)    Input array
        - m (int)        Degree of spline 
        """
        if m==1:
            augmKnots = self.generate_augmKnots(1)

            number_of_intervals = self.K + 1
            intervals = np.zeros((number_of_intervals,2))
            for i in range(number_of_intervals):
                for j in range(2):
                    intervals[i,j] = augmKnots[i+j]
            basis_1 = [1 if (x >= intervals[i,0] and x < intervals[i,1]) else 0  
                        for i in  range(number_of_intervals)]
            basis_1 = np.array(basis_1)

            return basis_1
        
        augmKnots = self.generate_augmKnots(m)
        prev_basis = self.generate_basis(x, m-1)
        prev_basis = np.pad(prev_basis, (1, 1), 'constant', constant_values=(0,0))
        
        basis_m = np.zeros(self.K + m)
        for i in range(self.K + m):
            if augmKnots[i+m-1] == augmKnots[i]  :
                if (augmKnots[i+m] - augmKnots[i+1]) == 0:
                    raise Exception('Division by zero')
                basis_m[i] = (
                ((augmKnots[i+m] - x) / (augmKnots[i+m] - augmKnots[i+1])) * prev_basis[i+1])
            elif augmKnots[i+m] == augmKnots[i+1]:  
                if (augmKnots[i+m-1] - augmKnots[i]) == 0:
                    raise Exception('Division by zero')
                basis_m[i] = (
                ((x-augmKnots[i]) / (augmKnots[i+m-1] - augmKnots[i])) * prev_basis[i])
            else:
                if (augmKnots[i+m-1] - augmKnots[i]) == 0 or (augmKnots[i+m] - augmKnots[i+1]) == 0:
                    raise Exception('Division by zero')
                basis_m[i] = (
                ((x-augmKnots[i]) / (augmKnots[i+m-1] - augmKnots[i])) * prev_basis[i] +
                ((augmKnots[i+m]-x) / (augmKnots[i+m] - augmKnots[i+1])) * prev_basis[i+1])
        return basis_m
    
    
    def fitGLM(self, response='', verbose=False, paint=False, showKnots=False, show_data=True, show=True, ax=None):
        """
        Fits the data with a GLM to get the best coefficient estimations.
        
        Attributes:
        - response (str : default='gaussian')   Distribution of the target variable (ie. gaussian, poisson..)
        - verbose (boolean : default=False)     True if the glm results should be visible
        - show (boolean : default=False)        True if one wants the prediction in a graph form to be shown
        - showKnots (boolean : default=False)   True if the knot positions should be displayed
        - show_data (boolean : default=False)   True if one wants the original dataset to be shown
        """
        self.ax = ax
        np.seterr(divide='ignore', invalid='ignore')
        response = response.lower().replace(" ", "")
        if response == '': response = 'gaussian'
        
        #print(self.dataset.y.shape, self.base.shape)
        if response == 'gaussian':
            self.model = sm.GLM(self.dataset.y, self.base, family=sm.families.Gaussian())
        elif response == 'poisson':
            self.model = sm.GLM(self.dataset.y, self.base, family=sm.families.Poisson())
        else:
            raise Exception('BSpline GLM fit -> Invalid response type.')
        
        try:
            self.result = self.model.fit()
            self.predicted = self.model.predict(self.result.params)
            RSS = np.sum(np.power(self.dataset.y - self.model.predict(self.result.params), 2))
        except:
            self.result = None
            RSS = None
            self.predicted = [np.mean(self.dataset.y) for i in range(len(self.base))]
        
        if verbose:
            print(self.result.summary())
            
        if paint:
            title = None
            if ax is not None:
                title = ''
            else:
                print(f'Response type: {response.title()}')
            self.plotRegression(showKnots=showKnots, show_data=show_data, show=show, title=title)
        
        return self.result, RSS
    
    
    def predict(self, X):
        """
        Given a value x, it returns the predicted value y by results of the GLM fit.
        
        Attributes:
        - x (float)     Point where the prediction should be performed
        """
        y = []
        if self.result is None:
            return np.array([np.mean(self.dataset.y)]*len(X))
        else:
            for x in X:
                if x < self.dataset.x[0]:
                    x = self.dataset.x[0]
                if x > self.dataset.x[-1]:
                    x = self.dataset.x[-1]
                y.append(np.dot(self.generate_basis(x, self.order + 1), self.result.params))
            return np.array(y)

    
    def plotBasis(self):
        x = self.dataset.x
        y = np.array(self.base)
        skip = 1
        for i in range(y.shape[1]):
            plt.plot(x,y[:,i])
        plt.title("Basis spline")
        plt.show()


    def plotRegression(self, showKnots=False, show_data=True, show=True, title=None):
        if show_data:
            self.ax = self.dataset.plot(self.ax)
            x = self.dataset.x
            y = self.predicted
        if self.result is not None:
            x = np.linspace(min(x), max(x), 100)
            y = self.predict(x)
        
        self.ax.plot(x, y, label= "prediction", c='red')
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        if title is None:
            self.ax.set_title("Data vs B-spline predictions")
            self.ax.legend()
        if showKnots:
            y_min = min(self.dataset.y)
            y_max = max(self.dataset.y)
            for x in self.knots:
                self.ax.plot([x,x], [y_min, y_max], c='purple')
        if show:
            plt.show()


def annotate_colname(x, **kws):
    ax = plt.gca()
    ax.annotate(x.name, xy=(0.05, 0.9), xycoords=ax.transAxes,
                fontweight='bold')


def corrfunc(x, y, **kws):
    ax = plt.gca()
    try:
        r, p = stats.pearsonr(x, y)
        p_stars = ''
        if p <= 0.05:
            p_stars = '*'
        if p <= 0.01:
            p_stars = '**'
        if p <= 0.001:
            p_stars = '***'
        ax.annotate('r = {:.2f} '.format(r) + p_stars,
                    xy=(0.05, 0.9), xycoords=ax.transAxes)
    except:
        return
    
    
def cor_plots(DF, dir_path, show):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    g = sns.PairGrid(DF)
    g.map_upper(sns.regplot, scatter_kws={'s': min(10, len(DF.columns))})
    g.map_diag(sns.histplot)
    for ax, col in zip(np.diag(g.axes), DF.columns):
        ax.annotate(col, xy=(0.05, 0.9), xycoords=ax.transAxes, fontweight='bold')
    g.map_lower(sns.kdeplot, cmap='Blues_d')
    g.map_lower(corrfunc)
    for ax in g.axes.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')
    plt.savefig(dir_path+"/CorrMap.png", format='png', dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
        

def paired_plots(DF, hue, dir_path, show):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    newDF = DF.drop(['target'], axis=1)
    for col in newDF.columns:
        allVal = np.sort(np.unique(newDF[col].values))
        if len(allVal) < 3:
            newDF[col] = newDF[col].values + (allVal[-1] - allVal[0])/10 * (2 * np.random.rand(len(DF)) - 1)
    n_perRow = 3
    ny = int(newDF.shape[1]*(newDF.shape[1]-1)/n_perRow /2)
    fig, ax = plt.subplots(ny, n_perRow, figsize=(18, int(18*ny/n_perRow)))
    dx = np.array(list(newDF.columns))
    dy = np.array(list(newDF.columns))
    couples = [x for x in itertools.product(dx, dy) if x[0] != x[1]]
    m=[]
    for x in couples:
        if (x[0], x[1]) not in m and (x[1], x[0]) not in m:
            m.append(x)
    
    i, j = 0, 0
    
    colors = {}
    h = 0
    for el in DF['target']:
        if el not in colors.keys():
            colors[el] = h
            h += 1
    
    col = [plt.get_cmap('tab20').colors[colors[el]] for el in DF['target']]
    for co in m:
        ax[i, j].scatter(newDF[co[0]].values, newDF[co[1]].values, color=col)
        ax[i, j].set_xlabel(co[0])
        ax[i, j].set_ylabel(co[1])
        j += 1
        if j == 3:
            i += 1
            j = 0
    
    plt.savefig(dir_path+"/PairedPlots.png", format='png', dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    

        
def correlationMatrix(data):
    corr_mat = data.drop(["target","object_id"], axis = 1).corr()
    mask = np.zeros_like(corr_mat)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style('white'):
        f, ax = plt.subplots(figsize=(13, 13))
        ax = sns.heatmap(corr_mat, mask=mask,vmax = 1.0, square=True, annot = True)
        plt.show()


def unique(df, col):
    
    """ return list of unique items"""
    
    objs = df.loc[:, col].unique().tolist()
    objs.sort()
    return objs


def normalize_fluxes(df, col='flux', col2='flux_err', by='passband'):
    
    """normalize the values of the variable 'col' within the df using 
    average means and standard deviations"""
    
    objs = unique(df, 'object_id')
    bands = unique(df, 'passband')
    objs_dict = {ob:{band:[] for band in bands} for ob in objs}
    newdf = df[["object_id", col, by]]
    for row in newdf.values:
        objs_dict[row[0]][row[2]].append(row[1])
    for ob in objs_dict.keys():
        elements = []
        for band in objs_dict[ob]:
            series = objs_dict[ob][band]
            for el in series:
                elements.append(el)
            count = len(series)
            objs_dict[ob][band] = [elements, count]
        elements = np.array(elements)
        mean = np.mean(elements)
        std = np.sqrt(np.sum(np.square(elements - mean))/len(elements))
        #mean = np.sum([objs_dict[ob][key][0]*objs_dict[ob][key][0] \
        #                 for key in objs_dict[ob].keys()])/len(objs_dict[ob].keys())
        #std = np.max([objs_dict[ob][key][1] for key in objs_dict[ob].keys()])
        objs_dict[ob] = [mean, std]
    means = []
    stds = []
    for row in df.values:
        means.append(objs_dict[row[0]][0])
        stds.append(objs_dict[row[0]][1])
    normalized = df.copy()
    normalized.loc[:, 'mean'] = np.array(means)
    normalized.loc[:, 'std'] = np.array(stds)
    normalized[col] = (df[col] - normalized['mean']) / normalized['std']
    normalized[col2] = (df[col2] - normalized['mean']) / normalized['std']
    return normalized


def updateMetadata(data, metadata):
    rows = []
    prev = None
    cols = ['object_id', 'mean', 'std']
    for row in data[cols].values:
        if prev != row[0]:
            rows.append(row)
            prev = row[0]
    df = pd.DataFrame(np.array(rows), columns=cols)
    new_metadata = metadata.merge(df, on = "object_id")
    return new_metadata


def plotTargetDistr(data, targets):
    justTargets = data[['target']]
    targets_dict = {targets[i]:i+1 for i in range(len(targets))}
    maskedTarg = []
    for val in justTargets['target'].values:
        maskedTarg.append(targets_dict[val])
    justTargets['mask_target'] = np.array(maskedTarg)
    plt.figure(figsize=(14,7))
    g = sns.histplot(data = justTargets['mask_target'], discrete = True)
    plt.yscale('log')
    plt.xticks(list(targets_dict.values()), targets)
    print('Target distribution:')
    plt.show()
    distr = justTargets.groupby(['mask_target']).count().values[:,0]
    distr = {'target ' + str(targets[i]):distr[i] for i in range(len(targets_dict))}
    print('Target distribution:\n')
    for item in distr.items():
        print(str(item[0]) + ' -> number: ' + str(item[1]) + '  <=>  ' + str(round(item[1] * 100/len(justTargets), 1)) + '%')
    return distr


def reduce_dataset(df, n_points, n_bands=-1, dir_path=''):
    """This function returns the final dataset with the object that satisfy all the requirements and it wraps
       the two other functions and gives them their arguments.
       It takes as arguments the dataset, the number of points required in each band, and the number of bands
       in which we want those point.
       This is the setting of n_bands
       n_bands = -1 -------> All the bands require to have a certain number of points
       n_bands = 1 -------> 1 Bands require to have a certain number of points
       n_bands = 2 -------> 2 bands require to have a certain number of points
       n_bands = 3 -------> 3 bands require to have a certain number of points
       n_bands = 4 -------> 4 bands require to have a certain number of points
       n_bands = 5 -------> 5 band requires to have a certain number of points"""
    
    df['dT'] = np.nan
    threshold = n_bands - 1
    all_bands = np.unique(df['passband'])
    pass_check = False
    if n_bands == 0 or n_points == 0:
        pass_check = True
    if n_bands == -1:
        threshold = len(all_bands) - 1
    rows_toSave = []
    ob_ind = list(df.columns).index('object_id')
    pb_ind = list(df.columns).index('passband')
    t_ind = list(df.columns).index('mjd')
    act_id = -1
    band_dict = {}
    rows_Temp = []
    j = 0
    act_band = -1
    for row in df.values:
        j += 1
        band = int(row[pb_ind])
        id_ = int(row[ob_ind])
        if id_ != act_id:
            sys.stdout.write('\r'+f'Reducing: {round(100*j/len(df), 0)}%      ')
            act_id = row[ob_ind]
            act_date = row[t_ind]
            
            if pass_check or np.sum(np.array(list(band_dict.values())) > n_points) > threshold:
                for r in rows_Temp:
                    rows_toSave.append(np.array(r))
            
            for i in all_bands:
                band_dict[i] = 0
            act_band = band
            band_dict[band] = 1
            rows_Temp = []
            row[-1] = 0
            rows_Temp.append(row)
        else:

            if band != act_band:
                act_band = band
                row[-1] = 0
            else:
                row[-1] = row[t_ind] - act_date
            band_dict[band] += 1
            rows_Temp.append(row)
            act_date = row[t_ind]
    print('Creating the dataframe..')
    newdf = pd.DataFrame(rows_toSave, columns=df.columns)
    df = df.drop(['dT'], axis=1)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    if dir_path != '':
        dir_path += '/'
    print('Saving to local..')
    newdf.to_csv(dir_path+'reduced_data.csv', sep=';', index=None)
    print('Reduced dataset successfully saved!')
    return newdf


def getSeries(series, target, bands = [], n='all'):  

    """Return fluxes of 'n' object choosed  from a specific target.
    With the attribute 'bands' we can specify a filter for the passbands we are currently considering.
    The attribute n defines number of objects in the list"""

    mask_target = series["target"] == target
    list_of_object_id = pd.unique(series[mask_target]["object_id"]).tolist()

    objects = []
    if n == 'all':
        object_to_filter = list_of_object_id
    else:
        object_to_filter = list_of_object_id[:n]
        
    for obj in object_to_filter:
        mask_object = series["object_id"] == obj
        single_object = series[mask_object]

        if bands == []:
            objects.append(single_object)
            continue

        mask_band = np.zeros((single_object.shape[0])).astype(dtype=bool)
        for band in bands:
            mask_band = mask_band | np.array(single_object["passband"] == band)
        objects.append(single_object[mask_band])
    
    return objects

def reduceK(k, ceil, x):
    if x < 10:
        return 2
    if x > 30:
        return ceil
    return round(k*x**2 + (0.05*ceil - 40*k - 0.1) * x + (3 + 300*k - 0.5*ceil))


def makeIntervals(interval, scores, threshold=50):
    news = []
    x_in = None
    x_fin = None
    mask = []
    for i in range(len(interval)):
        if scores[i] < threshold:
            mask.append(False)
            if x_in is None:
                continue
            x_fin = interval[i]
            news.append((x_in, x_fin))
            x_in = None
            x_fin = None
        else:
            mask.append(True)
            if x_in is None:
                x_in = interval[i]
            else:
                x_fin = interval[i]
    return news


def plotApproximation(model, x_interval, y_predicted, ax):
    model.dataset.plot(ax)
    ax.plot(x_interval, y_predicted, label= "prediction", c='red')
    ax.set_xlabel("")
    ax.set_ylabel("")

    
def plotRegions(intervals, ax):
    y0, yl = ax.get_ylim()
    for i_ in intervals:
        rect = patches.Rectangle((i_[0], y0), i_[1]-i_[0], yl-y0, linewidth=1, color='r', fill=True)
        ax.add_patch(rect)


def plotSeries(data, class_, params, n_points=100, n=3):
    series = getSeries(data, class_, bands=params.bands, n=n)
    bands = list(pd.unique(series[0]['passband']))
    bands.sort()
    distr=[]
    scors = []
    ints = []
    fig, axs = plt.subplots(len(bands), len(series), figsize=(18,7))
    print(f'Class: {class_}')
    for i in range(len(series)):
        distribution = []
        errors = []
        minX = min(series[i]['mjd'])
        maxX = max(series[i]['mjd'])
        x_interval = np.linspace(minX, maxX, 100)
        for j in range(len(bands)):
            b = series[i][series[i]['passband'] == bands[j]]
            x = b['mjd'].values
            y = b['flux'].values
            y_err = b['flux_err'].values
            distribution.append(x)
            if len(x) == 0:
                continue
            
            k = params.spl_K[j]
            k = reduceK(0.02, k, len(x))
            order = params.spl_Ord[j]
            y, y_err = applySplines(x, x_interval, y, y_err=y_err, 
                                    k=k, order=order, ax=axs[j][i])
        xinterval, scores = evaluateFidelity(distribution, x_interval, bands, params)
        
        distr.append(distribution)
        scors.append(scores)
        ints.append(x_interval)
        
        intervals = makeIntervals(xinterval, scores, threshold=params.ts)
        for j in range(len(bands)):
            plotRegions(intervals=intervals, ax=axs[j][i])
            
    plt.show()
    return distr, scors, ints


def getProportions_perBand(df, show=False):
    grouped = df.groupby(['object_id', 'passband']).count().reset_index()
    #sns.histplot(grouped, x='passband', palette=sns.color_palette("tab10", 6))
    grouped = grouped[['passband','flux']].groupby(['passband']).sum().reset_index()
    grouped['weights'] = grouped['flux'].values / np.sum(grouped['flux'].values)
    fig, ax = plt.subplots(1,1, figsize=(16, 6))
    ax.bar(grouped['passband'].values, grouped['flux'].values, color='red')
    ax.set_xlabel('Passband')
    plt.show()
    prop = {int(el[0]):el[1] for el in grouped.values[:, [0,2]]}
    return prop


class spl_param:
    
    def __init__(self, threshold, weights={}):
        self.bands = []
        self.spl_K = []
        self.spl_Ord = []
        self.weights = []
        self.weights_dict = weights
        self.ts = threshold
        
    def newParam(self, band, K, Ord, weight=0):
        self.bands.append(band)
        self.spl_K.append(K)
        self.spl_Ord.append(Ord)
        self.weights.append(self.weights_dict.get(band, 0)**2)
        return self


def saveLocal(data, metadata, n_points, path):
    ids = list(data.keys())
    dfs = []
    for i in range(len(ids)):
        df = pd.DataFrame(data[ids[i]], columns=range(n_points))
        df['object_id'] = np.array([ids[i]]*data[ids[0]].shape[0])
        dfs.append(df)
    df = pd.concat(dfs)
    df = df.merge(metadata, on = "object_id")
    df.to_csv(path, index=False, sep=';')
    return


def maskIntervals(on, scores, intervals, threshold=50):
    masked_indexes = scores >= threshold
    dT = [1]
    val = 0
    for i in range(1,len(masked_indexes)):
        if masked_indexes[i-1] != 1:
            val = 1
        else:
            val += 1
        dT.append(val)
    dT = np.array(dT)
    dT[masked_indexes] = -10
    
    for interval in intervals:
        interval[masked_indexes] = -10
    intervals.insert(0, dT)
    return intervals


def evaluateFidelity(distributions, x_interval, bands, params):
    suspicion = np.zeros(shape=(len(x_interval,)))
    sum_bandWeights = sum([params.weights[params.bands.index(bands[i])] for i in range(len(distributions))])
    for i in range(len(distributions)):
        band_ind = bands.index(bands[i])
        distro = distributions[i]
        dist = np.min(np.fabs(x_interval - distro[:, np.newaxis]), axis=0)
        suspicion = suspicion + dist * params.weights[band_ind]/sum_bandWeights
    return x_interval, suspicion


def evalFidelityExample(plots):
    distr, scors, ints = plots
    num = min(3, len(distr))
    fig, ax = plt.subplots(1, num, figsize=(20, 4))
    for j in range(num):
        for i in range(len(distr[j])):
            x = distr[j][i]
            y = np.zeros(x.shape) - i*4
            ax[j].scatter(x, y)
        scores = scors[j]
        xlin = ints[j]
        ax[j].plot(xlin, scores)
    plt.show()


def applySplines(xi, x_interval, yi, k, order, y_err=None, ax=None):
    r_data = my_data(xi, yi)
    r_spline = B_spline(r_data, K=k, order=order, K_optimization=False)
    result, RSS = r_spline.fitGLM(response='gaussian', verbose=False)
    y = r_spline.predict(x_interval)
    y[y > 5] = 5.
    y[y < -5] = -5.

    if y_err is not None:
        r_data_err = my_data(xi, y_err)
        r_spline_err = B_spline(r_data_err, K=2, order=2, K_optimization=True)
        r_spline_err.fitGLM(response='gaussian', verbose=False)
        y_err = r_spline_err.predict(x_interval)
    
    if ax is not None:
        plotApproximation(model=r_spline, x_interval=x_interval, y_predicted=y, ax=ax)
    return (y, y_err)

def createTensors(data_, metadata, params, errFlux=False, dir_path='', n_points=100):
    list_of_targets = pd.unique(metadata["target"]).tolist()
    flux_tensor = {}
    flux_err_tensor = {}
    n_objects = 0
        
    print('Creating the reference system..')
    group = ['object_id', 'passband', 'target']
    series_ = {el[0]:el[1][['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'target']] 
              for el in data_.groupby(group)}
    allKeys = list(series_.keys())
    allKeys_df = pd.DataFrame(allKeys, columns=group)
    bands = list(np.unique([el[1] for el in allKeys]))
    bands.sort()
    missing_bands = [el for el in bands if el not in params.bands]
    for i in missing_bands:
        allKeys_df = allKeys_df[allKeys_df['passband'] != i]
        bands.pop(i)

    t0 = time.time()
    for class_ in list_of_targets:
        classKeys = [tuple(el) for el in allKeys_df[allKeys_df['target'] == class_].values]
        objs = np.unique([el[0] for el in classKeys])

        for i in range(len(objs)):
            distribution = []
            results = []
            name = objs[i]
            objKeys = [el for el in classKeys if el[0] == name]
            mins = [min(series_[el]['mjd']) for el in objKeys]
            maxs = [max(series_[el]['mjd']) for el in objKeys]
            x_interval = np.linspace(min(mins), max(maxs), n_points)
            for j in range(len(bands)):
                bandKey = [el for el in objKeys if el[1] == bands[j]]
                b = series_[bandKey[0]]
                x = b['mjd'].values
                y = b['flux'].values
                if errFlux:
                    y_err = b['flux_err'].values
                else:
                    y_err = None
                distribution.append(x)

                band_ind = params.bands.index(bands[j])
                k = params.spl_K[band_ind]
                k = reduceK(0.02, k, len(x))
                order = params.spl_Ord[band_ind]
                y, y_err = applySplines(x, x_interval, y, y_err=y_err, k=k, order=order)
                results.append(y)
                if errFlux:
                    results.append(y_err)

            xinterval, scores = evaluateFidelity(distribution, x_interval, bands, params)
            masked = maskIntervals(on=xinterval, scores=scores, intervals=results, threshold=params.ts)
            
            if errFlux:
                flux_tensor[name] = np.array(masked[::2])
                flux_err_tensor[name] = np.array(masked[1::2])
            else:
                flux_tensor[name] = np.array(masked)
            n_objects += 1
            perc = round(n_objects * 100/ len(metadata), 1)
            dt = round(time.time() - t0, 0)
            time_r = int(dt * len(df_metadata) / n_objects - dt)
            
            if time_r > 120: min_r = str(int(time_r/60)) +'min'
            else: min_r = str(int(time_r)) +'s'
            if dt > 120: dt_min = str(int(dt/60)) +'min'
            else: dt_min = str(int(dt)) +'s'
            
            sys.stdout.write('\r'+f'Preprocessing on the object {n_objects} out of {len(metadata)} ... {perc}%'
            + f'      remaining time: {min_r}  elapsed: {dt_min}           ')
    
    sys.stdout.write('\r'+f'\n')
    print('Preprocessing completed!')
    sys.stdout.flush()
    
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    if dir_path != '':
        dir_path += '/'
    print('Saving flux_tensor')
    t = dir_path+'flux.csv'
    print(f'at {t}')
    saveLocal(flux_tensor, metadata, n_points, path=dir_path+'flux.csv')
    if errFlux:
        print('Saving flux_err_tensor')
        saveLocal(flux_err_tensor, metadata, n_points, path=dir_path+'flux_err.csv')
    print('Saving normalized data')
    data_.to_csv(dir_path+'normalized_data.csv', sep=';', index=None)
    print('Tensors successfully saved')
    return flux_tensor

print('Functions for the preprocessing phase successfully imported!')