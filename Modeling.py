import os
import os.path
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pandas as pd
import numpy as np
import random
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
from os import listdir
from os.path import isfile, join
from tabulate import tabulate
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
tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
#config = tf.compat.v1.ConfigProto(
#                                    intra_op_parallelism_threads=4,
#                                    inter_op_parallelism_threads=4,
#                                    allow_soft_placement=True,
#                                    device_count = {'CPU' : 1,
#                                                    'GPU' : 0}
#                                   )
#session = tf.compat.v1.Session(config=config)


def obj_band_dic(object_list_unique):
    """This function takes as argument the list of unique objects in order to create
        a dictionary that has as keys the object_id and as values a list like this
        one ['object_id', passband]"""
    obj_dic = {}
    for an_object in object_list_unique:
        obj_dic[str(an_object)] = []
        for a_band in range(0,6):
            obj_dic[str(an_object)].append([str(an_object), str(a_band)])
    return obj_dic


def get_values(df, column, a_dic):
    """ This function adds the values from a column of a dataset to an existing dictionary.
        It takes as argument the dataset, the name of the column as string, and the dictionary."""
    for index,row in df.iterrows():
        a_dic[str(row['object_id'])][int(row['passband'])].append(row[column])

        
def modify_dates(a_dic,skip_num):
    """This function returns a dictionary with the time differences (dt) and adds 10 at the end of each list in order to
        have lists with the same lenght.
        It takes as argument the dictionary that needs to be modified and an integer that will be put in each list
        to reach the fixed length. It is preferable to set a number that is less equal than -10."""
    a_dic_copy = copy.deepcopy(a_dic)
    for a_key in a_dic_copy.keys():
        for i in range(6):
            temp_list = a_dic_copy[a_key][i].copy()
            for j in range(3, len(a_dic_copy[a_key][i]),1):
                a_dic_copy[a_key][i][j] = temp_list[j] - temp_list[j-1]
            a_dic_copy[a_key][i][2] = 0
            if len(a_dic_copy[a_key][i]) != 74:
                for k in range(74 - len(a_dic_copy[a_key][i])):
                    a_dic_copy[a_key][i].append(skip_num)
    return a_dic_copy


def add_values(a_dic, skip_num):
    """This function adds 10 at the end of each list in order to have lists with the same lenght.
        It takes as argument the dictionary that needs to be modified and an integer that will be put in each list
        to reach the fixed length. It is preferable to set a number that is less equal than -10."""
    a_dic_copy = copy.deepcopy(a_dic)
    for a_key in a_dic_copy.keys():
        for i in range(6):
            if len(a_dic_copy[a_key][i]) != 74:
                for k in range(74 - len(a_dic_copy[a_key][i])):
                    a_dic_copy[a_key][i].append(skip_num)
    return a_dic_copy


def create_matrix(a_dic, a_list):
    """This function creates the two matrices that will form the tensor that we need to put into the encoder."""
    for a_key in a_dic:
        a = np.vstack(tuple(a_dic[a_key]))
        a_list.append(a)
    final_matrix = np.vstack(tuple(a_list))
    return final_matrix

def retrieveReducedMetadata(reduced_data, metadata):
    ids = pd.unique(reduced_data['object_id'].values)
    m = reduced_data[['object_id']].merge(metadata, how='left', on='object_id').groupby(['object_id'])
    frames = [el[1].values[0, :len(metadata.columns)] for el in m]
    reduced_metadata = pd.DataFrame(frames, columns=metadata.columns)
    return reduced_metadata


def create_tensor(df, lenght=76, mask_val=-10.):
    """This function returns the tensor that has the fluxes and the dt matrices. It needs the dataset 
    as input and an integer that will be put in each list to reach the fixed length. It is preferable 
    to set a number that is less equal than -10. as mask_val attribute"""
    
    all_bands = np.unique(df['passband'])
    lenghts = df.value_counts(subset=['object_id', 'passband'], sort=False).values
    grouped = df.groupby(['object_id', 'passband'])
    final = []
    residuals = 0
    cutted_series = 0
    band = 0
    i = 0
    for group in grouped:
        i+=1
        p = round(100*i/len(grouped), 0)
        sys.stdout.write('\r'+f'Tensor creation: {p}%      ')
        band += 1
        vals = group[1][['flux', 'dT']].values

        cut = lenght-len(vals)
        if cut < 0:
          residuals -= cut
          cutted_series += 1
          final.append(vals[:lenght, :])
        else:
          fill = np.ones(shape=(cut, vals.shape[1])) * mask_val
          final.append(np.r_[vals, fill])
        if band == len(all_bands):
            band = 0
    print('->  aggregation..   ', end='')
    final = np.array(final)
    print(f'Done!\nTensor shape: {final.shape}')
    return final, residuals, cutted_series


def encodeData(encoder, input_tensor, metadata, target, n_bands, name=None):
    if encoder is not None:
        X = encoder.predict([input_tensor, input_tensor[:,:,1]])
        X = np.swapaxes(X.reshape(int(X.shape[0]/n_bands), n_bands, X.shape[1]), 1, 2)
    else:
        X = np.swapaxes(input_tensor, 1, 2)
    print('Data encoded.')
    OneHot = OneHotEncoder(sparse=False)
    targ = OneHot.fit(target.reshape(-1, 1)).transform(target.reshape(-1, 1))
    encoded = k_data(X=X, dT=metadata, Y=targ, name=name)
    return encoded


def predictData(model, input):
    X_train = model.predict([input.x_train, input.dT_train])
    X_test = model.predict([input.x_test, input.dT_test])
    print('Data predicted.')
    predicted = k_data(X=np.r_[X_train, X_test], Y=np.r_[input.y_train, input.y_test], shuffle=False, split=input.split)
    return predicted


def pushMasked(tensor, mask_val=-10.):
    width = tensor[0].shape[0]
    matrices = []
    for i in range(tensor.shape[0]):
        masked = tensor[i][0] != mask_val
        height = tensor[0].shape[1] - np.sum(masked)
        valid = np.c_[tensor[i][:, masked], mask_val * np.ones(shape=(width, height))]
        matrices.append(valid)
    return np.array(matrices)


class k_data():
    def __init__(self, X=None, dT=None, Y=None, split=0.8, name=None, shuffle=True, mask_value=-10.):
        self.x_train = np.empty(shape=(0,))
        self.x_test = np.empty(shape=(0,))
        self.dT_train = np.empty(shape=(0,))
        self.dT_test = np.empty(shape=(0,))
        self.y_train = np.empty(shape=(0,))
        self.y_test = np.empty(shape=(0,))
        self.name = name
        self.mask_value = mask_value
        self.split = split
        if shuffle:
            indexes = [i for i in range(len(X))]
            random.shuffle(indexes)
            if X is not None:
                nX = X[indexes]
            if Y is not None:
                nY = Y[indexes]
            if dT is not None:
                nT = dT[indexes]
        else:
            nX = X
            nY = Y
            nT = dT
        ind_split = int(len(X)*split)
        if X is not None:
            self.x_train = nX[:ind_split]
            self.x_test = nX[ind_split:]
        if Y is not None:
            self.y_train = nY[:ind_split]
            self.y_test = nY[ind_split:]
        if dT is not None:
            self.dT_train = nT[:ind_split]
            self.dT_test = nT[ind_split:]
            
    def reduce_to(self, n_points=None):
        if n_points is not None:
            self.x_train = self.x_train[:n_points]
            self.x_test = self.x_test[:n_points]
            self.dT_train = self.dT_train[:n_points]
            self.dT_test = self.dT_test[:n_points]
            self.y_train = self.y_train[:n_points]
            self.y_test = self.y_test[:n_points]
        return self
    
    
    def getShapes(self, print_=True):
        text = f'x_train: {self.x_train.shape}\ndT_train: {self.dT_train.shape}\ny_train: {self.y_train.shape}\n'
        text += f'x_test: {self.x_test.shape}\ndT_test: {self.dT_test.shape}\ny_test: {self.y_test.shape}\n'
        if print_:
            print(text)
        return (self.x_train.shape, self.dT_train.shape, self.y_train.shape, 
                self.x_test.shape, self.dT_test.shape, self.y_test.shape)
        

def plotModel(model, dir_path, name, show=True):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    if dir_path != '':
        dir_path += '/'
    plot_model(model, to_file=dir_path+name+'.png', show_shapes=True, show_layer_names=True)
    time.sleep(3)
    if show:
        fig, ax = plt.subplots(figsize=(27, 22))
        ax.axis('off')
        print(f'Model scheme -> {name}')
        ax.imshow(plt.imread(dir_path+name+'.png'))
        #plt.show()


def plotRebuiltSeries(model, data, n_series=12, dir_path=''):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    if n_series < 2:
        n_series = 2
    Y = model.predict([data.x_test[:n_series,:,:], data.dT_test[:n_series,:]])
    T = data.dT_test[:n_series,:]
    X = data.x_test[:n_series,:,0]
    Y = Y.reshape((Y.shape[0], Y.shape[1]))

    fig, ax = plt.subplots(math.ceil(len(T)/4), 4, figsize=(20, math.ceil(len(T)/4)*4))
    print('Real vs Rebuilt sequences')
    for i in range(len(T)):
        t = np.cumsum(T[i, :][X[i, :] != -10.])
        x = X[i, :][X[i, :] != -10.]
        y = Y[i, :][X[i, :] != -10.]
        line= ax[(i)//4, (i)%4].scatter(t, x)
        line.set_label('True curve')
        line2, = ax[(i)//4, (i)%4].plot(t, y, c='orange')
        line2.set_label('Rebuilt')
        ax[(i)//4, (i)%4].legend()
      
    plt.savefig(dir_path+'/AutoEnc_rebuilt.png')
    plt.show()


def getFreq(df, column):
    absolutes = df.groupby(column).count().values[:,0]
    return absolutes/np.sum(absolutes)




def retrieveFeaturesIndexes(df):
    features = df.columns
    data_ind = []
    feat_ind = [i for i in range(len(features))]

    for i in range(len(features)-1,-1,-1):
        try:
            string_integer = int(features[i])
            data_ind.append(feat_ind.pop(i))
        except ValueError:
            pass

    data_ind.reverse()
    return data_ind, feat_ind

def loadLocal(path):
    df = pd.read_csv(path, sep=';')
    data_ind, feat_ind = retrieveFeaturesIndexes(df)
    lenRow = df.shape[1]
    ids = list(pd.unique(df['object_id']))
    grouped = df.groupby(df['object_id'])
    tensor_flux = []
    n_objects = 0
    for id_ in ids:
        n_objects += 1
        df_new = grouped.get_group(id_)
        tensor_flux.append(df_new[np.array(df.columns)[data_ind]])
        sys.stdout.write('\r'+f'Loading object {n_objects} out of {len(ids)}          ')
    sys.stdout.write('\r'+f'\n')
    return np.array(tensor_flux), df

def retrieveMetadata(df):
    data_ind, feat_ind = retrieveFeaturesIndexes(df)
    id_ = -1
    rows = []
    ind_id = list(df.columns).index('object_id')
    for row in df.values:
        if row[ind_id] != id_:
            rows.append(row[feat_ind])
            id_ = row[ind_id]
            
    return pd.DataFrame(rows, columns=df.columns[feat_ind])

def runDeepModel(data, fold, name, params={}, model=None, prev_modelLoss=10**10,
                 save_model=True, force_train=False, show_plots=True):
    dir_path = 'models'
    extra = ''
    if data.name is not None:
        extra = data.name
        
    arch = ['LSTM - AutoEncoder', 'CNN - joined', 'Exotic', 'ResNet', 'MLP']
    loc_par = {'batch_size':32,
               'eval_batch':1,
               'epochs':80,
               'verbose':1,
               'validation_split':0.2,
               'n_bottleneck':20,
               'earlyS':False,
               'optimizer':'adam'
               }
    for k, v in params.items():
        loc_par[k] = v

    #if 'weights' in loc_par.keys():
    #    weights = np.array(loc_par['weights'])
    #else:
    #    weights = 1
    
    loss = params['loss']

    def plotMod():
        if show_plots:
            img = plt.imread(fold + dir_path+'/'+name+'.png')
            fig, ax = plt.subplots(figsize=(18, int(18*img.shape[0]/img.shape[1])))
            ax.imshow(img, aspect='auto')
            ax.axis('off')
            print(f'Model scheme -> {name}')
            plt.show()
            print('Model imported')

    def plotHist(history):
        if show_plots:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.plot(history.history["loss"], label = "train")
            ax.plot(history.history["val_loss"], label = "val")
            ax.legend()
            plt.savefig(fold +'images/' + name+'_Loss.png', dpi=150)
            plt.show()

    def evaluate(model, input_, target_):
        print('\n..Evaluating the model..')
        eval = model.evaluate([data.x_test, data.dT_test], data.y_test, batch_size=loc_par['eval_batch'])
        return eval

    if not force_train:
        try:
            print('Trying to load a previous model')
            if model is not None:
                model = keras.models.load_model(fold+dir_path+'/'+name+'_'+extra, compile=True,
                                                custom_objects={'loss': loss})
                plotMod()
                evaluate(model, [data.x_test, data.dT_test], data.y_test)
                output = model
            elif name == arch[0]:
                autoencod = keras.models.load_model(fold+dir_path+'/'+name+'_'+extra, compile=True, custom_objects={'loss': loss})        
                encod = keras.models.load_model(fold+dir_path+'/'+name+'_'+extra+'_core', compile=True, custom_objects={'loss': loss})
                plotMod()
                output = autoencod, encod
            elif name == arch[1]:
                model = keras.models.load_model(fold+dir_path+'/'+name+'_'+extra, compile=True, custom_objects={'loss': loss})
                plotMod()
                evaluate(model, [data.x_test, data.dT_test], data.y_test)
                output = model
            elif name == arch[2]:
                model = keras.models.load_model(fold+dir_path+'/'+name+'_'+extra, compile=True, custom_objects={'loss': loss})
                plotMod()
                evaluate(model, [data.x_test, data.dT_test], data.y_test)
                output = model
            elif name == arch[3]:
                model = keras.models.load_model(fold+dir_path+'/'+name+'_'+extra, compile=True, custom_objects={'loss': loss})
                plotMod()
                evaluate(model, [data.x_test, data.dT_test], data.y_test)
                output = model
            elif name == arch[4]:
                model = keras.models.load_model(fold+dir_path+'/'+name+'_'+extra, compile=True, custom_objects={'loss': loss})
                plotMod()
                evaluate(model, [data.x_test, data.dT_test], data.y_test)
                output = model
            else:
                print(f'Architecture not recognized. Try one of those below:\n{arch}')
                return
            
            print(f'[{name}] model ready!')
            return (output, None, None)
        except:
            pass
    
    if force_train:
        print(f'[{name}] model forced to train!')
    else:
        print(f'[{name}] model not found - New architecture to train!')
    calls = []
    if loc_par.get('earlyS', False) is not False:
      earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=loc_par['earlyS'])
      calls.append(earlyStopping)

    if name == arch[0]:
        autoencod, encod = AutoEncoder_model(data.x_train.shape, n_bottleneck=params['n_bottleneck'],
                                             params=loc_par, mask_value=data.mask_value)
        autoencod.compile(optimizer=loc_par['optimizer'], loss=loss)
        if show_plots:
            plotModel(autoencod, fold + dir_path, name=name)
        history = autoencod.fit([data.x_train, data.dT_train], data.y_train, epochs=loc_par['epochs'], 
                                batch_size=loc_par['batch_size'], validation_split=loc_par['validation_split'], 
                                verbose=loc_par['verbose'], callbacks=calls)
        if save_model:
            autoencod.save(fold + dir_path + '/' + name)
            encod.save(fold + dir_path + '/' + name + '_core')
        eval = None
        output = (autoencod, encod)

    if name == arch[1]:
        convnet = CNNjoined_model(data.x_train.shape, data.dT_train.shape, n_classes=data.y_test.shape[1], params=loc_par)
        convnet.compile(loss=loss, optimizer=loc_par['optimizer'])
        if show_plots:
            plotModel(convnet, fold + dir_path, name=name)
        history = convnet.fit([data.x_train, data.dT_train], data.y_train, epochs=loc_par['epochs'], 
                                batch_size=loc_par['batch_size'], validation_split=loc_par['validation_split'], 
                                verbose=loc_par['verbose'], callbacks=calls)
        eval = evaluate(convnet, [data.x_test, data.dT_test], data.y_test)
        output = convnet

    if name == arch[2]:
        exotic = Exotic_model(data.x_train.shape, data.dT_train.shape, n_classes=data.y_test.shape[1], params=loc_par)
        exotic.compile(loss=loss, optimizer=loc_par['optimizer'])
        if show_plots:
            plotModel(exotic, fold + dir_path, name=name)
        history = exotic.fit([data.x_train, data.dT_train], data.y_train, epochs=loc_par['epochs'], 
                                batch_size=loc_par['batch_size'], validation_split=loc_par['validation_split'], 
                                verbose=loc_par['verbose'], callbacks=calls)
        eval = evaluate(exotic, [data.x_test, data.dT_test], data.y_test)
        output = exotic

    if name == arch[3]:
        resnet = ResNet_model(data.x_train.shape, data.dT_train.shape, n_classes=data.y_test.shape[1], params=loc_par)
        resnet.compile(loss=loss, optimizer=loc_par['optimizer'])
        if show_plots:
            plotModel(resnet, fold + dir_path, name=name)
        history = resnet.fit([data.x_train, data.dT_train], data.y_train, epochs=loc_par['epochs'], 
                                batch_size=loc_par['batch_size'], validation_split=loc_par['validation_split'], 
                                verbose=loc_par['verbose'], callbacks=calls)
        eval = evaluate(resnet, [data.x_test, data.dT_test], data.y_test)
        output = resnet

    if name == arch[4]:
        mlp = MLP_model(data.x_train.shape, n_classes=data.y_test.shape[1], params=loc_par)
        mlp.compile(loss=loss, optimizer=loc_par['optimizer'])
        if show_plots:
            plotModel(mlp, fold + dir_path, name=name)
        history = mlp.fit(data.x_train, data.y_train, epochs=loc_par['epochs'], 
                                batch_size=loc_par['batch_size'], validation_split=loc_par['validation_split'], 
                                verbose=loc_par['verbose'], callbacks=calls)
        eval = evaluate(resnet, data.x_test, data.y_test)
        output = mlp

    try:
        if prev_modelLoss > eval or save_model:
            print('Saving the model..')
            output.save(fold + dir_path + '/' + name + '_' + extra)
    except:
        None

    plotHist(history)
    return (output, eval, len(history.history['val_loss']))



def randomSearch(input, folder, name, params, n_iter):
    input_name = input.name
    prev_loss = 10**10
    if name == 'CNN - joined':
        other_params = [['f_filter', 32, 64, 96, 128, 156, 256],
                        ['f_kernel', 1, 2, 3, 4, 5],
                        ['s_kernel', 1, 2, 3, 4, 5],
                        ['t_kernel', 1, 2, 3, 4, 5],
                        ['f_drop', 0.1, 0.2, 0.3, 0.4, 0.5],
                        ['s_filter', 32, 64, 96, 128, 156, 256],
                        ['f_dense', 32, 64, 96, 128],
                        ['s_drop', 0.1, 0.2, 0.3, 0.4, 0.5],
                        ['t_filter', 32, 64, 96, 128, 256]]
    elif name == 'Exotic':
        other_params = [['f_gru', 32, 64, 96, 128, 156],
                        ['f_drop', 0.1, 0.2, 0.3, 0.4, 0.5],
                        ['s_gru', 32, 64, 96, 128, 156],
                        ['f_dense', 32, 64, 96, 128],
                        ['s_dense', 32, 64, 96, 128],
                        ['s_drop', 0.1, 0.2, 0.3, 0.4, 0.5],
                        ['t_dense', 32, 64, 96, 128]]
    cols = np.array([v[0] for v in other_params] + ['test_loss', 'Epochs'])
    cols_str = str([str(el) for el in cols])
    cols_str = str(cols_str)[1:-1].replace("'", '').replace(', ', ';')
    text_file = folder + 'models/'+ name + '_' + input_name+'.txt'

    if not os.path.isfile(text_file):
        with open(text_file, "a+") as file:
            file.write(cols_str + '\n')
    else:
        data = pd.read_csv(text_file, sep=';')
        if len(data['test_loss'].values) > 0:
            prev_loss = np.min(data['test_loss'].values)
    print(f'Previous loss: {prev_loss}')
    p=[]
    for i in range(n_iter):
        p.append([(l[0], random.choice(l[1:])) for l in other_params])

    overall_eval = []
    i = 0
    for cop in p:
        i += 1
        values = np.array(cop).T
        print(f'\n\nIteration:{i}/{n_iter}\nParams:')
        print(pd.DataFrame(values))
        cur_params = params.copy()
        for couple in cop:
            cur_params[couple[0]] = couple[1]
        exotic, loss, n = runDeepModel(input, fold=folder, name=name, params=cur_params, save_model=False,
                                        force_train=True, show_plots=False, prev_modelLoss=prev_loss)
        overall_eval.append((cop, loss))
        if loss < prev_loss:
            prev_loss = loss

        values = np.array(np.r_[np.array(cop).T[1], np.array([loss, n])])
        values_str = str(values)[1:-1].replace("'", '').replace(' ', ';')
        with open(text_file, "a+") as file:
            file.write(values_str + '\n')


def ResNet_model(input_shape, input_shape_metadata, params, n_classes):
    input_metadata = Input((input_shape_metadata[1], ))
    input_layer = Input((input_shape[1], input_shape[2]))
    
   # metadata = Activation("tanh")(input_metadata)

    convx = Conv1D(filters = 16, kernel_size = 3, padding = 'same')(input_layer)
    convx = BatchNormalization()(convx)
    convx= Activation('relu')(convx)
    convy = Conv1D(filters = 16, kernel_size = 3, padding = 'same')(convx)
    convy = BatchNormalization()(convy)
    convy = Activation('relu')(convy)
    convz = Conv1D(filters = 16, kernel_size = 3, padding = 'same')(convy)
    convz = BatchNormalization()(convz)
    shortcut_y = Conv1D(filters = 16, kernel_size = 1, padding = 'same')(input_layer)
    shortcut_y  = BatchNormalization()(shortcut_y)
    output_block_1 = Add()([shortcut_y, convz])
    output_block_1 = Activation('relu')(output_block_1)
    convx = Conv1D(filters = 16 *2, kernel_size = 5, padding = 'same')(output_block_1)
    convx = BatchNormalization()(convx)
    convx=Activation('relu')(convx)
    convy = Conv1D(filters = 16*2, kernel_size = 5, padding = 'same')(convx)
    convy = BatchNormalization()(convy)
    convy = Activation('relu')(convy)
    convz = Conv1D(filters = 16*2, kernel_size = 5, padding = 'same')(convy)
    convz = BatchNormalization()(convz)
    shortcut_y = Conv1D(filters = 16*2, kernel_size = 1, padding = 'same')(output_block_1)
    shortcut_y  = BatchNormalization()(shortcut_y)
    output_block_2 = Add()([shortcut_y, convz])
    output_block_2 = Activation('relu')(output_block_1)
    convx = Conv1D(filters = 32 *2, kernel_size = 5, padding = 'same')(output_block_2)
    convx = BatchNormalization()(convx)
    convx = Activation('relu')(convx)
    convy = Conv1D(filters = 32*2, kernel_size = 5, padding = 'same')(convx)
    convy = BatchNormalization()(convy)
    convy = Activation('relu')(convy)
    convz = Conv1D(filters = 32*2, kernel_size = 5, padding = 'same')(convy)
    convz = BatchNormalization()(convz)
    shortcut_y = Conv1D(filters = 32*2, kernel_size = 1, padding = 'same')(output_block_2)
    output_block_3 = Add()([shortcut_y, convz])
    output_block_3 = Activation('relu')(output_block_3)
    final_layer = AveragePooling1D()(output_block_3)
    flatten_layer = Flatten()(final_layer)
    #flatten_layer = Activation("tanh")(flatten_layer)
    flatten_layer = Concatenate(axis = 1)([flatten_layer,input_metadata])#prova metadata
    #flatten_layer = Dense(50, activation = "relu")(flatten_layer)
    flatten_layer = Dropout(0.5)(flatten_layer)
    output_layer = Dense(n_classes, activation="softmax")(flatten_layer)
    
    model = Model(inputs=[input_layer, input_metadata], outputs=output_layer)
    return model


def AutoEncoder_model(inp_shape, n_bottleneck, params, mask_value):
    x = Input(shape=(inp_shape[1], inp_shape[2]))
    dt = Input(shape=(inp_shape[1], 1))

    if mask_value is not None: e = Masking(mask_value=mask_value)(x)
    else: e = x

    e = Bidirectional(GRU(96, return_sequences=True))(e)
    e = Dropout(0.25)(e)
    e = Bidirectional(GRU(96))(e)
    bottleneck = Dense(n_bottleneck)(e)
    d = RepeatVector(inp_shape[1])(bottleneck)
    d = Concatenate(axis=2)([d, dt])
    d = Bidirectional(GRU(96, return_sequences=True))(d)
    d = Dropout(0.25)(d)
    d = Bidirectional(GRU(96, return_sequences=True))(d)
    output = TimeDistributed(Dense(1))(d)

    autoencod = Model(inputs=[x, dt], outputs=output)
    encoder = Model(inputs=[x, dt], outputs=bottleneck)
    return autoencod, encoder


def CNNjoined_model(input_shape, input_shape_metadata, params, n_classes):
    input_metadata = Input((input_shape_metadata[1],))
    input_layer = Input((input_shape[1], input_shape[2]))
    normMeta = BatchNormalization(axis=1)(input_metadata)

    c = Conv1D(filters=params['f_filter'], kernel_size=params['f_kernel'], padding='same')(input_layer)
    c = Activation("relu")(c)
    c = tf.keras.layers.MaxPooling1D()(c)
    c = Conv1D(filters=params['s_filter'], kernel_size=params['s_kernel'], padding='same')(c)
    c = Dropout(params['f_drop'])(c)
    c = Activation("relu")(c)
    c = tf.keras.layers.MaxPooling1D()(c)
    #c = Conv1D(filters=params['t_filter'], kernel_size=params['t_kernel'], padding='same')(c)
    #c = Activation("relu")(c)
    #c = tf.keras.layers.MaxPooling1D()(c)
    flatten_layer = Flatten()(c)
    last = Concatenate(axis=1)([flatten_layer, normMeta])
    last = Dense(params['f_dense'], activation="relu")(last)
    last = Dropout(params['s_drop'])(last)
    output_layer = Dense(n_classes, activation="softmax")(last)

    model = Model(inputs=[input_layer, input_metadata], outputs=output_layer)
    return model


def Exotic_model(input_shape, input_shape_metadata, params, n_classes):
    input_metadata = Input((input_shape_metadata[1],))
    input_layer = Input((input_shape[1], input_shape[2]))
    e = input_layer
    e = Masking(mask_value=-10.)(e)

    e = GRU(params['f_gru'], return_sequences=True)(e)
    e = Dropout(params['f_drop'])(e)
    e = GRU(params['s_gru'], return_sequences=False)(e)
    flatten_layer = Dense(params['f_dense'], activation="relu")(e)

    normMeta = BatchNormalization(axis=1)(input_metadata)
    flatten_layer = Concatenate(axis=1)([flatten_layer, normMeta])
    flatten_layer = Dense(params['s_dense'], activation="relu")(flatten_layer)
    flatten_layer = Dropout(params['s_drop'])(flatten_layer)
    flatten_layer = Dense(params['s_dense'], activation="relu")(flatten_layer)
    flatten_layer = Dropout(params['s_drop'])(flatten_layer)
    flatten_layer = Dense(params['t_dense'], activation="relu")(flatten_layer)
    output_layer = Dense(n_classes, activation="softmax")(flatten_layer)

    model = Model(inputs=[input_layer, input_metadata], outputs=output_layer)
    return model


def MLP_model(input_shape, params, n_classes):
    input_layer = Input((input_shape[1], ))

    flatten_layer = Dense(100, activation = "relu")(input_layer)
    flatten_layer = Dense(100, activation = "relu")(flatten_layer)
    output_layer = Dense(n_classes, activation = "softmax")(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def showRandomSearch_results(folder=os.getcwd()):
    folder = os.getcwd() + '/'
    onlyfiles = [f for f in listdir(folder+'models') if isfile(join(folder+'/models', f))]
    onlyText = [f[:len(f)-4] for f in onlyfiles if f[len(f)-3:] == 'txt']
    configuration = {f: pd.read_csv('models/'+f+'.txt', sep=';') for f in onlyText}
    best = {f: np.min(np.nan_to_num(configuration[f]['test_loss'].values, nan=10**10)) for f in onlyText}
    
    best = {f: pd.DataFrame(configuration[f].values[configuration[f]['test_loss'].values == best[f]],
                            columns=[nam[:5] for nam in configuration[f].columns]) for f in onlyText}

    for k, v in best.items():
        print(str(k) + ':  ' + tabulate(v, headers='keys', tablefmt='psql'), end='\n\n')

print('Functions for the modeling and evaluation phase successfully imported!')