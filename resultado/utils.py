import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
from eolearn.core import EOPatch
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from numpy.random import seed
from sklearn.model_selection import cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

def reshape_bands(bands):
    '''
    Calculates the mean value for every band and timestamp images.
    '''
    t, h, w, b = bands.shape
    return bands.reshape(t, h*w, b).mean(axis=1)

def band_to_dataframe(data, index):
    '''
    Creates a dataframe with dates as index and
    bands as columns.
    '''
    return pd.DataFrame(data=data, index=index)

def band_interpolate(df, days, method, order):
    '''
    Interpolates values for missing dates.
    '''
    return df.resample('1D').mean().reindex(days).interpolate(method=method, order = 3)

def pipeline(eopatch, days, layers):
    '''
    Transform one eopatch into a dataframe with dates as
    index and bands as columns.
    '''
    p = np.concatenate([eopatch.data[layer] for layer in layers], axis=-1)
    p = reshape_bands(p)
    p = band_to_dataframe(p, eopatch.timestamp)
    p = band_interpolate(p, days, method = 'akima', order = 3)
    return p

def band_selection(df, bands):
    '''
    Selects bands
    '''
    return df[bands]

def resampling(df, freq):
    '''
    Resamples a dataframe with the mean
    '''
    return df.resample(freq).mean()

def flattener(df):
    '''
    Converts a dataframe to a flatten numpy array
    '''
    return df.values.flatten()

def create_geodataframe(buffer):
    '''
    Creates a GeoDataframe with test and train points.
    '''
    train_1819 = pd.read_csv('data/train_1819.csv')
    train_1920 = pd.read_csv('data/train_1920.csv')
    train_new_points = pd.read_csv('data/new_points_with_global_ids.csv')
    df_test = pd.read_csv('data/test_corregido.csv').drop_duplicates()
    df_train = pd.concat([train_1920, train_1819.drop_duplicates(), train_new_points]).drop(columns='field_1')
    df = pd.concat([df_train, df_test]).set_index('GlobalId')
    df.drop(df[(df.Cultivo == 'S/M') | (df.Cultivo == 'T')].index, inplace=True)
    geometry = [Point(xy) for xy in zip(df.Longitud, df.Latitud)]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326').to_crs('EPSG:32720')
    gdf.geometry = gdf.geometry.buffer(buffer)
    
    return gdf

class Preprocessor:
    '''
    Class to read and preprocess EoPatches downloaded
    '''
    def __init__(self, PATH, FREQ, TIME_INTERVAL, BANDS, GDF, LAYERS):
        self.PATH = PATH
        self.FREQ = FREQ
        self.TIME_INTERVAL = TIME_INTERVAL
        self.BANDS = BANDS
        self.GDF = GDF
        self.LAYERS = LAYERS
        self.X_train, self.y_train = self.train_preprocessing()
        self.X_test = self.test_preprocessing()

    def train_preprocessing(self):
        
        X_train, y_train = [], []
        eopatches = os.listdir(self.PATH)
        train_global_ids = self.GDF[~self.GDF.Cultivo.isna()].index.tolist()
        test_global_ids = self.GDF[self.GDF.Cultivo.isna()].index.tolist()
        
        for patch in tqdm( eopatches, position=0, leave=True, desc='Train preprocessing'):

            global_id = int(patch.split('_')[1])

            if global_id in train_global_ids:

                eopatch = EOPatch.load(f'{self.PATH}/{patch}', lazy_loading=True)
                df = pipeline(eopatch, self.TIME_INTERVAL[self.GDF.loc[global_id, 'Campania']], self.LAYERS)
                df = band_selection(df, self.BANDS)
                df = resampling(df, self.FREQ)
                X_train.append(flattener(df.ffill().bfill()))

                y_train.append(self.GDF.loc[global_id, 'Cultivo'])
        
        return np.array(X_train), np.array(y_train)

    def test_preprocessing(self):

        X_test = []
        global_ids = []
        eopatches = os.listdir(self.PATH)
        train_global_ids = self.GDF[~self.GDF.Cultivo.isna()].index.tolist()
        test_global_ids = self.GDF[self.GDF.Cultivo.isna()].index.tolist()

        for patch in tqdm( eopatches, position=0, leave=True, desc='Test preprocessing'):

            global_id = int(patch.split('_')[1])

            if global_id in test_global_ids:
                global_ids.append(global_id)
                eopatch = EOPatch.load(f'{self.PATH}/{patch}', lazy_loading=True)
                df = pipeline(eopatch, self.TIME_INTERVAL[self.GDF.loc[global_id, 'Campania']], self.LAYERS)
                df = band_selection(df, self.BANDS)
                df = resampling(df, self.FREQ)
                X_test.append(flattener(df.ffill().bfill()))
            
        return np.array(X_test)
    
def grid_search(X, y):
    
    '''
    Grid search for 5 SVC models using bootstrapping
    '''
    
    folds = StratifiedShuffleSplit(n_splits=5, random_state=26)
    bootstrap_idx = [x for x, _ in folds.split(X, y)]

    params = {
        'model__gamma': np.arange(0.005,0.1,0.005),
        'model__C': np.arange(5,100,5),
    }

    os_dict = {}

    for k,v in pd.Series(y).value_counts().to_dict().items():
        if v < 25:
            os_dict[k] = 25
        else:
            os_dict[k] = v
            
    steps = [('oversampling', SMOTE(sampling_strategy=os_dict, random_state=26, k_neighbors=6)),
         ('model', SVC())]

    pipe = Pipeline(steps)

    results = []
    cv_folds = StratifiedKFold(n_splits=3, random_state=26, shuffle=True)

    for bootstrap in tqdm( bootstrap_idx ):
        X_, y_ = X[bootstrap], y[bootstrap]
        grid = GridSearchCV(pipe, params, scoring='balanced_accuracy', cv=cv_folds.split(X_, y_), n_jobs=-1, verbose=0)
        grid.fit(X_, y_)
        results.append(pd.DataFrame(grid.cv_results_).sort_values('rank_test_score').iloc[0])
    
    results_dict = (
        pd.DataFrame(results)[['param_model__C','param_model__gamma']]
        .rename(columns={'param_model__C':'C', 'param_model__gamma':'gamma'})
        .to_dict('records')
    )

    for dict_value in results_dict:
        for k, v in dict_value.items():
            dict_value[k] = round(v, 3)
    
    models = []
    
    for params in results_dict:
        models.append(SVC(**params))
    
    return models, bootstrap_idx

def predict_and_submit(preprocessor, models, metamodel, bootstrap_idx):

    os_dict = {}

    for k,v in pd.Series(preprocessor.y_train).value_counts().to_dict().items():
        if v < 25:
            os_dict[k] = 25
        else:
            os_dict[k] = v

    predictions = [preprocessor.y_train]
    predictions_test = []

    for model, idx in zip(models, bootstrap_idx):
        X_, y_ = SMOTE(
            random_state=26, 
            sampling_strategy=os_dict, 
            k_neighbors=6).fit_resample(preprocessor.X_train[idx], preprocessor.y_train[idx])

        model.fit(X_, y_)     
        predictions.append(model.predict(preprocessor.X_train))
        predictions_test.append(model.predict(preprocessor.X_test))

    metamodel_train = pd.DataFrame(predictions).T
    le = LabelEncoder().fit(preprocessor.y_train)
    metamodel_train = metamodel_train.apply(lambda x: le.transform(x))

    metamodel.fit(metamodel_train.iloc[:,1:].values, metamodel_train.iloc[:,0].values)

    metamodel_test = pd.DataFrame(predictions_test).T
    metamodel_test = metamodel_test.apply(lambda x: le.transform(x))

    predictions = metamodel.predict(metamodel_test.values)
    predictions = le.inverse_transform(predictions)

    cultivos_id = pd.read_csv('data/Etiquetas.csv')
    dict_map = pd.Series(cultivos_id.Cultivo.values ,index=cultivos_id.CultivoId).to_dict()
    dict_map = {v:int(k) for k,v in dict_map.items()}
    predictions = pd.Series(predictions).map(dict_map)
    sub = pd.DataFrame(columns = ['globalid','pred'])

    global_ids = []
    eopatches = os.listdir(preprocessor.PATH)
    test_global_ids = preprocessor.GDF[preprocessor.GDF.Cultivo.isna()].index.tolist()

    for patch in eopatches:

        global_id = int(patch.split('_')[1])

        if global_id in test_global_ids:
            global_ids.append(global_id)

    sub['globalid'] = global_ids
    sub['pred'] = predictions
    sub.to_csv('data/final_submission.csv', header = False, index = False)
    print('Submission file created.')