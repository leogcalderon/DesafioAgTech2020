import warnings
import pickle
from sklearn.ensemble import RandomForestClassifier
from numpy.random import seed
from utils import *
  
seed(26)
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    LAYERS = ['BANDS', 'NDVI', 'NDWI', 'NDBI']
    TIME_INTERVAL = {
        '18/19' : pd.date_range(start='2018-11-01', end='2019-04-15', freq='D'),
        '19/20' : pd.date_range(start='2019-11-01', end='2020-04-15', freq='D')
    }
    BANDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    BUFFER = 50

    gdf = create_geodataframe(BUFFER)
    preprocessor = Preprocessor('eopatches/', '5D', TIME_INTERVAL, BANDS, gdf, LAYERS)
    metamodel = RandomForestClassifier(random_state=26)
    
    with open('pickle_files/models.pkl', 'rb') as f:
        models = pickle.load(f)

    with open('pickle_files/bootstrap_idx.pkl', 'rb') as f:
        bootstrap_idx = pickle.load(f)

    metamodel = RandomForestClassifier(random_state=26)
    predict_and_submit(preprocessor, models, metamodel, bootstrap_idx)
