import pickle
from utils import *
import warnings
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
    
    models, bootstrap_idx = grid_search(preprocessor.X_train, preprocessor.y_train)
    
    with open('pickle_files/models.pkl', 'wb') as f:
        pickle.dump(models, f)

    with open('pickle_files/bootstrap_idx.pkl', 'wb') as f:
        pickle.dump(bootstrap_idx, f)