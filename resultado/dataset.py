import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import datetime
import matplotlib.pyplot as plt
import sys
import os

from sentinelhub import (
    UtmZoneSplitter,
    BBox,
    CRS,
    DataCollection,
)

from eolearn.core import (
    EOTask,
    EOPatch,
    LinearWorkflow,
    FeatureType,
    OverwritePermission,
    LoadTask,
    SaveTask,
    EOExecutor,
    #ExtractBandsTask,
    MergeFeatureTask,
)

from eolearn.io import (
    SentinelHubInputTask,
    #ExportToTiff,
)

from eolearn.features import (
    LinearInterpolation,
    #SimpleFilterTask,
    NormalizedDifferenceIndexTask,
)

from eolearn.geometry import PointSamplingTask

if __name__ == '__main__':

    SAVE_PATH = 'eopatches/'
    BUFFER = 50
    BANDS = '[2,3,4,5,6,8,11,12]'
    MAXCC = 0.2

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    train_1819 = pd.read_csv('data/train_1819.csv')
    train_1920 = pd.read_csv('data/train_1920.csv')
    train_new_points = pd.read_csv('data/new_points_with_global_ids.csv')
    df_test = pd.read_csv('data/test_corregido.csv').drop_duplicates()
    df_train = pd.concat([train_1920, train_1819.drop_duplicates(), train_new_points]).drop(columns='field_1')
    df = pd.concat([df_train, df_test]).set_index('GlobalId')
    
    #Geopandas df
    geometry = [Point(xy) for xy in zip(df.Longitud, df.Latitud)]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326').to_crs('EPSG:32720')
    gdf.geometry = gdf.geometry.buffer(BUFFER)
    
    df.drop(['Longitud', 'Latitud', 'Elevacion', 'Dataset', 'Id'], axis=1, inplace=True)
    
    #SentinelHubInputTask
    band_names = []
    for band in list(map(int, BANDS.strip('[]').split(','))):
        if int(band) < 10:
            band_names.append(f'B0{band}')
        else:
            band_names.append(f'B{band}')
    
    add_data = SentinelHubInputTask(
        bands_feature=(FeatureType.DATA, 'BANDS'),
        bands=band_names,
        resolution=10,
        maxcc=MAXCC,
        data_collection=DataCollection.SENTINEL2_L1C,
        additional_data=[(FeatureType.MASK, 'dataMask'),
                         (FeatureType.MASK, 'CLM'),
                         (FeatureType.DATA, 'CLP')],
        max_threads=5
    )
    
    #IndexTasks
    ndvi = NormalizedDifferenceIndexTask(
        (FeatureType.DATA, 'BANDS'), (FeatureType.DATA, 'NDVI'),
        [band_names.index('B08'),band_names.index('B04')]
    )

    ndwi = NormalizedDifferenceIndexTask(
        (FeatureType.DATA, 'BANDS'), (FeatureType.DATA, 'NDWI'), 
        [band_names.index('B03'), band_names.index('B08')]
    )

    ndbi = NormalizedDifferenceIndexTask(
        (FeatureType.DATA, 'BANDS'), (FeatureType.DATA, 'NDBI'), 
        [band_names.index('B11'), band_names.index('B08')]
    )
    
    #SaveTask
    save = SaveTask(SAVE_PATH,
                overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
                
    #Workflow
    workflow = LinearWorkflow(
        add_data,
        ndvi,
        ndwi,
        ndbi,
        save,
    )
    
    time_interval = {
        '18/19' : ['2018-11-01', '2019-05-01'],
        '19/20' : ['2019-11-01', '2020-05-01'],
    }
    
    downloaded = os.listdir(SAVE_PATH)
    downloaded = list(map(lambda x: int(x.split('_')[1]), downloaded))
    execution_args = []

    for id, row in gdf.loc[~gdf.index.isin(downloaded),:].iterrows():
        bbox = row.geometry.bounds
        bbox = BBox(bbox, CRS('32720'))
        execution_args.append({
            add_data:{'bbox': bbox, 'time_interval': time_interval[row.Campania]},
            save: {'eopatch_folder': f'eopatch_{id}'}
        })
        
    executor = EOExecutor(workflow, execution_args)
    executor.run(workers=None, multiprocess=True)
    executor.make_report()
