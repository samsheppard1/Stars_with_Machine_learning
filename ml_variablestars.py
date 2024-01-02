!pip install astroquery
!pip install tensorflow
!pip install lightkurve

import astroquery
from tensorflow import keras 
import matplotlib.pyplot as plt 
import os

cwd = os.getcwd()
print("Current Working Directory:", cwd)
files = os.listdir(cwd)
print("Files and directories in '", cwd, "':")
print(files)

modellocation = '/home/jovyan/TESS/variabilitymodel.h5'
model = keras.models.load_model(modellocation)

from astroquery.mast import Observations
# Define the sector number you're interested in
sector_number = 26
# Query the MAST archive for data from this sector
obsTable = Observations.query_criteria(provenance_name="TESS-SPOC", sequence_number=sector_number)

# Extract the first 'n' TIC IDs from the results
n = 20000 # Set 'n' to the number of TIC IDs you want tic_ids = obsTable['target_name'][:n]
print(tic_ids)
dataset = [(tic_id, sector_number) for tic_id in tic_ids]

from lightkurve import search_targetpixelfile 
import lightkurve as lk
import io

import tensorflow as tf 
from PIL import Image 
import numpy as np

for tic_id, sector_id in dataset:
    pixelfile = search_targetpixelfile(f'TIC {tic_id}', sector=sector_id, quarter=16).download();
    lc = pixelfile.to_lightcurve(aperture_mask='all') lc = lc.remove_outliers(sigma=10)
    lc = lc.normalize()
    fig, ax2 = plt.subplots(figsize=(6,4))
    lc_bin = lc.bin(time_bin_size = 0.05) plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = image.convert('RGB')
    resize = tf.image.resize(image, (256,256)) np.expand_dims(resize, 0)
    yhat = model.predict(np.expand_dims(resize/255, 0))
    if yhat > 0.5:
        print('Tic', tic_id, 'is a variable star')
        with open('/home/jovyan/TESS/bigdata.txt', 'a') as file:
        file.write(f'Tic {tic_id} is a variable star\n') else:
        print('Tic', tic_id, 'is not a variable star')
    else:
        print('Tic', tic_id, 'is not a variable star')