#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 18:53:02 2024

@author: samsheppard
"""

import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import os
from lightkurve import search_targetpixelfile import random
from PIL import Image
from collections import defaultdict

!pip install lightkurve

# We next need to build our folder structure to save our results

import os
root_folder = r"/Data" ## ENTER ROOT FOLDER HERE
# List of subdirectories to be created
subdirs = [
    "fourierAnalysis",
    "InitalStarData",
    "InitalStarData/StarList",
    "InitalStarData/StarMetaData",
    "VaraibleStars",
    "VaraibleStars/StarList"
]
for subdir in subdirs:
    os.makedirs(os.path.join(root_folder, subdir), exist_ok=True)

# Next You should add your list of target stars (TIC ID and Sector number) to the location specified below

targetStars = 'Data/InitalStarData/Starlist/targetstars.txt'
targetStarsPlot = 'Data/InitalStarData/StarMetaData'

0.0.1 Defining functions

# Here we will define general usecase functions, these include some bolier plate code to read and write to text files, others are extracting data from the TESS database.


def read_target_file(file_path): 
    targets = []
    with open(file_path, 'r') as file: for line in file:
        if line.strip():
            tic_id, sector_id = map(int, line.split()) 
            targets.append((tic_id, sector_id))
    return targets
def plot_target(tic_id, sector_id, outputfolder):
    # Inital try and except function to allow smooth worflow for itteration through dataset
    try:
        pixelfile = lk.search_targetpixelfile(f'TIC {tic_id}', sector=sector_id).download() 
    except lk.search.SearchError:
        print(f"No data found for TIC {tic_id}, Sector {sector_id}.")
        return
    except Exception as e:
        print(f"An error occurred while trying to download data for TIC␣ ↪{tic_id}, Sector {sector_id}: {str(e)}")
        return
    ## If star data exists, then the following fill run
    try:
        pixelfile = search_targetpixelfile(f'TIC {tic_id}', sector=sector_id, quarter=16).download(); 
        pixelfile.plot(frame=1);
        plt.title(f'TIC {tic_id} Pixel Map') plt.savefig(f'{outputfolder}/{tic_id}_pixels.png', format='png') 
        plt.close()
        lc = pixelfile.to_lightcurve(aperture_mask='all')
        lc = lc.remove_outliers(sigma=10)
        lc = lc.normalize()
        lc.time
        lc.flux
        fig = plt.figure(figsize=(6,4))
        plt.errorbar(lc.time.mjd-lc.time[0].mjd, lc.flux, lc.flux_err, fmt='.', ecolor='LightGrey')
        plt.xlabel("Time (days)")
        plt.ylabel('Normalized Flux')
        plt.title(f'TIC {tic_id} Normalized Flux vs. Time') plt.savefig(f'{outputfolder}/{tic_id}_fluxtimegraph.png', format='png')
        plt.close()
    except Exception as e:
        print(f"An error occurred while processing TIC {tic_id}, Sector {sector_id}: {str(e)}")
        
def plotimages(folder_path, num_to_plot):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png'))]
    selected_images = random.sample(image_files, min(num_to_plot,len(image_files)))
    fig, ax = plt.subplots(ncols=min(num_to_plot, len(selected_images)), figsize=(20, 20))
    for idx, img_file in enumerate(selected_images):
        img_path = os.path.join(folder_path, img_file) 
        img = Image.open(img_path)
        ax[idx].imshow(img) 
        ax[idx].title.set_text(img_file)
        
    plt.show()
    
def plot_all_targets(file_path, endfile): 
    targets = read_target_file(file_path) 
    for tic_id, sector_id in targets:
        plot_target(tic_id, sector_id, endfile)
        
def calculate_cdpp_errors(target_list):
    cdpp_errors = {}
    for tic_id, sector_id in target_list:
        try:
            pixelfile = search_targetpixelfile(f'TIC {tic_id}',sector=sector_id, quarter=16).download();
            lc = pixelfile.to_lightcurve(aperture_mask='all') 
            if lc is not None:
                lc = pixelfile.to_lightcurve(aperture_mask='all')
                lc = lc.remove_outliers(sigma=10)
                lc = lc.normalize()
                cdpp_error = lc.estimate_cdpp(transit_duration=13)
                cdpp_errors[(tic_id, sector_id)] = cdpp_error.value
            else:
                cdpp_errors[(tic_id, sector_id)] = None
        except Exception as e:
            print(f"An error occurred for TIC {tic_id}, Sector {sector_id}: {str(e)}")
            cdpp_errors[(tic_id, sector_id)] = None
        return cdpp_errors
    
def calculate_cdpp_errors(target_list): 
    cdpp_errors = {}
    cess each target
    for tic_id, sector_id in target_list:
        try:
            # Attempt to download the pixel file and convert to a light curve
            pixelfile = lk.search_targetpixelfile(f'TIC {tic_id}', sector=sector_id).download()
            lc = pixelfile.to_lightcurve(aperture_mask='all')
            lc = lc.remove_outliers(sigma=10)
            lc = lc.normalize()
            # Estimate CDPP and store the value
            cdpp_error = lc.estimate_cdpp(transit_duration=13)
            cdpp_errors[(tic_id, sector_id)] = cdpp_error.value 
        except Exception as e:
            # Handle exceptions and mark CDPP as None for failures
            print(f"An error occurred for TIC {tic_id}, Sector {sector_id}:␣ ↪{str(e)}")
            cdpp_errors[(tic_id, sector_id)] = None
            
    # Organize CDPP values by TIC ID and calculate average CDPP per TIC
    cdpp_by_tic = defaultdict(list)
    for (tic_id, sector_id), cdpp in cdpp_errors.items():
        if cdpp is not None: 
            cdpp_by_tic[tic_id].append((sector_id, cdpp))
    # Calculate the average CDPP for each TIC ID and sum total errors
    average_cdpp_by_tic = {}
    for tic_id, sector_cdpps in cdpp_by_tic.items():
        sectors, cdpps = zip(*sector_cdpps)
        average_cdpp = sum(cdpps) / len(cdpps)
        total_error = sum(cdpps)
        average_cdpp_by_tic[tic_id] = {
            'average_cdpp': average_cdpp,
            'sectors': sectors,
            'total_error': total_error
        }
    return average_cdpp_by_tic


targets = read_target_file("interestingstars.txt")
average_cdpp_errors = calculate_cdpp_errors(targets)
for tic_id, data in average_cdpp_errors.items():
    print(f"TIC {tic_id} in sectors {data['sectors']} has an average CDPP of {data['average_cdpp']} and a total error of {data['total_error']}")

targets = read_target_file("targetstars.txt")
cdpp_errors = calculate_cdpp_errors(targets)
print(cdpp_errors)

plot_all_targets(targetStars, targetStarsPlot)
plotimages(targetStarsPlot, 10)

#0.1 Data
#After selecting 5 stars to analyse, we will focus on increasing the underlying signal whilst reducing noise, in order to give a more precise timeseries graph to analyse. TESS provide two main methods to clean data, which are flattening the curve and binning data. We will start looking at reducing the standard deviation on the data by binning the time. By averageing the flux values over a time period, we reduce the standard deviation massivley, but risk information loss. There are a number of methods we could use to predict the potential period of these stars, but we will focus on two, the Gaussian Process Regression and Wavelet Transfrom method. Flattening data is not of much use as we arent looking for underlying signals, but rather the complete collection.

import pywt
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel 
import scipy.signal
import scipy
from lightkurve import search_targetpixelfile

# Next we will analyse our binned data, TESS has a build in periodogram function, which performs a fourier transfomration to the timeseries data, highlighting its peak underlying frequnecies. We then apply the box least Square method to amplify the underlying frequnecies. Once we have found the maximum power frequency, we fold the frequency over all its


def fourier_analysis(tic_id, sector_id, output_dir): try:
    pixelfile = lk.search_targetpixelfile(f'TIC {tic_id}', sector=sector_id).download()
    except lk.search.SearchError:
        print(f"No data found for TIC {tic_id}, Sector {sector_id}.") 
        return
    except Exception as e:
        print(f"An error occurred while trying to download data for TIC {tic_id}, Sector {sector_id}: {str(e)}") return
    try:
        ## Initalise data payload
        lc = pixelfile.to_lightcurve(method="pld").remove_outliers().normalize().remove_nans()
        lc_file = os.path.join(output_dir, f'TIC_{tic_id}_Sector_{sector_id}_lc.png')
        plt.savefig(lc_file) 
        #plt.show()
        lc2 = lc.scatter()
        lc2_file = os.path.join(output_dir, f'TIC_{tic_id}_Sector_{sector_id}_scatter.png')
        plt.savefig(lc2_file) 
        #plt.show()
        ##fourier analysis using periodogram
        lc.to_periodogram("bls").plot()
        period_file = os.path.join(output_dir, f'TIC_{tic_id}_Sector_{sector_id}_periodogram.png')
        plt.savefig(period_file) 
        #plt.show()
        
        period_power = lc.to_periodogram("bls").period_at_max_power
        power = lc.to_periodogram("bls").power
        periods = lc.to_periodogram("bls").period
        print(power.shape)
        print(power)
        distance = int(round(len(power.value) / 32))
        threshold = round(max(power.value) * 0.65)
        peak_indices, _ = scipy.signal.find_peaks(power, prominence=threshold)
        peak_periods = periods [peak_indices]
        
        print("Peak indices:", peak_indices)
        print("Power at peaks:", power[peak_indices])
        print("Periods at peaks:", peak_periods)
        
        for period in peak_periods:
            lc.fold(period = period).scatter() 
            plt.title(f'TIC {tic_id}, {period}') 
            power_file = os.path.join(output_dir, f'TIC_{tic_id}_Sector_{sector_id}_powerfold{period}.png') 
            plt.savefig(power_file)
        
       period_frequency = lc.to_periodogram("bls").frequency_at_max_power
       lc.fold(period = period_power).scatter()
       power_file = os.path.join(output_dir, f'TIC_{tic_id}_Sector_{sector_id}_powerfold.png')
       plt.savefig(power_file)
       #plt.show()
       return(tic_id, period_power)
   except Exception as e:
       print(f"An error occurred while processing TIC {tic_id}, Sector, {sector_id}: {str(e)}")

def getFourierTransform(file_path, endfile): 
    targets = read_target_file(file_path) 
    for tic_id, sector_id in targets:
        data = fourier_analysis(tic_id, sector_id, endfile)
        print(data)
        
binnedStarData = r"Data\VaraibleStars\StarList\interestingstars.txt"
fouierData = r"Data/fourierAnalysis/"
getFourierTransform("interestingstars.txt", "FourierAnalysis")

from astroquery.mast import Observations, Catalogs 
import pandas as pd

def stardata(tic_id, sector, targetfile):
    # Query for star data
    star_data = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC")[0]
    # Check if required parameters have values
    if pd.notnull(star_data['Teff']) and pd.notnull(star_data['Vmag']) and pd.notnull(star_data['Kmag']) and pd.notnull(star_data['d']):
        # Compile the parameters of interest into a list
        star_info = [tic_id, star_data['Vmag'], star_data['Teff'], star_data['Kmag'], star_data['d']]
        
        return star_info 
    else:
        return None

def read_target_file_coma(file_path):
    targets = []
    with open(file_path, 'r') as file: 
        for line in file:
            if line.strip():
                tic_id, sector_id = map(int, line.split(',')) targets.append((tic_id, sector_id))
    return targets

def getStardata(file_path, targetfile): 
    targets = read_target_file_coma(file_path) 
    all_star_data = []
    for tic_id, sector_id in targets:
        star_info = stardata(tic_id, sector_id, targetfile) 
        if star_info:
            all_star_data.append(star_info)
            star_df = pd.DataFrame([star_info], columns=['TIC ID', 'Vmag','Teff', 'Kmag', 'Distance'])
            file_path_to_write = 'star_data.csv' if targetfile == '' else f'{targetfile}/star_data.csv'
            with open(file_path_to_write, 'a') as f:
                star_df.to_csv(f, index=False, header=f.tell()==0)
    return all_star_data


ggg

def absolute_magnitude(Vmag, distance):
    return Vmag - 2.5 * (np.log10(distance/10)**2)

HV_Data = []
for star in star_data_array:
    TIC_ID, Vmag, T_eff, _, distance = star
    M = absolute_magnitude(Vmag, distance)
    HV_Data.append([TIC_ID, T_eff, M])
    
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

HV_Data = np.array(HV_Data)
fig, ax = plt.subplots(figsize=(10, 6))
for star in HV_Data:
    TIC_ID, T_eff, abs_mag = star ax.scatter(T_eff, abs_mag, c='red')
    ax.set_xscale('log')
    
ax.invert_xaxis()
ax.invert_yaxis()
ax.set_xlim(30000, 3000)
ax.set_ylim(15, -15)

main_sequence_vertices = [(30000, -10), (12500, 1), (4800, 5), (3000, 10), (20000, -6)]
giants_vertices = [(6500, -2), (3400, -2), (3000, -7), (4800, -6)]
dwarfs_vertices = [(27000, 10), (27000, 5), (7500, 10)]
main_sequence_patch = patches.Polygon(main_sequence_vertices, closed=True, color='grey', alpha=0.2, label='Main Sequence')
giants_patch = patches.Polygon(giants_vertices, closed=True, color='orange', alpha=0.2, label='Giants')
dwarfs_patch = patches.Polygon(dwarfs_vertices, closed=True, color='green', alpha=0.2, label='Dwarfs')
ax.add_patch(main_sequence_patch)
ax.add_patch(giants_patch)
ax.add_patch(dwarfs_patch)
plt.xlabel('Effective Temperature [K]')
plt.ylabel('Absolute Magnitude')
plt.title('Hertzsprung-Russell Diagram')
plt.show()

from astroquery.mast import Catalogs
def stardata(tic_id, sector):
# Query for star data
    star_data = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC")[0]
    # Check if required parameters have values
    if pd.notnull(star_data['Teff']) and pd.notnull(star_data['Vmag']) and pd.notnull(star_data['Kmag']) and pd.notnull(star_data['d']) and pd.notnull(star_data['ra']) and pd.notnull(star_data['dec']):
        # Compile the parameters of interest into a list
        star_info = [
            tic_id,
            star_data['Vmag'],
            star_data['Teff'],
            star_data['Kmag'],
            star_data['d'],
            star_data['ra'],
            star_data['dec']
        ]
        return star_info
    else:
        return None
    
def getStardata(file_path):
    targets = read_target_file(file_path)
    all_star_data = []
    for tic_id, sector_id in targets:
        star_info = stardata(tic_id, sector_id) 
        if star_info:
            all_star_data.append(star_info)
    return all_star_data

file_path = r'Data/Classifaction/variablestarCNN.txt'
new_delimiter = ' '

with open(file_path, 'r') as file: 
    lines = file.readlines()
    new_lines = [line.replace(',', new_delimiter) for line in lines]
    new_file_path = r'Data/Classifaction/variablestarCNN.txt' 
    with open(new_file_path, 'w') as file:
        file.writelines(new_lines)
        
def spherical_to_cartesian(ra, dec, distance):
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    x = distance * np.cos(dec_rad) * np.cos(ra_rad) 
    y = distance * np.cos(dec_rad) * np.sin(ra_rad) 
    z = distance * np.sin(dec_rad)
return x, y, z

CNNtargets = r"Data/Classifaction/variablestarCNN.txt"
star_data_array = getStardata(CNNtargets)
star_data_np = np.array(star_data_array, dtype='float')

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

Teff = np.array([star[2] for star in star_data_array]) norm = plt.Normalize(min(Teff), max(Teff))
cmap = plt.cm.viridis
# Plot each star
for star in star_data_array:

    tic_id, Vmag, T_eff, Kmag, distance, ra, dec = star
    x, y, z = spherical_to_cartesian(ra, dec, distance) 
    color = cmap(norm(T_eff))
    ax.scatter(x, y, z, color=color)

ax.set_xlabel('X (parsecs)')
ax.set_ylabel('Y (parsecs)')
ax.set_zlabel('Z (parsecs)')
plt.show()

# Create a grid of temperatures and absolute magnitudes
temp = np.linspace(3000, 30000, 100) # X-axis: Effective Temperature 
mag = np.linspace(-5, 15, 100) # Y-axis: Absolute Magnitude
T, M = np.meshgrid(temp, mag)

# Function to simulate star density
def star_density(T, M):
# This is a simplified function to mimic the star density on the HR diagram. 
    return np.exp(-((M - 5*np.log10(T/5800))**2 / (2 * (0.5 + 0.001*(T-5800)**2))))

# Calculate the star density
density = star_density(T, M)

# Plotting the heatmap
fig, ax = plt.subplots(figsize=(10, 6))
heatmap = ax.contourf(T, M, density, levels=50, cmap='Spectral')

# Add a color band for the main sequence
main_sequence_mask = (M > 5*np.log10(T/5800) - 1) & (M < 5*np.log10(T/5800) + 1)
ax.contourf(T, M, main_sequence_mask, levels=[0.5, 1], colors=['blue'], alpha=0.2)


# Label general areas
ax.text(10000, 1, 'Giants', color='black', ha='center', fontsize=12)
ax.text(5800, 4.85, 'Sun', color='black', ha='center', fontsize=12)
ax.text(7000, -3, 'Main Sequence', color='black', ha='center', fontsize=12, rotation=45)
ax.text(15000, 12, 'White Dwarfs', color='black', ha='center', fontsize=12)

# Adjust the scales and orientation
ax.set_xscale('log')
ax.invert_xaxis() # Invert the x-axis so hot stars are on the left ax.invert_yaxis()
ax.set_xlim(30000, 3000)
ax.set_ylim(15, -5)

# Set labels and title
ax.set_xlabel('Effective Temperature [K]')
ax.set_ylabel('Absolute Magnitude')
ax.set_title('Hertzsprung-Russell Diagram')

# Add a colorbar for the heatmap
cbar = plt.colorbar(heatmap, ax=ax)
cbar.set_label('Star Density')

# Show the plot
plt.show()
        
        