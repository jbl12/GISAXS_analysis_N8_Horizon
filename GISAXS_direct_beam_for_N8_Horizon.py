# -*- coding: utf-8 -*-
"""
Last modified date 07/2025

@author: B. Jiang
credit: deepseek-R1


Can be used to do:
- using kmeans to find the center of the direct beam


"""
#For arrays and mathematical operations
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import figure,plot,imshow
from matplotlib.widgets import Slider, Button,RectangleSelector, Button

#For interactive plotting for defining ROI and Cutting perposes
from ipywidgets import interactive

from tqdm import tqdm #for progress bar

from scipy.ndimage import rotate #for img rotation
from scipy.signal import savgol_filter as sg1
from PIL import Image

import os

#For data extraction
import struct

#k means clustering for finding clusters
from sklearn.cluster import KMeans
from sklearn import metrics

import scipy.stats as stats
from operator import itemgetter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from matplotlib import colors

# Function allows to draw rectangle as ROI
from IPython.display import clear_output, display

########### Defining plot style ##############
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.family'] = "serif"
# mpl.rcParams.update({'font.size': 12})
mpl.rcParams.update({'figure.figsize': (7.2,4.45)})
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)


mpl.rcParams.update({'xtick.labelsize': 15})

mpl.rcParams.update({'ytick.labelsize': 15})

mpl.rcParams.update({'font.size': 14})

mpl.rcParams.update({'figure.autolayout': True})

mpl.rcParams.update({'axes.titlesize': 15})

mpl.rcParams.update({'axes.labelsize': 15})

mpl.rcParams.update({'lines.linewidth': 2})

mpl.rcParams.update({'lines.markersize': 6})

mpl.rcParams.update({'legend.fontsize': 13})

###############################################


########### Defining ROI for locating the direct beam ##############

# Initialize the rect as None
rect = None


def onselect(eclick, erelease):
    global default_x, default_y, rect

    # Remove previous rectangle if it exists
    if rect:
        rect.remove()
        plt.draw()  # Redraw to reflect the removal

    # Check that the selection occurred within the plot boundaries
    if eclick.xdata is not None and erelease.xdata is not None:
        # Set selected coordinates
        default_x = [int(eclick.xdata), int(erelease.xdata)]
        default_y = [int(eclick.ydata), int(erelease.ydata)]

        # Display the selected region coordinates for confirmation
        clear_output(wait=True)
        display(f"Selected X coordinates: {default_x}")
        display(f"Selected Y coordinates: {default_y}")

        # Draw a red rectangle to visualize the selected area
        rect = plt.Rectangle((default_x[0], default_y[0]),
                             default_x[1] - default_x[0],
                             default_y[1] - default_y[0],
                             linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.draw()  # Redraw to show the new rectangle
    else:
        # Inform user if selection was out of bounds
        clear_output(wait=True)
        display("Selection out of bounds. Please select within the image boundaries.")


# Callback function to reset the selection
def reset(event):
    global default_x, default_y, rect
    # Clear coordinates
    default_x, default_y = None, None
    # Remove rectangle if it exists
    if rect:
        rect.remove()
        rect = None
    clear_output(wait=True)
    print("Selection reset.")

###############################################


########### Filtering Data & Finding the Centroid using kmeans ##############
# Various filter types - use whichever you would like
# You can see what the various types do here: https://www.desmos.com/calculator/0bbwgth7cq

def exp_filter(pixel_val,filter_level):
    """
    Takes an exponential value for input data and cuts off at certain value

    Args:
        pixel_val: The original value of the pixel from the video feed.
        filter_level: Value for which all results below will be mapped to 0.

    Returns:
        A transformed value that has either been mapped to an exponential function or mapped to 0.
    """
    exp = ((2 ** pixel_val) / (2 ** 255)) * 255
    if exp > filter_level:
        return exp
    else:
        return 0

def poly_filter(pixel_val,filter_level):
    """
    Takes a polynomial value for input data with a cutoff for threshold. The default/original is a quadratic mapping.

    Args:
        pixel_val: The original value of the pixel from the video feed.
        filter_level: Value for which all results below will be mapped to 0.

    Returns:
        A transformed value that has either been mapped to a polynomial function or mapped to 0.
    """
    poly = (pixel_val ** 2) / 255
    if poly > filter_level:
        return poly
    else:
        return 0

def sigmoid_filter(pixel_val,filter_level):
    """
    Applies a sigmoid function to pixel values for filtering.

    Args:
        pixel_val: The original value of the pixel from the video feed.
        filter_level: Midpoint of the sigmoid function. This will change how many low intensity pixels you want to map
            to higher values.

    Returns:
        A transformed value that has either mostly been filtered upwards to the max intensity or downwards to 0.
    """
    sig = 255 / (1 + 0.1 ** (-(pixel_val - filter_level)))
    return sig

def simple_filter(pixel_val,filter_level):
    """
    Strict threshold filter that scales up all values above the threshold.

    Args:
        pixel_val: The original value of the pixel from the video feed.
        filter_level: Value for which all results below will be mapped to 0.

    Returns:
        255 if above the threshold, 0 if below.
    """
    if pixel_val > filter_level:
        return MAX_VAL
    else:
        return 0

def filter_image(image):
    """
    Filters all of the values in a frame/image.

    Args:
        image: the 2D array with intensity values for each pixel.

    Returns:
        An array for input to the kMeans algorithm. This will still be able to plot as a scatter, but using imshow()
            will no longer work.
    """
    # This is the array that will store all of the values for kMeans.
    cluster_data = []

    # We will use the maximum intensity value to determine which pixels to filter out.
    max_val = np.amax(image[1:,1:])

    # This variable is just to ensure that kMeans does not crash due to an empty feature set. It will still produce
    # an error when you try to identify clusters, but the error will make more sense and it will be easier
    # to identify the solution.
    count = 0

    # Loop over every pixel in the image, except for the 0 edges, which for some reason will have a hight pixel
    # intensity value, even though there's nothing there.
    for i in range(1,np.shape(image)[0]):
        for j in range(1,np.shape(image)[1]):

            raw_data = float(image[i,j])

            # Ignore 0 values to speed up slightly.
            if raw_data != 0:
                # This is where you change the filter type. If you put "True" to the filter_data parameter, it will run
                # the data through the filter you specify here.
                if filter_data:
                    new_point = sigmoid_filter(raw_data,max_val / 4)
                else:
                    new_point = raw_data

                # This transforms all the filtered data points on a log scale, which turns all the values into small
                # integers, drastically speeding up computational time for the kMeans algorithm.
                if log_scale:
                    log_val = float(np.log(raw_data))
                else:
                    log_val = new_point

                # Append data to the cluster array for the kMeans algorithm.
                if log_val > float(0):
                    cluster_data += [[i,j]] * int(round(log_val))
                    count += 1

    # If there is no data to process, then the method will return an empty array and kMeans will return an error.
    if count == 0:
        return [[]]

    return cluster_data

def find_centroid_positions(dots,image, **keyword_parameters):
    """
    Finds the coordinates of dot centers.

    Args:
        dots: the total number of dots to look for.
        image: the numpy array of the cropped image.
        init: the previous frame centers, helps speed up calculation by giving the algorithm a reference.

    Returns:
        A tuple containing the center coordinates, an array with the filtered data, and labels.
    """

    K = int(dots)

    # First filter the data to make identifying centers easier.
    filtered_data = np.asarray(filter_image(image))

    # Using kMeans to find the centers. Will use either depending on if you provide the initial center positions.
    if ('init' in keyword_parameters):
        kmeans_model = KMeans(n_clusters=K,init=np.asarray(keyword_parameters['init'])).fit(filtered_data)
    else:
        kmeans_model = KMeans(n_clusters=K).fit(filtered_data)

    # The algorithm will return the center coordinates as (y,x), so keep that in mind when trying to manipulate it.
    centers = np.array(kmeans_model.cluster_centers_)
    labels = kmeans_model.labels_

    return (centers,filtered_data,labels)
