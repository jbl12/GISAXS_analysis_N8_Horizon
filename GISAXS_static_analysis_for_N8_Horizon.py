# -*- coding: utf-8 -*-
"""
Last modified date 07/2025

@author: B. Jiang
credit: deepseek-R1


Can be used to do:
- convert tiff format from rgba to grayscale and save
- q calculation
- Visualizations for the GISAXS data and intensity vs q// plotting

"""
#For arrays and mathematical operations
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm #for progress bar


from scipy.ndimage import rotate #for img rotation
from scipy.signal import savgol_filter as sg1
from PIL import Image

import os

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


def convert_rgba_tiff_to_grayscale(input_path, output_path):
    """
    Converts an RGBA TIFF image to a grayscale TIFF image.

    Args:
        input_path (str): The path to the input RGBA TIFF file.
        output_path (str): The path to save the output grayscale TIFF file.
    """
    try:
        # Open the RGBA TIFF image
        img = Image.open(input_path)

        # Convert the image to grayscale ('L' mode for 8-bit grayscale, 'LA' for grayscale with alpha)
        # Using 'L' discards the alpha channel and converts to a single grayscale channel.
        # If you need to preserve the alpha channel and have a grayscale image with transparency,
        # you would use 'LA' and then potentially handle the alpha channel separately if needed.
        grayscale_img = img.convert('L')

        # Save the grayscale image as a TIFF
        grayscale_img.save(output_path)
        print(f"Successfully converted '{input_path}' to grayscale and saved as '{output_path}'")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")


############################################################################################################
# #######   xxxxxxxxxxxxxxxxxxxxxxxxxxxxx  q calculation start   xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx ############
# ###########################################################################################################

def convert_pixel_to_q(x,y):#,x0,y0,d,Energy,alpha_i):
    """Notice that x comes first here. sorry for inconsistency. normally Y is taken first because of
    the way data is sliced. but for computing I would like to keep things in x and y order"""
    Psi = ((x - x0)*(pixel_size*1E-6)) / (d*1E-3)
    alpha_f = ((y - y0)*(pixel_size*1E-6)) / (d*1E-3)


    lambda_ = photon_wavelength*1e-10
    q_x = ((2*np.pi)/lambda_)*(np.cos(alpha_f)*np.cos(Psi)-np.cos(alpha_i))
    q_y = ((2*np.pi)/lambda_)*(np.cos(alpha_f)*np.sin(Psi))
    q_z = ((2*np.pi)/lambda_)*(np.sin(alpha_f)+np.sin(alpha_i))#+ omitted  selecting qz = 0 at incident beam also sin i + sin f = sin if in small angle


    #fix the alpha_f since now you have to count pixels from the plane of sample. not from incident beam
    alpha_f = (((y - y0)*(pixel_size*1E-6)) / (d*1E-3)) - alpha_i

    #to calculate qprime
    if alpha_f < alpha_c:
        alpha_f_prime = 0
    else:
        alpha_f_prime = np.sqrt(alpha_f**2-alpha_c**2)
    alpha_i_prime = np.sqrt(alpha_i**2-alpha_c**2)

    #calcuate qprime
    qx_prime = ((2*np.pi)/lambda_)*(np.cos(alpha_f_prime)*np.cos(Psi)-np.cos(alpha_i))
    qy_prime = ((2*np.pi)/lambda_)*(np.cos(alpha_f_prime)*np.sin(Psi))
    qz_prime = ((2*np.pi)/lambda_)*(np.sin(alpha_f_prime) + np.sin(alpha_i_prime)) #np.sin(alpha_i)+ omitted  selecting qz = 0 at incident beam

#     print(alpha_f_prime,alpha_f )

    #convert m^-1 to nm^-1
    q_x,q_y,q_z,qx_prime,qy_prime,qz_prime = q_x*1E-9,q_y*1E-9,q_z*1E-9, qx_prime*1E-9,qy_prime*1E-9,qz_prime*1E-9
    #calculate q magnitude
    q = (q_x**2+q_y**2+q_z**2)**(0.5)


    return(q_x,q_y,q_z,q,alpha_f,qx_prime,qy_prime,qz_prime)


############################################################################################################
# #######   xxxxxxxxxxxxxxxxxxxxxxxxxxxxx  q calculation end   xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx ############
# ###########################################################################################################

def get_ROI_Ivsq_avg(img, xslice, yslice, logscale = True, SG=False, box_size = 59, polynomial_order = 1, save = False):
    y_avg = (yslice[1]+yslice[0])//2

    image=np.flip(img,0)

    pixelinfo = np.linspace(np.round(convert_pixel_to_q(xslice[0],y_avg)[1],decimals=2), np.round(convert_pixel_to_q(xslice[1],y_avg)[1],decimals=2),len(image[y_avg,xslice[0]:xslice[1]]))

    if SG:
        image = sgolay2d(image, window_size=box_size, order=polynomial_order)

    sq_avg = np.average(image[yslice[0]:yslice[1],xslice[0]:xslice[1]],axis=0)
    sq_sum = np.sum(image[yslice[0]:yslice[1],xslice[0]:xslice[1]],axis=0)

    fig, ax = plt.subplots(figsize=(16,5))
    ax.grid(color='grey', linestyle='-', which='both', axis = 'y', linewidth=1, alpha=0.2)

    ax.scatter(pixelinfo, sq_avg, marker = '.', s=100, label='average')

    # ax.set_title(r'$q_z′$ = ' + np.str_(np.round(convert_pixel_to_q(0,y_avg)[7],decimals=2)) + r' $nm^{-1}$')
    q_z = np.str_(np.round(convert_pixel_to_q(0,y_avg)[2],decimals=2))
    print('qz and y_avg are %s %s' %(q_z,y_avg))

    ax.set_ylabel(r'Intensity [a.u.]')

    ax.set_xlabel(r'$q_{//}$ [nm$^{-1}$]')

    if logscale:
        # if ylim:
        #     ax.set_ylim(ylim[0], ylim[1])
        ax.set_yscale('log')

    if save:
        file_name = save_master_filename + '_Ivq'
        fp = data_dir + '/' + file_name + '.png'
        plt.savefig(fp,bbox_inches='tight',transparent=True)
        fp = data_dir + '/' + file_name + 'pixelinfo.npy'
        np.save(fp, pixelinfo)
        fp = data_dir + '/' + file_name + 'sq_avg.npy'
        np.save(fp, sq_avg)
        fp = data_dir + '/' + file_name + 'sq_sum.npy'
        np.save(fp, sq_sum)
        print('Saving in respective folder.')

    plt.show()

    return(pixelinfo,sq_avg)
