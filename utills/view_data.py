import matplotlib.pyplot as plt
import numpy as np
from data_managment import get_data
import matplotlib.cm as cm

def display_colormap(arr: np.ndarray,name=""):
    """
    Display the colormap of a given RGBA array.

    :param arr: A numpy array of shape [4, n, m] representing an RGBA image.
    :param title: The title of the plot.
    :param xlabel: The label for the x-axis.
    :param ylabel: The label for the y-axis.
    """
    if arr.shape[0] != 4:
        raise ValueError("The first dimension of the input array must be 4 (representing RGBA channels).")

    # Combine the RGBA channels to create an image
    img = np.transpose(arr, (1,2, 0))
    plt.imshow(img,aspect='auto')
    plt.title('Doppler Matrix Heatmap')  # Set the title of the plot
    plt.xlabel('Doppler Cell')  # Set the label for the x-axis
    plt.ylabel('Distance Cell')  # Set the label for the y-axis
    plt.colorbar(label='Normalized Magnitude')  # Add a colorbar with a label
    plt.axis('on')  # Turn off axis numbers and ticks
    plt.savefig(f'{name}.pdf', format='pdf')
    plt.show()


import numpy as np


def time_doppler_to_speed_range(time_doppler_matrix, f0, prf, range_resolution):
    c = 3e8  # Speed of light in m/s

    # Number of range bins and Doppler bins
    _,num_range_bins, num_doppler_bins = time_doppler_matrix.shape

    # Calculate the speed for each Doppler bin
    doppler_freqs = np.linspace(-prf / 2, prf / 2, num_doppler_bins)
    speeds = (doppler_freqs * c) / (2 * f0)

    # Calculate the range for each range bin
    ranges = np.arange(num_range_bins) * range_resolution

    # Initialize the speed-range matrix
    speed_range_matrix = np.zeros((num_range_bins, num_doppler_bins))

    # Populate the speed-range matrix with values from the time-Doppler matrix


    return time_doppler_matrix, ranges, speeds
def plot_speed_range_matrix(arr, ranges, speeds):

    if arr.shape[0] != 4:
        raise ValueError("The first dimension of the input array must be 4 (representing RGBA channels).")

    # Combine the RGBA channels to create an image
    img = np.transpose(arr, (1,2, 0))
    plt.imshow(img,aspect='auto',extent=(speeds.min(), speeds.max(), ranges.min(), ranges.max()))
    plt.title('Speed-Range Matrix Heatmap')
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Range (m)')
    plt.axis('on')  # Turn off axis numbers and ticks
    plt.show()








a,b,c = get_data()
# Radar Frequency (f0) in Hz
f0 = 8.75e9  # 8.75 GHz = 8.75 * 10^9 Hz

# Pulse Repetition Frequency (PRF) in Hz
prf = 2.86e3  # 2.86 kHz = 2.86 * 10^3 Hz

# Range Resolution in meters
range_resolution = 0.75  # meters

#time_doppler_matrix, ranges, speeds  = time_doppler_to_speed_range(a[123],f0,prf,range_resolution)
#plot_speed_range_matrix(time_doppler_matrix, ranges, speeds)


display_colormap(a[123],'car1')
display_colormap(a[126],'car2')

display_colormap(b[123],'drone1')
display_colormap(b[126],'drone2')

display_colormap(c[123],'people1')
display_colormap(c[126],'people2')


