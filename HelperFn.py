import scipy
import pycbc
import numpy as np
from pycbc import noise 
import scipy.signal
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from pycbc.waveform import get_fd_waveform , get_waveform_filter_length_in_time as chirplen
from pycbc.waveform.generator import (FDomainDetFrameGenerator , FDomainCBCGenerator)
from pycbc.psd import aLIGOZeroDetHighPower , AdVDesignSensitivityP1200087 
from pycbc.noise.gaussian import frequency_noise_from_psd
from pycbc.filter.matchedfilter import matched_filter 


def calculate_signal_duration(approximant, m1, m2, s1z, s2z, flow, fsamp):
    """
    Calculate the signal duration for a binary system using the given waveform approximant.
    
    Parameters:
    - approximant: String specifying the waveform model (e.g., 'IMRPhenomD')
    - m1: Mass of the first object
    - m2: Mass of the second object
    - s1z: Spin of the first object along the z-axis
    - s2z: Spin of the second object along the z-axis
    - flow: The lower frequency cutoff for the signal
    - fsamp: Sampling frequency (typically in Hz)
    
    Returns:
    - signalDuration: Duration of the gravitational wave signal
    - delta_f: Frequency resolution in the frequency domain (1/seglen)
    - Nt: Number of time-domain samples (fsamp * seglen)
    - Nf: Number of frequency-domain samples (Nt // 2 + 1)
    """
    # Calculate the signal duration in seconds using the chirp length approximation
    signalDuration = chirplen(approximant=approximant, mass1=m1, mass2=m2, spin1z=s1z, spin2z=s2z, f_lower=flow)
    
    # Segment length is the smallest power of 2 greater than or equal to the signal duration
    seglen = 2 ** (int(np.log2(signalDuration)) + 1)
    
    # Delta_f is the frequency resolution, defined as 1 / seglen
    delta_f = 1 / seglen
    
    # Nt is the number of time samples, which is the segment length times the sampling frequency
    Nt = fsamp * seglen
    
    # Nf is the number of frequency-domain points, typically half of the time-domain samples plus one
    Nf = int(Nt // 2 + 1)
    
    return signalDuration, delta_f, Nt, Nf



def Signal_Generation(m1, m2, s1z, s2z, distance, alpha, delta, signalDuration, frozenParams, tStart, psi, Nf):
    """
    Generate a gravitational wave signal across detectors using PyCBC's frequency domain generator.
    
    Parameters:
    - m1, m2: Masses of the binary objects
    - s1z, s2z: Spins of the binary objects
    - distance: Distance to the source (in megaparsecs)
    - alpha: Right ascension of the source (in radians)
    - delta: Declination of the source (in radians)
    - signalDuration: Duration of the gravitational wave signal
    - frozenParams: Dictionary of parameters that remain fixed during signal generation
    - Nf: Represents the number of frequency bins in the frequency domain
    - frozenParams = {
            'approximant': "IMRPhenomD",
            'mass1': m1,
            'mass2': m2,
            'spin1z': s1z,
            'spin2z': s2z,
            'delta_f': delta_f,
            'f_lower': flow,
            'f_final': fhigh,  # Example upper cutoff frequency
            'inclination': iota,
            'distance': distance,  # Example distance
            'mode_array': mode_array,
            'coa_phase': phic
        }
    - tStart: GPS start time of the signal
    - psi: Polarization angle of the gravitational wave

    
    Returns:
    - signal_L1V1H1: Dictionary of the generated gravitational wave signals for each detector in frequency domain
    """

    # Parameters related to the binary system's location and timing
    locationParams = ['ra', 'dec', 'tc', 'polarization']
    ifos = ['L1', 'H1', 'V1']  # Detectors (LIGO-Hanford, LIGO-Livingston, Virgo)
    
    # Generate the signal across the detectors
    generator = FDomainDetFrameGenerator(FDomainCBCGenerator, detectors=ifos, epoch=tStart, 
                                         variable_args=locationParams, **frozenParams)
    
    # Polarization angle and GPS time of trigger
    tTrigger = tStart + signalDuration
    
    # Generate the signal in the detectors
    signal_L1V1H1 = generator.generate(ra=alpha, dec=delta, tc=tTrigger, polarization=psi)
    
    # Resize the signal in each detector to match the frequency domain length (Nf)
    if len(signal_L1V1H1['L1']) != Nf:
        for ifo in ifos:
            signal_L1V1H1[ifo].resize(Nf)
    
    return signal_L1V1H1


def Noisy_data(ifos, signal_L1V1H1_fd , psd):
    """
    Generate noisy data for gravitational wave signals by adding Gaussian noise to the signals in the frequency domain.

    Parameters:
    - ifos (list of str): List of detector names (e.g., ['L1', 'H1', 'V1']).
    - signal_L1V1H1_fd (dict): Dictionary containing the gravitational wave signals in the frequency domain for each detector.
                               Each key should correspond to a detector name, and the value should be a frequency domain signal object.
    - psd (dict): Dictionary containing the power spectral density (PSD) for each detector.
                  Each key should correspond to a detector name, and the value should be the PSD data used to generate noise.

    Returns:
    - data_fd (dict): Dictionary containing the noisy data in the frequency domain for each detector.
    - signal_L1V1H1_fd (dict): Dictionary containing the original gravitational wave signals in the frequency domain for each detector.
    """
    
    noise = {}
    data_fd = {}

    for detector in ifos:
        # Generate Gaussian noise based on the PSD for each detector
        noise[detector] = frequency_noise_from_psd(psd[detector])

        # Add Gaussian noise to the pure signal
        data_fd[detector] = signal_L1V1H1_fd[detector] + noise[detector]  # Data in frequency domain

    return data_fd, signal_L1V1H1_fd


def whitened(detectors, strain_data_fd, PSD, dt):
    """
    Whiten strain data for multiple detectors using the provided PSD.

    Args:
        detectors (list of str): List of detector names (e.g., ['L1', 'H1', 'V1']).
        strain_data_fd (dict): Dictionary containing strain data in frequency domain for each detector.
                               Each key should correspond to a detector name.
        PSD (dict): Dictionary containing interpolating PSD functions for each detector.
        dt (float): Sample time interval of data. (1/f_samp)

    Returns:
        dict: Dictionary of whitened strain data for each detector in timeseries. (dtype = pycbc format)
    """
    whitened_data = {}

    for ifo in detectors:

        # Avoid division by zero by replacing zero values in PSD with a very small value
        PSD[ifo].data[np.where(PSD[ifo].data == 0)[0]] = 10**(-50)

        # Sqrt of  Nyquist Frequency
        norm = 1./np.sqrt(1./(dt*2))   

        # Whiten the strain in the frequency domain and Normalize 
        whiten_strain_f = strain_data_fd[ifo] / (np.sqrt(PSD[ifo]))*norm
        
        whiten_strain_f = whiten_strain_f/norm

        # Convert to time series (using to_timeseries method)
        whiten_strain_t = whiten_strain_f.to_timeseries()

        # Store the whitened data
        whitened_data[ifo] = whiten_strain_t

    return whitened_data
    

import matplotlib.pyplot as plt


def plot_whitened_data(whitened_data_td, whitened_h_td, tStart, tTrigger, detectators , xmin = - 0.1 , xmax = 0.1 , figsize = (20 , 9)):
    """
    Plot the whitened strain data and the corresponding signals for multiple detectors.

    Args:
        whitened_data_td (dict): Dictionary containing whitened strain data in the time domain for each detector.
                                  The keys should correspond to detector names (e.g., 'L1', 'H1', 'V1').
        whitened_h_td (dict): Dictionary containing the true gravitational wave signal in the time domain for each detector.
                               The keys should correspond to detector names (e.g., 'L1', 'H1', 'V1').
        tStart (float): The start time of the signal (in seconds).
        tTrigger (float): The trigger time of the signal (in seconds).
        xmin = Minimum xlim for plot(default = 0.1)
        xmax = Maximum xlim for plot(default = 0.1)
        detectators (list): List of detector names to plot.

    Returns:
        None: Displays the plots for the whitened strain data and the signals for the specified detectors.
    """
    
    fig, ax = plt.subplots(len(detectators), 1, sharex=True, figsize= figsize)
    fig.subplots_adjust(hspace=0)

    for i, ifo in enumerate(detectators):
        # Plot the whitened data and signal
        ax[i].plot(whitened_data_td[ifo].sample_times, whitened_data_td[ifo], c='darkgray', label='Noisy Data')
        ax[i].plot(whitened_h_td[ifo].sample_times, whitened_h_td[ifo], c='navy', label='Signal')
        
        # Vertical lines for trigger and start times
        ax[i].axvline(x=tTrigger, c='k', ls='--', label='Trigger Time')
        # ax[i].axvline(x=tStart, c='k', label='Start Time')

        # Set labels and titles
        ax[i].set_ylabel('Whitened Strain', fontsize=10)
        ax[i].set_title(ifo, x=0.99, y=0.95, pad=-10, fontsize=10, loc='right')
        ax[i].set_xlim(tTrigger + xmin, tTrigger + xmax)

        ax[-1].set_xlabel('Time (s)', fontsize=10)  # Label for the x-axis on the last plot

    plt.legend(loc = 'lower right' , fontsize=8)
    plt.tight_layout()
    plt.show()



def clip_signal_around_merger(ifos, timeseries, tTrigger, pre_merger_time= 3.5, post_merger_time=0.5):
    """
    Clip the signal and extract a portion of the time series data around the binary merger event.

    Parameters:
    - ifos: List of interferometers (e.g., ['L1', 'H1', 'V1'])
    - timeseries: Dictionary of time series data for each interferometer, including sample times
    - tTrigger: Time of the binary merger (trigger time)
    - pre_merger_time: Duration before the trigger to include in the clipping (default 3.5 seconds)
    - post_merger_time: Duration after the trigger to include in the clipping (default 0.5 seconds)

    Returns:
    - time_around_event: Time values clipped around the merger event
    - clipped_data_around_event: Dictionary of clipped time series data for each interferometer in time domain (dtype = Numpy array)
    """

    # Get the time array from one of the interferometers (assuming time is the same across detectors)
    sample_times_key = next(iter(timeseries))           # Get the key of the first interferometer (e.g., 'L1')
    time = timeseries[sample_times_key].sample_times    # Extract the sample times from this interferometer

    # Dictionary to store the clipped data around the merger event for each interferometer
    clipped_data_around_event = {}

    # Define the index range for clipping (pre-merger time to post-merger time)

    index = np.where((time >= tTrigger - pre_merger_time) & (time < tTrigger + post_merger_time))[0]
    time_around_event = time[index]  # Extract corresponding time values

    # Clip the data for each interferometer in the provided range
    for ifo in ifos:
        clipped_data_around_event[ifo] = timeseries[ifo][index]

    return time_around_event, clipped_data_around_event





def apply_tukey_window(ifos, data_td, time_length = 4, fs = 4096,  alpha=1./4):
    """
    Apply a Tukey window to the given signals for multiple detectors.
    
    Parameters:
    - ifos: List of detectors (e.g., ['L1', 'H1', 'V1']).
    - data_td: Dictionary containing time-series data for each detector.
    - window_length: Length of the window in samples. (Default 4 Seconds)
    - alpha: Shape parameter of the Tukey window (Default is 1/4).
    - fs: Smapling frequency in Hz. (Default 4096)
    
    Returns:
    - windowed_data: Dictionary with windowed data for each detector in time domain.
    """
    # Generate the Tukey window with specified length and alpha parameter

    window_length = fs*time_length
    tukey_window = scipy.signal.windows.tukey(window_length, alpha=alpha)
    
    # Apply the Tukey window to each detector's signal
    windowed_data = {}
    for ifo in ifos:
        windowed_data[ifo] = tukey_window * data_td[ifo]
    
    return windowed_data


def bandpass_filter(data, fband, ifos , fs = 4096):
    """
    Apply a bandpass filter and normalize the signal for multiple detectors.
    
    Parameters:
    - data: Dictionary containing time-series data for each detector.
    - fband: List or tuple containing the lower and upper frequency cutoff [f_low, f_high].
    - fs: Sampling rate (in Hz). (Default : 4096 Hz)
    - ifos: List of detectors (e.g., ['L1', 'H1', 'V1']).
    
    Returns:
    - bandpassed_data: Dictionary with bandpass filtered and normalized data for each detector.
    """
    # Create a dictionary to store the bandpassed and normalized data
    bandpassed_data = {}
    
    # Design a Butterworth bandpass filter
    b, a = butter(4, [fband[0] * 2 / fs, fband[1] * 2 / fs], btype='band')
    
    # Apply the filter and normalization to each detector's data
    for ifo in ifos:
        # Get the data for the current detector
        current_data = data[ifo]
        
        # Apply the bandpass filter to the current detector's data
        filtered_data = filtfilt(b, a, current_data)
        
        # Normalize the filtered data
        normalization = np.sqrt((fband[1] - fband[0]) / (fs / 2))
        normalized_data = filtered_data / normalization
        
        # Store the normalized data in the output dictionary
        bandpassed_data[ifo] = normalized_data
    
    return bandpassed_data



def plot_windowed_and_bandpassed(windowed_data_clean, windowed_data_noisy, time_around_merger, tStart, tTrigger, detectors, fband , xmin = -0.1 ,xmax = 0.1 , figsize = (20,9) , fs = 4096):
    """
    Plot the windowed and bandpassed strain data for multiple detectors.

    Args:
        windowed_data_clean (dict): Dictionary containing clean windowed strain data for each detector.
        windowed_data_noisy (dict): Dictionary containing noisy windowed strain data for each detector.
        band_passed_data_clean (dict): Dictionary containing clean bandpassed strain data for each detector.
        band_passed_data_noisy (dict): Dictionary containing noisy bandpassed strain data for each detector.
        time_around_merger (array): Time vector around the merger event (in seconds).
        tStart (float): The start time of the signal (in seconds).
        tTrigger (float): The trigger time of the signal (in seconds).
        xmin = Minimum xlim for plot(default = 0.1)
        xmax = Maximum xlim for plot(default = 0.1)
        detectors (list): List of detector names to plot.

    Returns:
        None: Displays the plots for the windowed and bandpassed strain data for the specified detectors.
    """
    band_passed_data_clean = bandpass_filter(windowed_data_clean, fband, detectors , fs = fs)
    band_passed_data_noisy = bandpass_filter(windowed_data_noisy, fband, detectors , fs = fs)
    # Plot Windowed Data
    fig, ax = plt.subplots(len(detectors), 1, sharex=True,  figsize=figsize) 
    fig.subplots_adjust(hspace=0)

    for i, ifo in enumerate(detectors):
        # Plot the windowed data (clean and noisy)
        ax[i].plot(time_around_merger, windowed_data_noisy[ifo], c='darkgray', label='Noisy Data')
        ax[i].plot(time_around_merger, windowed_data_clean[ifo], c='navy', label='Clean Data')
        
        # Vertical lines for trigger and start times
        ax[i].axvline(x=tTrigger, c='k', ls='--', label='Trigger Time')
        ax[i].axvline(x=tStart, c='k', label='Start Time')

        # Set labels and titles
        ax[i].set_ylabel('Windowed Strain', fontsize=10)
        ax[i].set_title(ifo, x=0.99, y=0.95, pad=-10, fontsize=10, loc='right')
        ax[i].set_xlim(tTrigger + xmin, tTrigger + xmax)

    ax[-1].set_xlabel('Time (s)', fontsize=10)  # Label for the x-axis on the last plot
    plt.legend(loc='lower right', fontsize=8)
    plt.suptitle('Windowed Strain Data', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Plot Bandpassed Data
    fig, ax = plt.subplots(len(detectors), 1, sharex=True, figsize=figsize)
    fig.subplots_adjust(hspace=0)

    for i, ifo in enumerate(detectors):
        # Plot the bandpassed data (clean and noisy)
        ax[i].plot(time_around_merger, band_passed_data_noisy[ifo], c='darkgray', label='Noisy Bandpassed')
        ax[i].plot(time_around_merger, band_passed_data_clean[ifo], c='navy', label='Clean Bandpassed')
        
        # Vertical lines for trigger and start times
        ax[i].axvline(x=tTrigger, c='k', ls='--', label='Trigger Time')
        ax[i].axvline(x=tStart, c='k', label='Start Time')

        # Set labels and titles
        ax[i].set_ylabel('Bandpassed Strain', fontsize=10)
        ax[i].set_title(ifo, x=0.99, y=0.95, pad=-10, fontsize=10, loc='right')
        ax[i].set_xlim(tTrigger + xmin, tTrigger + xmax)

    ax[-1].set_xlabel('Time (s)', fontsize=10)  # Label for the x-axis on the last plot
    plt.legend(loc='lower right', fontsize=8)
    plt.suptitle('Bandpassed Strain Data', fontsize=12)
    plt.tight_layout()
    plt.show()



def cross_correlate(data1, data2):
    """Perform cross-correlation between two data arrays."""
    return plt.xcorr(data1, data2, maxlags=len(data1) - 1)


def get_correlation(data1, data2, time, t_min, t_max, timedelay, tTrigger, fs = 4096):
    """
    Get the cross-correlation between two datasets within a specified time range,
    restricted to a specific time delay.
    
    Args:
        data1: First data array.
        data2: Second data array.
        time: Time vector.
        t_min: Minimum time range around tTrigger. (before tTrigger)
        t_max: Maximum time range around tTrigger. (after tTrigger)
        tTrigger: The time around which to compute the correlation.
        timedelay: The time delay range for filtering the correlation.
        fs: Sampling frequency. (Default: 4096)
    
    Returns:
        corr: Cross-correlation values.
        time_lag: The time lag corresponding to the maximum correlation within the restricted range.
        lags: Time lags associated with the correlation.
    """
    time_int = np.where((time >= tTrigger + t_min) & (time < tTrigger + t_max))
    data1_int = data1[time_int]
    data2_int = data2[time_int]

    correlation = cross_correlate(data1_int, data2_int)
    plt.close()
    lags = correlation[0] / fs
    corr = correlation[1]

    ind = np.where((lags >= - timedelay) & (lags < timedelay))
    filtered_corr = corr[ind]
    filtered_lag = lags[ind]
    max_corr_index = np.argmax(np.abs(filtered_corr))
    time_lag = filtered_lag[max_corr_index]

    return corr, time_lag, lags



def get_cross_correlation(bandpass_signal_clean, bandpass_signal_noisy, ifo_set, Timedelays, time, t_min, t_max , tTrigger, fs = 4096):
    """
    Plot the cross-correlation between clean and noisy signals for given detectors.

    Args:
        bandpass_signal_clean: Dictionary of clean bandpassed signals for each detector.
        bandpass_signal_noisy: Dictionary of noisy bandpassed signals for each detector.
        ifo_set: List of tuples representing pairs of detectors.
        Timedelays: Dictionary mapping detector pairs to their corresponding time delays
        time: Time vector associated with the signals.
        tTrigger: The time around which to compute the correlation.
        fs = Sampling frequency Hz (Default 4096)
        t_min: Minimum time range for correlation.
        t_max: Maximum time range for correlation.
    """
    
    # Loop over each pair in ifo_set and compute cross-correlation
    Time_lag = { 'Clean': {} ,'Noisy': {} }
    for ifo in ifo_set:  

        delay = Timedelays[ifo]

        # Extract data for both clean and noisy signals
        clean_data_1 = bandpass_signal_clean[ifo[0]]
        clean_data_2 = bandpass_signal_clean[ifo[1]]
        noisy_data_1 = bandpass_signal_noisy[ifo[0]]
        noisy_data_2 = bandpass_signal_noisy[ifo[1]]

        # Compute cross-correlation for clean signals and noisy signals
        clean_corr_vals, clean_time_lag, clean_lags = get_correlation(clean_data_1, clean_data_2, time, t_min, t_max, delay , tTrigger, fs)
        noisy_corr_vals, noisy_time_lag, noisy_lags = get_correlation(noisy_data_1, noisy_data_2, time, t_min, t_max, delay , tTrigger, fs)
        
        Time_lag['Clean'][ifo] = round(clean_time_lag*1000 , 2)
        Time_lag['Noisy'][ifo] = round(noisy_time_lag*1000 , 2)
        
    return Time_lag


def plot_cross_correlation(bandpass_signal_clean, bandpass_signal_noisy, ifo_set, Timedelays, time, t_min, t_max , tTrigger, fs = 4096 , figsize = (14,8)):
    """
    Plot the cross-correlation between clean and noisy signals for given detectors.

    Args:
        bandpass_signal_clean: Dictionary of clean bandpassed signals for each detector.
        bandpass_signal_noisy: Dictionary of noisy bandpassed signals for each detector.
        ifo_set: List of tuples representing pairs of detectors.
        Timedelays: Dictionary mapping detector pairs to their corresponding time delays
        time: Time vector associated with the signals.
        tTrigger: The time around which to compute the correlation.
        fs = Sampling frequency Hz (Default 4096)
        t_min: Minimum time range for correlation.
        t_max: Maximum time range for correlation.
    """
    
    # Loop over each pair in ifo_set and compute cross-correlation
    Time_lag = { 'Clean': {} ,'Noisy': {} }
    for ifo in ifo_set:  

        delay = Timedelays[ifo]

        # Extract data for both clean and noisy signals
        clean_data_1 = bandpass_signal_clean[ifo[0]]
        clean_data_2 = bandpass_signal_clean[ifo[1]]
        noisy_data_1 = bandpass_signal_noisy[ifo[0]]
        noisy_data_2 = bandpass_signal_noisy[ifo[1]]

        # Compute cross-correlation for clean signals and noisy signals
        clean_corr_vals, clean_time_lag, clean_lags = get_correlation(clean_data_1, clean_data_2, time, t_min, t_max, delay , tTrigger, fs)
        noisy_corr_vals, noisy_time_lag, noisy_lags = get_correlation(noisy_data_1, noisy_data_2, time, t_min, t_max, delay , tTrigger, fs)
        Time_lag['Clean'][ifo] = round(clean_time_lag*1000 , 2)
        Time_lag['Noisy'][ifo] = round(noisy_time_lag*1000 , 2)
        
        # Create plot
        fig = plt.figure(figsize=figsize)

        # Plot noisy and clean cross-correlation together 
        plt.plot(noisy_lags * 1000, noisy_corr_vals, label=f'Noisy Cross-Correlation {ifo[0]} and {ifo[1]}', color='red', linewidth=0.6)
        plt.plot(clean_lags * 1000, clean_corr_vals, label=f'Clean Cross-Correlation {ifo[0]} and {ifo[1]}', color='blue', linewidth=0.6)

        # Add vertical lines for time lags
        plt.axvline(x=clean_time_lag * 1000, color='black', label=f'Clean Time Lag = {clean_time_lag*1000:.2f} ms', ls='--')
        plt.axvline(x=noisy_time_lag * 1000, color='silver', label=f'Noisy Time Lag = {noisy_time_lag*1000:.2f} ms', ls='--')

        plt.xlabel('Time Lag (ms)')
        plt.ylabel('Cross-correlation')
        plt.title(f'Cross-Correlation of {ifo[0]} and {ifo[1]} Strain Data')
        plt.xlim(-100, 100)
        plt.ylim(-1, 1)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.close()
        plt.show()
        
    return Time_lag


def matchfiltering(frozenParams , psd , data_fd , fband,  ifos , Nf):
    flow , fhigh = fband[0] , fband[1]
    hp, hc = get_fd_waveform(**frozenParams)
    hp.resize(Nf)
    matchedFilterSNR = {}
    matchedFiltersnr = {}
    for detector in ifos:
        
        matchedFilterSNR[detector] = abs(matched_filter(template = hp, data = data_fd[detector], \
                                                    psd=psd[detector], low_frequency_cutoff = flow, high_frequency_cutoff = fhigh))
        # print('SNR for Detector %s : %0.4f'%(detector, abs(matchedFilterSNR[detector]).max()))
        matchedFiltersnr[detector] = round(abs(matchedFilterSNR[detector]).max(), 2)
    return matchedFiltersnr









