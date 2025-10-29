import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import fft

# --- Configuration ---
# Sampling frequency inferred from the raw data (0.005s interval = 200 Hz)
FS = 200 
WINDOW_SAMPLES = FS # 1 second window
# Define frequency bands for Band Energy calculation (in Hz)
FREQ_BANDS = [(0, 10), (10, 30), (30, 50), (50, FS/2)] # Example bands

# --- Helper Functions for Frequency Domain Features ---

def calculate_fft_features(data, fs, bands):
    """Calculates FFT-based features for a single window of data."""
    data = np.asarray(data)
    if len(data) < 2:
        return {'spectral_kurtosis': np.nan, 'cepstrum_peak': np.nan, **{f'band_energy_{b[0]}-{b[1]}hz': np.nan for b in bands}}

    # 1. FFT
    N = len(data)
    yf = fft(data)
    # Take the magnitude of the positive frequency components
    yf_mag = 2.0/N * np.abs(yf[0:N//2])
    # Corresponding frequencies
    xf = np.linspace(0.0, fs/2, N//2, endpoint=False) # Use endpoint=False for correct frequency bins

    # 2. Spectral Kurtosis
    # Power Spectral Density (PSD) is proportional to yf_mag**2
    psd = yf_mag**2
    if len(psd) < 4: # Need enough points for kurtosis
        spectral_kurtosis = np.nan
    else:
        # Spectral Kurtosis is the kurtosis of the power spectral density
        spectral_kurtosis = kurtosis(psd)

    # 3. Cepstrum (Real Cepstrum)
    # The real cepstrum is the inverse Fourier transform of the logarithm of the magnitude of the Fourier transform
    # Using the magnitude spectrum (yf_mag)
    if np.any(yf_mag == 0):
        cepstrum_peak = np.nan # Avoid log(0)
    else:
        log_mag_spectrum = np.log(yf_mag + 1e-10) # Add small epsilon to avoid log(0)
        cepstrum = np.abs(np.fft.ifft(log_mag_spectrum))
        # We are interested in the peak of the cepstrum (excluding the first component)
        cepstrum_peak = np.max(cepstrum[1:])
        
    # 4. FFT Band Energies
    band_energies = {}
    total_energy = np.sum(psd)
    
    for low_freq, high_freq in bands:
        # Find indices corresponding to the frequency band
        idx_start = np.argmax(xf >= low_freq)
        # argmax returns 0 if no value is True, so we need to handle the case where high_freq is beyond max freq
        idx_end_candidates = np.where(xf >= high_freq)[0]
        idx_end = idx_end_candidates[0] if len(idx_end_candidates) > 0 else len(xf)
        
        band_psd = psd[idx_start:idx_end]
        band_energy = np.sum(band_psd)
        
        # Normalize by total energy if it's not zero
        normalized_energy = band_energy / total_energy if total_energy > 0 else 0
        band_energies[f'band_energy_{low_freq}-{high_freq}hz'] = normalized_energy

    return {
        'spectral_kurtosis': spectral_kurtosis,
        'cepstrum_peak': cepstrum_peak if isinstance(cepstrum_peak, (float, np.float64)) else np.nan,
        **band_energies
    }


# --- Main Feature Extraction Function ---

def extract_features(df, window_size='1s', fs=FS, freq_bands=FREQ_BANDS):
    """
    Extracts time-domain and frequency-domain features from raw sensor data.
    The features are calculated over a rolling window.
    """
    
    # Set timestamp as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Columns to process for features
    vibration_cols = ['accel_x_m_s2', 'accel_y_m_s2', 'accel_z_m_s2', 'velocity_mm_s', 'displacement_um']
    process_cols = vibration_cols + ['temp_C', 'current_A']
    
    # Initialize features DataFrame with tag and rpm
    # Use last() and ffill() to get the tag and rpm for the end of the window
    features = df[['tag', 'rpm']].resample(window_size).last().ffill()
    
    # --- Time-Domain Features ---
    for col in process_cols:
        # RMS (Root Mean Square)
        features[f'{col}_rms'] = df[col].rolling(window_size).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True).resample(window_size).last()
        
        # Peak-to-Peak (Max - Min)
        features[f'{col}_peak_to_peak'] = df[col].rolling(window_size).apply(lambda x: np.max(x) - np.min(x), raw=True).resample(window_size).last()
        
        # Skewness
        features[f'{col}_skew'] = df[col].rolling(window_size).apply(lambda x: skew(x), raw=True).resample(window_size).last()
        
        # Kurtosis
        features[f'{col}_kurtosis'] = df[col].rolling(window_size).apply(lambda x: kurtosis(x), raw=True).resample(window_size).last()
        
        # Mean
        features[f'{col}_mean'] = df[col].rolling(window_size).mean().resample(window_size).last()
        
        # Standard Deviation
        features[f'{col}_std'] = df[col].rolling(window_size).std().resample(window_size).last()

    # --- Frequency-Domain Features (Advanced) ---
    # We will use a dictionary to collect all FFT features and then merge them.
    fft_features_list = []
    
    # Iterate over the resampled indices (time windows)
    for window_end_time in features.index:
        # Define the start of the window
        window_start_time = window_end_time - pd.Timedelta(window_size)
        
        # Get the raw data for the current window
        window_data = df.loc[window_start_time:window_end_time].iloc[:-1] # Exclude the end time to match rolling behavior
        
        # Dictionary to store features for the current window
        window_fft_features = {'timestamp': window_end_time}
        
        for col in vibration_cols:
            raw_signal = window_data[col].values
            
            # Calculate FFT features
            features_dict = calculate_fft_features(raw_signal, fs, freq_bands)
            
            # Rename keys and store
            for key, value in features_dict.items():
                window_fft_features[f'{col}_{key}'] = value
        
        fft_features_list.append(window_fft_features)

    # Convert the list of dictionaries to a DataFrame
    fft_df = pd.DataFrame(fft_features_list).set_index('timestamp')
    
    # Merge the FFT features with the time-domain features
    features = features.merge(fft_df, left_index=True, right_index=True, how='left')

    # Drop rows with NaN values that result from the rolling window (the first few rows)
    features = features.dropna(how='all', subset=[c for c in features.columns if c not in ['tag', 'rpm']]).reset_index()
    
    return features

if __name__ == '__main__':
    # Load the raw data
    RAW_DATA_CSV = 'stranding_machine_sample_data.csv'
    try:
        # Assuming the raw data file is now in the current directory after previous step
        raw_df = pd.read_csv(RAW_DATA_CSV)
    except FileNotFoundError:
        print(f"Error: Raw data file '{RAW_DATA_CSV}' not found. Please ensure it is in the same directory.")
        exit()

    print("Starting advanced feature extraction...")
    
    # Assuming a 1-second window size for feature calculation
    features_df = extract_features(raw_df, window_size='1s')
    
    # Save the features to a new CSV file
    OUTPUT_CSV = 'stranding_features_advanced.csv'
    features_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"Advanced feature extraction complete. Features saved to '{OUTPUT_CSV}'.")
    print(f"Shape of the features DataFrame: {features_df.shape}")
    print("First 5 rows of the features DataFrame:")
    print(features_df.head())
