import numpy as np
import random
from scipy.io import wavfile

# Default options dictionary
defopts = {
    "win_nfloor_s": 0.05,            # Window size for noise floor analysis (in seconds)
    "win_skip_s": 0.02,              # Window skip size (in seconds)
    "nfloor_max_frac": 0.1,          # Fraction of amplitude for noise floor threshold
    "smooth_amp_frac": 0.1,          # Amplitude fraction for smooth transition near noise floor
    "amplitude_adjust_range": (0.8, 1.2)  # Range for random amplitude adjustment multiplier
}

class AudioPreprocessor:
    def __init__(
        self, 
        win_nfloor_s=defopts["win_nfloor_s"],        
        win_skip_s=defopts["win_skip_s"],           
        nfloor_max_frac=defopts["nfloor_max_frac"],  
        smooth_amp_frac=defopts["smooth_amp_frac"],  
        amplitude_adjust_range=defopts["amplitude_adjust_range"]
    ):
        self.win_nfloor_s = win_nfloor_s
        self.win_skip_s = win_skip_s
        self.nfloor_max_frac = nfloor_max_frac
        self.smooth_amp_frac = smooth_amp_frac
        self.amplitude_adjust_range = amplitude_adjust_range

    def load_audio(self, file_path):
        """Load audio file and return sample rate and normalized audio data."""
        sample_rate, data = wavfile.read(file_path)
        data = data / np.max(np.abs(data))  # Normalize to -1 to 1
        return sample_rate, data

    def calculate_noise_floor(self, data, sample_rate):
        """Calculate noise floor amplitude and max amplitude from the audio data."""
        window_size = int(self.win_nfloor_s * sample_rate)
        skip_size = int(self.win_skip_s * sample_rate)
        
        max_amp = np.max(np.abs(data))
        nfloor_amplitudes = []
        
        for i in range(0, len(data) - window_size, skip_size):
            window = data[i:i + window_size]
            window_max = np.max(np.abs(window))
            
            # Stop calculating noise floor if above the threshold
            if window_max > self.nfloor_max_frac * (max_amp - min(nfloor_amplitudes, default=max_amp)):
                break
            
            nfloor_amplitudes.append(window_max)

        # Noise floor amplitude is the minimum of calculated max amplitudes within each window
        nfloor_amp = min(nfloor_amplitudes) if nfloor_amplitudes else 0
        return nfloor_amp, max_amp

    def adjust_amplitude(self, data, nfloor_amp, max_amp):
        """Adjust amplitude of data above noise floor with smooth transition and random variation."""
        adjusted_data = np.copy(data)
        
        for i in range(len(data)):
            abs_val = np.abs(data[i])
            
            # Only adjust amplitude if the value is above the noise floor
            if abs_val > nfloor_amp:
                smooth_transition_factor = max(
                    0, 
                    min(1, (abs_val - nfloor_amp) / (self.smooth_amp_frac * (max_amp - nfloor_amp)))
                )
                
                # Generate random amplitude adjustment within the specified range
                random_adjust = random.uniform(*self.amplitude_adjust_range)
                amplitude_adjustment = (1 - smooth_transition_factor) + (smooth_transition_factor * random_adjust)
                
                # Apply adjustment
                adjusted_data[i] = data[i] * amplitude_adjustment

        return adjusted_data

    def preprocess(self, file_path, output_path):
        """Main preprocessing function to load audio, calculate noise floor, adjust amplitude, and save output."""
        sample_rate, data = self.load_audio(file_path)
        nfloor_amp, max_amp = self.calculate_noise_floor(data, sample_rate)
        
        # Apply amplitude adjustment
        adjusted_data = self.adjust_amplitude(data, nfloor_amp, max_amp)
        
        # Save the adjusted audio to output path
        wavfile.write(output_path, sample_rate, np.int16(adjusted_data * 32767))  # Scale back to int16 for saving
        return nfloor_amp, max_amp

# Example usage in CLI code:
# from audio_signal_floor_augment import defopts as asfaopts
# parser.add_argument('-w', '--win_nfloor_s', default=asfaopts["win_nfloor_s"], help=f'Window size ({asfaopts["win_nfloor_s"]})')
# parser.add_argument('-s', '--win_skip_s', default=asfaopts["win_skip_s"], help=f'Window skip ({asfaopts["win_skip_s"]})')
# ...
# processor = AudioPreprocessor(win_nfloor_s=args.win_nfloor_s, ...)
# processor.preprocess(args.input_file, args.output_file)
