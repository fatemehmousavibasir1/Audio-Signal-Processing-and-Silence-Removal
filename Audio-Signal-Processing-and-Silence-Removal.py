import wave
import numpy as np
import librosa
def remove_silence(input_path, output_path):

    with wave.open(input_path, 'rb') as wav_file:

        params = wav_file.getparams()
        num_frames = params.nframes
        frame_rate = params.framerate
        num_channels = params.nchannels
        sample_width = params.sampwidth
        
        raw_data = wav_file.readframes(num_frames)
        audio_data = np.frombuffer(raw_data, dtype=np.int16)

        threshold = 1000 
        non_silent_indices = np.where(np.abs(audio_data) > threshold)[0]
        start_index = non_silent_indices[0]
        end_index = non_silent_indices[-1]

        trimmed_audio = audio_data[start_index:end_index]

        with wave.open(output_path, 'wb') as trimmed_wav_file:
            trimmed_wav_file.setparams((num_channels, sample_width, frame_rate, len(trimmed_audio), params.comptype, params.compname))
            trimmed_wav_file.writeframes(trimmed_audio.tobytes())

input_audio_path = "C:/Users/Asus/Desktop/a.wav"
output_audio_path = "C:/Users/Asus/Desktop/newa.wav"

remove_silence(input_audio_path, output_audio_path)

import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
def plot_audio_waveform(audio_file):

    if not os.path.exists(audio_file):
        print("not exist ")
        return
    
    sr, signala = wavfile.read(audio_file)

    plt.figure(figsize=(10, 6))
    plt.plot(signala)
    plt.title('Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    audio_file = "C:/Users/Asus/Desktop/a.wav"
    plot_audio_waveform(audio_file)
   

if __name__ == "__main__":
    audio_file = "C:/Users/Asus/Desktop/newa.wav"
    plot_audio_waveform(audio_file)
    



def plot_audio_fft(input_path):
    with wave.open(input_path, 'rb') as wav_file:
        params = wav_file.getparams()
        num_frames = params.nframes
        frame_rate = params.framerate
        num_channels = params.nchannels
        sample_width = params.sampwidth

        raw_data = wav_file.readframes(num_frames)
  
        audio_data = np.frombuffer(raw_data, dtype=np.int16)

        fft_result = np.fft.fft(audio_data)
        frequencies = np.fft.fftfreq(len(fft_result), 1 / frame_rate)

        plt.figure(figsize=(10, 6))
        plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result)[:len(frequencies)//2])
        plt.title("FFT of Audio Signal")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        peaks = np.where(np.abs(fft_result) > np.mean(np.abs(fft_result)) + 3 * np.std(np.abs(fft_result)))[0]
        peak_frequencies = frequencies[peaks]
        peak_amplitudes = np.abs(fft_result[peaks])
        plt.show()

input_audio_path = "C:/Users/Asus/Desktop/newa.wav"

plot_audio_fft(input_audio_path)
from scipy.signal import find_peaks
signal, fs = librosa.load(input_audio_path, sr=None)

print("Sampling Rate:", fs)
import numpy as np

fft_result = np.fft.fft(signal)
fft_length = len(fft_result)
print("Length of FFT:", fft_length)

import numpy as np
from scipy.signal import argrelextrema

peaks = argrelextrema(signal, np.greater)[0]
peak_distances = np.diff(peaks)
mean_peak_distance = np.mean(peak_distances)

signal, sample_rate = librosa.load(input_audio_path, sr=None)
magnitude_spectrum = np.abs(fft_result)
peaks, _ = find_peaks(magnitude_spectrum)

frequencies = peaks * sample_rate / fft_length
print("Frequencies of formants (Hz):", frequencies)

peak_distances = np.diff(peaks)

mean_peak_distance = np.mean(peak_distances)
pitch_freq = sample_rate / mean_peak_distance
print("peak_distance:" ,mean_peak_distance)
print(sample_rate)
print("Pitch frequency extracted from peak distances (Hz):", pitch_freq)











from scipy.signal import find_peaks
from scipy.signal import lfilter
audio_path = "C:/Users/Asus/Desktop/newa.wav"
signal, sample_rate = librosa.load(audio_path, sr=None)
order = 16 
lpc_coeffs = librosa.lpc(signal, order)
residuals = lfilter(np.append(0, -lpc_coeffs[1:]), [1], signal)
fft_result_residuals = np.fft.fft(residuals)
magnitude_spectrum_residuals = np.abs(fft_result_residuals)
peaks_residuals, _ = find_peaks(magnitude_spectrum_residuals)

plt.plot(magnitude_spectrum_residuals)
plt.plot(peaks_residuals, magnitude_spectrum_residuals[peaks_residuals], 'x')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Residuals with Identified Peaks')
plt.show()
frequencies_residuals = peaks_residuals * sample_rate / len(signal)
print("Formants frequencies extracted using LPC (Hz):", frequencies_residuals)
lpc_coeffs = librosa.lpc(signal, order)
print("LPC Coefficients:", lpc_coeffs)
residuals = lfilter(np.append(0, -lpc_coeffs[1:]), [1], signal)
plt.figure(figsize=(10, 6))
plt.plot(np.abs(np.fft.fft(lpc_coeffs, 2048)))
plt.title('Spectrum of LPC Model')
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()


import wave
import numpy as np
import librosa
def remove_silence(input_path, output_path):

    with wave.open(input_path, 'rb') as wav_file:

        params = wav_file.getparams()
        num_frames = params.nframes
        frame_rate = params.framerate
        num_channels = params.nchannels
        sample_width = params.sampwidth
        
        raw_data = wav_file.readframes(num_frames)
        audio_data = np.frombuffer(raw_data, dtype=np.int16)

        threshold = 1000 
        non_silent_indices = np.where(np.abs(audio_data) > threshold)[0]
        start_index = non_silent_indices[0]
        end_index = non_silent_indices[-1]

        trimmed_audio = audio_data[start_index:end_index]

        with wave.open(output_path, 'wb') as trimmed_wav_file:
            trimmed_wav_file.setparams((num_channels, sample_width, frame_rate, len(trimmed_audio), params.comptype, params.compname))
            trimmed_wav_file.writeframes(trimmed_audio.tobytes())

input_audio_path = "C:/Users/Asus/Desktop/e.wav"
output_audio_path = "C:/Users/Asus/Desktop/newe.wav"

remove_silence(input_audio_path, output_audio_path)

import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
def plot_audio_waveform(audio_file):

    if not os.path.exists(audio_file):
        print("not exist ")
        return
    
    sr, signala = wavfile.read(audio_file)

    plt.figure(figsize=(10, 6))
    plt.plot(signala)
    plt.title('Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    audio_file = "C:/Users/Asus/Desktop/e.wav"
    plot_audio_waveform(audio_file)
   

if __name__ == "__main__":
    audio_file = "C:/Users/Asus/Desktop/newe.wav"
    plot_audio_waveform(audio_file)
    



def plot_audio_fft(input_path):
    with wave.open(input_path, 'rb') as wav_file:
        params = wav_file.getparams()
        num_frames = params.nframes
        frame_rate = params.framerate
        num_channels = params.nchannels
        sample_width = params.sampwidth

        raw_data = wav_file.readframes(num_frames)
  
        audio_data = np.frombuffer(raw_data, dtype=np.int16)

        fft_result = np.fft.fft(audio_data)
        frequencies = np.fft.fftfreq(len(fft_result), 1 / frame_rate)

        plt.figure(figsize=(10, 6))
        plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result)[:len(frequencies)//2])
        plt.title("FFT of Audio Signal")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        peaks = np.where(np.abs(fft_result) > np.mean(np.abs(fft_result)) + 3 * np.std(np.abs(fft_result)))[0]
        peak_frequencies = frequencies[peaks]
        peak_amplitudes = np.abs(fft_result[peaks])
        plt.show()

input_audio_path = "C:/Users/Asus/Desktop/newe.wav"

plot_audio_fft(input_audio_path)
from scipy.signal import find_peaks
signal, fs = librosa.load(input_audio_path, sr=None)

print("Sampling Rate:", fs)
import numpy as np

fft_result = np.fft.fft(signal)
fft_length = len(fft_result)
print("Length of FFT:", fft_length)

import numpy as np
from scipy.signal import argrelextrema

peaks = argrelextrema(signal, np.greater)[0]
peak_distances = np.diff(peaks)
mean_peak_distance = np.mean(peak_distances)

signal, sample_rate = librosa.load(input_audio_path, sr=None)
magnitude_spectrum = np.abs(fft_result)
peaks, _ = find_peaks(magnitude_spectrum)

frequencies = peaks * sample_rate / fft_length
print("Frequencies of formants (Hz):", frequencies)

peak_distances = np.diff(peaks)

mean_peak_distance = np.mean(peak_distances)
pitch_freq = sample_rate / mean_peak_distance
print("peak_distance:" ,mean_peak_distance)
print(sample_rate)












from scipy.signal import find_peaks
from scipy.signal import lfilter
audio_path = "C:/Users/Asus/Desktop/newe.wav"
signal, sample_rate = librosa.load(audio_path, sr=None)
order = 16 
lpc_coeffs = librosa.lpc(signal, order)
residuals = lfilter(np.append(0, -lpc_coeffs[1:]), [1], signal)
fft_result_residuals = np.fft.fft(residuals)
magnitude_spectrum_residuals = np.abs(fft_result_residuals)
peaks_residuals, _ = find_peaks(magnitude_spectrum_residuals)

plt.plot(magnitude_spectrum_residuals)
plt.plot(peaks_residuals, magnitude_spectrum_residuals[peaks_residuals], 'x')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Residuals with Identified Peaks')
plt.show()
frequencies_residuals = peaks_residuals * sample_rate / len(signal)
print("Formants frequencies extracted using LPC (Hz):", frequencies_residuals)
lpc_coeffs = librosa.lpc(signal, order)
print("LPC Coefficients:", lpc_coeffs)
residuals = lfilter(np.append(0, -lpc_coeffs[1:]), [1], signal)
plt.figure(figsize=(10, 6))
plt.plot(np.abs(np.fft.fft(lpc_coeffs, 2048)))
plt.title('Spectrum of LPC Model')
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

import wave
import numpy as np
import librosa
def remove_silence(input_path, output_path):

    with wave.open(input_path, 'rb') as wav_file:

        params = wav_file.getparams()
        num_frames = params.nframes
        frame_rate = params.framerate
        num_channels = params.nchannels
        sample_width = params.sampwidth
        
        raw_data = wav_file.readframes(num_frames)
        audio_data = np.frombuffer(raw_data, dtype=np.int16)

        threshold = 1000 
        non_silent_indices = np.where(np.abs(audio_data) > threshold)[0]
        start_index = non_silent_indices[0]
        end_index = non_silent_indices[-1]

        trimmed_audio = audio_data[start_index:end_index]

        with wave.open(output_path, 'wb') as trimmed_wav_file:
            trimmed_wav_file.setparams((num_channels, sample_width, frame_rate, len(trimmed_audio), params.comptype, params.compname))
            trimmed_wav_file.writeframes(trimmed_audio.tobytes())

input_audio_path = "C:/Users/Asus/Desktop/ee.wav"
output_audio_path = "C:/Users/Asus/Desktop/newee.wav"

remove_silence(input_audio_path, output_audio_path)

import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
def plot_audio_waveform(audio_file):

    if not os.path.exists(audio_file):
        print("not exist ")
        return
    
    sr, signala = wavfile.read(audio_file)

    plt.figure(figsize=(10, 6))
    plt.plot(signala)
    plt.title('Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    audio_file = "C:/Users/Asus/Desktop/ee.wav"
    plot_audio_waveform(audio_file)
   

if __name__ == "__main__":
    audio_file = "C:/Users/Asus/Desktop/newee.wav"
    plot_audio_waveform(audio_file)
    



def plot_audio_fft(input_path):
    with wave.open(input_path, 'rb') as wav_file:
        params = wav_file.getparams()
        num_frames = params.nframes
        frame_rate = params.framerate
        num_channels = params.nchannels
        sample_width = params.sampwidth

        raw_data = wav_file.readframes(num_frames)
  
        audio_data = np.frombuffer(raw_data, dtype=np.int16)

        fft_result = np.fft.fft(audio_data)
        frequencies = np.fft.fftfreq(len(fft_result), 1 / frame_rate)

        plt.figure(figsize=(10, 6))
        plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result)[:len(frequencies)//2])
        plt.title("FFT of Audio Signal")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        peaks = np.where(np.abs(fft_result) > np.mean(np.abs(fft_result)) + 3 * np.std(np.abs(fft_result)))[0]
        peak_frequencies = frequencies[peaks]
        peak_amplitudes = np.abs(fft_result[peaks])
        plt.show()

input_audio_path = "C:/Users/Asus/Desktop/newee.wav"

plot_audio_fft(input_audio_path)
from scipy.signal import find_peaks
signal, fs = librosa.load(input_audio_path, sr=None)

print("Sampling Rate:", fs)
import numpy as np

fft_result = np.fft.fft(signal)
fft_length = len(fft_result)
print("Length of FFT:", fft_length)

import numpy as np
from scipy.signal import argrelextrema

peaks = argrelextrema(signal, np.greater)[0]
peak_distances = np.diff(peaks)
mean_peak_distance = np.mean(peak_distances)

signal, sample_rate = librosa.load(input_audio_path, sr=None)
magnitude_spectrum = np.abs(fft_result)
peaks, _ = find_peaks(magnitude_spectrum)

frequencies = peaks * sample_rate / fft_length
print("Frequencies of formants (Hz):", frequencies)

peak_distances = np.diff(peaks)

mean_peak_distance = np.mean(peak_distances)
pitch_freq = sample_rate / mean_peak_distance
print("peak_distance:" ,mean_peak_distance)
print(sample_rate)












from scipy.signal import find_peaks
from scipy.signal import lfilter
audio_path = "C:/Users/Asus/Desktop/newee.wav"
signal, sample_rate = librosa.load(audio_path, sr=None)
order = 16 
lpc_coeffs = librosa.lpc(signal, order)
residuals = lfilter(np.append(0, -lpc_coeffs[1:]), [1], signal)
fft_result_residuals = np.fft.fft(residuals)
magnitude_spectrum_residuals = np.abs(fft_result_residuals)
peaks_residuals, _ = find_peaks(magnitude_spectrum_residuals)

plt.plot(magnitude_spectrum_residuals)
plt.plot(peaks_residuals, magnitude_spectrum_residuals[peaks_residuals], 'x')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Residuals with Identified Peaks')
plt.show()
frequencies_residuals = peaks_residuals * sample_rate / len(signal)
print("Formants frequencies extracted using LPC (Hz):", frequencies_residuals)
lpc_coeffs = librosa.lpc(signal, order)
print("LPC Coefficients:", lpc_coeffs)
residuals = lfilter(np.append(0, -lpc_coeffs[1:]), [1], signal)
plt.figure(figsize=(10, 6))
plt.plot(np.abs(np.fft.fft(lpc_coeffs, 2048)))
plt.title('Spectrum of LPC Model')
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()


import wave
import numpy as np
import librosa
def remove_silence(input_path, output_path):

    with wave.open(input_path, 'rb') as wav_file:

        params = wav_file.getparams()
        num_frames = params.nframes
        frame_rate = params.framerate
        num_channels = params.nchannels
        sample_width = params.sampwidth
        
        raw_data = wav_file.readframes(num_frames)
        audio_data = np.frombuffer(raw_data, dtype=np.int16)

        threshold = 1000 
        non_silent_indices = np.where(np.abs(audio_data) > threshold)[0]
        start_index = non_silent_indices[0]
        end_index = non_silent_indices[-1]

        trimmed_audio = audio_data[start_index:end_index]

        with wave.open(output_path, 'wb') as trimmed_wav_file:
            trimmed_wav_file.setparams((num_channels, sample_width, frame_rate, len(trimmed_audio), params.comptype, params.compname))
            trimmed_wav_file.writeframes(trimmed_audio.tobytes())

input_audio_path = "C:/Users/Asus/Desktop/aa.wav"
output_audio_path = "C:/Users/Asus/Desktop/newaa.wav"

remove_silence(input_audio_path, output_audio_path)

import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
def plot_audio_waveform(audio_file):

    if not os.path.exists(audio_file):
        print("not exist ")
        return
    
    sr, signala = wavfile.read(audio_file)

    plt.figure(figsize=(10, 6))
    plt.plot(signala)
    plt.title('Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    audio_file = "C:/Users/Asus/Desktop/aa.wav"
    plot_audio_waveform(audio_file)
   

if __name__ == "__main__":
    audio_file = "C:/Users/Asus/Desktop/newaa.wav"
    plot_audio_waveform(audio_file)
    



def plot_audio_fft(input_path):
    with wave.open(input_path, 'rb') as wav_file:
        params = wav_file.getparams()
        num_frames = params.nframes
        frame_rate = params.framerate
        num_channels = params.nchannels
        sample_width = params.sampwidth

        raw_data = wav_file.readframes(num_frames)
  
        audio_data = np.frombuffer(raw_data, dtype=np.int16)

        fft_result = np.fft.fft(audio_data)
        frequencies = np.fft.fftfreq(len(fft_result), 1 / frame_rate)

        plt.figure(figsize=(10, 6))
        plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result)[:len(frequencies)//2])
        plt.title("FFT of Audio Signal")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        peaks = np.where(np.abs(fft_result) > np.mean(np.abs(fft_result)) + 3 * np.std(np.abs(fft_result)))[0]
        peak_frequencies = frequencies[peaks]
        peak_amplitudes = np.abs(fft_result[peaks])
        plt.show()

input_audio_path = "C:/Users/Asus/Desktop/newaa.wav"

plot_audio_fft(input_audio_path)
from scipy.signal import find_peaks
signal, fs = librosa.load(input_audio_path, sr=None)

print("Sampling Rate:", fs)
import numpy as np

fft_result = np.fft.fft(signal)
fft_length = len(fft_result)
print("Length of FFT:", fft_length)

import numpy as np
from scipy.signal import argrelextrema

peaks = argrelextrema(signal, np.greater)[0]
peak_distances = np.diff(peaks)
mean_peak_distance = np.mean(peak_distances)

signal, sample_rate = librosa.load(input_audio_path, sr=None)
magnitude_spectrum = np.abs(fft_result)
peaks, _ = find_peaks(magnitude_spectrum)

frequencies = peaks * sample_rate / fft_length
print("Frequencies of formants (Hz):", frequencies)

peak_distances = np.diff(peaks)

mean_peak_distance = np.mean(peak_distances)
pitch_freq = sample_rate / mean_peak_distance
print("peak_distance:" ,mean_peak_distance)
print(sample_rate)












from scipy.signal import find_peaks
from scipy.signal import lfilter
audio_path = "C:/Users/Asus/Desktop/newaa.wav"
signal, sample_rate = librosa.load(audio_path, sr=None)
order = 16 
lpc_coeffs = librosa.lpc(signal, order)
residuals = lfilter(np.append(0, -lpc_coeffs[1:]), [1], signal)
fft_result_residuals = np.fft.fft(residuals)
magnitude_spectrum_residuals = np.abs(fft_result_residuals)
peaks_residuals, _ = find_peaks(magnitude_spectrum_residuals)

plt.plot(magnitude_spectrum_residuals)
plt.plot(peaks_residuals, magnitude_spectrum_residuals[peaks_residuals], 'x')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Residuals with Identified Peaks')
plt.show()
frequencies_residuals = peaks_residuals * sample_rate / len(signal)
print("Formants frequencies extracted using LPC (Hz):", frequencies_residuals)
lpc_coeffs = librosa.lpc(signal, order)
print("LPC Coefficients:", lpc_coeffs)
residuals = lfilter(np.append(0, -lpc_coeffs[1:]), [1], signal)
plt.figure(figsize=(10, 6))
plt.plot(np.abs(np.fft.fft(lpc_coeffs, 2048)))
plt.title('Spectrum of LPC Model')
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()


import wave
import numpy as np
import librosa
def remove_silence(input_path, output_path):

    with wave.open(input_path, 'rb') as wav_file:

        params = wav_file.getparams()
        num_frames = params.nframes
        frame_rate = params.framerate
        num_channels = params.nchannels
        sample_width = params.sampwidth
        
        raw_data = wav_file.readframes(num_frames)
        audio_data = np.frombuffer(raw_data, dtype=np.int16)

        threshold = 1000 
        non_silent_indices = np.where(np.abs(audio_data) > threshold)[0]
        start_index = non_silent_indices[0]
        end_index = non_silent_indices[-1]

        trimmed_audio = audio_data[start_index:end_index]

        with wave.open(output_path, 'wb') as trimmed_wav_file:
            trimmed_wav_file.setparams((num_channels, sample_width, frame_rate, len(trimmed_audio), params.comptype, params.compname))
            trimmed_wav_file.writeframes(trimmed_audio.tobytes())

input_audio_path = "C:/Users/Asus/Desktop/ao.wav"
output_audio_path = "C:/Users/Asus/Desktop/newao.wav"

remove_silence(input_audio_path, output_audio_path)

import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
def plot_audio_waveform(audio_file):

    if not os.path.exists(audio_file):
        print("not exist ")
        return
    
    sr, signala = wavfile.read(audio_file)

    plt.figure(figsize=(10, 6))
    plt.plot(signala)
    plt.title('Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    audio_file = "C:/Users/Asus/Desktop/ao.wav"
    plot_audio_waveform(audio_file)
   

if __name__ == "__main__":
    audio_file = "C:/Users/Asus/Desktop/newao.wav"
    plot_audio_waveform(audio_file)
    



def plot_audio_fft(input_path):
    with wave.open(input_path, 'rb') as wav_file:
        params = wav_file.getparams()
        num_frames = params.nframes
        frame_rate = params.framerate
        num_channels = params.nchannels
        sample_width = params.sampwidth

        raw_data = wav_file.readframes(num_frames)
  
        audio_data = np.frombuffer(raw_data, dtype=np.int16)

        fft_result = np.fft.fft(audio_data)
        frequencies = np.fft.fftfreq(len(fft_result), 1 / frame_rate)

        plt.figure(figsize=(10, 6))
        plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result)[:len(frequencies)//2])
        plt.title("FFT of Audio Signal")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        peaks = np.where(np.abs(fft_result) > np.mean(np.abs(fft_result)) + 3 * np.std(np.abs(fft_result)))[0]
        peak_frequencies = frequencies[peaks]
        peak_amplitudes = np.abs(fft_result[peaks])
        plt.show()

input_audio_path = "C:/Users/Asus/Desktop/newao.wav"

plot_audio_fft(input_audio_path)
from scipy.signal import find_peaks
signal, fs = librosa.load(input_audio_path, sr=None)

print("Sampling Rate:", fs)
import numpy as np

fft_result = np.fft.fft(signal)
fft_length = len(fft_result)
print("Length of FFT:", fft_length)

import numpy as np
from scipy.signal import argrelextrema

peaks = argrelextrema(signal, np.greater)[0]
peak_distances = np.diff(peaks)
mean_peak_distance = np.mean(peak_distances)

signal, sample_rate = librosa.load(input_audio_path, sr=None)
magnitude_spectrum = np.abs(fft_result)
peaks, _ = find_peaks(magnitude_spectrum)

frequencies = peaks * sample_rate / fft_length
print("Frequencies of formants (Hz):", frequencies)

peak_distances = np.diff(peaks)

mean_peak_distance = np.mean(peak_distances)
pitch_freq = sample_rate / mean_peak_distance
print("peak_distance:" ,mean_peak_distance)
print(sample_rate)












from scipy.signal import find_peaks
from scipy.signal import lfilter
audio_path = "C:/Users/Asus/Desktop/newao.wav"
signal, sample_rate = librosa.load(audio_path, sr=None)
order = 16 
lpc_coeffs = librosa.lpc(signal, order)
residuals = lfilter(np.append(0, -lpc_coeffs[1:]), [1], signal)
fft_result_residuals = np.fft.fft(residuals)
magnitude_spectrum_residuals = np.abs(fft_result_residuals)
peaks_residuals, _ = find_peaks(magnitude_spectrum_residuals)

plt.plot(magnitude_spectrum_residuals)
plt.plot(peaks_residuals, magnitude_spectrum_residuals[peaks_residuals], 'x')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Residuals with Identified Peaks')
plt.show()
frequencies_residuals = peaks_residuals * sample_rate / len(signal)
print("Formants frequencies extracted using LPC (Hz):", frequencies_residuals)
lpc_coeffs = librosa.lpc(signal, order)
print("LPC Coefficients:", lpc_coeffs)
residuals = lfilter(np.append(0, -lpc_coeffs[1:]), [1], signal)
plt.figure(figsize=(10, 6))
plt.plot(np.abs(np.fft.fft(lpc_coeffs, 2048)))
plt.title('Spectrum of LPC Model')
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()
