import os
import math
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='band')
  return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
  b, a = butter_bandpass(lowcut, highcut, fs, order=order)
  y = lfilter(b, a, data)
  return y


def plotSignals(fig, data, dmin, dmax, n_samples, sampling_duration, signal_labels, xLabel, subplot_loc, xticks_offset = 50) :
  n_rows = len(data[0])
  time = sampling_duration * np.arange(n_samples) / n_samples

  ticklocs = []
  ax = fig.add_subplot(subplot_loc)
  ax.set_xlim(0, sampling_duration)
  ax.set_xticks(np.arange(0, sampling_duration, xticks_offset))
  dr = (dmax - dmin)  # Crowd them a bit.
  y0 = dmin
  y1 = (n_rows - 1) * dr + dmax
  ax.set_ylim(y0, y1)

  segs = []
  for i in range(n_rows):
    segs.append(np.column_stack((time, data[:, i])))
    ticklocs.append(i * dr)

  offsets = np.zeros((n_rows, 2), dtype=float)
  offsets[:, 1] = ticklocs

  lines = LineCollection(segs, offsets=offsets, transOffset=None)
  ax.add_collection(lines)

  # Set the yticks to use axes coordinates on the y axis
  ax.set_yticks(ticklocs)
  ax.set_yticklabels(signal_labels)

  ax.set_xlabel(xLabel)


def initPlots(file_name) :
  fig = plt.figure(file_name, constrained_layout=True, dpi=100)
  fig.set_size_inches(16.5, 8.5)
  spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
  return {
    "fig": fig,
    "spec": spec
  }


def showPlots() :
  plt.show()


def generatePlotsToPNG(file_name) :
  plt.savefig("./signals_plot/{}.png".format(file_name))
  plt.close()


def parseEDF(file_name) :
  file                = pyedflib.EdfReader(file_name)
  numberOfSignals     = file.signals_in_file
  samplingFrequency   = file.getSampleFrequencies()
  samplingDuration    = file.getFileDuration()
  signalLabels        = file.getSignalLabels()
  n_samples           = samplingDuration * samplingFrequency

  return {
    "file": file,
    "n_signals": numberOfSignals,
    "n_samples": n_samples[0],
    "signal_labels": signalLabels,
    "sampling_duration": samplingDuration,
    "sampling_frequency": samplingFrequency,
  }


def getSignals(file, n_samples, filtered_labels = None) :
  if filtered_labels != None :
    parsedSignals = np.zeros((len(filtered_labels), n_samples))
  else :
    parsedSignals = np.zeros((len(file.getSignalLabels()), n_samples))

  position = 0
  added_signals = list()
  for i, label in enumerate(file.getSignalLabels()) :
    if filtered_labels != None :
      if label in filtered_labels and (label not in added_signals) :
        parsedSignals[position, :] = file.readSignal(i)
        added_signals.append(label)
        position += 1  
    else :
      parsedSignals[i, :] = file.readSignal(i)

  return parsedSignals.transpose()


def downsampleSignals(signals, n_samples) :
  parsedSignals = signals.transpose()

  actual_n_samples = len(parsedSignals[0])
  if n_samples < actual_n_samples :
    downsampledSignals = np.zeros((len(parsedSignals), n_samples))
    offset = int(actual_n_samples / n_samples)
    for i in range(len(parsedSignals)) :
      for j in range(n_samples) :
        downsampledSignals[i, j] = parsedSignals[i, j * offset]
  else :
    print("[!] Sample frequency of downsampling need to be lower than the actual sample frequency")
    return signals
  
  return downsampledSignals.transpose()


def bpfSignals(signals, lowcut, highcut, fs) :
  parsedSignals = signals.transpose()

  for i in range(len(parsedSignals)) :
    parsedSignals[i] = butter_bandpass_filter(parsedSignals[i], lowcut, highcut, fs)

  return parsedSignals.transpose()


def fftSignals(signals, sampling_freq) :
  parsedSignals = signals.transpose()

  transformed_signals = np.zeros((len(parsedSignals), len(np.absolute(np.fft.rfft(parsedSignals[0])))))
  for i in range(len(transformed_signals)) :
    transformed_signals[i] = np.absolute(np.fft.rfft(parsedSignals[i]))

  freq = np.fft.rfftfreq(len(parsedSignals[0]), 1.0/sampling_freq)
    
  return  {
    "signals": transformed_signals.transpose(),
    "frequency": freq
  }


def processEDFFile(file_name) :
  properties = parseEDF(file_name)

  # signal_labels = properties["signal_labels"]
  # sampling_freq = properties["sampling_frequency"]

  signal_labels = ["EEG PO8", "EEG PO7", "EEG P8", "EEG P7", "EEG O1", "EEG O2", "EEG PO3", "EEG PO4"]
  sampling_freq = 64 # in Hertz
  sampling_duration = properties["sampling_duration"]
  n_samples = sampling_freq * sampling_duration 

  # Cutoff Frequencies (in Hertz)
  lowcut = 1
  highcut = 30

  print("[|] Plotting FFT Signal")

  plotData = initPlots(file_name)
  signals_data = getSignals(properties["file"], properties["n_samples"], signal_labels)
  plotSignals(plotData["fig"],
                signals_data, 
                signals_data.min(),
                signals_data.max(),
                properties["n_samples"], 
                sampling_duration, 
                signal_labels, 
                "Original signals - Time (s)",
                plotData["spec"][0, 0])

  downsampled_signals = downsampleSignals(signals_data, n_samples)
  plotSignals(plotData["fig"],
                downsampled_signals, 
                signals_data.min(),
                signals_data.max(),
                n_samples, 
                sampling_duration, 
                signal_labels, 
                "Downsampled signals - Time (s)",
                plotData["spec"][0, 1])

  filtered_signals = bpfSignals(downsampled_signals, lowcut, highcut, sampling_freq)
  plotSignals(plotData["fig"],
                filtered_signals, 
                signals_data.min(),
                signals_data.max(),
                n_samples, 
                sampling_duration, 
                signal_labels, 
                "Filtered signals - Time (s)",
                plotData["spec"][1, 0])

  transformed_signals = fftSignals(filtered_signals, sampling_freq)
  plotSignals(plotData["fig"],
                transformed_signals["signals"], 
                transformed_signals["signals"].min(),
                transformed_signals["signals"].max(),
                len(transformed_signals["signals"]), 
                transformed_signals["frequency"].max(), 
                signal_labels, 
                "FFT signals - Frequency (Hz)",
                plotData["spec"][1, 1],
                1)

  print("[|] Plotting PSD Signal")

  # Create signals_plot folder if not exist
  try:  
    os.mkdir("./signals_plot")  
  except OSError as error:  
    pass

  generatePlotsToPNG(file_name)

  # Define EEG band frequencies
  eeg_bands = {
    "Delta": (0, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (12, 30),
    "Gamma": (30, 45)
  }

  signals = transformed_signals["signals"].transpose()
  fft_freq = transformed_signals["frequency"]

  try:  
    os.makedirs("./psd_plot/{}".format(file_name.split(".edf")[0]))  
  except OSError as error:  
    pass

  # Take the mean of the fft amplitude for each EEG band
  for i in range(len(signals)) :
    fft_vals = signals[i]

    eeg_band_fft = dict()
    for band in eeg_bands:  
      freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                        (fft_freq <= eeg_bands[band][1]))[0]
      eeg_band_fft[band] = np.mean(fft_vals[freq_ix])

    fig = plt.figure(figsize=(20,10), dpi=75)
    ax = fig.add_subplot(111)

    df = pd.DataFrame(columns=['band', 'val'])
    df['band'] = list(eeg_bands.keys())
    df['val'] = [eeg_band_fft[band] for band in eeg_bands]
    df.plot.bar(x='band', y='val', legend=False, ax=ax)
    ax.set_title("EEG Power Bands")
    ax.set_xlabel("Frequency band")
    ax.set_ylabel("Mean band Amplitude")

    plt.savefig("./psd_plot/{}/{}.png".format(file_name.split(".edf")[0], signal_labels[i]))
    plt.close()

def main() :
  print("[+] Starting ...")
  isExist = False
  files = [f for f in os.listdir('.') if os.path.isfile(f)]
  for f in files :
    if(f.split(".")[-1] == "edf") :
      isExist = True
      print("\n[|] Processing {}".format(f))
      processEDFFile(f)
      print("[|] Graph for {} has been generated to {}.png".format(f, f.split(".edf")[0]))

  if not isExist :
    print("\n[!] There are no .edf file found")
  else :
    print("\n[+] All process finished successfully")

if __name__ == "__main__" :
  main()