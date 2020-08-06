import os
import math
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


def plotSignals(fig, data, dmin, dmax, n_samples, sampling_duration, signal_labels, xLabel, subplot_loc) :
  n_rows = len(data[0])
  time = sampling_duration * np.arange(n_samples) / n_samples

  ticklocs = []
  ax = fig.add_subplot(subplot_loc)
  ax.set_xlim(0, sampling_duration)
  ax.set_xticks(np.arange(0, sampling_duration, 50))
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
  plt.savefig("{}.png".format(file_name))


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
    
  return  {
    "signals": transformed_signals.transpose(),
    "frequency": len(transformed_signals.transpose())
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
                transformed_signals["frequency"], 
                sampling_duration, 
                signal_labels, 
                "FFT signals - Frequency (Hz)",
                plotData["spec"][1, 1])
  # generatePlotsToPNG(file_name)
  showPlots()


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
      break

  if not isExist :
    print("\n[!] There are no .edf file found")
  else :
    print("\n[+] All process finished successfully")

if __name__ == "__main__" :
  main()