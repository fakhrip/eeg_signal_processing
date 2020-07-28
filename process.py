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
  ax.set_xticks(np.arange(sampling_duration))
  dr = (dmax - dmin) * 0.7  # Crowd them a bit.
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

def initPlots() :
  fig = plt.figure("EEG_Signals_Graphing", constrained_layout=True)
  spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
  return {
    "fig": fig,
    "spec": spec
  }

def showPlots() :
  plt.show()

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

def main() :
  # file_name = input("Tuliskan nama file nya = ")
  file_name = "rsvp_5Hz_02a.edf"
  properties = parseEDF(file_name)

  # signal_labels = properties["signal_labels"]
  signal_labels = ["EEG P8", "EEG P7", "EEG O1", "EEG O2"]

  # sampling_freq = properties["sampling_frequency"]
  sampling_freq = 64 # in Hertz

  # sampling_duration = properties["sampling_duration"]
  sampling_duration = 20 # in Second

  n_samples = sampling_freq * sampling_duration 

  # Cutoff Frequencies (in Hertz)
  lowcut = 1
  highcut = 30

  plotData = initPlots()

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
  plotSignals(plotData["fig"],
                filtered_signals, 
                signals_data.min(),
                signals_data.max(),
                n_samples, 
                sampling_duration, 
                signal_labels, 
                "Filtered signals - Time (s)",
                plotData["spec"][1, 1])
  showPlots()

if __name__ == "__main__" :
  main()