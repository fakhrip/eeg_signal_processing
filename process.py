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
  b, a = butter(order, [low, high], btype="band")
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

  # signal_labels = ["EEG PO8", "EEG PO7", "EEG P8", "EEG P7", "EEG O1", "EEG O2", "EEG PO3", "EEG PO4"]
  signal_labels = ["EEG P8", "EEG P7", "EEG O1", "EEG O2"]
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

  # Create signals_plot folder if not exist
  try:  
    os.mkdir("./signals_plot")  
  except OSError as error:  
    pass

  generatePlotsToPNG(file_name)

  return {
    "fft_signals": transformed_signals["signals"].transpose(),
    "fft_freq": transformed_signals["frequency"],
    "channels": signal_labels
  }


def plotPSDSignals(fft_results, type, result_name) :
  print("[|] Plotting {} PSD Signal(s)".format(type))

  # Define EEG band frequencies
  eeg_bands = {
    # "Delta": (0, 4),
    # "Theta": (4, 8),
    "Alpha": (8, 12),
    # "Beta": (12, 30),
    "Gamma": (30, 45)
  }

  if type == "single" :

    fft_signals = fft_results["fft_signals"]
    fft_freq = fft_results["fft_freq"]

    signal_labels = fft_results["channels"]    
    
    try:  
      os.makedirs("./psd_single_plot/{}".format(result_name.split(".edf")[0]))  
    except OSError as error:  
      pass

    # Take the mean of the fft amplitude for 
    # each EEG band in each channel
    for i in range(len(fft_signals)) :
      fft_vals = fft_signals[i]

      eeg_band_fft = dict()
      for band in eeg_bands:  
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                          (fft_freq <= eeg_bands[band][1]))[0]
        eeg_band_fft[band] = np.mean(fft_vals[freq_ix])

      fig = plt.figure(figsize=(20,10), dpi=75)
      ax = fig.add_subplot(111)

      df = pd.DataFrame(columns=["band", "val"])
      df["band"] = list(eeg_bands.keys())
      df["val"] = [eeg_band_fft[band] for band in eeg_bands]
      df.plot.bar(x="band", y="val", legend=False, ax=ax)
      ax.set_title("EEG Power Bands")
      ax.set_xlabel("Frequency band")
      ax.set_ylabel("Mean band Amplitude")

      plt.savefig("./psd_single_plot/{}/{}.png".format(result_name.split(".edf")[0], signal_labels[i]))
      plt.close()

  elif type == "multiple" :
      
    signal_labels = fft_results[0]["channels"]  

    try:  
      os.makedirs("./psd_multiple_plot/subject_{}".format(result_name[0].split(".")[0].split("_")[-1]))  
    except OSError as error:  
      pass
    
    psd_values = list()
    for result in fft_results :
      fft_signals = result["fft_signals"]
      fft_freq = result["fft_freq"]  

      # Take the mean of the fft amplitude for 
      # each EEG band in each channel
      channel_values = list()
      for i in range(len(fft_signals)) :
        fft_vals = fft_signals[i]

        eeg_band_fft = dict()
        for band in eeg_bands:  
          freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                            (fft_freq <= eeg_bands[band][1]))[0]
          eeg_band_fft[band] = np.mean(fft_vals[freq_ix])

        channel_values.append(eeg_band_fft)

      psd_values.append(channel_values)

    frequencies = list()
    for result in result_name :
      frequencies.append(result.split(".")[0].split("_")[1])

    # -------------
    # TODO: Change this crappy lines of codes  
    #    
    # So many things that can be more efficient than
    # all these lines of codes (particularly this part below)
    # -------------
    for pos, channel in enumerate(signal_labels) :
      fig = plt.figure(figsize=(20,10), dpi=75)
      ax_alpha = fig.add_subplot(121)
      ax_gamma = fig.add_subplot(122)

      df_alpha = pd.DataFrame(columns=["band", "val"])
      df_alpha["band"] = frequencies
      df_alpha["val"] = [(signal[pos]["Alpha"]) for signal in psd_values]
      df_alpha.plot.bar(x="band", y="val", legend=False, ax=ax_alpha)
      ax_alpha.set_title("Alpha EEG Power Bands")
      ax_alpha.set_xlabel("Frequency band")
      ax_alpha.set_ylabel("Mean band Amplitude")

      for p in ax_alpha.patches:                 
        ax_alpha.annotate(
          np.round(p.get_height(),decimals=2), 
          (p.get_x()+p.get_width()/2., p.get_height()),      
          ha='center',                              
          va='center',                             
          xytext=(0, 10),                               
          textcoords='offset points')

      df_gamma = pd.DataFrame(columns=["band", "val"])
      df_gamma["band"] = frequencies
      df_gamma["val"] = [(signal[pos]["Gamma"]) for signal in psd_values]
      df_gamma.plot.bar(x="band", y="val", legend=False, ax=ax_gamma)
      ax_gamma.set_title("Gamma EEG Power Bands")
      ax_gamma.set_xlabel("Frequency band")
      ax_gamma.set_ylabel("Mean band Amplitude")

      for p in ax_gamma.patches:                 
        ax_gamma.annotate(
          np.round(p.get_height(),decimals=2), 
          (p.get_x()+p.get_width()/2., p.get_height()),                              
          ha='center',
          va='center',                             
          xytext=(0, 10),                               
          textcoords='offset points') 

      plt.savefig("./psd_multiple_plot/subject_{}/{}.png".format(result_name[0].split(".")[0].split("_")[-1], channel))
      plt.close()


"""
Process all edf files one by one

Will resulted to a plot of all selected channels 
in a file
"""
def processEachFile() :
  isExist = False

  files = [f for f in os.listdir(".") if os.path.isfile(f)]
  for f in files :
    if f.split(".")[-1] == "edf" :
      isExist = True
      print("\n[|] Processing {}".format(f))
      result = processEDFFile(f)
      plotPSDSignals(result, "single", f)
      print("[|] Graph for {} has been generated to psd_single_plot/{}".format(f.split(".edf")[0]))

  if not isExist :
    print("\n[!] There are no .edf file found")
  else :
    print("\n[+] All process finished successfully")


"""
Process all edf files based on subject and frequency

Will resulted to a plot of a subject with 
some frequencies on the corresponding channel

File name format must be:
[a-zA-Z0-9]_[frequency]_[subject].edf

Example:
rsvp_5Hz_02a.edf

param: 
- subject (required) [subject to process]
- frequencies (required) [array of all selected frequencies (if exist)]
"""
def processEachSubject(subject, frequencies) :
  isExist = False

  files = [f for f in os.listdir(".") if os.path.isfile(f)]
  filtered_files = list()
  for f in files :
    if f.split(".")[-1] == "edf" :
      if f.split(".")[0].split("_")[1] in frequencies and f.split(".")[0].split("_")[-1] == subject :
        filtered_files.append(f)

  print("[|] Found {} files matching subject".format(len(filtered_files)))

  results_arr = list()
  for file in filtered_files :
    print("[|] Processing {}".format(file))
    results_arr.append(processEDFFile(file))

  plotPSDSignals(results_arr, "multiple", filtered_files)
  print("[|] All graphs has been generated to psd_multiple_plot/subject_{}".format(subject))


def main() :
  print("[+] Starting ...")

  # -----------
  # Uncomment below if you want to parse each files
  # one by one
  # -----------
  # processEachFile()

  # -----------
  # Uncomment below if you want to parse each subject
  # in multiple files
  # -----------
  subject_arr = ["02a", "03a", "04a", "06a", "08a", "09a"]
  frequencies = ["5Hz", "6Hz", "10Hz"]
  for subject in subject_arr :
    print("\n[|] Parsing files for {} subject".format(subject))
    processEachSubject(subject, frequencies)


if __name__ == "__main__" :
  main()