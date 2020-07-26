import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# file_name = input("Tuliskan nama file nya = ")
file_name = "rsvp_5Hz_02a.edf"
file = pyedflib.EdfReader(file_name)
numberOfSignals = file.signals_in_file
samplingFrequncy = file.getSampleFrequencies()
samplingDuration = file.getFileDuration()
n_samples = samplingDuration * samplingFrequncy
n_rows = 4

signalLabels = file.getSignalLabels()
parsedSignals = np.zeros((n_rows, file.getNSamples()[0]))

position = 0
for i, label in enumerate(signalLabels) :
  if label == "EEG P8" or label == "EEG P7" or label == "EEG O1" or label == "EEG O2" :
    parsedSignals[position, :] = file.readSignal(i)
    position += 1

data = parsedSignals.transpose()

fig = plt.figure("EEG_Signals_Graphing")
time = samplingDuration * np.arange(n_samples[0]) / n_samples[0]

# Plot the EEG
ticklocs = []
ax2 = fig.add_subplot(1, 1, 1)
ax2.set_xlim(0, samplingDuration)
ax2.set_xticks(np.arange(samplingDuration))
dmin = data.min()
dmax = data.max()
dr = (dmax - dmin) * 0.7  # Crowd them a bit.
y0 = dmin
y1 = (n_rows - 1) * dr + dmax
ax2.set_ylim(y0, y1)

segs = []
for i in range(n_rows):
  segs.append(np.column_stack((time, data[:, i])))
  ticklocs.append(i * dr)

offsets = np.zeros((n_rows, 2), dtype=float)
offsets[:, 1] = ticklocs

lines = LineCollection(segs, offsets=offsets, transOffset=None)
ax2.add_collection(lines)

# Set the yticks to use axes coordinates on the y axis
ax2.set_yticks(ticklocs)
ax2.set_yticklabels(['P8', 'P7', 'O1', 'O2'])

ax2.set_xlabel('Time (s)')

plt.tight_layout()
plt.show()