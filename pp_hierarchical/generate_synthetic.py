import matplotlib.pyplot as plt
import numpy as np


getHour = lambda t: t // 3600 % 24
hr_to_gaps = dict()
for hr in range(24):
    hr_to_gaps[hr] = 2 * (hr + 1) * 60


num_discrete_states = len(hr_to_gaps)
sequence_length = 10000
begin_ts = 1431777686

cur_ts = begin_ts
timestamps = list()
markers = list()
for i in range(sequence_length):
    cur_ts += hr_to_gaps[getHour(cur_ts)]
    timestamps.append(cur_ts)
    markers.append(getHour(cur_ts))


#markers = np.ones_like(timestamps)
event_tuples = list()
for i, (m, t) in enumerate(zip(markers, timestamps)):
    event_tuples.append([m, t, i+1])

np.savetxt('synthetic.txt', event_tuples, delimiter=" ", fmt="%s")
