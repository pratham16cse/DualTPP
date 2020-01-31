import matplotlib.pyplot as plt
import numpy as np


getHour = lambda t: t // 3600 % 24
gaps_list = [4, 8, 3, 5, 2, 10, 15]

num_discrete_states = len(gaps_list)
sequence_length = 10000
begin_ts = 1431777686
begin_ts = 0000000000

cur_ts = begin_ts
timestamps = list()
markers = list()
gap_idx = 1
for i in range(sequence_length):
    cur_ts += gaps_list[gap_idx-1]
    timestamps.append(cur_ts)
    markers.append(gap_idx)
    gap_idx = (gap_idx+1) % len(gaps_list)


#markers = np.ones_like(timestamps)
event_tuples = list()
for i, (m, t) in enumerate(zip(markers, timestamps)):
    event_tuples.append([m, t, i+1])

np.savetxt('testdata.txt', event_tuples, delimiter=" ", fmt="%s")
