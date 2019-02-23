import numpy as np

event = np.ones((20, 300))
time = np.zeros((20, 300))
for i in range(20):
    step = np.random.normal()
    stop = step*300
    time[i, :] = np.arange(0, stop, step)[:300]

event_train = event[:, :200]
time_train = time[:, :200]
event_test = event[:,200:]
time_test = time[:,200:]

np.savetxt('event.txt', event, delimiter=' ', fmt='%.4d')
np.savetxt('time.txt', time, delimiter=' ', fmt='%.4f')
np.savetxt('event-train.txt', event_train, delimiter=' ', fmt='%.4d')
np.savetxt('time-train.txt', time_train, delimiter=' ', fmt='%.4f')
np.savetxt('event-test.txt', event_test, delimiter=' ', fmt='%.4d')
np.savetxt('time-test.txt', time_test, delimiter=' ', fmt='%.4f')
