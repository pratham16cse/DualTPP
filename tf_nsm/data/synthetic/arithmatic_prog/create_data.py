import numpy as np

event = np.ones((40, 1000))
time = np.zeros((40, 1000))
for i in range(40):
    step = np.random.normal()
    stop = step*1000
    time[i, :] = np.arange(0, stop, step)[:1000]

event_train = event[:, :900]
time_train = time[:, :900]
event_test = event[:,900:]
time_test = time[:,900:]

np.savetxt('event.txt', event, delimiter=' ', fmt='%.4d')
np.savetxt('time.txt', time, delimiter=' ', fmt='%.4f')
np.savetxt('event-train.txt', event_train, delimiter=' ', fmt='%.4d')
np.savetxt('time-train.txt', time_train, delimiter=' ', fmt='%.4f')
np.savetxt('event-test.txt', event_test, delimiter=' ', fmt='%.4d')
np.savetxt('time-test.txt', time_test, delimiter=' ', fmt='%.4f')
