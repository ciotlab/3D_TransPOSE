import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

'''------------------------------------------------------------------------------------------------------------------'''
radar_dir = 'D:/revised_train/subtracted_radar_one_person_C/'  #1~119000
cnt = 37067
'''------------------------------------------------------------------------------------------------------------------'''

signal = np.load(radar_dir + '%08d.npy' % cnt)
num_tx, num_rx, _ = signal.shape
fig = plt.figure(figsize=(15, 9))
title = plt.suptitle('%08d.npy' % cnt)

gs1 = GridSpec(num_tx, num_rx, left=0.03, right=0.97, wspace=0.10)

temp = np.zeros(shape=(num_tx, num_rx))
temp_signal_list = np.array(temp, dtype=object)
signal_list = np.array(temp, dtype=object)

for tx in range(num_tx):
    for rx in range(num_rx):
        temp_signal_list[tx][rx] = fig.add_subplot(gs1[tx, rx])
        temp_signal_list[tx][rx].axis(ymin=-2, ymax=2)
        temp_signal_list[tx][rx].axes.xaxis.set_visible(False)
        temp_signal_list[tx][rx].axes.yaxis.set_visible(False)

        signal_list[tx][rx], = temp_signal_list[tx][rx].plot(signal[tx][rx],
                                                             label='tx:{}, rx:{}'.format(tx + 1, rx + 1))
        temp_signal_list[tx][rx].legend(loc='lower right', fontsize=5)


def animate(i):
    animate_cnt = i + cnt + 1
    title.set_text('%08d.npy' % animate_cnt)
    animate_signal = np.load(radar_dir + '%08d.npy' % animate_cnt)

    for tx in range(num_tx):
        for rx in range(num_rx):
            signal_list[tx][rx].set_ydata(animate_signal[tx][rx])

    return signal_list


ani = animation.FuncAnimation(fig, animate, frames=1000, interval=200, blit=False)
plt.show()
