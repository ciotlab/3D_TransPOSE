from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
import json
import re
from tqdm import tqdm
import logging
import torch
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from typing import Union, Optional, Tuple

# Radar hardware data for imaging
tx_antenna_position = [[-0.5, 0, 1.050],
                       [-0.5, 0, 0.800],
                       [-0.5, 0, 0.550],
                       [-0.5, 0, 0.300],
                       [0.5, 0, 1.050],
                       [0.5, 0, 0.800],
                       [0.5, 0, 0.550],
                       [0.5, 0, 0.300]]
rx_antenna_position = [[-0.3, 0, 1.050],
                       [-0.1, 0, 1.050],
                       [0.1, 0, 1.050],
                       [0.3, 0, 1.050],
                       [-0.3, 0, 0.300],
                       [-0.1, 0, 0.300],
                       [0.1, 0, 0.300],
                       [0.3, 0, 0.300]]
area_min_xyz = [-1.5, 0.5, 0.0]
area_max_xyz = [1.5, 3.5, 2.5]
area_num_step = [100, 100, 1]
imaging_height = 0.85
radar_sample_spacing = 0.026  # ns
radar_start_sample_time = 0  # ns
radar_num_sample = 1024


class Person:
    def __init__(self, ax):
        self._edge_conf = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
                                    (13, 14), (14, 15), (16, 17), (17, 18), (3, 5), (3, 9), (0, 13), (0, 16), (15, 19),
                                    (18, 20)])
        pt = np.zeros((21, 3))
        self._points = ax.scatter(pt[:, 0], pt[:, 1], pt[:, 2], s=5, alpha=0)
        self._lines = []
        for edge in self._edge_conf:
            tmp = pt[(edge[0], edge[1]), :]
            self._lines.append(ax.plot(tmp[:, 0], tmp[:, 1], tmp[:, 2], alpha=0))

    def hide(self):
        self._points.set_alpha(0)
        for line in self._lines:
            line[0].set_alpha(0)

    def show(self, pt, color='red', alpha=1):
        self._points._offsets3d = (pt[:, 0], pt[:, 1], pt[:, 2])
        self._points.set_color(color)
        self._points.set_alpha(alpha)
        for line, edge in zip(self._lines, self._edge_conf):
            tmp = pt[(edge[0], edge[1]), :]
            line[0].set_data(tmp[:, 0], tmp[:, 1])
            line[0].set_3d_properties(tmp[:, 2])
            line[0].set_color(color)
            line[0].set_alpha(alpha)


class Skeleton:
    def __init__(self, location, num_person, session_idx, type):
        self._location = location
        self._num_person = num_person
        self._session_idx = session_idx
        self._type = type
        file_dir = Path(__file__).parents[1].resolve() / 'data' / f'label_{type}' / location / num_person / f'session_{session_idx}'
        self._files = list(file_dir.glob('*'))
        self._ax = None
        self._person_pool = []

    def set_plot(self, ax):
        self._ax = ax
        ax.set_xlim3d([area_min_xyz[0], area_max_xyz[0]])
        ax.set_xlabel('X')
        ax.set_ylim3d([area_min_xyz[1], area_max_xyz[1]])
        ax.set_ylabel('Y')
        ax.set_zlim3d([area_min_xyz[2], area_max_xyz[2]])
        ax.set_zlabel('Z')
        for _ in range(10):
            self._person_pool.append(Person(ax))

    def show_person(self, pt, color, count):
        num_person = pt.shape[0] // 21
        for p in range(num_person):
            tmp_pt = pt[p*21:(p+1)*21, :]
            self._person_pool[count].show(tmp_pt, color=color, alpha=1)
            count += 1
            if count >= len(self._person_pool):
                break
        return count

    def animate_plot(self, frame):
        count = 0
        file = self._files[frame]
        pt = np.load(file)
        count = self.show_person(pt, 'red', count)
        if count >= len(self._person_pool):
            return
        parts = list(file.parts)
        parts[-5] = 'predicted_label'
        pred_file = Path('').joinpath(*parts)
        if pred_file.exists():
            pt = np.load(str(pred_file))
            count = self.show_person(pt, 'blue', count)
            if count >= len(self._person_pool):
                return
        while count < len(self._person_pool):
            self._person_pool[count].hide()
            count += 1


class RadarSignal:
    def __init__(self, location, num_person, session_idx, type):
        self._location = location
        self._num_person = num_person
        self._session_idx = session_idx
        self._type = type
        file_dir = Path(__file__).parents[1].resolve() / 'data' / f'radar_{type}' / location / num_person / f'session_{session_idx}'
        self._files = list(file_dir.glob('*'))
        signal = np.load(self._files[0])
        self._num_tx, self._num_rx, self._num_samples = signal.shape
        self._num_frames = len(self._files)
        self._sig_plots = np.array(np.zeros((self._num_tx, self._num_rx)), dtype=object)

    def get_info(self):
        return self._num_tx, self._num_rx, self._num_frames

    def set_plot(self, ax, start_frame):
        signal = np.load(self._files[start_frame])
        for tx in range(self._num_tx):
            for rx in range(self._num_rx):
                ax[tx, rx].axis(ymin=-2, ymax=2)
                ax[tx, rx].axes.xaxis.set_visible(False)
                ax[tx, rx].axes.yaxis.set_visible(False)
                self._sig_plots[tx][rx], = ax[tx][rx].plot(signal[tx][rx], label='tx:{}, rx:{}'.format(tx + 1, rx + 1))
                ax[tx, rx].legend(loc='lower right', fontsize=5)

    def animate_plot(self, frame):
        signal = np.load(self._files[frame])
        for tx in range(self._num_tx):
            for rx in range(self._num_rx):
                self._sig_plots[tx][rx].set_ydata(signal[tx][rx])


class RadarImaging:
    def __init__(self, location, num_person, session_idx, type):
        self._tx_antenna_position = np.array(tx_antenna_position)
        self._rx_antenna_position = np.array(rx_antenna_position)
        self._num_tx_antenna = self._tx_antenna_position.shape[0]
        self._num_rx_antenna = self._rx_antenna_position.shape[0]
        self._sample_spacing = radar_sample_spacing  # ns
        self._start_sample_time = radar_start_sample_time  # ns
        self._num_sample = radar_num_sample
        self._sample_time = self._start_sample_time + np.arange(self._num_sample) * self._sample_spacing
        self._speed_of_light = 0.299792458  # speed of light in m/ns
        self._sample_time_delay = np.empty(0)
        self._sample_time_delay_invalid = np.empty(0)
        self._step_size = 0
        self._area_min_xyz = np.array(area_min_xyz)
        self._area_max_xyz = np.array(area_max_xyz)
        self._area_num_step = np.array(area_num_step)
        self._area_min_xyz[2] = imaging_height
        self._area_max_xyz[2] = imaging_height
        self._area_num_step[2] = 1
        self._set_area()
        self._location = location
        self._num_person = num_person
        self._session_idx = session_idx
        self._type = type
        file_dir = Path(__file__).parents[1].resolve() / 'data' / f'radar_{type}' / location / num_person / f'session_{session_idx}'
        self._files = list(file_dir.glob('*'))
        self._plot_image = 0

    def _set_area(self):
        x_space, x_step_size = np.linspace(self._area_min_xyz[0], self._area_max_xyz[0], self._area_num_step[0], retstep=True)
        y_space, y_step_size = np.linspace(self._area_min_xyz[1], self._area_max_xyz[1], self._area_num_step[1], retstep=True)
        z_space, z_step_size = np.linspace(self._area_min_xyz[2], self._area_max_xyz[2], self._area_num_step[2], retstep=True)
        self._step_size = np.array([x_step_size, y_step_size, z_step_size])
        x_grid, y_grid, z_grid = np.meshgrid(x_space, y_space, z_space, indexing='ij')
        area_grid = np.stack((x_grid, y_grid, z_grid), axis=3)
        tx_distance = np.linalg.norm(area_grid - self._tx_antenna_position[:, np.newaxis, np.newaxis, np.newaxis, :], axis=4)
        rx_distance = np.linalg.norm(area_grid - self._rx_antenna_position[:, np.newaxis, np.newaxis, np.newaxis, :], axis=4)
        distance = tx_distance[:, np.newaxis, ...] + rx_distance
        self._sample_time_delay = np.round(((distance / self._speed_of_light) - self._start_sample_time) /
                                           self._sample_spacing).astype(int)
        self._sample_time_delay_invalid = (self._sample_time_delay < 0) | (self._sample_time_delay >= self._num_sample)
        self._sample_time_delay[self._sample_time_delay_invalid] = 0

    def _get_radar_image(self, radar_signal):
        tx_antenna_index = np.arange(self._num_tx_antenna)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        rx_antenna_index = np.arange(self._num_rx_antenna)[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
        tx_rx_radar_image = radar_signal[tx_antenna_index, rx_antenna_index, self._sample_time_delay]
        tx_rx_radar_image[self._sample_time_delay_invalid] = 0
        radar_image = np.sum(tx_rx_radar_image, axis=(0, 1))
        radar_image = np.square(radar_image)
        radar_image = radar_image / np.amax(radar_image)
        return radar_image

    def set_plot(self, ax, start_frame):
        signal = np.load(self._files[start_frame])
        radar_image = self._get_radar_image(signal)
        extent = [self._area_min_xyz[0], self._area_max_xyz[0], self._area_min_xyz[1], self._area_max_xyz[1]]
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        self._plot_image = ax.imshow(np.transpose(radar_image[:, :, 0]), origin='lower', extent=extent)

    def animate_plot(self, frame):
        signal = np.load(self._files[frame])
        radar_image = self._get_radar_image(signal)
        self._plot_image.set_data(np.transpose(radar_image[:, :, 0]))


def visualize_dataset(location, num_person, session_idx, radar_type, skeleton_type, start_frame=0):
    radar_signal = RadarSignal(location, num_person, session_idx, radar_type)
    skeleton = Skeleton(location, num_person, session_idx, skeleton_type)
    imaging = RadarImaging(location, num_person, session_idx, radar_type)
    num_tx, num_rx, num_frames = radar_signal.get_info()

    fig = plt.figure(figsize=(15, 9))
    title = plt.suptitle(f'{0}/{num_frames}')
    gs = GridSpec(num_tx, num_rx+4, left=0.03, right=0.97, wspace=0.10)

    radar_ax = np.array(np.zeros((num_tx, num_rx)), dtype=object)
    for tx in range(num_tx):
        for rx in range(num_rx):
            radar_ax[tx, rx] = fig.add_subplot(gs[tx, rx])
    radar_signal.set_plot(radar_ax, start_frame)
    skeleton_ax = fig.add_subplot(gs[0:4, num_rx:num_rx+4], projection='3d')
    skeleton.set_plot(skeleton_ax)
    imaging_ax = fig.add_subplot(gs[4:8, num_rx:num_rx+4])
    imaging.set_plot(imaging_ax, start_frame)

    def animate(frame):
        frame = frame + start_frame
        title.set_text(f'{frame}/{num_frames-1}')
        radar_signal.animate_plot(frame)
        skeleton.animate_plot(frame)
        imaging.animate_plot(frame)
        return radar_signal, skeleton, imaging,

    anim = animation.FuncAnimation(fig, animate, frames=500, interval=200, blit=False, save_count=100)

    # anim.save('{}_{}_{}_{}.mp4'.format(location, num_person, session_idx, start_frame), writer='imagemagick')
    # print('done')

    plt.show()


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    visualize_dataset('A', 'two', 1, radar_type='declutter', skeleton_type='raw_revised', start_frame=1800)