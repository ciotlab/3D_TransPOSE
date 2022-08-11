import json
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from datasets.radar import TargetProcessing

cnt = 12001
dots = np.load('D:/revised_test/motive/' + '%08d.npy' % cnt)

radar_signal = np.load('D:/revised_test/radar/' + '%08d.npy' % cnt)
with open('D:/revised_test/annotation_keypoint_test_scaled_two_people_B.json') as f:
    data = json.load(f)

a = TargetProcessing(radar_imaging_size=(3, 3, 2))
_, target = a(radar_signal=radar_signal, target=data)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_xlim3d([-1.5, 1.5])
ax.set_xlabel('X')
ax.set_ylim3d([0, 3])
ax.set_ylabel('Y')
ax.set_zlim3d([0, 2.5])
ax.set_zlabel('Z')

title = plt.title('%08d.npy' % cnt)

line1 = [ax.plot([dots[3][0], dots[5][0]], [dots[3][1], dots[5][1]], [dots[3][2], dots[5][2]], c='Red')[0],
         ax.plot([dots[3][0], dots[9][0]], [dots[3][1], dots[9][1]], [dots[3][2], dots[9][2]], c='Red')[0],
         ax.plot([dots[0][0], dots[13][0]], [dots[0][1], dots[13][1]], [dots[0][2], dots[13][2]], c='Red')[0],
         ax.plot([dots[0][0], dots[16][0]], [dots[0][1], dots[16][1]], [dots[0][2], dots[16][2]], c='Red')[0],
         ax.plot([dots[15][0], dots[19][0]], [dots[15][1], dots[19][1]], [dots[15][2], dots[19][2]], c='Red')[0],
         ax.plot([dots[18][0], dots[20][0]], [dots[18][1], dots[20][1]], [dots[18][2], dots[20][2]], c='Red')[0]]

line2 = [
    ax.plot([dots[21 + 3][0], dots[21 + 5][0]], [dots[21 + 3][1], dots[21 + 5][1]], [dots[21 + 3][2], dots[21 + 5][2]],
            c='Blue')[0],
    ax.plot([dots[21 + 3][0], dots[21 + 9][0]], [dots[21 + 3][1], dots[21 + 9][1]], [dots[21 + 3][2], dots[21 + 9][2]],
            c='Blue')[0],
    ax.plot([dots[21 + 0][0], dots[21 + 13][0]], [dots[21 + 0][1], dots[21 + 13][1]],
            [dots[21 + 0][2], dots[21 + 13][2]], c='Blue')[0],
    ax.plot([dots[21 + 0][0], dots[21 + 16][0]], [dots[21 + 0][1], dots[21 + 16][1]],
            [dots[21 + 0][2], dots[21 + 16][2]], c='Blue')[0],
    ax.plot([dots[21 + 15][0], dots[21 + 19][0]], [dots[21 + 15][1], dots[21 + 19][1]],
            [dots[21 + 15][2], dots[21 + 19][2]], c='Blue')[0],
    ax.plot([dots[21 + 18][0], dots[21 + 20][0]], [dots[21 + 18][1], dots[21 + 20][1]],
            [dots[21 + 18][2], dots[21 + 20][2]], c='Blue')[0]]

x = [0, 5, 9, 13, 16]
y = [4, 8, 12, 15, 18]
for n1, n2 in zip(x, y):
    for i in range(n1, n2):
        line1.append(
            ax.plot([dots[i][0], dots[i + 1][0]], [dots[i][1], dots[i + 1][1]], [dots[i][2], dots[i + 1][2]], c='Red')[
                0])
        line2.append(
            ax.plot([dots[21 + i][0], dots[21 + i + 1][0]], [dots[21 + i][1], dots[21 + i + 1][1]],
                    [dots[21 + i][2], dots[21 + i + 1][2]], c='Blue')[
                0])



# def animate(i):
#     animate_cnt = i + cnt
#     adots = np.load('D:/revised_test/motive/' + '%08d.npy' % animate_cnt)
#     title.set_text('%08d.npy' % animate_cnt)
#
#     line1[0].set_data_3d([adots[3][0], adots[5][0]], [adots[3][1], adots[5][1]], [adots[3][2], adots[5][2]])
#     line1[1].set_data_3d([adots[3][0], adots[9][0]], [adots[3][1], adots[9][1]], [adots[3][2], adots[9][2]])
#     line1[2].set_data_3d([adots[0][0], adots[13][0]], [adots[0][1], adots[13][1]], [adots[0][2], adots[13][2]])
#     line1[3].set_data_3d([adots[0][0], adots[16][0]], [adots[0][1], adots[16][1]], [adots[0][2], adots[16][2]])
#     line1[4].set_data_3d([adots[15][0], adots[19][0]], [adots[15][1], adots[19][1]], [adots[15][2], adots[19][2]])
#     line1[5].set_data_3d([adots[18][0], adots[20][0]], [adots[18][1], adots[20][1]], [adots[18][2], adots[20][2]])
#
#     line2[0].set_data_3d([adots[21+3][0], adots[21+5][0]], [adots[21+3][1], adots[21+5][1]], [adots[21+3][2], adots[21+5][2]])
#     line2[1].set_data_3d([adots[21+3][0], adots[21+9][0]], [adots[21+3][1], adots[21+9][1]], [adots[21+3][2], adots[21+9][2]])
#     line2[2].set_data_3d([adots[21+0][0], adots[21+13][0]], [adots[21+0][1], adots[21+13][1]], [adots[21+0][2], adots[21+13][2]])
#     line2[3].set_data_3d([adots[21+0][0], adots[21+16][0]], [adots[21+0][1], adots[21+16][1]], [adots[21+0][2], adots[21+16][2]])
#     line2[4].set_data_3d([adots[21+15][0], adots[21+19][0]], [adots[21+15][1], adots[21+19][1]], [adots[21+15][2], adots[21+19][2]])
#     line2[5].set_data_3d([adots[21+18][0], adots[21+20][0]], [adots[21+18][1], adots[21+20][1]], [adots[21+18][2], adots[21+20][2]])
#
#     x = [0, 5, 9, 13, 16]
#     y = [4, 8, 12, 15, 18]
#     l = 6
#     for n1, n2 in zip(x, y):
#         for k in range(n1, n2):
#             line1[l].set_data_3d([adots[k][0], adots[k + 1][0]], [adots[k][1], adots[k + 1][1]], [adots[k][2], adots[k + 1][2]])
#             line2[l].set_data_3d([adots[21+k][0], adots[21+k + 1][0]], [adots[21+k][1], adots[21+k + 1][1]],
#                                  [adots[21+k][2], adots[21+k + 1][2]])
#             l += 1
#
#     return title, line1, line2
#
#
# ani = animation.FuncAnimation(fig, animate, frames=100000, interval=300, blit=False)
plt.show()
