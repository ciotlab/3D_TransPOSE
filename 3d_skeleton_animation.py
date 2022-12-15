import os
import json
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


cnt = 18001
link = 'C:/Users/kl008/PycharmProjects/3D_TransPOSE/data/'
link1 = link + 'label_raw_revised/B/one/session_1/'
link2 = link + 'predicted_label_num_layers_6/B/one/session_1/'
dots1 = np.load(link1 + '%08d.npy' % cnt)
dots2 = np.load(link2 + '%08d.npy' % cnt)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_xlim3d([-1.5, 1.5])
ax.set_xlabel('X')
ax.set_ylim3d([0, 3])
ax.set_ylabel('Y')
ax.set_zlim3d([0, 2.5])
ax.set_zlabel('Z')

title = plt.title('%08d.npy' % cnt)

line1 = [ax.plot([dots1[3][0], dots1[5][0]], [dots1[3][1], dots1[5][1]], [dots1[3][2], dots1[5][2]], c='Red')[0],
         ax.plot([dots1[3][0], dots1[9][0]], [dots1[3][1], dots1[9][1]], [dots1[3][2], dots1[9][2]], c='Red')[0],
         ax.plot([dots1[0][0], dots1[13][0]], [dots1[0][1], dots1[13][1]], [dots1[0][2], dots1[13][2]], c='Red')[0],
         ax.plot([dots1[0][0], dots1[16][0]], [dots1[0][1], dots1[16][1]], [dots1[0][2], dots1[16][2]], c='Red')[0],
         ax.plot([dots1[15][0], dots1[19][0]], [dots1[15][1], dots1[19][1]], [dots1[15][2], dots1[19][2]], c='Red')[0],
         ax.plot([dots1[18][0], dots1[20][0]], [dots1[18][1], dots1[20][1]], [dots1[18][2], dots1[20][2]], c='Red')[0]]

line_dot2 = [ax.plot([dots2[3][0], dots2[5][0]], [dots2[3][1], dots2[5][1]], [dots2[3][2], dots2[5][2]], c='Green')[0],
             ax.plot([dots2[3][0], dots2[9][0]], [dots2[3][1], dots2[9][1]], [dots2[3][2], dots2[9][2]], c='Green')[0],
             ax.plot([dots2[0][0], dots2[13][0]], [dots2[0][1], dots2[13][1]], [dots2[0][2], dots2[13][2]], c='Green')[
                 0],
             ax.plot([dots2[0][0], dots2[16][0]], [dots2[0][1], dots2[16][1]], [dots2[0][2], dots2[16][2]], c='Green')[
                 0],
             ax.plot([dots2[15][0], dots2[19][0]], [dots2[15][1], dots2[19][1]], [dots2[15][2], dots2[19][2]],
                     c='Green')[0],
             ax.plot([dots2[18][0], dots2[20][0]], [dots2[18][1], dots2[20][1]], [dots2[18][2], dots2[20][2]],
                     c='Green')[0]]

if dots1.shape[0] > 21:
    line2 = [
        ax.plot([dots1[21 + 3][0], dots1[21 + 5][0]], [dots1[21 + 3][1], dots1[21 + 5][1]], [dots1[21 + 3][2], dots1[21 + 5][2]],
                c='Red')[0],
        ax.plot([dots1[21 + 3][0], dots1[21 + 9][0]], [dots1[21 + 3][1], dots1[21 + 9][1]], [dots1[21 + 3][2], dots1[21 + 9][2]],
                c='Red')[0],
        ax.plot([dots1[21 + 0][0], dots1[21 + 13][0]], [dots1[21 + 0][1], dots1[21 + 13][1]],
                [dots1[21 + 0][2], dots1[21 + 13][2]], c='Red')[0],
        ax.plot([dots1[21 + 0][0], dots1[21 + 16][0]], [dots1[21 + 0][1], dots1[21 + 16][1]],
                [dots1[21 + 0][2], dots1[21 + 16][2]], c='Red')[0],
        ax.plot([dots1[21 + 15][0], dots1[21 + 19][0]], [dots1[21 + 15][1], dots1[21 + 19][1]],
                [dots1[21 + 15][2], dots1[21 + 19][2]], c='Red')[0],
        ax.plot([dots1[21 + 18][0], dots1[21 + 20][0]], [dots1[21 + 18][1], dots1[21 + 20][1]],
                [dots1[21 + 18][2], dots1[21 + 20][2]], c='Red')[0]]

    line2_dot2 = [
        ax.plot([dots2[21 + 3][0], dots2[21 + 5][0]], [dots2[21 + 3][1], dots2[21 + 5][1]], [dots2[21 + 3][2], dots2[21 + 5][2]],
                c='Green')[0],
        ax.plot([dots2[21 + 3][0], dots2[21 + 9][0]], [dots2[21 + 3][1], dots2[21 + 9][1]], [dots2[21 + 3][2], dots2[21 + 9][2]],
                c='Green')[0],
        ax.plot([dots2[21 + 0][0], dots2[21 + 13][0]], [dots2[21 + 0][1], dots2[21 + 13][1]],
                [dots2[21 + 0][2], dots2[21 + 13][2]], c='Green')[0],
        ax.plot([dots2[21 + 0][0], dots2[21 + 16][0]], [dots2[21 + 0][1], dots2[21 + 16][1]],
                [dots2[21 + 0][2], dots2[21 + 16][2]], c='Green')[0],
        ax.plot([dots2[21 + 15][0], dots2[21 + 19][0]], [dots2[21 + 15][1], dots2[21 + 19][1]],
                [dots2[21 + 15][2], dots2[21 + 19][2]], c='Green')[0],
        ax.plot([dots2[21 + 18][0], dots2[21 + 20][0]], [dots2[21 + 18][1], dots2[21 + 20][1]],
                [dots2[21 + 18][2], dots2[21 + 20][2]], c='Green')[0]]


x = [0, 5, 9, 13, 16]
y = [4, 8, 12, 15, 18]
for n1, n2 in zip(x, y):
    for i in range(n1, n2):
        line1.append(
            ax.plot([dots1[i][0], dots1[i + 1][0]], [dots1[i][1], dots1[i + 1][1]], [dots1[i][2], dots1[i + 1][2]],
                    c='Red')[
                0])

        line_dot2.append(
            ax.plot([dots2[i][0], dots2[i + 1][0]], [dots2[i][1], dots2[i + 1][1]], [dots2[i][2], dots2[i + 1][2]],
                    c='Green')[
                0])
        if dots1.shape[0] > 21:
            line2.append(
                ax.plot([dots1[21 + i][0], dots1[21 + i + 1][0]], [dots1[21 + i][1], dots1[21 + i + 1][1]],
                        [dots1[21 + i][2], dots1[21 + i + 1][2]], c='Red')[0])

            line2_dot2.append(
                ax.plot([dots2[21 + i][0], dots2[21 + i + 1][0]], [dots2[21 + i][1], dots2[21 + i + 1][1]],
                        [dots2[21 + i][2], dots2[21 + i + 1][2]], c='Green')[0])


def animate(i):
    animate_cnt = 3 * i + cnt
    if os.path.isfile(link1 + '%08d.npy' % animate_cnt):
        adots1 = np.load(link1 + '%08d.npy' % animate_cnt)
        if adots1.shape[0] >= 21:
            adots2 = np.load(link2 + '%08d.npy' % animate_cnt)

            title.set_text('%08d.npy' % animate_cnt)

            line1[0].set_data_3d([adots1[3][0], adots1[5][0]], [adots1[3][1], adots1[5][1]], [adots1[3][2], adots1[5][2]])
            line1[1].set_data_3d([adots1[3][0], adots1[9][0]], [adots1[3][1], adots1[9][1]], [adots1[3][2], adots1[9][2]])
            line1[2].set_data_3d([adots1[0][0], adots1[13][0]], [adots1[0][1], adots1[13][1]], [adots1[0][2], adots1[13][2]])
            line1[3].set_data_3d([adots1[0][0], adots1[16][0]], [adots1[0][1], adots1[16][1]], [adots1[0][2], adots1[16][2]])
            line1[4].set_data_3d([adots1[15][0], adots1[19][0]], [adots1[15][1], adots1[19][1]], [adots1[15][2], adots1[19][2]])
            line1[5].set_data_3d([adots1[18][0], adots1[20][0]], [adots1[18][1], adots1[20][1]], [adots1[18][2], adots1[20][2]])

            line_dot2[0].set_data_3d([adots2[3][0], adots2[5][0]], [adots2[3][1], adots2[5][1]], [adots2[3][2], adots2[5][2]])
            line_dot2[1].set_data_3d([adots2[3][0], adots2[9][0]], [adots2[3][1], adots2[9][1]], [adots2[3][2], adots2[9][2]])
            line_dot2[2].set_data_3d([adots2[0][0], adots2[13][0]], [adots2[0][1], adots2[13][1]],
                                     [adots2[0][2], adots2[13][2]])
            line_dot2[3].set_data_3d([adots2[0][0], adots2[16][0]], [adots2[0][1], adots2[16][1]],
                                     [adots2[0][2], adots2[16][2]])
            line_dot2[4].set_data_3d([adots2[15][0], adots2[19][0]], [adots2[15][1], adots2[19][1]],
                                     [adots2[15][2], adots2[19][2]])
            line_dot2[5].set_data_3d([adots2[18][0], adots2[20][0]], [adots2[18][1], adots2[20][1]],
                                     [adots2[18][2], adots2[20][2]])

            if adots1.shape[0] > 21:
                line2[0].set_data_3d([adots1[21+3][0], adots1[21+5][0]], [adots1[21+3][1], adots1[21+5][1]], [adots1[21+3][2], adots1[21+5][2]])
                line2[1].set_data_3d([adots1[21+3][0], adots1[21+9][0]], [adots1[21+3][1], adots1[21+9][1]], [adots1[21+3][2], adots1[21+9][2]])
                line2[2].set_data_3d([adots1[21+0][0], adots1[21+13][0]], [adots1[21+0][1], adots1[21+13][1]], [adots1[21+0][2], adots1[21+13][2]])
                line2[3].set_data_3d([adots1[21+0][0], adots1[21+16][0]], [adots1[21+0][1], adots1[21+16][1]], [adots1[21+0][2], adots1[21+16][2]])
                line2[4].set_data_3d([adots1[21+15][0], adots1[21+19][0]], [adots1[21+15][1], adots1[21+19][1]], [adots1[21+15][2], adots1[21+19][2]])
                line2[5].set_data_3d([adots1[21+18][0], adots1[21+20][0]], [adots1[21+18][1], adots1[21+20][1]], [adots1[21+18][2], adots1[21+20][2]])

                line2_dot2[0].set_data_3d([adots2[21 + 3][0], adots2[21 + 5][0]], [adots2[21 + 3][1], adots2[21 + 5][1]],
                                     [adots2[21 + 3][2], adots2[21 + 5][2]])
                line2_dot2[1].set_data_3d([adots2[21 + 3][0], adots2[21 + 9][0]], [adots2[21 + 3][1], adots2[21 + 9][1]],
                                     [adots2[21 + 3][2], adots2[21 + 9][2]])
                line2_dot2[2].set_data_3d([adots2[21 + 0][0], adots2[21 + 13][0]], [adots2[21 + 0][1], adots2[21 + 13][1]],
                                     [adots2[21 + 0][2], adots2[21 + 13][2]])
                line2_dot2[3].set_data_3d([adots2[21 + 0][0], adots2[21 + 16][0]], [adots2[21 + 0][1], adots2[21 + 16][1]],
                                     [adots2[21 + 0][2], adots2[21 + 16][2]])
                line2_dot2[4].set_data_3d([adots2[21 + 15][0], adots2[21 + 19][0]], [adots2[21 + 15][1], adots2[21 + 19][1]],
                                     [adots2[21 + 15][2], adots2[21 + 19][2]])
                line2_dot2[5].set_data_3d([adots2[21 + 18][0], adots2[21 + 20][0]], [adots2[21 + 18][1], adots2[21 + 20][1]],
                                     [adots2[21 + 18][2], adots2[21 + 20][2]])

            x = [0, 5, 9, 13, 16]
            y = [4, 8, 12, 15, 18]
            l = 6
            for n1, n2 in zip(x, y):
                for k in range(n1, n2):
                    line1[l].set_data_3d([adots1[k][0], adots1[k + 1][0]], [adots1[k][1], adots1[k + 1][1]],
                                         [adots1[k][2], adots1[k + 1][2]])

                    line_dot2[l].set_data_3d([adots2[k][0], adots2[k + 1][0]], [adots2[k][1], adots2[k + 1][1]],
                                             [adots2[k][2], adots2[k + 1][2]])
                    if adots1.shape[0] > 21:
                        line2[l].set_data_3d([adots1[21 + k][0], adots1[21 + k + 1][0]],
                                             [adots1[21 + k][1], adots1[21 + k + 1][1]],
                                             [adots1[21 + k][2], adots1[21 + k + 1][2]])

                        line2_dot2[l].set_data_3d([adots2[21 + k][0], adots2[21 + k + 1][0]],
                                             [adots2[21 + k][1], adots2[21 + k + 1][1]],
                                             [adots2[21 + k][2], adots2[21 + k + 1][2]])

                    l += 1
            #
            if adots1.shape[0] <= 21:
                return title, line1, line_dot2
            else:
               return title, line1, line_dot2, line2, line2_dot2
        else:
            i += 1
    else:
        i += 1


#
ani = animation.FuncAnimation(fig, animate, frames=100000, interval=300, blit=False)
plt.show()
