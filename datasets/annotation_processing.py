import json
import numpy as np
from slack_sdk.webhook import WebhookClient
from slack_sdk.errors import SlackApiError

alpha = 0.3

def get_data_motive_3d_one(pos):
    motive_x = []
    motive_y = []
    motive_z = []

    for i in range(pos.shape[0]):
        motive_x.append(pos[i][0])
        motive_y.append(pos[i][1])
        motive_z.append(pos[i][2])

    max_x = np.max(motive_x) + alpha
    max_y = np.max(motive_y) + alpha
    max_z = np.max(motive_z)

    min_x = np.min(motive_x) - alpha
    min_y = np.min(motive_y) - alpha
    min_z = np.min(motive_z)

    return max_x, max_y, max_z, min_x, min_y, min_z


def get_data_motive_3d_two(pos):
    motive_x1 = []
    motive_y1 = []
    motive_z1 = []
    motive_x2 = []
    motive_y2 = []
    motive_z2 = []

    for i in range(21):
        motive_x1.append(pos[i][0])
        motive_y1.append(pos[i][1])
        motive_z1.append(pos[i][2])

    for i in range(21, 42):
        motive_x2.append(pos[i][0])
        motive_y2.append(pos[i][1])
        motive_z2.append(pos[i][2])

    max_x1 = np.max(motive_x1) + alpha
    max_y1 = np.max(motive_y1) + alpha
    max_z1 = np.max(motive_z1)
    min_x1 = np.min(motive_x1) - alpha
    min_y1 = np.min(motive_y1) - alpha
    min_z1 = np.min(motive_z1)

    max_x2 = np.max(motive_x2) + alpha
    max_y2 = np.max(motive_y2) + alpha
    max_z2 = np.max(motive_z2)
    min_x2 = np.min(motive_x2) - alpha
    min_y2 = np.min(motive_y2) - alpha
    min_z2 = np.min(motive_z2)

    return max_x1, max_y1, max_z1, min_x1, min_y1, min_z1, max_x2, max_y2, max_z2, min_x2, min_y2, min_z2


def xyxy_to_xywh(max_x, max_y, max_z, min_x, min_y, min_z):
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    center_z = (max_z + min_z) / 2
    width = max_x - min_x
    depth = max_y - min_y
    height = max_z - min_z

    return center_x, center_y, center_z, width, depth, height


def data_train_annotation(file_name):
    json_data = {'images': [], 'annotations': [], 'categories': []}
    anno_dir = 'D:/revised_train/'
    file_path = anno_dir + file_name

    start_index = [1, 150, 850, 1600, 6000, 9500, 13000, 18000, 21000, 26000, 31000, 35000, 37000, 43000, 47000, 55000, 59000, 64000, 72000, 77000, 79000, 84000, 88000, 92000, 99000, 105000, 109000, 116000]
    end_index = [150, 850, 1600, 6000, 9500, 13000, 18000, 21000, 26000, 31000, 35000, 37000, 43000, 47000, 55000, 59000, 64000, 72000, 77000, 79000, 84000, 88000, 92000, 99000, 105000, 109000, 116000, 119000]

    for r1, r2 in zip(start_index, end_index):
        for i in range(r1 + 1 + 64, r2 + 1):
            print(i)
            motive_dir = anno_dir + 'motive/'
            file_name = '%08d.npy' % i
            pos = np.load(motive_dir + '%08d.npy' % i)
            category_id = 0

            json_data['images'].append({
                "file_name": "{}".format(file_name),
                "id": i
            })

            if pos.shape[0] == 21:
                max_x, max_y, max_z, min_x, min_y, min_z = get_data_motive_3d_one(pos)
                center_x, center_y, center_z, width, depth, height = xyxy_to_xywh(max_x, max_y, max_z, min_x, min_y,
                                                                                  min_z)

                json_data['annotations'].append({
                    "is_crowd": 0,
                    "image_id": i,
                    "3d_bbox": [
                        center_x,
                        center_y,
                        center_z,
                        width,
                        depth,
                        height
                    ],
                    "keypoint": pos.tolist(),
                    "category_id": category_id,
                    "id": i
                })

            else:
                max_x1, max_y1, max_z1, min_x1, min_y1, min_z1, max_x2, max_y2, max_z2, min_x2, min_y2, min_z2 = get_data_motive_3d_two(pos)
                center_x1, center_y1, center_z1, width1, depth1, height1 = xyxy_to_xywh(max_x1, max_y1, max_z1, min_x1, min_y1, min_z1)
                center_x2, center_y2, center_z2, width2, depth2, height2 = xyxy_to_xywh(max_x2, max_y2, max_z2, min_x2, min_y2, min_z2)

                json_data['annotations'].append({
                    "is_crowd": 0,
                    "image_id": i,
                    "3d_bbox": [
                        center_x1,
                        center_y1,
                        center_z1,
                        width1,
                        depth1,
                        height1
                    ],
                    "keypoint": pos[:21].tolist(),
                    "category_id": category_id,
                    "id": 2 * i - 1
                })

                json_data['annotations'].append({
                    "is_crowd": 0,
                    "image_id": i,
                    "3d_bbox": [
                        center_x2,
                        center_y2,
                        center_z2,
                        width2,
                        depth2,
                        height2
                    ],
                    "keypoint": pos[21:42].tolist(),
                    "category_id": category_id,
                    "id": 2 * i
                })

        json_data['categories'].append({
            "id": 0,
            "name": "person"
        })

        with open(file_path, 'w') as outfile:
            json.dump(json_data, outfile)

        print("done")


def data_test_annotation(file_name):
    json_data = {'images': [], 'annotations': [], 'categories': []}
    anno_dir = 'D:/revised_test/'
    file_path = anno_dir + file_name

    start_index = [1, 2000, 5000, 7000, 10000, 12000, 15000, 17000]
    end_index = [2000, 5000, 7000, 10000, 12000, 15000, 17000, 19400]

    for r1, r2 in zip(start_index, end_index):
        for i in range(r1 + 1 + 64, r2 + 1):
            print(i)
            motive_dir = anno_dir + 'motive/'
            file_name = '%08d.npy' % i
            pos = np.load(motive_dir + '%08d.npy' % i)
            category_id = 0

            json_data['images'].append({
                "file_name": "{}".format(file_name),
                "id": i
            })

            if pos.shape[0] == 21:
                max_x, max_y, max_z, min_x, min_y, min_z = get_data_motive_3d_one(pos)
                center_x, center_y, center_z, width, depth, height = xyxy_to_xywh(max_x, max_y, max_z, min_x, min_y,
                                                                                  min_z)

                json_data['annotations'].append({
                    "is_crowd": 0,
                    "image_id": i,
                    "3d_bbox": [
                        center_x,
                        center_y,
                        center_z,
                        width,
                        depth,
                        height
                    ],
                    "keypoint": pos.tolist(),
                    "category_id": category_id,
                    "id": i
                })

            else:
                max_x1, max_y1, max_z1, min_x1, min_y1, min_z1, max_x2, max_y2, max_z2, min_x2, min_y2, min_z2 = get_data_motive_3d_two(pos)
                center_x1, center_y1, center_z1, width1, depth1, height1 = xyxy_to_xywh(max_x1, max_y1, max_z1, min_x1, min_y1, min_z1)
                center_x2, center_y2, center_z2, width2, depth2, height2 = xyxy_to_xywh(max_x2, max_y2, max_z2, min_x2, min_y2, min_z2)

                json_data['annotations'].append({
                    "is_crowd": 0,
                    "image_id": i,
                    "3d_bbox": [
                        center_x1,
                        center_y1,
                        center_z1,
                        width1,
                        depth1,
                        height1
                    ],
                    "keypoint": pos[:21].tolist(),
                    "category_id": category_id,
                    "id": 2 * i - 1
                })

                json_data['annotations'].append({
                    "is_crowd": 0,
                    "image_id": i,
                    "3d_bbox": [
                        center_x2,
                        center_y2,
                        center_z2,
                        width2,
                        depth2,
                        height2
                    ],
                    "keypoint": pos[21:42].tolist(),
                    "category_id": category_id,
                    "id": 2 * i
                })

        json_data['categories'].append({
            "id": 0,
            "name": "person"
        })

        with open(file_path, 'w') as outfile:
            json.dump(json_data, outfile)

        print("done")


train_file_name = 'annotation_keypoint_train_scaled_all.json'
test_file_name = 'annotation_keypoint_test_scaled_all.json'
data_train_annotation(file_name=train_file_name)
data_test_annotation(file_name=test_file_name)

url = "https://hooks.slack.com/services/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"
webhook = WebhookClient(url)
msg = 'annotation_done'

try:
    response = webhook.send(text=msg)
except SlackApiError as e:
    print(e)
