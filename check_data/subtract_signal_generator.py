import numpy as np
from slack_sdk.webhook import WebhookClient
from slack_sdk.errors import SlackApiError


def get_radar_subtracted_signal(index, radar_dir):
    radar_signal_before = np.load(radar_dir + '%08d.npy' % index)
    radar_buffer = np.zeros(radar_signal_before.shape)
    for k in range(index - 64, index):
        radar_buffer += np.load(radar_dir + '%08d.npy' % k)
    radar_buffer /= 64
    # radar_signal_after = np.load(radar_dir + '%08d.npy' % int(index + 1))
    radar_signal = np.subtract(radar_signal_before, radar_buffer)
    return radar_signal


radar_dir = 'D:/revised_train/radar/'
output_radar_dir = 'D:/revised_train/subtracted_radar_two_person/'
test_radar_dir = 'D:/revised_test/radar/'
test_output_radar_dir = 'D:/revised_test/subtracted_radar_two_person/'

train_start_index = [77000, 79000, 84000, 88000, 92000, 99000, 105000, 109000, 116000]
train_end_index = [79000, 84000, 88000, 92000, 99000, 105000, 109000, 116000, 119000]

for r1, r2 in zip(train_start_index, train_end_index):
    for index in range(r1 + 1 + 64, r2 + 1):
        print(index)
        subtracted_radar_signal = get_radar_subtracted_signal(index=index, radar_dir=radar_dir)
        np.save(output_radar_dir + '%08d.npy' % int(index), subtracted_radar_signal)

test_start_index = [10000, 12000, 15000, 17000]
test_end_index = [12000, 15000, 17000, 19400]

for r1, r2 in zip(test_start_index, test_end_index):
    for ind in range(r1 + 1 + 64, r2 + 1):
        print(ind)
        test_subtracted_radar_signal = get_radar_subtracted_signal(index=ind, radar_dir=test_radar_dir)
        np.save(test_output_radar_dir + '%08d.npy' % int(ind), test_subtracted_radar_signal)

url = "https://hooks.slack.com/services/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"
webhook = WebhookClient(url)
msg = 'subtract_radar_data_done'

try:
    response = webhook.send(text=msg)
except SlackApiError as e:
    print(e)
