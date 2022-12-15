from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
import logging


def generate_index_dataframe():
    file_dir = Path(__file__).parents[1].resolve() / 'data'
    label_file_dir = file_dir / 'label_raw'
    radar_file_dir = file_dir / 'radar_declutter'
    df = pd.DataFrame()
    location_list = ['A', 'B', 'C', 'D']
    num_person_list = ['one', 'two']
    for n in product(location_list, num_person_list):
        label_dir = label_file_dir / n[0] / n[1]
        radar_dir = radar_file_dir / n[0] / n[1]
        label_sessions = list(label_dir.glob('*'))
        radar_sessions = list(radar_dir.glob('*'))
        for label_session_dir, radar_session_dir in zip(label_sessions, radar_sessions):
            label_files = list(label_session_dir.glob('*'))
            radar_files = list(radar_session_dir.glob('*'))
            location = n[0]
            num_person = 1 if n[1] == 'one' else 2
            session_idx = int(str(label_session_dir)[-1])
            if len(label_files) != len(radar_files):
                logging.info(f"Label and radar inconsistency found (location: {location}, num_person: {num_person}, session: {session_idx}")
            for seq_idx, (label_file, radar_file) in enumerate(zip(label_files, radar_files)):
                tmp_dict = {'location': [location], 'num_person': [num_person], 'session': [session_idx],
                            'sequence': [seq_idx], 'label': [label_file], 'radar': [radar_file]}
                if seq_idx % 1000 == 0:
                    logging.info(f"location: {location}, num_person: {num_person}, session: {session_idx}, "
                                 f"sequence: {seq_idx}")
                df = pd.concat((df, pd.DataFrame(tmp_dict)), axis=0, ignore_index=True)
    df.to_csv(file_dir / 'index_dataframe.csv')


def eliminate_radar_clutter(hist_len):
    load_file_dir = Path(__file__).parents[1].resolve() / 'data' / 'radar_raw'
    save_file_dir = Path(__file__).parents[1].resolve() / 'data' / 'radar_declutter'
    location_list = ['A', 'B', 'C', 'D']
    num_person_list = ['one', 'two']
    for n in product(location_list, num_person_list):
        tmp_dir = load_file_dir / n[0] / n[1]
        sessions = list(tmp_dir.glob('*'))
        for session_dir in sessions:
            radar_files = list(session_dir.glob('*'))
            sig_hist = np.empty(0)
            for seq_idx, radar_file in enumerate(radar_files):
                if seq_idx % 100 == 0:
                    logging.info(f"location: {n[0]}, num_person: {n[1]}, session: {int(str(session_dir)[-1])}, "
                                 f"sequence: {seq_idx}")
                signal = np.load(radar_file)
                if seq_idx == 0:
                    sig_hist = np.zeros((hist_len, *signal.shape))
                sig_hist[seq_idx % hist_len] = signal
                dec_signal = signal - np.mean(sig_hist, axis=0)
                tmp_dir = save_file_dir / n[0] / n[1] / session_dir.name
                Path(tmp_dir).mkdir(parents=True, exist_ok=True)
                np.save(tmp_dir / radar_file.name, dec_signal)


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    #eliminate_radar_clutter(64)
    generate_index_dataframe()



