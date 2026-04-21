import os
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import muscima
import muscima.io
from collections import defaultdict
from tqdm import tqdm

class_vocab = {
    '<UNK>': 0, '16th_flag': 1, '16th_rest': 2, '8th_flag': 3, '8th_rest': 4,
    'accent': 5, 'beam': 6, 'c-clef': 7, 'double_sharp': 8, 'duration-dot': 9,
    'dynamics_text': 10, 'f-clef': 11, 'flat': 12, 'g-clef': 13, 'grace_strikethrough': 14,
    'grace-notehead-full': 15, 'hairpin-cresc.': 16, 'hairpin-decr.': 17, 'half_rest': 18,
    'key_signature': 19, 'ledger_line': 20, 'letter_a': 21, 'letter_c': 22, 'letter_d': 23,
    'letter_e': 24, 'letter_f': 25, 'letter_i': 26, 'letter_l': 27, 'letter_M': 28,
    'letter_m': 29, 'letter_n': 30, 'letter_o': 31, 'letter_P': 32, 'letter_p': 33,
    'letter_r': 34, 'letter_s': 35, 'letter_t': 36, 'letter_u': 37, 'measure_separator': 38,
    'multi-staff_brace': 39, 'multi-staff_bracket': 40, 'multiple-note_tremolo': 41,
    'natural': 42, 'notehead-empty': 43, 'notehead-full': 44, 'numeral_2': 45,
    'numeral_3': 46, 'numeral_4': 47, 'numeral_5': 48, 'numeral_6': 49, 'numeral_7': 50,
    'numeral_8': 51, 'ornament(s)': 52, 'other_text': 53, 'other-dot': 54, 'quarter_rest': 55,
    'repeat': 56, 'repeat-dot': 57, 'sharp': 58, 'slur': 59, 'staccato-dot': 60,
    'staff_grouping': 61, 'stem': 62, 'tempo_text': 63, 'tenuto': 64, 'thin_barline': 65,
    'tie': 66, 'time_signature': 67, 'trill': 68, 'tuple': 69, 'tuple_bracket/line': 70,
    'whole_rest': 71
}

def get_deterministic_splits(root_dir, train_ratio=0.75, val_ratio=0.10):
    pieces = sorted([p for p in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, p))])
    rng = random.Random(42)
    rng.shuffle(pieces)
    n = len(pieces)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)
    return pieces[:train_end], pieces[train_end:val_end], pieces[val_end:]

class MSMDDataset(Dataset):
    def __init__(self, root_dir, split_pieces, class_vocab, mode='train', num_crops=2, min_frames=80, max_frames=160, val_frames=100, transform=None, img_save_dir='./local_data'):
        self.root_dir = root_dir
        self.mode = mode
        self.num_crops = num_crops if mode == 'train' else 1
        self.min_frames = min_frames 
        self.max_frames = max_frames 
        self.val_frames = val_frames 

        self.transform = transform or T.Compose([T.ToTensor()])
        self.class_vocab = class_vocab
        self.data_index = []

        os.makedirs(img_save_dir, exist_ok=True)
        print(f"Indexing {len(split_pieces)} pieces")

        for piece_name in tqdm(split_pieces, desc="Pre-building Graph Tensors"):
            piece_dir = os.path.join(root_dir, piece_name)
            if not os.path.isdir(piece_dir): continue

            perf_dir = os.path.join(piece_dir, 'performances')
            performances = [p for p in os.listdir(perf_dir) if os.path.isdir(os.path.join(perf_dir, p))]

            score_dir_base = os.path.join(piece_dir, 'scores')
            score_folders = os.listdir(score_dir_base)
            if not score_folders: continue
            score_path = os.path.join(score_dir_base, score_folders[0])

            xml_files = sorted(glob.glob(os.path.join(score_path, 'mung', '*.xml')))

            for xml_file in xml_files:
                page_id = os.path.basename(xml_file).replace('.xml', '')
                systems_npy_path = os.path.join(score_path, 'coords', f"systems_{page_id}.npy")
                if not os.path.exists(systems_npy_path): continue

                img_path = os.path.join(score_path, 'img', f"{page_id}.png")
                with Image.open(img_path) as img:
                    image_width = float(img.size[0])

                system_boxes = np.load(systems_npy_path)
                nodes = muscima.io.parse_cropobject_list(xml_file)

                for perf_name in performances:
                    spec_path = os.path.join(perf_dir, perf_name, 'features', f"{perf_name}.flac_spec.npy")
                    notes_path = os.path.join(perf_dir, perf_name, 'features', f"{perf_name}.flac_notes.npy")

                    if not os.path.exists(notes_path) or not os.path.exists(spec_path):
                        continue

                    notes_array = np.load(notes_path)
                    onset_key = f"{perf_name}_onset_frame"
                    event_idx_key = f"{perf_name}_note_event_idx"

                    for box in system_boxes:
                        if np.ptp(box[:, 0]) < np.ptp(box[:, 1]):
                            sys_top, sys_bottom = float(np.min(box[:, 0])), float(np.max(box[:, 0]))
                        else:
                            sys_top, sys_bottom = float(np.min(box[:, 1])), float(np.max(box[:, 1]))

                        crop_top = max(0, sys_top - 30)
                        crop_bottom = sys_bottom + 30
                        line_id = f"{piece_name}_{page_id}_{sys_top}"

                        valid_nodes = [
                            n for n in nodes
                            if crop_top <= n.top <= crop_bottom
                            and n.clsname not in ['system', 'staff', 'measure', 'staff_space']
                        ]
                        if not valid_nodes: continue

                        min_start_frame, max_end_frame = float('inf'), 0
                        played_notes = []

                        for n in valid_nodes:
                            if n.clsname == 'notehead-full' and onset_key in n.data:
                                played_notes.append(n)
                                onset_frame = int(n.data[onset_key])
                                min_start_frame = min(min_start_frame, onset_frame)

                                event_idx = n.data.get(event_idx_key)
                                if event_idx is not None and event_idx < len(notes_array):
                                    duration_frames = int(float(notes_array[event_idx, 2]) * 20.0)
                                    max_end_frame = max(max_end_frame, onset_frame + duration_frames)
                                else:
                                    max_end_frame = max(max_end_frame, onset_frame + 20)

                        if min_start_frame == float('inf'): continue
                        start_frame = int(min_start_frame)
                        end_frame = int(max_end_frame) + 5

                        record_info = {
                            'crop_top': crop_top, 'crop_bottom': crop_bottom,
                            'start_frame': start_frame, 'end_frame': end_frame, 'perf_name': perf_name
                        }

                        x_cont, x_class, x_pitch, edge_index = self._build_graph_from_nodes(
                            valid_nodes, played_notes, record_info, image_width, notes_array, onset_key, event_idx_key
                        )

                        self.data_index.append({
                            'spec_path': spec_path,
                            'image_path': os.path.join(img_save_dir, f"{line_id}.png"),
                            'start_frame': start_frame, 'end_frame': end_frame,
                            'graph_x_cont': x_cont, 'graph_x_class': x_class,
                            'graph_x_pitch': x_pitch, 'graph_edge_index': edge_index,
                            'line_id': line_id
                        })

        print(f"Successfully cached {len(self.data_index)} graph tensors in RAM.")

    def __len__(self):
        return len(self.data_index)

    def _build_graph_from_nodes(self, valid_nodes, played_notes, record, image_width, notes_array, onset_key, event_idx_key):
        id_to_idx = {node.objid: i for i, node in enumerate(valid_nodes)}
        crop_height = record['crop_bottom'] - record['crop_top']
        max_x = float(image_width)

        x_cont_list, x_class_list, x_pitch_list = [], [], []
        src_edges, dst_edges = [], []

        valid_onsets = [notes_array[n.data[event_idx_key], 0] for n in played_notes if event_idx_key in n.data]
        base_time_sec = min(valid_onsets) if valid_onsets else 0.0
        line_duration_sec = (record['end_frame'] - record['start_frame']) / 20.0

        for node in valid_nodes:
            top_in_crop = max(0, node.top - record['crop_top'])
            bottom_in_crop = min(crop_height, (node.top + node.height) - record['crop_top'])

            norm_top = top_in_crop / crop_height
            norm_height = max(0, bottom_in_crop - top_in_crop) / crop_height
            norm_left = node.left / max_x
            norm_width = node.width / max_x

            event_idx = node.data.get(event_idx_key)
            if event_idx is not None and event_idx < len(notes_array):
                onset_sec = float(notes_array[event_idx, 0])
                duration_sec = float(notes_array[event_idx, 2])
                norm_onset = max(0.0, min(1.0, (onset_sec - base_time_sec) / line_duration_sec))
                norm_duration = max(0.0, min(1.0, duration_sec / line_duration_sec))
            else:
                norm_duration, norm_onset = 0.0, 0.0

            x_cont_list.append([norm_top, norm_left, norm_width, norm_height, norm_duration, norm_onset])
            x_class_list.append(self.class_vocab.get(node.clsname, 0))
            x_pitch_list.append(int(node.data.get('midi_pitch_code', 0)))

            src_idx = id_to_idx[node.objid]
            for target_id in node.outlinks:
                if target_id in id_to_idx:
                    dst_idx = id_to_idx[target_id]
                    src_edges.extend([src_idx, dst_idx])
                    dst_edges.extend([dst_idx, src_idx])

        time_groups = defaultdict(list)
        for n in played_notes:
            time_groups[n.data[onset_key]].append(n)

        sorted_times = sorted(time_groups.keys())
        for i, current_time in enumerate(sorted_times):
            current_chord_notes = time_groups[current_time]
            if len(current_chord_notes) > 1:
                for idx_a in range(len(current_chord_notes)):
                    for idx_b in range(idx_a + 1, len(current_chord_notes)):
                        src_idx, dst_idx = id_to_idx[current_chord_notes[idx_a].objid], id_to_idx[current_chord_notes[idx_b].objid]
                        src_edges.extend([src_idx, dst_idx])
                        dst_edges.extend([dst_idx, src_idx])

            if i < len(sorted_times) - 1:
                next_time = sorted_times[i + 1]
                for curr_n in current_chord_notes:
                    for next_n in time_groups[next_time]:
                        src_edges.append(id_to_idx[curr_n.objid])
                        dst_edges.append(id_to_idx[next_n.objid])

        x_cont = torch.tensor(x_cont_list, dtype=torch.float32)
        x_class = torch.tensor(x_class_list, dtype=torch.long)
        x_pitch = torch.tensor(x_pitch_list, dtype=torch.long)
        edge_index = torch.tensor([src_edges, dst_edges], dtype=torch.long) if src_edges else torch.empty((2, 0), dtype=torch.long)

        return x_cont, x_class, x_pitch, edge_index

    def __getitem__(self, idx):
        record = self.data_index[idx]
        start_frame, end_frame = record['start_frame'], record['end_frame']
        total_frames = end_frame - start_frame

        full_spec = np.load(record['spec_path'], mmap_mode='r')
        spec_slice = full_spec[:, start_frame : end_frame]
        spec_slice = (spec_slice - np.mean(spec_slice)) / (np.std(spec_slice) + 1e-6)
        spec_tensor = torch.tensor(np.copy(spec_slice), dtype=torch.float32).unsqueeze(0)

        crops = []
        if self.mode == 'val':
            if total_frames <= self.val_frames:
                crops.append(spec_tensor)
            else:
                center_point = total_frames // 2
                start_idx = center_point - (self.val_frames // 2)
                crops.append(spec_tensor[:, :, start_idx : start_idx + self.val_frames])
        else:
            if total_frames <= self.min_frames:
                for _ in range(self.num_crops): crops.append(spec_tensor)
            else:
                for _ in range(self.num_crops):
                    current_crop_len = random.randint(self.min_frames, min(self.max_frames, total_frames))
                    random_start = random.randint(0, total_frames - current_crop_len)
                    crops.append(spec_tensor[:, :, random_start : random_start + current_crop_len])

        img = Image.open(record['image_path']).convert('RGB')
        
        return {
            'image': self.transform(img),
            'spectrogram_crops': crops,
            'graph_x_cont': record['graph_x_cont'],
            'graph_x_class': record['graph_x_class'],
            'graph_x_pitch': record['graph_x_pitch'],
            'graph_edge_index': record['graph_edge_index'],
            'line_id': record['line_id']
        }

def custom_collate_fn(batch):
    max_time_frames = max([crop.shape[2] for item in batch for crop in item['spectrogram_crops']])
    padded_spectrograms, images = [], []
    x_cont_list, x_class_list, x_pitch_list, edge_index_list, batch_index_list = [], [], [], [], []
    node_offset, batch_index_counter = 0, 0
    line_ids = []

    for item in batch:
        for spec_crop in item['spectrogram_crops']:
            images.append(item['image'])
            pad_amount = max_time_frames - spec_crop.shape[2]
            padded_spectrograms.append(F.pad(spec_crop, (0, pad_amount)))

            num_nodes = item['graph_x_cont'].shape[0]
            x_cont_list.append(item['graph_x_cont'])
            x_class_list.append(item['graph_x_class'])
            x_pitch_list.append(item['graph_x_pitch'])

            if item['graph_edge_index'].numel() > 0:
                edge_index_list.append(item['graph_edge_index'] + node_offset)

            batch_index_list.append(torch.full((num_nodes,), batch_index_counter, dtype=torch.long))

            node_offset += num_nodes
            batch_index_counter += 1
            line_ids.append(item['line_id'])

    return {
        'images': torch.stack(images),
        'spectrograms': torch.stack(padded_spectrograms), 
        'graph_x_cont': torch.cat(x_cont_list, dim=0),
        'graph_x_class': torch.cat(x_class_list, dim=0),
        'graph_x_pitch': torch.cat(x_pitch_list, dim=0),
        'graph_edge_index': torch.cat(edge_index_list, dim=1) if edge_index_list else torch.empty((2,0), dtype=torch.long),
        'graph_batch_index': torch.cat(batch_index_list, dim=0),
        'line_id': line_ids
    }