# NTU RGB+D 60 - Full processing pipeline
# Usage:
#   python run_pipeline.py --zip_path /path/to/nturgbd_skeletons_s001_to_s017.zip --output_path /path/to/output
import os
import os.path as osp
import argparse
import zipfile
import tempfile
import shutil
import numpy as np
import pickle
import logging

# ── numpy compatibility ──────────────────────────────────────────────────────
try:
    np_int = np.int
except AttributeError:
    np_int = int   # numpy >= 1.24 removed np.int


# ── helpers shared across steps ──────────────────────────────────────────────

STAT_DIR = osp.join(osp.dirname(osp.abspath(__file__)), 'statistics')


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  get_raw_skes_data
# ─────────────────────────────────────────────────────────────────────────────

def _get_raw_bodies_data(zf, zip_prefix, ske_name, frames_drop_skes, frames_drop_logger):
    zip_entry = zip_prefix + ske_name + '.skeleton'
    print('Reading data from %s' % ske_name)
    with zf.open(zip_entry) as f:
        str_data = f.read().decode('utf-8').splitlines()

    num_frames = int(str_data[0].strip('\r\n'))
    frames_drop = []
    bodies_data = dict()
    valid_frames = -1
    current_line = 1

    for f in range(num_frames):
        num_bodies = int(str_data[current_line].strip('\r\n'))
        current_line += 1
        if num_bodies == 0:
            frames_drop.append(f)
            continue
        valid_frames += 1
        joints = np.zeros((num_bodies, 25, 3), dtype=np.float32)
        colors = np.zeros((num_bodies, 25, 2), dtype=np.float32)

        for b in range(num_bodies):
            bodyID = str_data[current_line].strip('\r\n').split()[0]
            current_line += 1
            num_joints = int(str_data[current_line].strip('\r\n'))
            current_line += 1
            for j in range(num_joints):
                temp_str = str_data[current_line].strip('\r\n').split()
                joints[b, j, :] = np.array(temp_str[:3], dtype=np.float32)
                colors[b, j, :] = np.array(temp_str[5:7], dtype=np.float32)
                current_line += 1

            if bodyID not in bodies_data:
                body_data = dict()
                body_data['joints'] = joints[b]
                body_data['colors'] = colors[b, np.newaxis]
                body_data['interval'] = [valid_frames]
            else:
                body_data = bodies_data[bodyID]
                body_data['joints'] = np.vstack((body_data['joints'], joints[b]))
                body_data['colors'] = np.vstack((body_data['colors'], colors[b, np.newaxis]))
                pre_frame_idx = body_data['interval'][-1]
                body_data['interval'].append(pre_frame_idx + 1)
            bodies_data[bodyID] = body_data

    num_frames_drop = len(frames_drop)
    assert num_frames_drop < num_frames, \
        'Error: All frames data (%d) of %s is missing or lost' % (num_frames, ske_name)
    if num_frames_drop > 0:
        frames_drop_skes[ske_name] = np.array(frames_drop, dtype=np_int)
        frames_drop_logger.info('{}: {} frames missed: {}\n'.format(
            ske_name, num_frames_drop, frames_drop))

    if len(bodies_data) > 1:
        for body_data in bodies_data.values():
            body_data['motion'] = np.sum(np.var(body_data['joints'], axis=0))

    return {'name': ske_name, 'data': bodies_data,
            'num_frames': num_frames - num_frames_drop}


def step1_get_raw_skes_data(zip_path, work_dir):
    """Read .skeleton files directly from zip → raw_data/raw_skes_data.pkl  (no extraction needed)"""
    raw_data_dir = osp.join(work_dir, 'raw_data')
    os.makedirs(raw_data_dir, exist_ok=True)

    skes_name_file = osp.join(STAT_DIR, 'skes_available_name.txt')
    save_data_pkl  = osp.join(raw_data_dir, 'raw_skes_data.pkl')
    frames_drop_pkl = osp.join(raw_data_dir, 'frames_drop_skes.pkl')

    frames_drop_logger = logging.getLogger('frames_drop')
    frames_drop_logger.setLevel(logging.INFO)
    if not frames_drop_logger.handlers:
        frames_drop_logger.addHandler(
            logging.FileHandler(osp.join(raw_data_dir, 'frames_drop.log')))
    frames_drop_skes = dict()

    skes_name = np.loadtxt(skes_name_file, dtype=str)
    num_files = skes_name.size
    print('\n[Step 1] Found %d available skeleton files.' % num_files)

    # Detect the prefix path inside the zip (e.g. "nturgb+d_skeletons/")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        all_entries = zf.namelist()
    ske_entries = [n for n in all_entries if n.endswith('.skeleton')]
    assert ske_entries, 'No .skeleton files found inside %s' % zip_path
    sample_entry = ske_entries[0]
    zip_prefix = sample_entry[: len(sample_entry) - len(osp.basename(sample_entry))]
    print('[Step 1] Zip prefix detected: "%s"' % zip_prefix)

    raw_skes_data = []
    frames_cnt = np.zeros(num_files, dtype=np_int)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for idx, ske_name in enumerate(skes_name):
            bodies_data = _get_raw_bodies_data(
                zf, zip_prefix, ske_name, frames_drop_skes, frames_drop_logger)
            raw_skes_data.append(bodies_data)
            frames_cnt[idx] = bodies_data['num_frames']
            if (idx + 1) % 1000 == 0:
                print('Processed: %.2f%% (%d / %d)' %
                      (100.0 * (idx + 1) / num_files, idx + 1, num_files))

    with open(save_data_pkl, 'wb') as fw:
        pickle.dump(raw_skes_data, fw, pickle.HIGHEST_PROTOCOL)
    np.savetxt(osp.join(raw_data_dir, 'frames_cnt.txt'), frames_cnt, fmt='%d')

    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)

    print('[Step 1] Saved raw bodies data → %s' % save_data_pkl)
    return save_data_pkl


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  get_raw_denoised_data
# ─────────────────────────────────────────────────────────────────────────────

_NOISE_LEN_THRES  = 11
_NOISE_SPR_THRES1 = 0.8
_NOISE_SPR_THRES2 = 0.69754
_NOISE_MOT_LO     = 0.089925
_NOISE_MOT_HI     = 2


def _make_logger(name, filepath, header=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.FileHandler(filepath))
    if header:
        logger.info(header)
    return logger


def _denoising_by_length(ske_name, bodies_data, noise_len_logger):
    new_bodies_data = bodies_data.copy()
    noise_info = ''
    for bodyID, body_data in new_bodies_data.items():
        length = len(body_data['interval'])
        if length <= _NOISE_LEN_THRES:
            noise_info += 'Filter out: %s, %d (length).\n' % (bodyID, length)
            noise_len_logger.info('{}\t{}\t{:.6f}\t{:^6d}'.format(
                ske_name, bodyID, body_data['motion'], length))
            del bodies_data[bodyID]
    if noise_info:
        noise_info += '\n'
    return bodies_data, noise_info


def _get_valid_frames_by_spread(points):
    num_frames = points.shape[0]
    valid_frames = []
    for i in range(num_frames):
        x = points[i, :, 0]
        y = points[i, :, 1]
        if (x.max() - x.min()) <= _NOISE_SPR_THRES1 * (y.max() - y.min()):
            valid_frames.append(i)
    return valid_frames


def _denoising_by_spread(ske_name, bodies_data, noise_spr_logger):
    noise_info = ''
    denoised_by_spr = False
    new_bodies_data = bodies_data.copy()
    for bodyID, body_data in new_bodies_data.items():
        if len(bodies_data) == 1:
            break
        valid_frames = _get_valid_frames_by_spread(
            body_data['joints'].reshape(-1, 25, 3))
        num_frames = len(body_data['interval'])
        num_noise = num_frames - len(valid_frames)
        if num_noise == 0:
            continue
        ratio = num_noise / float(num_frames)
        motion = body_data['motion']
        if ratio >= _NOISE_SPR_THRES2:
            del bodies_data[bodyID]
            denoised_by_spr = True
            noise_info += 'Filter out: %s (spread rate >= %.2f).\n' % (bodyID, _NOISE_SPR_THRES2)
            noise_spr_logger.info('%s\t%s\t%.6f\t%.6f' % (ske_name, bodyID, motion, ratio))
        else:
            joints = body_data['joints'].reshape(-1, 25, 3)[valid_frames]
            body_data['motion'] = min(motion, np.sum(np.var(joints.reshape(-1, 3), axis=0)))
            noise_info += '%s: motion %.6f -> %.6f\n' % (bodyID, motion, body_data['motion'])
    if noise_info:
        noise_info += '\n'
    return bodies_data, noise_info, denoised_by_spr


def _denoising_bodies_data(bodies_data, noise_len_logger, noise_spr_logger):
    ske_name   = bodies_data['name']
    bodies_data = bodies_data['data']

    bodies_data, noise_info_len = _denoising_by_length(ske_name, bodies_data, noise_len_logger)
    if len(bodies_data) == 1:
        return bodies_data.items(), noise_info_len

    bodies_data, noise_info_spr, _ = _denoising_by_spread(ske_name, bodies_data, noise_spr_logger)
    if len(bodies_data) == 1:
        return bodies_data.items(), noise_info_len + noise_info_spr

    bodies_motion = sorted(
        {bid: bd['motion'] for bid, bd in bodies_data.items()}.items(),
        key=lambda x: x[1], reverse=True)
    denoised = [(bid, bodies_data[bid]) for bid, _ in bodies_motion]
    return denoised, noise_info_len + noise_info_spr


def _get_bodies_info(bodies_data):
    info = '{:^17}\t{}\t{:^8}\n'.format('bodyID', 'Interval', 'Motion')
    for bodyID, body_data in bodies_data.items():
        start, end = body_data['interval'][0], body_data['interval'][-1]
        info += '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start, end]), body_data['motion'])
    return info + '\n'


def _get_one_actor_points(body_data, num_frames):
    joints = np.zeros((num_frames, 75), dtype=np.float32)
    colors = np.ones((num_frames, 1, 25, 2), dtype=np.float32) * np.nan
    start, end = body_data['interval'][0], body_data['interval'][-1]
    joints[start:end + 1] = body_data['joints'].reshape(-1, 75)
    colors[start:end + 1, 0] = body_data['colors']
    return joints, colors


def _remove_missing_frames(ske_name, joints, colors, missing_skes_logger,
                           missing_skes_logger1, missing_skes_logger2):
    num_frames = joints.shape[0]
    num_bodies = colors.shape[1]

    if num_bodies == 2:
        mi1 = np.where(joints[:, :75].sum(axis=1) == 0)[0]
        mi2 = np.where(joints[:, 75:].sum(axis=1) == 0)[0]
        cnt1, cnt2 = len(mi1), len(mi2)
        start = 1 if 0 in mi1 else 0
        end   = 1 if num_frames - 1 in mi1 else 0
        if max(cnt1, cnt2) > 0:
            if cnt1 > cnt2:
                missing_skes_logger1.info(
                    '{}\t{:^10d}\t{:^6d}\t{:^6d}\t{:^5d}\t{:^3d}'.format(
                        ske_name, num_frames, cnt1, cnt2, start, end))
            else:
                missing_skes_logger2.info(
                    '{}\t{:^10d}\t{:^6d}\t{:^6d}'.format(ske_name, num_frames, cnt1, cnt2))

    valid_indices   = np.where(joints.sum(axis=1) != 0)[0]
    missing_indices = np.where(joints.sum(axis=1) == 0)[0]
    if len(missing_indices) > 0:
        joints = joints[valid_indices]
        colors[missing_indices] = np.nan
        missing_skes_logger.info('{}\t{:^10d}\t{:^11d}'.format(
            ske_name, num_frames, len(missing_indices)))
    return joints, colors, len(missing_indices) > 0


def _get_two_actors_points(bodies_data, actors_info_dir,
                           noise_len_logger, noise_spr_logger,
                           fail_logger_1, fail_logger_2,
                           missing_skes_logger, missing_skes_logger1, missing_skes_logger2):
    ske_name   = bodies_data['name']
    label      = int(ske_name[-2:])
    num_frames = bodies_data['num_frames']
    bodies_info = _get_bodies_info(bodies_data['data'])

    bodies_data, noise_info = _denoising_bodies_data(
        bodies_data, noise_len_logger, noise_spr_logger)
    bodies_info += noise_info
    bodies_data = list(bodies_data)

    if len(bodies_data) == 1:
        if label >= 50:
            fail_logger_2.info(ske_name)
        bodyID, body_data = bodies_data[0]
        joints, colors = _get_one_actor_points(body_data, num_frames)
        bodies_info += 'Main actor: %s' % bodyID
    else:
        if label < 50:
            fail_logger_1.info(ske_name)
        joints = np.zeros((num_frames, 150), dtype=np.float32)
        colors = np.ones((num_frames, 2, 25, 2), dtype=np.float32) * np.nan

        bodyID, actor1 = bodies_data[0]
        start1, end1 = actor1['interval'][0], actor1['interval'][-1]
        joints[start1:end1 + 1, :75] = actor1['joints'].reshape(-1, 75)
        colors[start1:end1 + 1, 0]   = actor1['colors']
        actor1_info = ('{:^17}\t{}\t{:^8}\n'.format('Actor1', 'Interval', 'Motion') +
                       '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start1, end1]), actor1['motion']))
        del bodies_data[0]

        actor2_info = '{:^17}\t{}\t{:^8}\n'.format('Actor2', 'Interval', 'Motion')
        start2, end2 = 0, 0

        while bodies_data:
            bodyID, actor = bodies_data[0]
            start, end = actor['interval'][0], actor['interval'][-1]
            if min(end1, end) - max(start1, start) <= 0:
                joints[start:end + 1, :75] = actor['joints'].reshape(-1, 75)
                colors[start:end + 1, 0]   = actor['colors']
                actor1_info += '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start, end]), actor['motion'])
                start1, end1 = min(start, start1), max(end, end1)
            elif min(end2, end) - max(start2, start) <= 0:
                joints[start:end + 1, 75:] = actor['joints'].reshape(-1, 75)
                colors[start:end + 1, 1]   = actor['colors']
                actor2_info += '{}\t{:^8}\t{:f}\n'.format(bodyID, str([start, end]), actor['motion'])
                start2, end2 = min(start, start2), max(end, end2)
            del bodies_data[0]

        bodies_info += '\n' + actor1_info + '\n' + actor2_info

    with open(osp.join(actors_info_dir, ske_name + '.txt'), 'w') as fw:
        fw.write(bodies_info + '\n')

    return joints, colors


def step2_get_raw_denoised_data(work_dir):
    """raw_data/raw_skes_data.pkl → denoised_data/"""
    raw_data_file = osp.join(work_dir, 'raw_data', 'raw_skes_data.pkl')
    save_path     = osp.join(work_dir, 'denoised_data')
    os.makedirs(save_path, exist_ok=True)

    rgb_ske_path   = osp.join(save_path, 'rgb+ske');     os.makedirs(rgb_ske_path, exist_ok=True)
    actors_info_dir = osp.join(save_path, 'actors_info'); os.makedirs(actors_info_dir, exist_ok=True)

    noise_len_logger  = _make_logger('noise_length', osp.join(save_path, 'noise_length.log'),
        '{:^20}\t{:^17}\t{:^8}\t{}'.format('Skeleton', 'bodyID', 'Motion', 'Length'))
    noise_spr_logger  = _make_logger('noise_spread', osp.join(save_path, 'noise_spread.log'),
        '{:^20}\t{:^17}\t{:^8}\t{:^8}'.format('Skeleton', 'bodyID', 'Motion', 'Rate'))
    fail_logger_1     = _make_logger('noise_outliers_1', osp.join(save_path, 'denoised_failed_1.log'))
    fail_logger_2     = _make_logger('noise_outliers_2', osp.join(save_path, 'denoised_failed_2.log'))
    missing_skes_logger  = _make_logger('missing_frames', osp.join(save_path, 'missing_skes.log'),
        '{:^20}\t{}\t{}'.format('Skeleton', 'num_frames', 'num_missing'))
    missing_skes_logger1 = _make_logger('missing_frames_1', osp.join(save_path, 'missing_skes_1.log'),
        '{:^20}\t{}\t{}\t{}\t{}\t{}'.format('Skeleton','num_frames','Actor1','Actor2','Start','End'))
    missing_skes_logger2 = _make_logger('missing_frames_2', osp.join(save_path, 'missing_skes_2.log'),
        '{:^20}\t{}\t{}\t{}'.format('Skeleton', 'num_frames', 'Actor1', 'Actor2'))

    with open(raw_data_file, 'rb') as fr:
        raw_skes_data = pickle.load(fr)

    num_skes = len(raw_skes_data)
    print('\n[Step 2] Found %d available skeleton sequences.' % num_skes)

    raw_denoised_joints = []
    raw_denoised_colors = []
    frames_cnt = []
    missing_count = 0

    for idx, bodies_data in enumerate(raw_skes_data):
        ske_name   = bodies_data['name']
        num_bodies = len(bodies_data['data'])
        print('Processing %s' % ske_name)

        if num_bodies == 1:
            num_frames = bodies_data['num_frames']
            body_data  = list(bodies_data['data'].values())[0]
            joints, colors = _get_one_actor_points(body_data, num_frames)
        else:
            joints, colors = _get_two_actors_points(
                bodies_data, actors_info_dir,
                noise_len_logger, noise_spr_logger,
                fail_logger_1, fail_logger_2,
                missing_skes_logger, missing_skes_logger1, missing_skes_logger2)
            joints, colors, had_missing = _remove_missing_frames(
                ske_name, joints, colors,
                missing_skes_logger, missing_skes_logger1, missing_skes_logger2)
            if had_missing:
                missing_count += 1
            num_frames = joints.shape[0]

        raw_denoised_joints.append(joints)
        raw_denoised_colors.append(colors)
        frames_cnt.append(num_frames)

        if (idx + 1) % 1000 == 0:
            print('Processed: %.2f%% (%d / %d), Missing count: %d' %
                  (100.0 * (idx + 1) / num_skes, idx + 1, num_skes, missing_count))

    joints_pkl = osp.join(save_path, 'raw_denoised_joints.pkl')
    with open(joints_pkl, 'wb') as f:
        pickle.dump(raw_denoised_joints, f, pickle.HIGHEST_PROTOCOL)

    colors_pkl = osp.join(save_path, 'raw_denoised_colors.pkl')
    with open(colors_pkl, 'wb') as f:
        pickle.dump(raw_denoised_colors, f, pickle.HIGHEST_PROTOCOL)

    frames_cnt_arr = np.array(frames_cnt, dtype=np_int)
    np.savetxt(osp.join(save_path, 'frames_cnt.txt'), frames_cnt_arr, fmt='%d')

    print('[Step 2] Saved denoised joints → %s' % joints_pkl)
    print('[Step 2] Files with missing data: %d' % missing_count)
    return joints_pkl, save_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  seq_transformation
# ─────────────────────────────────────────────────────────────────────────────

def _remove_nan_frames(ske_name, ske_joints, nan_logger):
    num_frames = ske_joints.shape[0]
    valid_frames = []
    for f in range(num_frames):
        if not np.any(np.isnan(ske_joints[f])):
            valid_frames.append(f)
        else:
            nan_indices = np.where(np.isnan(ske_joints[f]))[0]
            nan_logger.info('{}\t{:^5}\t{}'.format(ske_name, f + 1, nan_indices))
    return ske_joints[valid_frames]


def _seq_translation(skes_joints):
    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 2:
            mf1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
            mf2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
            cnt1, cnt2 = len(mf1), len(mf2)

        i = 0
        while i < num_frames:
            if np.any(ske_joints[i, :75] != 0):
                break
            i += 1

        origin = np.copy(ske_joints[i, 3:6])
        for f in range(num_frames):
            if num_bodies == 1:
                ske_joints[f] -= np.tile(origin, 25)
            else:
                ske_joints[f] -= np.tile(origin, 50)

        if num_bodies == 2 and cnt1 > 0:
            ske_joints[mf1, :75] = np.zeros((cnt1, 75), dtype=np.float32)
        if num_bodies == 2 and cnt2 > 0:
            ske_joints[mf2, 75:] = np.zeros((cnt2, 75), dtype=np.float32)

        skes_joints[idx] = ske_joints
    return skes_joints


def _align_frames(skes_joints, frames_cnt):
    num_skes = len(skes_joints)
    max_num_frames = frames_cnt.max()
    aligned = np.zeros((num_skes, max_num_frames, 150), dtype=np.float32)
    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 1:
            aligned[idx, :num_frames] = np.hstack((ske_joints, np.zeros_like(ske_joints)))
        else:
            aligned[idx, :num_frames] = ske_joints
    return aligned


def _one_hot_vector(labels, num_classes=60):
    labels_vector = np.zeros((len(labels), num_classes))
    for idx, l in enumerate(labels):
        labels_vector[idx, l] = 1
    return labels_vector


def _get_indices(performer, camera, evaluation='CS'):
    test_indices  = np.empty(0)
    train_indices = np.empty(0)
    if evaluation == 'CS':
        train_ids = [1,  2,  4,  5,  8,  9,  13, 14, 15, 16,
                     17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        test_ids  = [3,  6,  7,  10, 11, 12, 20, 21, 22, 23,
                     24, 26, 29, 30, 32, 33, 36, 37, 39, 40]
        for idx in test_ids:
            test_indices = np.hstack((test_indices, np.where(performer == idx)[0])).astype(np_int)
        for idx in train_ids:
            train_indices = np.hstack((train_indices, np.where(performer == idx)[0])).astype(np_int)
    else:  # CV
        test_indices = np.hstack(
            (test_indices, np.where(camera == 1)[0])).astype(np_int)
        for train_id in [2, 3]:
            train_indices = np.hstack(
                (train_indices, np.where(camera == train_id)[0])).astype(np_int)
    return train_indices, test_indices


def step3_seq_transformation(work_dir, output_path):
    """denoised_data/ + statistics/ → NTU60_CS.npz, NTU60_CV.npz in output_path"""
    denoised_path      = osp.join(work_dir, 'denoised_data')
    raw_skes_joints_pkl = osp.join(denoised_path, 'raw_denoised_joints.pkl')
    frames_file        = osp.join(denoised_path, 'frames_cnt.txt')

    camera    = np.loadtxt(osp.join(STAT_DIR, 'camera.txt'),    dtype=np_int)
    performer = np.loadtxt(osp.join(STAT_DIR, 'performer.txt'), dtype=np_int)
    label     = np.loadtxt(osp.join(STAT_DIR, 'label.txt'),     dtype=np_int) - 1
    skes_name = np.loadtxt(osp.join(STAT_DIR, 'skes_available_name.txt'), dtype=np.bytes_)
    frames_cnt = np.loadtxt(frames_file, dtype=np_int)

    with open(raw_skes_joints_pkl, 'rb') as fr:
        skes_joints = pickle.load(fr)

    nan_logger = logging.getLogger('nan_skes')
    nan_logger.setLevel(logging.INFO)
    if not nan_logger.handlers:
        nan_logger.addHandler(logging.FileHandler(osp.join(work_dir, 'nan_frames.log')))
    nan_logger.info('{}\t{}\t{}'.format('Skeleton', 'Frame', 'Joints'))

    print('\n[Step 3] Applying sequence translation ...')
    skes_joints = _seq_translation(skes_joints)

    print('[Step 3] Aligning frames ...')
    skes_joints = _align_frames(skes_joints, frames_cnt)

    os.makedirs(output_path, exist_ok=True)

    for evaluation in ['CS', 'CV']:
        train_indices, test_indices = _get_indices(performer, camera, evaluation)
        train_x = skes_joints[train_indices]
        train_y = _one_hot_vector(label[train_indices])
        test_x  = skes_joints[test_indices]
        test_y  = _one_hot_vector(label[test_indices])

        save_name = osp.join(output_path, 'NTU60_%s.npz' % evaluation)
        np.savez(save_name, x_train=train_x, y_train=train_y,
                             x_test=test_x,  y_test=test_y)
        print('[Step 3] Saved → %s' % save_name)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='NTU RGB+D 60 full processing pipeline (single script)')
    parser.add_argument('--zip_path',    required=True,
                        help='Path to nturgbd_skeletons_s001_to_s017.zip')
    parser.add_argument('--output_path', required=True,
                        help='Directory where NTU60_CS.npz and NTU60_CV.npz will be saved')
    parser.add_argument('--work_dir', default=None,
                        help='Working directory for intermediate files '
                             '(default: <output_path>/work)')
    args = parser.parse_args()

    zip_path    = osp.abspath(args.zip_path)
    output_path = osp.abspath(args.output_path)
    work_dir    = osp.abspath(args.work_dir) if args.work_dir \
                  else osp.join(output_path, 'work')

    assert osp.isfile(zip_path), 'Zip file not found: %s' % zip_path
    os.makedirs(work_dir, exist_ok=True)

    # ── Run pipeline (reads .skeleton directly from zip, no extraction) ──────
    step1_get_raw_skes_data(zip_path, work_dir)
    step2_get_raw_denoised_data(work_dir)
    step3_seq_transformation(work_dir, output_path)

    print('\nDone! Output files are in: %s' % output_path)


if __name__ == '__main__':
    main()
