from __future__ import division

import os


class Image(object):
    def __init__(self, path, time, label):
        self.path = os.path.realpath(path)
        self.time = time
        self.label = label

    def __repr__(self):
        return 'Image(path: ' + repr(self.path) + ', time: ' + repr(self.time) + ', label: ' + repr(self.label) + ')'


class Day(object):
    def __init__(self, date, images, user=None):
        self.date = date
        self.images = images
        self.user = user

    @property
    def num_images(self):
        return len(self.images)

    def __repr__(self):
        return 'Date(date: ' + repr(self.date) + ', Images: ' + repr(self.images) + ')'

    def __eq__(self, other):
        return self.date == other.date

    def __cmp__(self, other):
        return cmp(self.date, other.date)

    def __len__(self):
        return self.num_images

    def __add__(self, other):
        return self.num_images + other

    def __radd__(self, other):
        return other + self.num_images

class User(object):
    def __init__(self, id_, days):
        self.id_= id_
        self.days = days

        for day in days:
            day.user = self

    @property
    def num_images(self):
        return sum([d.num_images for d in self.days])

    def __repr__(self):
        return 'User(id: ' + repr(self.id_) + ', Days: ' + repr(self.days) + ')'

    def __eq__(self, other):
        return self.id_ == other.id_

import IO

filepaths = IO.Filepaths(ntcir_dir='datasets/ntcir')

# import collections
# import itertools
# import os
# import re
#
# import numpy as np
# import scipy.io as sio
# import skimage.io as skio

# import chiro.IO.labelMe as labelMe
# import chiro.IO.mcg as mcg
# import chiro.commons as commons
# import chiro.commons.utils as utils
# import chiro.commons.utils.files as ufiles
# from chiro.detection import BoundingBox
#
# Frame = collections.namedtuple('Frame', 'number boxes')
#
# GTEAVideo = collections.namedtuple('GTEAVideo', 'name num_frames frame_size annotated_frames')
#
# Sequence = collections.namedtuple('Sequence', 'start end')
#
# Action = collections.namedtuple('Action', 'name sequence')
#
# regex = re.compile(r"(><)|,")
#
#
# def parse_actions(labels_filepath):
#     actions = list()
#     with open(labels_filepath) as labels:
#         for line in labels:
#             action_object, image_indices = line.split()[0:2]
#             if '><' not in action_object:
#                 continue
#
#             action_object = regex.sub('_', action_object[1:-1])
#             image_indices = [int(ind) for ind in image_indices[1:-1].split('-')]
#
#             actions.append(Action(action_object, Sequence(*image_indices)))
#     return actions
#
#
# def load_actions(labels_dirpath):
#     actions = dict()
#
#     for filename in os.listdir(labels_dirpath):
#
#         filepath = os.path.join(labels_dirpath, filename)
#         if not os.path.isfile(filepath):
#             continue
#
#         if not filename.endswith('txt'):
#             continue
#
#         videoname = os.path.splitext(filename)[0]
#         actions[videoname] = parse_actions(filepath)
#     return actions
#
#
# def load_action_classes(labels_dirpath):
#     actions = set()
#     for file_ in os.listdir(labels_dirpath):
#         filepath = os.path.join(labels_dirpath, file_)
#         if not os.path.isfile(filepath):
#             continue
#         if not filepath.endswith('txt'):
#             continue
#         with open(filepath) as fp:
#             for line in fp:
#                 action_object = line.split()[0][1:-1]
#                 if '><' not in action_object:
#                     continue
#                 action_object = regex.sub('_', action_object)
#                 actions.add(action_object)
#     num_actions = len(actions)
#     return dict(zip(sorted(list(actions)), xrange(num_actions)))
#
#
# def split(gtea_filename):
#     basename = os.path.basename(gtea_filename)
#     filename = os.path.splitext(basename)[0]
#     videoname, frame_number = filename.rsplit('_', 1)
#     return filename, videoname, int(frame_number)
#
#
# def get_basename(gtea_video, frame_number):
#     num_padding_zeros = utils.num_digits(gtea_video.num_frames)
#     basename = gtea_video.name + '_' + str(frame_number).zfill(num_padding_zeros)
#     return basename
#
#
# def load_frames(images_dir, hands_annotations_dir):
#     gtea_videos = dict()
#     for img_dir in ufiles.dirs_from(images_dir):
#         if not img_dir.endswith('_C1'):
#             continue
#         name = os.path.split(img_dir)[-1]
#
#         img_files = ufiles.files_from(img_dir)
#         num_frames = len(img_files)
#         frame_size = utils.size(img_files[0])
#         gtea_videos[name] = GTEAVideo(name, num_frames, frame_size, dict())
#
#     annotated_frames = set()
#     for annotation_xml in ufiles.files_from(hands_annotations_dir, True):
#         if not annotation_xml.endswith('.xml'):
#             continue
#
#         filename, videoname, frame_number = split(annotation_xml)
#         annotated_frames.add(filename)
#
#         frame = Frame(int(frame_number), list())
#         annotation = labelMe.parse(annotation_xml)
#         for o in annotation.objects:
#             box = labelMe.Polygon.to_bounding_box(o.polygon)
#             frame.boxes.append(box)
#         gtea_videos[videoname].annotated_frames[frame.number] = frame
#
#     return gtea_videos, annotated_frames
#
#
# class Filepaths(object):
#     def __init__(self, gtea_dir):
#         self.images_dir = os.path.join(gtea_dir, 'images')
#         self.masks_dir_dir = os.path.join(gtea_dir, 'masks')
#         self.hands_annotations_dir = os.path.join(gtea_dir, 'hand_annotations')
#
#         self.mcg_dir = os.path.join(gtea_dir, 'mcg')
#         self.mcg_train = os.path.join(self.mcg_dir, 'mcg_train.mat')
#         self.mcg_test = os.path.join(self.mcg_dir, 'mcg_test.mat')
#
#         self.groundtruth_dir = os.path.join(gtea_dir, 'ground_truth')
#         self.groundtruth_train = os.path.join(self.groundtruth_dir, 'gt_train.mat')
#         self.groundtruth_test = os.path.join(self.groundtruth_dir, 'gt_test.mat')
#
#     def get_filepath(self, gtea_video, frame_number, filetype='frame', dirpath=None):
#         basename = get_basename(gtea_video, frame_number)
#         if filetype == 'mask':
#             if not dirpath:
#                 dirpath = self.masks_dir
#             extension = '.png'
#         elif filetype == 'frame':
#             if not dirpath:
#                 dirpath = self.images_dir
#             extension = '.jpg'
#         elif filetype == 'annotation':
#             if not dirpath:
#                 dirpath = self.hands_annotations_dir
#             extension = '.xml'
#         elif filetype == 'mcg':
#             if not dirpath:
#                 dirpath = self.mcg_dir
#             extension = '.csv'
#         return os.path.join(dirpath, gtea_video.name, basename + extension)
#
#
# def _count_action_hands_frames(gtea_video, actions):
#     num_action_hands_frames = 0
#     frames = gtea_video.annotated_frames
#     for action in actions:
#         sequence = action.sequence
#         for ind in xrange(sequence.start, sequence.end + 1):
#             if frames[ind].boxes:
#                 num_action_hands_frames += 1
#     return num_action_hands_frames
#
#
# def count_action_hands_frames(gtea_videos, actions, videonames):
#     num_action_hands_frames = 0
#     for video in videonames:
#         num_action_hands_frames += _count_action_hands_frames(gtea_videos[video], actions[video])
#     return num_action_hands_frames
#
#
# def count_action_frames(actions):
#     num_action_frames = 0
#     for action in actions:
#         sequence = action.sequence
#         num_action_frames += sequence.end - sequence.start + 1
#     return num_action_frames
#
#
# def create_groundtruth_file(gtea_videos, actions, videonames, gt_filepath):
#     num_frames = count_action_hands_frames(gtea_videos, actions, videonames)
#
#     images = np.zeros((1, num_frames), dtype=np.object)
#     action_labels = np.zeros((1, num_frames), dtype=np.object)
#     boxes = np.zeros((num_frames, 1), dtype=np.object)
#
#     counter = itertools.count()
#     for videoname in videonames:
#         gtea_video = gtea_videos[videoname]
#         frames = gtea_video.annotated_frames
#
#         for action in actions[videoname]:
#             sequence = action.sequence
#             for ind in xrange(sequence.start, sequence.end + 1):
#                 if not frames[ind].boxes:
#                     continue
#
#                 hand_boxes = frames[ind].boxes
#                 if len(hand_boxes) == 2:
#                     coords = np.asarray([box.coords for box in hand_boxes])
#                     upper_left = np.min(coords[:, :2], axis=0) + 1
#                     lower_right = np.max(coords[:, 2:], axis=0) + 1
#                     action_coords = np.concatenate((upper_left, lower_right))
#
#                 else:
#                     action_coords = hand_boxes[0].coords + 1
#
#                 i = next(counter)
#                 images[0, i] = get_basename(gtea_video, ind)
#                 action_labels[0, i] = action.name
#                 boxes[i, 0] = action_coords
#
#     sio.savemat(gt_filepath, {'images': images, 'actions': action_labels, 'boxes': boxes})
#
#
# def create_mcg_file(gtea_filepaths, gtea_videos, actions, videonames, mcg_mat_filepath, dtype=None):
#     if not dtype:
#         dtype = np.int16
#
#     num_frames = count_action_hands_frames(gtea_videos, actions, videonames)
#
#     images = np.zeros((1, num_frames), dtype=np.object)
#     action_labels = np.zeros((1, num_frames), dtype=np.object)
#     boxes = np.zeros((num_frames, 1), dtype=np.object)
#
#     counter = itertools.count()
#     for videoname in videonames:
#         gtea_video = gtea_videos[videoname]
#         frames = gtea_video.annotated_frames
#
#         for action in actions[videoname]:
#             sequence = action.sequence
#             for ind in xrange(sequence.start, sequence.end + 1):
#                 if not frames[ind].boxes:
#                     continue
#
#                 mcg_filepath = gtea_filepaths.get_filepath(gtea_video, ind, 'mcg')
#                 mcg_boxes = mcg.read(mcg_filepath)[0]
#
#                 mcg_coords = np.zeros((len(mcg_boxes), 4), dtype=dtype)
#                 for i, mcg_box in enumerate(mcg_boxes):
#                     mcg_coords[i, :] = mcg_box.coords + 1
#
#                 i = next(counter)
#                 images[0, i] = get_basename(gtea_video, ind)
#                 action_labels[0, i] = action.name
#                 boxes[i, 0] = mcg_coords
#     sio.savemat(mcg_mat_filepath, {'images': images, 'actions': action_labels, 'boxes': boxes})
#
#
# def draw_hands(gtea_filepaths, gtea_video, frame_number, alpha=0.3, sec_roi=None, sec_roi_ind=None, num_sec_rois=None,
#                output_img=None):
#     img_filepath = gtea_filepaths.get_filepath(gtea_video, frame_number)
#     hand_boxes = gtea_video.annotated_frames[frame_number].boxes
#
#     coords = np.asarray([box.coords for box in hand_boxes])
#     upper_left = np.min(coords[:, :2], axis=0)
#     lower_right = np.max(coords[:, 2:], axis=0)
#     action_box = BoundingBox(left_upper_corner=upper_left, right_lower_corner=lower_right)
#
#     img = commons.add_overlays(img_filepath, commons.Overlay(action_box, (255, 0, 0)), alpha)
#
#     if sec_roi:
#         sec_rois = [sec_roi]
#     elif sec_roi_ind or num_sec_rois:
#         mcg_filepath = gtea_filepaths.get_filepath(gtea_video, frame_number, 'mcg')
#         mcg_boxes = mcg.read(mcg_filepath)[0]
#
#         if sec_roi_ind:
#             sec_rois = [mcg_boxes[sec_roi_ind]]
#         else:
#             sec_rois = mcg_boxes[:num_sec_rois]
#     else:
#         sec_rois = None
#
#     img = commons.draw_detections(img, hand_boxes, sec_rois)
#
#     if output_img:
#         skio.imsave(output_img, img)
#
#     return img
#
#
# def plot_hands(gtea_filepaths, gtea_video, frame_number, alpha=0.3, sec_roi=None, sec_roi_ind=None, num_sec_rois=None,
#                output_img=None):
#     img = draw_hands(gtea_filepaths, gtea_video, frame_number, alpha, sec_roi, sec_roi_ind, num_sec_rois, output_img)
#     commons.plot_img(img)
#
#
# def to_list(gtea_videos, actions, action_indices, filepaths, videonames):
#     img_paths = list()
#     targets = list()
#
#     for videoname in sorted(videonames):
#         gtea_video = gtea_videos[videoname]
#
#         for action in actions[videoname]:
#             seq = action.sequence
#             action_ind = action_indices[action.name]
#             for ind in xrange(seq.start, seq.end + 1):
#                 img_path = filepaths.get_filepath(gtea_video, ind)
#                 img_paths.append(img_path)
#                 targets.append(action_ind)
#
#     targets = np.asarray(targets, np.int)
#     return img_paths, targets
#
#
