import json
import os
from pathlib import Path

import decord
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from kinetics import VideoClsDataset, VideoMAE
from masking_generator import TubeMaskingGenerator
from ssv2 import SSVideoClsDataset
from transforms import *


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False)
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 400
    
    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


# ======================================= CODE FOR EMODIVERSITY DATASETS ======================================= #


def get_n_evenly_spaced(arr, n):
    """
    get_n_evenly_spaced(lst=[0,1,2,3,4,5,6,7,8,9], n=3) -> [0,4,9]
    get_n_evenly_spaced(lst=[0,1,2], n=9) -> [0,0,0,1,1,1,2,2,2]
    """
    idx = np.round(np.linspace(0, len(arr) - 1, n)).astype(int)
    return list(np.array(arr)[idx])

def loadvideo_decord(fname, width, height, num_frames, frame_sample_rate=1):
    """
    Load video content using Decord
    Output: numpy array of shape (T H W C)
    """
    try:
        vr = decord.VideoReader(fname, width=width, height=height, num_threads=1, ctx=decord.cpu(0))
        # with open(fname, 'rb') as f:
        #     vr = decord.VideoReader(f, width=width, height=height, num_threads=1, ctx=decord.cpu(0))
    except FileNotFoundError:
        print("video cannot be loaded by decord: ", fname)
        return np.zeros((1, height, width, 3))

    all_idxs = [x for x in range(0, len(vr), frame_sample_rate)]
    chosen_idxs = get_n_evenly_spaced(all_idxs, num_frames)
    buffer = vr.get_batch(chosen_idxs).asnumpy()
    return buffer


class VCEDataset(Dataset):

    def __init__(self, dataset_path, split, video_width, video_height, num_frames):
        assert split == "train" or split == "test"
        self.video_width = video_width
        self.video_height = video_height
        self.num_frames =  num_frames
        self.data_path = Path(dataset_path)
        assert self.data_path.is_dir()

        with open(self.data_path / "metadata.json") as f:
            metadata_dict = json.load(f)
        
        self.video_paths = []
        self.labels = []
        self.metadata = []
        with open(self.data_path / f"{split}_labels.json") as f:
            labels_dict = json.load(f)
            for key, obj in labels_dict.items():
                path, label = self.get_path_and_label(obj)
                self.video_paths.append(path)
                self.labels.append(label)
                self.metadata.append(metadata_dict[key])

        assert len(self.video_paths) == len(self.metadata)
        assert len(self.video_paths) == len(self.labels)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video = loadvideo_decord(self.video_paths[idx], self.video_width, self.video_height, self.num_frames)
        label = self.labels[idx]
        md = self.metadata[idx]
        return video, label, md

    def get_path_and_label(self, label_obj):
        # Convert from 27-vector of emotion scores to a single classification label corresponding to the max-scoring emotion
        emotions_and_scores = sorted(list(label_obj["emotions"].items())) # Make sure they are sorted alphabetically
        scores = [score for emotion, score in emotions_and_scores]
        label = int(np.argmax(scores))
        path = str(self.data_path / label_obj['file'])
        return path, label

class V2VPairwiseDataset(Dataset):

    def __init__(self, dataset_path, split, video_width, video_height, num_frames):
        assert split == "train" or split == "test"
        self.video_width = video_width
        self.video_height = video_height
        self.num_frames =  num_frames
        self.data_path = Path(dataset_path)
        assert self.data_path.is_dir()

        with open(self.data_path / "metadata.json") as f:
            metadata_dict = json.load(f)
        
        self.video_paths_1 = []
        self.video_paths_2 = []
        self.metadata_1 = []
        self.metadata_2 = []
        with open(self.data_path / f"{split}_labels.json") as f:
            labels_dict = json.load(f)
            for comparison_list in labels_dict["comparisons"]:
                video_id_1, video_id_2 = comparison_list
                path_1 = str(self.data_path / f"videos/{video_id_1}.mp4")
                path_2 = str(self.data_path / f"videos/{video_id_2}.mp4")

                self.video_paths_1.append(path_1)
                self.video_paths_2.append(path_2)
                self.metadata_1.append(metadata_dict[video_id_1])
                self.metadata_2.append(metadata_dict[video_id_2])

        assert len(self.video_paths_1) == len(self.metadata_1)
        assert len(self.video_paths_2) == len(self.metadata_2)
        assert len(self.video_paths_1) == len(self.video_paths_2)

        # self.video_cache = {}

    def __len__(self):
        return len(self.video_paths_1)

    # def cached_loadvideo(self, fname, width, height, num_frames, frame_sample_rate=1):
    #     key = (fname, width, height, num_frames, frame_sample_rate)
    #     if key in self.video_cache:
    #         return self.video_cache[key]
    #     else:
    #         self.video_cache[key] = loadvideo_decord(fname, width, height, num_frames, frame_sample_rate)
    #         print("Added to cache. Cache size currently:", sys.getsizeof(self.video_cache))
    #         return self.video_cache[key]

    def __getitem__(self, idx):
        """
        returns (video_1, video_2, md_1, md_2)
        where 
        - video_{1,2} are numpy arrays of shape (num_frames, video_height, video_width, channels)
        - video_2 is preferred over video1
        - md_{1,2} are dictionaries containing metadata for video_{1,2} respectively
        """
        video_1 = loadvideo_decord(self.video_paths_1[idx], self.video_width, self.video_height, self.num_frames)
        video_2 = loadvideo_decord(self.video_paths_2[idx], self.video_width, self.video_height, self.num_frames)
        md_1 = self.metadata_1[idx]
        md_2 = self.metadata_2[idx]
        return video_1, video_2, md_1, md_2

class V2VListwiseDataset(Dataset):

    def __init__(self, dataset_path, video_width, video_height, num_frames):
        self.video_width = video_width
        self.video_height = video_height
        self.num_frames =  num_frames
        self.data_path = Path(dataset_path)
        assert self.data_path.is_dir()

        with open(self.data_path / "metadata.json") as f:
            metadata_dict = json.load(f)
        
        self.nested_video_paths = []
        self.nested_metadata = []
        with open(self.data_path / f"listwise_labels.json") as f:
            labels_dict = json.load(f)
            for comparison_list in labels_dict["comparisons"]:
                video_paths = []
                metadata = []
                for video_id in comparison_list:
                    path = str(self.data_path / f"videos/{video_id}.mp4")
                    video_paths.append(path)
                    metadata.append(metadata_dict[video_id])

                self.nested_video_paths.append(video_paths)
                self.nested_metadata.append(metadata)

        assert len(self.nested_video_paths) == len(self.nested_metadata)

    def __len__(self):
        return len(self.nested_video_paths)

    def __getitem__(self, idx):
        """
        returns (videos_list, metadata_list)
        - videos_list:      is a list of [videoA, videoB, …]
        - metadata_list:    is a list of [metadataA, metadataB, …]
        where the last video in the list is the most-preferred one.
        """
        videos_list = []
        metadata_list = []
        for video_path in self.nested_video_paths[idx]:
            video = loadvideo_decord(video_path, self.video_width, self.video_height, self.num_frames)
            videos_list.append(video)
        for metadata in self.nested_metadata[idx]:
            metadata_list.append(metadata)
        return videos_list, metadata_list
