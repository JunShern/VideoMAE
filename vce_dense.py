import csv
import datetime
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import torchvision
import tqdm
from einops.layers.torch import Rearrange, Reduce
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.video_utils import VideoClips

import video_transforms
from random_erasing import RandomErasing

cudnn.benchmark = True
torchvision.set_video_backend('video_reader')

class VCEDataset(Dataset):
    """
    =========================== OUR DOCUMENTATION ===========================
    This class takes a root folder containing all of the videos, e.g. /data/emotion_videos/all_videos.
    It also takes the CSV containing the 27 emotion labels for these videos. The CSV can contain entries
    for other videos in the dataset that are not present in the root folder, e.g. if we only want to
    train on half-minute videos with no audio.

    With these inputs, we break each video into clips (see torchvision documentation below). These clips
    are what we pass to the network during training, and we can index into the dataset of clips with
    self.__getitem__. With self.getvideo, we load all the clips corresponding to a certain video. This
    is what we use for evaluation.


    =========================== TORCHVISION DOCUMENTATION ===========================
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

    Internally, it uses a VideoClips object to handle clip creation.

    Args:
        root (string): root directory of the dataset
        label_csv_path: path to the label csv, e.g. results.csv
        frames_per_clip (int): number of frames in a clip
        step_between_clips (int): number of frames between each clip
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        label (int): class of the video clip
    """

    def __init__(self, dataset_path, split, frames_per_clip, step_between_clips=1,
                 frame_rate=None, extensions=('avi',), transform=None, _precomputed_metadata=None,
                 num_workers=1, _video_width=0, _video_height=0, _video_min_dimension=0,
                 _audio_samples=0, _audio_channels=0, cached_video_clips=None,
                 mode='train', crop_size=224, args=None):
        self.crop_size = crop_size
        self.aug = False
        self.rand_erase = False
        self.mode = mode
        self.args = args
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        
        # ================== GET PATHS AND LABELS ==================
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

        # ================== LOAD CLIPS ==================
        if cached_video_clips is not None:
            with open(cached_video_clips, 'rb') as f:
                self.video_clips = pickle.load(f)
            print('Loaded cached video_clips.')
        else:
            print('Loading video_clips..')
            self.video_clips = VideoClips(
                self.video_paths, # TODO: REMOVE
                frames_per_clip,
                step_between_clips,
                frame_rate,
                _precomputed_metadata,
                num_workers=num_workers,
                _video_width=_video_width,
                _video_height=_video_height,
                _video_min_dimension=_video_min_dimension,
                _audio_samples=_audio_samples,
                _audio_channels=_audio_channels,
            )
        print(len(self.video_clips.video_pts), len(self.video_clips.clips))
        #print(self.video_clips.video_pts)
        self.transform = transform

        # ================== GET CLIP INDICES FOR EACH VIDEO INDEX ==================
        num_videos = self.video_clips.num_videos()
        print("num_videos: " + str(num_videos))
        num_clips = self.video_clips.num_clips()
        print("num_clips: " + str(num_clips))

        video_to_clip_indices = [[] for i in range(num_videos)]
        for i in range(num_clips):
            video_idx, _ = self.video_clips.get_clip_location(i)
            video_to_clip_indices[video_idx].append(i)

        self.video_to_clip_indices = video_to_clip_indices

    # @property
    # def metadata(self):
    #     return self.video_clips.metadata

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        if video is None:
            return None, None
        label = torch.FloatTensor(self.labels[video_idx])

        if self.transform is not None:
            video = self.transform(video)
        
        # print("Before aug_frame: video.shape", video.shape)
        video = self._aug_frame(video.numpy(), self.args)
        # print("after aug_frame: video.shape", video.shape)
        return video, label, idx, {}

    def getvideo(self, idx, sample_every=1, desired_num_clips=-1):
        """
        Takes the video index and returns the batch of clips, the label, and the file path.
        """
        if idx >= self.video_clips.num_videos():
            assert False, 'index out of range'

        clip_indices = self.video_to_clip_indices[idx]

        if desired_num_clips == -1:
            # use sample_every argument to subsample
            clip_indices = clip_indices[::sample_every]
        else:
            # use desired_num_clips argument to try to get a fixed number of clips, evenly spaced
            clip_indices = clip_indices[::max(1, (len(clip_indices) // desired_num_clips))]

        clips = []
        for clip_idx in clip_indices:
            video, label = self.__getitem__(clip_idx)
            if video is not None:
                clips.append((video, label))

        videos = torch.cat([clips[i][0].unsqueeze(0) for i in range(len(clips))], axis=0)
        labels = clips[0][1]

        return videos, labels, idx, {}
    
    def get_path_and_label(self, label_obj):
        emotions_and_scores = sorted(list(label_obj["emotions"].items())) # Make sure they are sorted alphabetically
        scores = [score for emotion, score in emotions_and_scores]
        label = scores
        path = str(self.data_path / label_obj['file'])
        return path, label

    def _aug_frame(
        self,
        buffer,
        args,
    ):

        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C 
        
        # T H W C 
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True ,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

class TrainDataset(Dataset):
    """
    Takes a VCEDataset object and defines __getitem__ as randomly sampling a clip from within the video index.
    For example, self.__getitem__(50) will return a random clip from the 50th full video.
    This mirrors the training setup from the R(2+1)D paper with the only difference being that we define an epoch
    as once through all videos instead of once through 1 million videos.
    """
    def __init__(self, labeled_videos):
        self.labeled_videos = labeled_videos
        vc = labeled_videos.video_clips
        # get a list of indices for valid videos; if the clip size is larger than the video length,
        # VideoClips doesn't include that video, and the clip indices will be an empty list, which
        # this was added to handle
        valid_videos = [i for i in range(vc.num_videos()) if len(labeled_videos.video_to_clip_indices[i]) > 0]
        self.valid_videos = valid_videos
    
    def __len__(self):
        return len(self.valid_videos)
    
    def __getitem__(self, idx):
        idx = self.valid_videos[idx]
        clip_indices = self.labeled_videos.video_to_clip_indices[idx]
        clip_idx = clip_indices[np.random.randint(0, len(clip_indices))]
        
        return self.labeled_videos[clip_idx]
