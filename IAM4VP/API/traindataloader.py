import os
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import random_split, Dataset, DataLoader
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class MovingObjectDataSet(data.Dataset):
    def __init__(self, root, n_frames_input=11, n_frames_output=11):
        super(MovingObjectDataSet, self).__init__()

        root = os.path.join(root,'test_unlabeled')
        unlabeled = os.listdir(root)
        self.videos.extend([os.path.join(root, video) + '/' for video in unlabeled if not video.endswith('.DS_Store')])

        self.length = len(self.videos)      
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output

    def __getitem__(self, index):
        length = self.n_frames_input + self.n_frames_output
        video_folder = os.listdir(self.videos[index])
        
        imgs = []
        for image in video_folder:
            imgs.append(np.array(Image.open(self.videos[index] + '/' + image)))

        past_clips = imgs[0:self.n_frames_input] 
        future_clips = imgs[-self.n_frames_output:] 

        past_clips = [torch.from_numpy(clip) for clip in past_clips]
        future_clips = [torch.from_numpy(clip) for clip in future_clips]

        past_clips = torch.stack(past_clips).permute(0, 3, 1, 2)
        future_clips = torch.stack(future_clips).permute(0, 3, 1, 2)
        
        return (past_clips).contiguous().float(), (future_clips).contiguous().float()

    def __len__(self):
        return self.length


def load_data(batch_size, val_batch_size, data_root, num_workers):

    data = MovingObjectDataSet(root=data_root, is_train=True, n_frames_input=11, n_frames_output=11)

    # train_size = int(0.9 * len(data))
    # val_size = int(0.09 * len(data))
    # test_size = int(0.01 * len(data))
    train_size = int(0.9 * len(data))
    val_size = int(0.1 * len(data))
    print(len(data), train_size, val_size)

    train_data, val_data = random_split(data, [train_size, val_size], generator=torch.Generator().manual_seed(2021))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    mean, std = 0, 1
    return train_loader, val_loader, mean, std