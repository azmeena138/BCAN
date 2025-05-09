import torch
import torch.utils.data as data
import os
import numpy as np
import json
import clip
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    ""Prefetch Data for Faster Loading""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class CLIPDataset(data.Dataset):
    """
    Dataset class for CLIP-based BCAN model
    Supports precomputed image and text features.
    """
    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = os.path.join(data_path, data_split)

        # Load CLIP model for processing images and text
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)

        # Load Captions
        self.captions = []
        with open(loc + '_caps.txt', 'r', encoding="utf-8") as f:
            for line in f:
                self.captions.append(line.strip())

        # Load Images (Preprocessed as NumPy Arrays)
        self.images = np.load(loc + '_ims.npy')

        self.length = len(self.captions)
        self.im_div = 5 if self.images.shape[0] != self.length else 1

        if data_split == 'dev':
            self.length = 5000  # Reduce validation set size for efficiency

    def __getitem__(self, index):
        img_id = int(index / self.im_div)
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption to CLIP tokens
        tokens = clip.tokenize(caption).to(self.device)
        
        return image, tokens, index, img_id

    def __len__(self):
        return self.length

def collate_fn(data):
    """Prepare mini-batches from (image, caption) pairs"""
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    images = torch.stack(images, 0)
    captions = torch.cat(captions, 0)  # CLIP tokenized captions

    return images, captions, ids

def get_clip_loader(data_path, data_split, vocab, opt, batch_size=64, shuffle=True, num_workers=4):
    """Returns DataLoader for CLIP-based dataset"""
    dataset = CLIPDataset(data_path, data_split, vocab)
    loader = DataLoaderX(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    return loader

def get_loaders(data_name, vocab, batch_size, workers, opt):
    """Get DataLoaders for training and validation"""
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_clip_loader(dpath, 'train', vocab, opt, batch_size, True, workers)
    val_loader = get_clip_loader(dpath, 'dev', vocab, opt, batch_size, False, workers)
    return train_loader, val_loader

def get_test_loader(split_name, data_name, vocab, batch_size, workers, opt):
    """Get DataLoader for testing"""
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_clip_loader(dpath, split_name, vocab, opt, batch_size, False, workers)
    return test_loader
