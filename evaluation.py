# -*- coding: UTF-8 -*-
"""Evaluation script for CLIP-Enhanced BCAN (SCAN)"""

import os
import torch
import numpy as np
import time
from collections import OrderedDict
from torch.autograd import Variable
from data import get_test_loader
from model import SCAN
from vocab import deserialize_vocab

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)

class LogCollector(object):
    """A collection of logging objects that can switch from train to eval mode"""
    def __init__(self):
        self.meters = OrderedDict()

    def update(self, k, v, n=1):
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        return '  '.join([f"{k}: {v}" for k, v in self.meters.items()])

def encode_data(model, data_loader, log_step=10):
    """Encodes all images and captions from `data_loader`"""
    batch_time = AverageMeter()
    val_logger = LogCollector()

    model.eval()
    end = time.time()

    img_embs, cap_embs = [], []
    cap_lens = []

    with torch.no_grad():
        for i, (images, captions) in enumerate(data_loader):
            images, captions = images.cuda(), captions.cuda()

            img_emb, img_mean, cap_emb, cap_mean = model.forward_emb(images, captions)

            img_embs.append(img_emb.cpu().numpy())
            cap_embs.append(cap_emb.cpu().numpy())

            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_step == 0:
                print(f"Test [{i}/{len(data_loader)}]  Time: {batch_time.avg:.3f}")

    img_embs = np.vstack(img_embs)
    cap_embs = np.vstack(cap_embs)

    return img_embs, cap_embs

def evalrank(model_path, data_path=None, split='test'):
    """
    Evaluate a trained model on Flickr30K.
    """
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    if data_path:
        opt.data_path = data_path

    vocab = deserialize_vocab(os.path.join(opt.vocab_path, f"{opt.data_name}_vocab.json"))
    opt.vocab_size = len(vocab)

    model = SCAN(opt)
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    model.eval()

    print("Loading test dataset...")
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.batch_size, 0, opt)

    print("Encoding data...")
    img_embs, cap_embs = encode_data(model, data_loader)

    print(f"Images: {img_embs.shape[0]}, Captions: {cap_embs.shape[0]}")

    print("Computing similarity scores...")
    sims = compute_similarity(img_embs, cap_embs, opt)

    print("Evaluating retrieval performance...")
    r, _ = i2t(img_embs, cap_embs, sims)
    ri, _ = t2i(img_embs, cap_embs, sims)

    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]

    print(f"rsum: {rsum:.1f}")
    print(f"Average Image-to-Text Recall: {ar:.1f}")
    print(f"Image to Text: {r}")
    print(f"Average Text-to-Image Recall: {ari:.1f}")
    print(f"Text to Image: {ri}")

def compute_similarity(img_embs, cap_embs, opt, shard_size=500):
    """Computes similarity scores in batches to handle memory constraints"""
    n_img, n_cap = img_embs.shape[0], cap_embs.shape[0]
    d = np.zeros((n_img, n_cap))

    for i in range(0, n_img, shard_size):
        im_start, im_end = i, min(i + shard_size, n_img)
        for j in range(0, n_cap, shard_size):
            cap_start, cap_end = j, min(j + shard_size, n_cap)
            im = Variable(torch.from_numpy(img_embs[im_start:im_end])).float().cuda()
            cap = Variable(torch.from_numpy(cap_embs[cap_start:cap_end])).float().cuda()

            with torch.no_grad():
                sim = im.mm(cap.t())

            d[im_start:im_end, cap_start:cap_end] = sim.cpu().numpy()

    return d

def i2t(images, captions, sims):
    """Image-to-Text Retrieval Evaluation"""
    npts = images.shape[0]
    ranks = np.zeros(npts)
    for i in range(npts):
        inds = np.argsort(sims[i])[::-1]
        ranks[i] = np.where(inds == i * 5)[0][0]

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    return (r1, r5, r10), ranks

def t2i(images, captions, sims):
    """Text-to-Image Retrieval Evaluation"""
    npts = images.shape[0]
    ranks = np.zeros(npts * 5)
    sims = sims.T  # Transpose for text-to-image retrieval

    for i in range(npts * 5):
        inds = np.argsort(sims[i])[::-1]
        ranks[i] = np.where(inds == i // 5)[0][0]

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    return (r1, r5, r10), ranks

def shard_xattn(model, images, img_means, captions, cap_means, opt, shard_size=128):
    """
    Compute pairwise image-text similarity with memory-efficient sharding.
    This prevents memory overflow by processing data in smaller batches.
    """
    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions)))

    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))

            im = Variable(torch.from_numpy(images[im_start:im_end])).float().cuda()
            cap = Variable(torch.from_numpy(captions[cap_start:cap_end])).float().cuda()

            with torch.no_grad():
                sim = model.forward_sim(im, img_means[im_start:im_end], cap, cap_means[cap_start:cap_end])

            d[im_start:im_end, cap_start:cap_end] = sim.cpu().numpy()

    return d

if __name__ == '__main__':
    model_path = "./runs/model_best.pth.tar"
    evalrank(model_path)
