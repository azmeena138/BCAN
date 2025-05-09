# -*- coding:UTF-8 -*-
"""Training script for CLIP-Enhanced BCAN (SCAN)"""

import os
import time
import shutil
import torch
import torch.nn as nn
import numpy as np
import random
from torch.optim import Adam
from torch.nn.utils.clip_grad import clip_grad_norm_
import logging
import argparse
from data import get_loaders
from vocab import deserialize_vocab
from model import SCAN, ContrastiveLoss
from evaluation import AverageMeter, encode_data, LogCollector, i2t, t2i, shard_xattn

def setup_seed(seed):
    """Ensure reproducibility by setting random seeds."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def logging_func(log_file, message):
    """Log messages to a file."""
    with open(log_file, 'a') as f:
        f.write(message)
    f.close()

def main():
    setup_seed(1024)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/content/drive/My Drive/BCAN/data/f30k_precomp/',
                        help='Path to dataset')
    parser.add_argument('--data_name', default='f30k_precomp',
                        help='Dataset name (coco_precomp or f30k_precomp)')
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to vocabulary json files')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin')
    parser.add_argument('--grad_clip', default=2.0, type=float,
                        help='Gradient clipping threshold')
    parser.add_argument('--num_epochs', default=20, type=int,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size for training')
    parser.add_argument('--embed_size', default=512, type=int,
                        help='Embedding size (should match CLIP)')
    parser.add_argument('--learning_rate', default=2e-4, type=float,
                        help='Initial learning rate')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loader workers')
    parser.add_argument('--log_step', default=1, type=int,
                        help='Log frequency (number of batches)')
    parser.add_argument('--logger_name', default='./runs/logs',
                        help='Directory to save Tensorboard logs')
    parser.add_argument('--model_name', default='./runs/model',
                        help='Directory to save models')
    parser.add_argument('--resume', default='',
                        help='Path to latest checkpoint (optional)')

    opt = parser.parse_known_args()[0]

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info('Starting training...')

    # Load Vocabulary
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, f'{opt.data_name}_vocab.json'))
    opt.vocab_size = len(vocab)

    # Load Dataset
    train_loader, val_loader = get_loaders(opt.data_name, vocab, opt.batch_size, opt.workers, opt)

    # Initialize Model
    model = SCAN(opt)
    model = model.cuda() if torch.cuda.is_available() else model
    model = nn.DataParallel(model)

    # Define Loss and Optimizer
    criterion = ContrastiveLoss(margin=opt.margin)
    optimizer = Adam(model.parameters(), lr=opt.learning_rate)

    best_rsum = 0
    start_epoch = 0

    # Resume Training if Required
    if opt.resume and os.path.isfile(opt.resume):
        checkpoint = torch.load(opt.resume)
        start_epoch = checkpoint['epoch'] + 1
        best_rsum = checkpoint['best_rsum']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Resumed from checkpoint {opt.resume} (Epoch {start_epoch})")

    # Train Model
    for epoch in range(start_epoch, opt.num_epochs):
        print(f"Epoch {epoch}/{opt.num_epochs}")
        if not os.path.exists(opt.model_name):
            os.makedirs(opt.model_name)

        log_file = os.path.join(opt.logger_name, "performance.log")
        logging_func(log_file, f"Epoch {epoch} started.\n")

        adjust_learning_rate(opt, optimizer, epoch)

        # Train for One Epoch
        train(opt, train_loader, model, criterion, optimizer, epoch, val_loader)

        # Validate Model
        rsum = validate(opt, val_loader, model)

        # Save Best Model
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=f'checkpoint_{epoch}.pth.tar', prefix=opt.model_name + '/')

def train(opt, train_loader, model, criterion, optimizer, epoch, val_loader):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    model.train()
    start_time = time.time()

    for i, (images, captions) in enumerate(train_loader):
        images, captions = images.cuda(), captions.cuda()

        optimizer.zero_grad()
        scores = model(images, captions)
        loss = criterion(scores)
        loss.backward()

        if opt.grad_clip > 0:
            clip_grad_norm_(model.parameters(), opt.grad_clip)

        optimizer.step()

        if (i + 1) % opt.log_step == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoch [{epoch}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}, Time: {elapsed_time:.2f}s")
            start_time = time.time()

def validate(opt, val_loader, model):
    model.eval()
    img_embs, img_means, cap_embs, cap_means = encode_data(model, val_loader, opt.log_step, logging.info)

    start = time.time()
    sims = shard_xattn(model, img_embs, img_means, cap_embs, cap_means, opt, shard_size=128)
    end = time.time()
    print(f"Validation similarity calculation time: {end - start:.2f}s")

    r1, r5, r10, medr, meanr = i2t(img_embs, cap_embs, sims)
    print(f"Image to Text Retrieval: R@1: {r1:.1f}, R@5: {r5:.1f}, R@10: {r10:.1f}")

    r1i, r5i, r10i, medri, meanr = t2i(img_embs, cap_embs, sims)
    print(f"Text to Image Retrieval: R@1: {r1i:.1f}, R@5: {r5i:.1f}, R@10: {r10i:.1f}")

    return r1 + r5 + r10 + r1i + r5i + r10i

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        print(f"Best model saved at epoch {state['epoch']}")
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')

def adjust_learning_rate(opt, optimizer, epoch):
    """Decay learning rate every few epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // 15))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
