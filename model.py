import torch
import torch.nn as nn
import clip
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
 
def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True)
    norm = torch.sqrt(norm + eps) + eps
    X = torch.div(X, norm)
    return X

class EncoderImageCLIP(nn.Module):
    def __init__(self, embed_size):
        super(EncoderImageCLIP, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model, _ = clip.load("ViT-B/16", device=self.device)
        self.fc = nn.Linear(512, embed_size)  # CLIP ViT-B/32 has a 512-dim output

    def forward(self, images):
        images = images.to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_image(images)  # Extract image features
        features = self.fc(features)  # Project to BCAN embedding space
        return l2norm(features, dim=-1), torch.mean(features, dim=1)

class EncoderTextCLIP(nn.Module):
    def __init__(self, embed_size):
        super(EncoderTextCLIP, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model, _ = clip.load("ViT-B/16", device=self.device)
        self.clip_model = self.clip_model.half()
        self.fc = nn.Linear(512, embed_size)  # CLIP ViT-B/32 has a 512-dim output

    def forward(self, captions):
        tokens = clip.tokenize(captions).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_text(tokens)  # Extract text features
        features = self.fc(features)  # Project to BCAN embedding space
        return l2norm(features, dim=-1), torch.mean(features, dim=1)

def func_attention(query, context, g_sim, opt, eps=1e-8):
    """ Attention mechanism for LCU and GCU """
    batch_size, queryL, sourceL = context.size(0), query.size(1), context.size(1)

    queryT = torch.transpose(query, 1, 2)
    attn = torch.bmm(context, queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn * opt.lambda_softmax)
    attn = attn.view(batch_size, queryL, sourceL)

    # LCU and GCU processing
    re_attn = g_sim.unsqueeze(1).unsqueeze(2) * attn
    attn_sum = torch.sum(re_attn, dim=-1, keepdim=True)
    re_attn = re_attn / attn_sum

    contextT = torch.transpose(context, 1, 2)
    re_attnT = torch.transpose(re_attn, 1, 2).contiguous()
    weightedContext = torch.bmm(contextT, re_attnT)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, re_attn

class SCAN(nn.Module):
    def __init__(self, opt):
        super(SCAN, self).__init__()
        self.img_enc = EncoderImageCLIP(opt.embed_size)
        self.txt_enc = EncoderTextCLIP(opt.embed_size)
        self.opt = opt

    def forward_emb(self, images, captions):
        img_emb, img_mean = self.img_enc(images)
        cap_emb, cap_mean = self.txt_enc(captions)
        return img_emb, img_mean, cap_emb, cap_mean

    def forward_sim(self, img_emb, img_mean, cap_emb, cap_mean):
        g_sims = cap_mean.mm(img_mean.t())  # Global similarity
        similarities = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)

        for i in range(n_caption):
            g_sim = g_sims[i]
            cap_i = cap_emb[i].unsqueeze(0).repeat(n_image, 1, 1)

            weiContext, _ = func_attention(cap_i, img_emb, g_sim, self.opt)
            t2i_sim = (cap_i * weiContext).sum(dim=2).mean(dim=1, keepdim=True)

            weiContext, _ = func_attention(img_emb, cap_i, g_sim, self.opt)
            i2t_sim = (img_emb * weiContext).sum(dim=2).mean(dim=1, keepdim=True)

            sim = t2i_sim + i2t_sim
            similarities.append(sim)

        similarities = torch.cat(similarities, 1)
        return similarities

    def forward(self, images, captions):
        img_emb, img_mean, cap_emb, cap_mean = self.forward_emb(images, captions)
        scores = self.forward_sim(img_emb, img_mean, cap_emb, cap_mean)
        return scores

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss for image-text similarity.
    """
    def __init__(self, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, scores):
        # Get the diagonal (correct pairs)
        diagonal = scores.diag().view(-1, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # Caption retrieval loss
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # Image retrieval loss
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # Remove diagonal (self-comparisons)
        mask = torch.eye(scores.size(0)) > .5
        I = mask.to(scores.device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # Max violation selection
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()
