import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - distances.sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


class OnlineTripletLossV2(nn.Module):
    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLossV2, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1).sqrt()  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1).sqrt()  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin).pow(2)

        return losses.mean(), len(triplets)


class CoupledClustersLoss(nn.Module):
    def __init__(self, margin, n_classes, n_samples):
        super(CoupledClustersLoss, self).__init__()
        self.margin = margin
        self.n_classes = n_classes
        self.n_samples = n_samples

    def forward(self, embeddings, target):
        # embeddings_len = embeddings.norm(dim=1, keepdim=True)
        # embeddings = torch.div(embeddings, embeddings_len)
        losses = torch.zeros(self.n_classes, dtype=torch.float32)
        for i in range(self.n_classes):
            positives = embeddings[2*i*self.n_samples:(2*i+1)*self.n_samples, :]
            negatives = embeddings[(2*i+1)*self.n_samples:(2*i+2)*self.n_samples, :]
            anchor = positives.mean(dim=0, keepdim=True)
            negative_anchor_distance = (negatives - anchor).pow(2).sum(1)
            ap_distances = (anchor - positives).pow(2).sum(1)
            an_distances = (anchor[0] - negatives[negative_anchor_distance.argmin(dim=0)]
                           ).pow(2).sum(dim=0, keepdim=True)
            losses[i] = F.relu(ap_distances - an_distances + self.margin).sum(0)

        return losses.mean()


class CoupledClustersLossV2(nn.Module):
    def __init__(self, margin, n_classes, n_samples):
        super(CoupledClustersLossV2, self).__init__()
        self.margin = margin
        self.n_classes = n_classes
        self.n_samples = n_samples

    def forward(self, embeddings, target):
        losses = torch.zeros(self.n_classes, dtype=torch.float32)
        for i in range(self.n_classes):
            positives = embeddings[2*i*self.n_samples:(2*i+1)*self.n_samples, :]
            negatives = embeddings[(2*i+1)*self.n_samples:(2*i+2)*self.n_samples, :]
            anchor = positives.mean(dim=0, keepdim=True)
            negative_anchor_distance = (negatives - anchor).pow(2).sum(1)
            ap_distances = (anchor - positives).pow(2).sum(1).sqrt()
            an_distances = (anchor[0] - negatives[negative_anchor_distance.argmin(dim=0)]
                            ).pow(2).sum(dim=0, keepdim=True).sqrt()
            losses[i] = F.relu(ap_distances - an_distances + self.margin).pow(2).sum(0)

        return losses.mean()


# class XentTripletLoss(nn.Module):
#     def __init__(self, margin, triplet_selector):
#         super(XentTripletLoss, self).__init__()
#         self.margin = margin
#         self.triplet_selector = triplet_selector
#
#     def forward(self, embeddings_classify, embeddings, target):
#         triplets = self.triplet_selector.get_triplets(embeddings, target)
#         if embeddings.is_cuda:
#             triplets = triplets.cuda()
#
#         ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
#         an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
#         losses = F.relu(ap_distances - an_distances + self.margin)
#
#         # loss_triplet = 1e-1 * loss.mean()
#         loss_triplet = losses.sum() / len(triplets)
#         loss_xent = nn.CrossEntropyLoss()(embeddings_classify, target)
#         total_loss = 0.75 * loss_xent + 0.25 * loss_triplet
#
#         return total_loss


class XentTripletLoss(nn.Module):
    def __init__(self, margin, n_class, each_cls_cnt, triplet_selector):
        super(XentTripletLoss, self).__init__()
        self.margin = margin
        self.n_class = n_class
        self.each_cls_cnt = each_cls_cnt
        self.triplet_selector = triplet_selector
        # self.xent = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]
        self.xent = nn.CrossEntropyLoss()

    def forward(self, embedding_classify, embedding, target, group_label):
        embedding_id = embedding
        triplets = self.triplet_selector.get_triplets(embedding_id, target)
        if len(triplets) == 0:
            inter_loss = torch.tensor(0.0, dtype=torch.float32).cuda()
            inter_loss.requires_grad = True
            print('There are no triplet in this batch.')
        else:
            if embedding_id.is_cuda:
                triplets = triplets.cuda()
            ap_distances = (embedding_id[triplets[:, 0]] - embedding_id[triplets[:, 1]]).pow(2).sum(1)
            an_distances = (embedding_id[triplets[:, 0]] - embedding_id[triplets[:, 2]]).pow(2).sum(1)
            inter_loss = F.relu(ap_distances - an_distances + self.margin)
            inter_loss = inter_loss.sum() / len(triplets)

        loss_xent = [self.xent(embed_clfy, target) for embed_clfy in embedding_classify]
        loss_xent = sum(loss_xent) / len(loss_xent)

        # total_loss = 0.375 * (loss_xent_p1 + loss_xent_p2) + 0.25 * inter_loss
        # total_loss = 0.5 * (loss_xent_p1 + loss_xent_p2) + inter_loss
        # total_loss = 0.5 * loss_xent + 0.5 * inter_loss
        total_loss = loss_xent + inter_loss
        # total_loss = inter_loss

        return total_loss


class XentTripletLossV2(nn.Module):
    def __init__(self, margin, n_class, each_cls_cnt, triplet_selector):
        super(XentTripletLossV2, self).__init__()
        self.margin = margin
        self.n_class = n_class
        self.each_cls_cnt = each_cls_cnt
        self.triplet_selector = triplet_selector
        # self.xent = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]
        self.xent = nn.CrossEntropyLoss()

    def forward(self, embedding_classify, embedding, target, group_label):
        # embedding_id = embedding[0]
        # triplets = self.triplet_selector.get_triplets(embedding_id, target)
        # if embedding_id.is_cuda:
        #     triplets = triplets.cuda()
        # ap_distances = (embedding_id[triplets[:, 0]] - embedding_id[triplets[:, 1]]).pow(2).sum(1).sqrt()
        # an_distances = (embedding_id[triplets[:, 0]] - embedding_id[triplets[:, 2]]).pow(2).sum(1).sqrt()
        # inter_loss1 = F.relu(ap_distances - an_distances + self.margin).pow(2)
        # inter_loss1 = inter_loss1.sum() * self.each_cls_cnt / (self.n_class * len(triplets))
        #
        # embedding_id = embedding[1]
        # triplets = self.triplet_selector.get_triplets(embedding_id, target)
        # if embedding_id.is_cuda:
        #     triplets = triplets.cuda()
        # ap_distances = (embedding_id[triplets[:, 0]] - embedding_id[triplets[:, 1]]).pow(2).sum(1).sqrt()
        # an_distances = (embedding_id[triplets[:, 0]] - embedding_id[triplets[:, 2]]).pow(2).sum(1).sqrt()
        # inter_loss2 = F.relu(ap_distances - an_distances + self.margin).pow(2)
        # inter_loss2 = inter_loss2.sum() * self.each_cls_cnt / (self.n_class * len(triplets))
        #
        # inter_loss = (inter_loss1 + inter_loss2) / 2.

        embedding_id = embedding
        triplets = self.triplet_selector.get_triplets(embedding_id, target)
        if len(triplets) == 0:
            inter_loss = torch.tensor(0.0, dtype=torch.float32).cuda()
            inter_loss.requires_grad = True
            print('There are no triplet in this batch.')
            # print('embedding_id:')
            # print(embedding_id)
        else:
            if embedding_id.is_cuda:
                triplets = triplets.cuda()
            ap_distances = (embedding_id[triplets[:, 0]] - embedding_id[triplets[:, 1]]).pow(2).sum(1).sqrt()
            an_distances = (embedding_id[triplets[:, 0]] - embedding_id[triplets[:, 2]]).pow(2).sum(1).sqrt()
            inter_loss = F.relu(ap_distances - an_distances + self.margin).pow(2)
            inter_loss = inter_loss.sum() / len(triplets)

        # verif_loss = torch.tensor(0.0, dtype=torch.float32).cuda()
        # for idx, cls in enumerate(target):
        #     pos_indices = target == cls
        #     neg_indices = target != cls
        #     pos_embed = embedding_id[pos_indices]
        #     neg_embed = embedding_id[neg_indices]
        #     ap_dist_squa = (embedding_id[idx].unsqueeze(dim=0) - pos_embed).pow(2)
        #     an_dist_sqrt = (embedding_id[idx].unsqueeze(dim=0) - neg_embed).pow(2).sum(dim=1).sqrt()
        #     verif_loss += ap_dist_squa.sum() / len(pos_embed)
        #     verif_loss += F.relu(1.4 - an_dist_sqrt).pow(2).sum() / len(neg_embed)
        # verif_loss = verif_loss / (2.0 * len(embedding_id))

        loss_xent = [self.xent(embed_clfy, target) for embed_clfy in embedding_classify]
        loss_xent = sum(loss_xent) / len(loss_xent)

        # total_loss = 0.375 * (loss_xent_p1 + loss_xent_p2) + 0.25 * inter_loss
        # total_loss = 0.5 * (loss_xent_p1 + loss_xent_p2) + inter_loss
        # total_loss = loss_xent + inter_loss
        # total_loss = 0.5 * (loss_xent + inter_loss)
        # total_loss = loss_xent + inter_loss + verif_loss
        total_loss = loss_xent + inter_loss
        # total_loss = inter_loss

        return total_loss


class XentTripletLossSqrt(nn.Module):
    def __init__(self, margin, n_class, each_cls_cnt, triplet_selector):
        super(XentTripletLossSqrt, self).__init__()
        self.margin = margin
        self.n_class = n_class
        self.each_cls_cnt = each_cls_cnt
        self.triplet_selector = triplet_selector
        self.xent = nn.CrossEntropyLoss()

    def forward(self, embedding_classify, embedding, target, group_label):
        embedding_id = embedding
        triplets = self.triplet_selector.get_triplets(embedding_id, target)
        if len(triplets) == 0:
            inter_loss = torch.tensor(0.0, dtype=torch.float32).cuda()
            print('There are no triplet in this batch.')
        else:
            if embedding_id.is_cuda:
                triplets = triplets.cuda()
            ap_distances = (embedding_id[triplets[:, 0]] - embedding_id[triplets[:, 1]]).pow(2).sum(1).sqrt()
            an_distances = (embedding_id[triplets[:, 0]] - embedding_id[triplets[:, 2]]).pow(2).sum(1).sqrt()
            inter_loss = F.relu(ap_distances - an_distances + self.margin)
            inter_loss = inter_loss.sum() / len(triplets)

        loss_xent = [self.xent(embed_clfy, target) for embed_clfy in embedding_classify]
        loss_xent = sum(loss_xent) / len(loss_xent)

        total_loss = loss_xent + inter_loss

        return total_loss

class XentTripletLossSqua(nn.Module):
    def __init__(self, margin, n_class, each_cls_cnt, triplet_selector):
        super(XentTripletLossSqua, self).__init__()
        self.margin = margin
        self.n_class = n_class
        self.each_cls_cnt = each_cls_cnt
        self.triplet_selector = triplet_selector
        self.xent = nn.CrossEntropyLoss()

    def forward(self, embedding_classify, embedding, target, group_label):
        embedding_id = embedding
        triplets = self.triplet_selector.get_triplets(embedding_id, target)
        if len(triplets) == 0:
            inter_loss = torch.tensor(0.0, dtype=torch.float32).cuda()
            print('There are no triplet in this batch.')
        else:
            if embedding_id.is_cuda:
                triplets = triplets.cuda()
            ap_distances = (embedding_id[triplets[:, 0]] - embedding_id[triplets[:, 1]]).pow(2).sum(1)
            an_distances = (embedding_id[triplets[:, 0]] - embedding_id[triplets[:, 2]]).pow(2).sum(1)
            inter_loss = F.relu(ap_distances - an_distances + self.margin).pow(2)
            inter_loss = inter_loss.sum() / len(triplets)

        loss_xent = [self.xent(embed_clfy, target) for embed_clfy in embedding_classify]
        loss_xent = sum(loss_xent) / len(loss_xent)

        total_loss = loss_xent + inter_loss

        return total_loss


class XentGroupTripletLossV2(nn.Module):
    def __init__(self, margin, n_class, each_cls_cnt, triplet_selector):
        super(XentGroupTripletLossV2, self).__init__()
        self.margin = margin
        self.n_class = n_class
        self.each_cls_cnt = each_cls_cnt
        self.triplet_selector = triplet_selector
        self.xent = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]

    def forward(self, embedding_classify, embedding, target, group_label):
        inter_loss = 0
        for gop in range(1, 5):
            gop_idx = group_label == gop
            gop_embedding = embedding[gop_idx]
            gop_target = target[gop_idx]
            if len(gop_embedding) == 0:
                continue
            triplets = self.triplet_selector.get_triplets(gop_embedding, gop_target)
            if len(triplets) == 0:
                # print('There are no triplet in group {} of this batch.'.format(gop))
                continue
            triplets = triplets.cuda()
            ap_distances = (gop_embedding[triplets[:, 0]] - gop_embedding[triplets[:, 1]]).pow(2).sum(1).sqrt()
            an_distances = (gop_embedding[triplets[:, 0]] - gop_embedding[triplets[:, 2]]).pow(2).sum(1).sqrt()
            inter_loss_gop = F.relu(ap_distances - an_distances + self.margin).pow(2)
            # inter_loss_gop = inter_loss_gop.sum() * self.each_cls_cnt / len(triplets)
            inter_loss_gop = inter_loss_gop.sum() / len(triplets)
            inter_loss += inter_loss_gop

        inter_loss = inter_loss / 4.0

        loss_xent_p1 = self.xent[0](embedding_classify[0], target)
        loss_xent_p2 = self.xent[1](embedding_classify[1], target)

        # total_loss = 0.375 * (loss_xent_p1 + loss_xent_p2) + 0.25 * inter_loss
        total_loss = 0.5 * (loss_xent_p1 + loss_xent_p2) + inter_loss

        return total_loss


