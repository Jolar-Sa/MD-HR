import torch
import torch.nn as nn

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    y = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return y

def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    x = normalize(x, axis=1)
    y = normalize(y, axis=1)
    distmat = (1. - torch.matmul(x, y.t())) / 2.
    distmat = distmat.clamp(min=1e-12)
    return distmat

def cosine(x):
    x = normalize(x, axis=1)
    distmat = torch.matmul(x, x.t())
    return distmat

class NSSD(nn.Module):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self,num_stripes,feat_dim,cluster_center_weight=[0.9,0.1],margin=0,weight_dict=None):
        super(NSSD, self).__init__()
        self.num_stripes=num_stripes
        self.feat_dim=feat_dim
        assert feat_dim%num_stripes==0
        # self.cluster_center=torch.zeros([num_stripes,int(feat_dim/num_stripes)])
        self.cluster_center_weight=cluster_center_weight
        self.margin=margin
        if not margin==0:
            print("MarginRankingLoss",margin)
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
            # exit()
        else:
            print("SoftMarginLoss")
            self.ranking_loss = nn.SoftMarginLoss()
        if weight_dict==None:
            self.weight_dict={}
            self.cluster_center=torch.zeros([num_stripes,int(feat_dim/num_stripes)]).cuda()
        else:
            self.weight_dict=weight_dict
            self.cluster_center=weight_dict['cluster_center'].cuda()

    def __call__(self, global_feat,iters,num_iter,epoch=0):
        N,L=global_feat.size()
        global_feat=global_feat.view(N,self.num_stripes,-1)
        global_feat=normalize(global_feat)
        loss1=torch.zeros(1).cuda()
        # print(global_feat.size())
        for i in range(N):
            # tmp_mat=cosine_dist(global_feat[i], global_feat[i])
            loss1=loss1+torch.sum(torch.abs(cosine(global_feat[i])-torch.eye(self.num_stripes).cuda()))/N/self.num_stripes/self.num_stripes
            # loss1=loss1+torch.sum(torch.abs(torch.ones([self.num_stripes,self.num_stripes]).cuda()-cosine_dist(global_feat[i], global_feat[i])-torch.eye(self.num_stripes).cuda()))/N

        loss2=torch.zeros(1).cuda()
        if torch.sum(self.cluster_center)==0:
            self.cluster_center=torch.sum(global_feat,0)/N
        for i in range(N):
            tmp_mat=cosine_dist(global_feat[i], self.cluster_center.detach())
            # print("tmp_mat",tmp_mat)
            diag = torch.diag(tmp_mat)
            # print("diag",diag)
            # a_diag = torch.diag_embed(diag)
            a_diag=torch.eye(self.num_stripes).cuda()
            # print("a_diag",a_diag)
            tmp_mat_max=torch.min(tmp_mat+a_diag,1)[0]
            # print("tmp_mat_max",tmp_mat_max)
            y = tmp_mat_max.new().resize_as_(tmp_mat_max).fill_(1)
            if not self.margin==0:
                loss2 =loss2+self.ranking_loss(tmp_mat_max,diag, y)/N
            else:
                loss2 =loss2+self.ranking_loss(tmp_mat_max-diag, y)/N
            # print(loss2)
        # exit()
        self.cluster_center=self.cluster_center_weight[0]*self.cluster_center+self.cluster_center_weight[1]*torch.sum(global_feat,0)/N
        self.weight_dict['cluster_center']=self.cluster_center
        return loss1+loss2,self.weight_dict


# class NSSD_loss_3(nn.Module):
#     """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
#     Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
#     Loss for Person Re-Identification'."""
#
#     def __init__(self,num_stripes,feat_dim,cluster_center_weight=[0.9,0.1],margin=None,weight_dict=None):
#         super(NSSD_loss_3, self).__init__()
#         print("NSSD_loss_3")
#         self.num_stripes=num_stripes
#         self.feat_dim=feat_dim
#         assert feat_dim%num_stripes==0
#         # self.cluster_center=torch.zeros([num_stripes,int(feat_dim/num_stripes)])
#         self.cluster_center_weight=cluster_center_weight
#         self.margin=margin
#         if not margin==0:
#             self.ranking_loss = nn.MarginRankingLoss(margin=margin)
#         else:
#             self.ranking_loss = nn.SoftMarginLoss()
#         if weight_dict==None:
#             self.weight_dict={}
#             self.cluster_center=torch.zeros([num_stripes,int(feat_dim/num_stripes)])
#         else:
#             self.weight_dict=weight_dict
#             self.cluster_center=weight_dict['cluster_center'].cuda()
#
#     def __call__(self, global_feat,iters,num_iter,epoch=0):
#         N,L=global_feat.size()
#         global_feat=global_feat.view(N,self.num_stripes,-1)
#         global_feat=normalize(global_feat)
#         loss1=torch.zeros(1).cuda()
#         for i in range(N):
#             loss1=loss1+torch.sum(torch.abs(cosine(global_feat[i])-torch.eye(self.num_stripes).cuda()))/N
#
#         loss2=torch.zeros(1).cuda()
#         if torch.sum(self.cluster_center)==0:
#             self.cluster_center=torch.sum(global_feat,0)/N
#         for i in range(N):
#             tmp_mat=cosine_dist(global_feat[i], self.cluster_center.detach())
#             diag = torch.diag(tmp_mat)
#             a_diag=torch.eye(self.num_stripes).cuda()
#             tmp_mat_max=torch.min(tmp_mat+a_diag,1)[0]
#             y = tmp_mat_max.new().resize_as_(tmp_mat_max).fill_(1)
#             if not self.margin==0:
#                 loss2 =loss2+self.ranking_loss(tmp_mat_max,diag, y)
#             else:
#                 loss2 =loss2+self.ranking_loss(tmp_mat_max-diag, y)
#         self.cluster_center=self.cluster_center_weight[0]*self.cluster_center+self.cluster_center_weight[1]*torch.sum(global_feat,0)/N
#         self.weight_dict['cluster_center']=self.cluster_center
#         return loss1+loss2,self.weight_dict
