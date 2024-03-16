import torch
import torch.nn as nn
import torch.nn.functional as F

def euclidean_dist(x, y, eps=1e-12):
	"""
	Args:
	  x: pytorch Tensor, with shape [m, d]
	  y: pytorch Tensor, with shape [n, d]
	Returns:
	  dist: pytorch Tensor, with shape [m, n]
	"""
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(1, -2, x, y.t())
	dist = dist.clamp(min=eps).sqrt()

	return dist


def hard_example_mining(dist_mat, target, return_inds=False):
	"""For each anchor, find the hardest positive and negative sample.
	Args:
	  dist_mat: pytorch Tensor, pair wise distance between samples, shape [N, N]
	  target: pytorch LongTensor, with shape [N]
	  return_inds: whether to return the indices. Save time if `False`(?)
	Returns:
	  dist_ap: pytorch Tensor, distance(anchor, positive); shape [N]
	  dist_an: pytorch Tensor, distance(anchor, negative); shape [N]
	  p_inds: pytorch LongTensor, with shape [N];
	    indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
	  n_inds: pytorch LongTensor, with shape [N];
	    indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
	NOTE: Only consider the case in which all target have same num of samples,
	  thus we can cope with all anchors in parallel.
	"""
	assert len(dist_mat.size()) == 2
	assert dist_mat.size(0) == dist_mat.size(1)
	N = dist_mat.size(0)

	# shape [N, N]
	is_pos = target.expand(N, N).eq(target.expand(N, N).t())
	is_neg = target.expand(N, N).ne(target.expand(N, N).t())

	# `dist_ap` means distance(anchor, positive)
	# both `dist_ap` and `relative_p_inds` with shape [N, 1]
	dist_ap, relative_p_inds = torch.max(
		dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
	# `dist_an` means distance(anchor, negative)
	# both `dist_an` and `relative_n_inds` with shape [N, 1]
	dist_an, relative_n_inds = torch.min(
		dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
	# shape [N]
	dist_ap = dist_ap.squeeze(1)
	dist_an = dist_an.squeeze(1)

	if return_inds:
		# shape [N, N]
		ind = (target.new().resize_as_(target)
			   .copy_(torch.arange(0, N).long())
			   .unsqueeze(0).expand(N, N))
		# shape [N, 1]
		p_inds = torch.gather(
			ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
		n_inds = torch.gather(
			ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
		# shape [N]
		p_inds = p_inds.squeeze(1)
		n_inds = n_inds.squeeze(1)

		return dist_ap, dist_an, p_inds, n_inds

	return dist_ap, dist_an


class TripletLoss(nn.Module):
	"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
	Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
	Loss for Person Re-Identification'."""
	def __init__(self, margin, feat_norm='no'):
		super(TripletLoss, self).__init__()
		self.margin = margin
		self.feat_norm = feat_norm
		if margin >= 0:
			self.ranking_loss = nn.MarginRankingLoss(margin=margin)
		else:
			self.ranking_loss = nn.SoftMarginLoss()

	def forward(self, global_feat1, global_feat2, target):
		if self.feat_norm == 'yes':
			global_feat1 = F.normalize(global_feat1, p=2, dim=-1)
			global_feat2 = F.normalize(global_feat2, p=2, dim=-1)

		dist_mat = euclidean_dist(global_feat1, global_feat2)
		dist_ap, dist_an = hard_example_mining(dist_mat, target)

		y = dist_an.new().resize_as_(dist_an).fill_(1)
		if self.margin >= 0:
			loss = self.ranking_loss(dist_an, dist_ap, y)
		else:
			loss = self.ranking_loss(dist_an - dist_ap, y)

		return loss


class OriTripletLoss(nn.Module):
	"""Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

	def __init__(self, margin=0.3):
		super(OriTripletLoss, self).__init__()
		self.margin = margin
		self.ranking_loss = nn.MarginRankingLoss(margin=margin)

	def forward(self, inputs, targets):
		"""
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
		n = inputs.size(0)

		# Compute pairwise distance, replace by the official when merged
		dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
		dist = dist + dist.t()
		dist.addmm_(1, -2, inputs, inputs.t())
		dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

		# For each anchor, find the hardest positive and negative
		mask = targets.expand(n, n).eq(targets.expand(n, n).t())
		dist_ap, dist_an = [], []
		for i in range(n):
			dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
			dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
		dist_ap = torch.cat(dist_ap)
		dist_an = torch.cat(dist_an)

		# Compute ranking hinge loss
		y = torch.ones_like(dist_an)
		loss = self.ranking_loss(dist_an, dist_ap, y)

		# compute accuracy
		correct = torch.ge(dist_an, dist_ap).sum().item()
		return loss, correct


	# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative  = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct


class TripletLoss_ADP(nn.Module):
	"""Weighted Regularized Triplet'."""

	def __init__(self, alpha=1, gamma=1, square=0):
		super(TripletLoss_ADP, self).__init__()
		self.ranking_loss = nn.SoftMarginLoss()
		self.alpha = alpha
		self.gamma = gamma
		self.square = square

	def forward(self, inputs, targets, normalize_feature=False):
		if normalize_feature:
			inputs = normalize(inputs, axis=-1)
		dist_mat = pdist_torch(inputs, inputs)

		N = dist_mat.size(0)
		# shape [N, N]
		is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
		is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

		# `dist_ap` means distance(anchor, positive)
		# both `dist_ap` and `relative_p_inds` with shape [N, 1]
		dist_ap = dist_mat * is_pos
		dist_an = dist_mat * is_neg

		weights_ap = softmax_weights(dist_ap * self.alpha, is_pos)
		weights_an = softmax_weights(-dist_an * self.alpha, is_neg)
		furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
		closest_negative = torch.sum(dist_an * weights_an, dim=1)

		# ranking_loss = nn.SoftMarginLoss(reduction = 'none')
		# loss1 = ranking_loss(closest_negative - furthest_positive, y)

		# squared difference
		if self.square == 0:
			y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
			loss = self.ranking_loss(self.gamma * (closest_negative - furthest_positive), y)
		else:
			diff_pow = torch.pow(furthest_positive - closest_negative, 2) * self.gamma
			diff_pow = torch.clamp_max(diff_pow, max=88)

			# Compute ranking hinge loss
			y1 = (furthest_positive > closest_negative).float()
			y2 = y1 - 1
			y = -(y1 + y2)

			loss = self.ranking_loss(diff_pow, y)

		# loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)

		# compute accuracy
		correct = torch.ge(closest_negative, furthest_positive).sum().item()
		return loss, correct

class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        # target = Variable(target_data.data.cuda(),requires_grad=False)
        target = target_data.data.cuda()
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss
