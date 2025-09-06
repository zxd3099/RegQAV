# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.misc import accuracy
# from utils import sigmoid_focal_loss
from .utils import sigmoid_focal_loss
from util.segment_ops import segment_cw_to_t1t2, segment_t1t2_to_cw, segment_iou, diou_loss, log_ratio_width_loss


class SetCriterion(nn.Module):
    """ This class computes the loss for Model.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth segments and the outputs of the models
        2) we supervise each pair of matched ground-truth / prediction (supervise class and segment)
    """
    def __init__(self, matcher, weight_dict, losses, focal_alpha=0.25, diou=True, label_smoothing=0, alpha=0.5):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.label_smoothing = label_smoothing
        self.diou = diou
        self.alpha = alpha

    def nll_loss(self, pred, target, filter_idx):
        device = pred.device

        mask = torch.full((pred.shape[0],), 5, device=device)
        mask[filter_idx] = 1

        base_loss = F.binary_cross_entropy_with_logits(pred, target.to(device), reduction='none')
        loss_mod = (base_loss * mask).mean()
        return {'loss_mod': loss_mod}

    def poisson_loss(self, pred, target, filter_idx):
        device = pred.device
        target = target.to(device)
        lambda_ = torch.exp(pred)

        base_loss = lambda_ - target * pred + torch.lgamma(target + 1)
        penalty = self.alpha * torch.relu(target - lambda_)
        loss_value = base_loss + penalty

        mask = torch.full((pred.shape[0],), 1.2, device=device)
        mask[filter_idx] = 1

        loss_pois = (loss_value * mask).mean()
        return {'loss_pois': loss_pois}

    def loss_actionness(self, outputs, targets):
        """Compute the actionness regression loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'pred_segments' in outputs
        assert 'pred_actionness' in outputs

        src_segments = outputs['pred_segments']
        src_segments = torch.stack([src_segments[..., 0], src_segments[..., 1].exp()], dim=-1).view((-1, 2))
        target_segments = torch.cat([t['segments'] for t in targets], dim=0)

        losses = {}

        iou_mat = segment_iou(segment_cw_to_t1t2(src_segments), target_segments)

        gt_iou = iou_mat.max(dim=1)[0]
        pred_actionness = outputs['pred_actionness']
        loss_actionness = F.l1_loss(pred_actionness.view(-1), gt_iou.view(-1).detach())

        losses['loss_actionness'] = loss_actionness
        return losses

    def loss_labels(self, outputs, targets, indices, num_segments, log=True):
        """
        Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_segments]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes_onehot = F.one_hot(target_classes_o, num_classes=src_logits.shape[2]).to(src_logits.dtype)

        target_onehot_shape = [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]]
        target_classes_onehot_full = torch.zeros(target_onehot_shape, dtype=src_logits.dtype, device=src_logits.device)
        target_classes_onehot_full[idx] = target_classes_onehot

        if self.label_smoothing != 0:
            target_classes_onehot_full *= 1 - self.label_smoothing
            target_classes_onehot_full += self.label_smoothing / (target_classes_onehot.size(-1) + 1)

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot_full, outputs['mask'], num_segments,
                                     alpha=self.focal_alpha, gamma=2)
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_segments(self, outputs, targets, indices, num_segments):
        """Compute the losses related to the segments, the L1 regression loss and the IoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'pred_segments' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_segments = outputs['pred_segments'][idx]
        target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_segment = log_ratio_width_loss(src_segments, segment_t1t2_to_cw(target_segments))

        src_segments = torch.stack([
            src_segments[..., 0], src_segments[..., 1].exp()
        ], dim=-1)

        if self.diou:
            loss_iou = diou_loss(segment_cw_to_t1t2(src_segments), target_segments)
        else:
            loss_iou = 1 - torch.diag(segment_iou(segment_cw_to_t1t2(src_segments), target_segments))

        losses = {'loss_segments': loss_segment.sum() / num_segments, 'loss_iou': loss_iou.sum() / num_segments}
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_segments, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "segments": self.loss_segments,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_segments, **kwargs)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def filter_preds_and_targets(self, predicts, targets, type, tar_labels):
        res_target, filter_idx, fake_nums = [], [], []
        for idx, target in enumerate(targets):
            if target["modified"] == 0 or (type == "_v" and target["v_modified"] == 0) or (
                    type == "_a" and target["a_modified"] == 0):
                fake_nums.append(0)
                continue

            if type == "_s":
                segments = target['merged_segments']
                labels = torch.zeros(segments.size(0), dtype=torch.int64).to(segments.device)
            else:
                labels = target['labels']
                mask = torch.isin(labels, tar_labels.to(labels.device))
                segments = target['segments'][mask]
                labels = labels[mask]

            res_target.append({'labels': labels, 'segments': segments})
            filter_idx.append(idx)
            fake_nums.append(len(segments))

        filtered_preds = {}
        for key, value in predicts.items():
            # if key in ["sampling_locations", "classifier_score", "proposals"]:
            #     filtered_preds[key] = value
            if key == "aux_outputs":
                filtered_preds[key] = [
                    {k: v[filter_idx] for k, v in aux_outputs.items()}
                    for aux_outputs in value
                ]
            else:
                filtered_preds[key] = value[filter_idx]

        return filtered_preds, res_target, filter_idx, fake_nums

    def cal_loss(self, outputs, targets, type, tar_labels=None):
        losses = {}

        filter_outputs, filter_targets, filter_idx, fake_nums = self.filter_preds_and_targets(outputs, targets, type, tar_labels)

        if type == "_v":
            modifies = torch.cat([torch.tensor(v["v_modified"]).unsqueeze(0) for v in targets]).float()
        elif type == "_a":
            modifies = torch.cat([torch.tensor(v["a_modified"]).unsqueeze(0) for v in targets]).float()
        elif type == "_s":
            modifies = torch.cat([torch.tensor(v["modified"]).unsqueeze(0) for v in targets]).float()

        if "classifier_score" in outputs:
            loss_modified = self.nll_loss(outputs["classifier_score"], modifies, filter_idx)
            loss_modified = {k + type: v for k, v in loss_modified.items()}
            losses.update(loss_modified)

        if "fake_num" in outputs:
            fake_nums = torch.cat([torch.tensor(v).unsqueeze(0) for v in fake_nums]).float()
            loss_poission = self.poisson_loss(outputs["fake_num"], fake_nums, filter_idx)
            loss_poission = {k + type: v for k, v in loss_poission.items()}
            losses.update(loss_poission)

        if len(filter_targets) == 0:
            return losses

        # Compute the average number of target segments accross all nodes, for normalization purposes
        num_segments = sum(len(t["labels"]) for t in filter_targets)
        num_segments = torch.as_tensor([num_segments], dtype=torch.float,
                                       device=next(iter(outputs.values())).device)
        if type == "_s" and "pred_actionness" in filter_outputs:
            outputs_without_aux = filter_outputs
            l_dict = self.loss_actionness(outputs_without_aux, filter_targets)
            losses.update(l_dict)
        else:
            outputs_without_aux = {k: v for k, v in filter_outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, filter_targets)

        for loss in self.losses:
            kwargs = {}
            if loss == 'labels':
                # Logging is enabled only for the last layer
                kwargs['log'] = False
            l_dict = self.get_loss(loss, outputs_without_aux, filter_targets, indices, num_segments, **kwargs)
            l_dict = {k + type: v for k, v in l_dict.items()}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in filter_outputs:
            for i, aux_outputs in enumerate(filter_outputs['aux_outputs']):
                indices_aux = self.matcher(aux_outputs, filter_targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, filter_targets, indices_aux, num_segments, **kwargs)
                    l_dict = {k + type + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

    def forward(self, outputs_v, outputs_a, targets, outputs_s=None):
        """ This performs the loss computation.
        Parameters:
            outputs_v: dict of tensors for visual, see the output specification of the models for the format
            outputs_a: dict of tensors for audio, see the output specification of the models for the format
            outputs_s: dict of tensors for stacking, see the output specification of the stacker for the format
            targets: list of dicts, such that len(targets) == batch_size.
                     The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = {}

        v_label = torch.tensor([0, 2])
        a_label = torch.tensor([1, 2])

        v_loss = self.cal_loss(outputs_v, targets, "_v", v_label)            # 0: visual, 1: audio, 2: both
        a_loss = self.cal_loss(outputs_a, targets, "_a", a_label)            # 0: visual, 1: audio, 2: both
        if outputs_s is not None:
            s_loss = self.cal_loss(outputs_s, targets, "_s")
            losses.update(s_loss)

        losses.update(v_loss)
        losses.update(a_loss)

        for key in self.weight_dict.keys():
            if key not in losses.keys():
                losses[key] = 0.

        return losses
