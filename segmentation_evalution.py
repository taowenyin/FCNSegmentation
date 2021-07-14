import cfg
import numpy as np


class SegmentationEvalution():
    def __init__(self, n_classes, ignore_index=None):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def _calc_confusion(self, label_true, label_pred, n_classes):
        mask = (label_true >= 0) & (label_true < n_classes)
        confusion = np.bincount(n_classes * label_true[mask].astype(int) + label_pred[mask],
                                minlength=n_classes ** 2).reshape(n_classes, n_classes)
        return confusion

    def update(self, label_true, label_pred):
        for lt, lp in zip(label_true, label_pred):
            self.confusion_matrix += self._calc_confusion(lt.flatten(), lp.flatten(), self.n_classes)

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - pixel_acc:
            - class_acc: class mean acc
            - mIou :     mean intersection over union
            - fwIou:     frequency weighted intersection union
        """

        confusion = self.confusion_matrix

        # ignore unlabel
        if self.ignore_index is not None:
            for index in self.ignore_index:
                confusion = np.delete(confusion, index, axis=0)
                confusion = np.delete(confusion, index, axis=1)

        acc = np.diag(confusion).sum() / confusion.sum()
        acc_cls = np.diag(confusion) / confusion.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(confusion) / (confusion.sum(axis=1) + confusion.sum(axis=0) - np.diag(confusion))
        mean_iou = np.nanmean(iu)
        freq = confusion.sum(axis=1) / confusion.sum()
        fw_iou = (freq[freq > 0] * iu[freq > 0]).sum()

        # set unlabel as nan
        if self.ignore_index is not None:
            for index in self.ignore_index:
                iu = np.insert(iu, index, np.nan)

        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "pixel_acc: ": acc,
                "class_acc: ": acc_cls,
                "mIou: ": mean_iou,
                "fwIou: ": fw_iou,
            },
            cls_iu,
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
