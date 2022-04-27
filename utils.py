import io
import logging
import os
import torch
import colorlog
import refile
from tabulate import tabulate


class TqdmToLogger(io.StringIO):
    logger = None
    level = None
    buf = ''

    def __init__(self):
        super(TqdmToLogger, self).__init__()
        self.logger = get_logger('tqdm')

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.info(self.buf)


def get_logger(logger_name='default', debug=False, save_to_dir=None):
    if debug:
        log_format = (
            '%(asctime)s - '
            '%(levelname)s : '
            '%(name)s - '
            '%(pathname)s[%(lineno)d]:'
            '%(funcName)s - '
            '%(message)s'
        )
    else:
        log_format = (
            '%(asctime)s - '
            '%(levelname)s : '
            '%(name)s - '
            '%(message)s'
        )
    bold_seq = '\033[1m'
    colorlog_format = f'{bold_seq} %(log_color)s {log_format}'
    colorlog.basicConfig(format=colorlog_format, datefmt='%y-%m-%d %H:%M:%S')
    logger = logging.getLogger(logger_name)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if save_to_dir is not None:
        fh = logging.FileHandler(os.path.join(save_to_dir, 'log', 'debug.log'))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        fh = logging.FileHandler(
            os.path.join(save_to_dir, 'log', 'warning.log'))
        fh.setLevel(logging.WARNING)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        fh = logging.FileHandler(os.path.join(save_to_dir, 'log', 'error.log'))
        fh.setLevel(logging.ERROR)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # pred(correct.shape)
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class TestLogger:
    def __init__(self):
        self.headers = ['Epoch', 'Acc@1', 'Acc@5', 'MRR', 'LR']
        self.data = []

    def add_record(self, record):
        self.data.append(record)

    def print_table(self, record):
        self.data.append(record)
        print(tabulate(self.data, headers=self.headers, tablefmt="github"))


class MgvSaveHelper(object):
    def __init__(self, save_oss=False, oss_path=''):
        self.oss_path = oss_path
        self.save_oss = save_oss

    def set_stauts(self, save_oss=False, oss_path=''):
        self.oss_path = oss_path
        self.save_oss = save_oss

    def get_s3_path(self, path):
        return refile.smart_path_join(self.oss_path, path)

    def check_s3_path(self, path):
        return refile.is_s3(path)

    def load_ckpt(self, path, rm_module=True):
        if self.check_s3_path(path):
            with refile.smart_open(path, "rb") as f:
                ckpt = torch.load(f)
        else:
            ckpt = torch.load(path)
        print(f"====> load checkpoint from {path}")
        return ckpt

    def save_ckpt(self, path, epoch, model, optimizer=None):
        if self.save_oss:
            # if not self.check_s3_path(path):
            if not refile.is_s3(path):
                path = self.get_s3_path(path)
            with refile.smart_open(path, "wb") as f:
                torch.save(
                    {"epoch": epoch,
                     "state_dict": model.state_dict(),
                     "optimizer": optimizer.state_dict()}, f)
        else:
            torch.save(
                {"epoch": epoch,
                 "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict()}, path)

        print(f"====> save checkpoint to {path}")

    def save_pth(self, path, file):
        if self.save_oss:
            if not self.check_s3_path(path):
                path = self.get_s3_path(path)
            with refile.smart_open(path, "wb") as f:
                torch.save(file, f)
        else:
            torch.save(file, path)

        print(f"====> save pth to {path}")

    def load_pth(self, path):
        if self.check_s3_path(path):
            with refile.smart_open(path, "rb") as f:
                ckpt = torch.load(f)
        else:
            ckpt = torch.load(path)
        print(f"====> load pth from {path}")
        return ckpt


