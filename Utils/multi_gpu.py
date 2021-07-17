import torch
import torch.distributed as dist
import os


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 多机情况下：RANK表示多机情况下使用的第几台主机，单机情况：RANK表示使用的第几块GPU
        args.rank = int(os.environ["RANK"])
        # 多机情况下：WORLD_SIZE表示主机数，单机情况：WORLD_SIZE表示GPU数量
        args.world_size = int(os.environ['WORLD_SIZE'])
        # 多机情况下：LOCAL_RANK表示某台主机下的第几块设备，单机情况：与RANK相同
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    # 为每个进程设置不同的GPU
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    # 创建进程组
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    # 等待每个GPU运行到此
    dist.barrier()


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value

def cleanup():
    dist.destroy_process_group()