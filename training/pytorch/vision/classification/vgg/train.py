import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms

import utils
from model.vgg import vgg16
import json
from tops_models.logger import tops_logger
from tops_models.avg_meter import AverageMeter
from tops_models.timers import Timers
from tops_models.bind_cpu import bind_cpu
from collections import OrderedDict
from tensorboardX import SummaryWriter

from optimizers import (
    get_optimizer,
    lr_cosine_policy,
    lr_linear_policy,
    lr_step_policy,
)

try:
    from apex import amp
except ImportError:
    amp = None

class global_var:
    def __init__(self):
        self.train_global_step = 0

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, lr_scheduler, tb_writer, scaler):
    model.train()
    local_step = 0
    for image, target in data_loader:
        if local_step == 0:
            timers('interval time').start()

        if args.device == 'gcu':
            image, target = image.to(device), target.int().to(device)
        else:
            image, target = image.to(device), target.to(device)
        if args.torch_amp:
            if args.device == 'gcu':
                with torch_gcu.amp.autocast(enabled=True):
                    output = model(image)
                    loss = criterion(output, target)
            elif args.device == 'cuda':
                with torch.cuda.amp.autocast(enabled=True):
                    output = model(image)
                    loss = criterion(output, target)
        else:
            output = model(image)
            loss = criterion(output, target)
        optimizer.zero_grad(True)
        lr_func = lambda i: lr_scheduler(optimizer, i, epoch)
        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        elif args.torch_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if args.device=='gcu':
            if args.enable_horovod:
                torch_gcu.sync_lived_tensor(device)
                grads = torch_gcu.fetch_gradients(optimizer)
                handles = []
                for grad in grads:
                    handle = hvd.allreduce_async_(grad, average=True)
                    handles.append(handle)
                for handle in handles:
                    hvd.synchronize(handle)
                torch_gcu.optimizer_step(optimizer, [loss, output], model=model,
                                         mode=torch_gcu.JitRunMode.ASYNC if args.enable_horovod else torch_gcu.JitRunMode.SAFEASYNC,
                                         device=device)
            elif args.torch_amp:
                scaler.step(optimizer)
                scaler.update()
                torch_gcu.unlazy(torch_gcu.collect_amp_training_params(model, optimizer, scaler)+[loss, output])
            else:
                torch_gcu.optimizer_step(optimizer, [loss, output],
                                         mode=torch_gcu.JitRunMode.ASYNC if args.distributed else torch_gcu.JitRunMode.SAFEASYNC,
                                         model=model)
        else:
            if args.torch_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        local_step += 1
        var.train_global_step += 1
        learning_rate = lr_func(var.train_global_step)
        if  local_step % args.print_freq == 0:
            loss = loss.item()
            timers('interval time').elapsed()
            train_fps_step = timers('interval time').step_fps_list[-1]
            acc1, acc5 = utils.accuracy(output.cpu().float(), target.cpu(), topk=(1, 5))
            logger.info(
                "[Model_Train] epoch: {}; local_step: {}; global_step: {}; loss: {}; train_accuracy_step: {}; train_fps_step: {}".format(
                    epoch,
                    local_step,
                    var.train_global_step,
                    loss,
                    acc1,
                    train_fps_step)
            )
            tb_writer.add_scalar('learning_rate', learning_rate, var.train_global_step)
            tb_writer.add_scalar('loss', loss, var.train_global_step)
            tb_writer.add_scalar('train_accuracy_step', acc1, var.train_global_step)
            tb_writer.add_scalar('train_fps_step', train_fps_step, var.train_global_step)
        if args.training_step_per_epoch > 0 and local_step >= args.training_step_per_epoch:
            logger.info("only train steps {} for one epoch.".format(args.training_step_per_epoch))
            break

def reduce_tensor_dtu(tensor, n):
    rt = tensor.clone()
    torch_gcu.distributed.all_reduce(rt, op=torch_gcu.distributed.ReduceOp.SUM)
    rt /= n
    return rt

def reduce_tensor_gpu(tensor, n):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= n
    return rt

def evaluate(model, criterion, data_loader, epoch, print_freq, device):
    loss_am = AverageMeter("evaluate loss")
    top1_am = AverageMeter("evaluate acc top1")
    top5_am = AverageMeter("evaluate acc top5")
    model.eval()
    local_step = 0
    with torch.no_grad():
        for image, target in data_loader:
            start_time = time.time()
            image = image.to(device, non_blocking=True)
            if args.device == 'gcu':
                target = target.int().to(device, non_blocking=True)
            else:
                target = target.to(device, non_blocking=True)
            if args.torch_amp:
                if args.device == 'gcu':
                    with torch_gcu.amp.autocast(enabled=True):
                        output = model(image)
                        loss = criterion(output, target)
                elif args.device == 'cuda':
                    with torch.cuda.amp.autocast(enabled=True):
                        output = model(image)
                        loss = criterion(output, target)
            else:
                output = model(image)
                loss = criterion(output, target)
            acc1, acc5 = utils.accuracy(output.float(), target, topk=(1, 5))
            ## allreduce loss acc1 acc5
            if args.distributed:
                if args.device == "gcu":
                    loss = reduce_tensor_dtu(loss, args.world_size)
                    acc1 = reduce_tensor_dtu(acc1, args.world_size)
                    acc5 = reduce_tensor_dtu(acc5, args.world_size)
                else:
                    loss = reduce_tensor_gpu(loss, args.world_size)
                    acc1 = reduce_tensor_gpu(acc1, args.world_size)
                    acc5 = reduce_tensor_gpu(acc5, args.world_size)
            loss = loss.item()
            acc1 = acc1.item()
            acc5 = acc5.item()
            loss_am.update(loss, output.size(0))
            top1_am.update(acc1, output.size(0))
            top5_am.update(acc5, output.size(0))
            local_step += 1
            if (local_step - 1) % print_freq == 0:
                batch_size = image.shape[0]
                eval_fps_step = batch_size / (time.time() - start_time)
                logger.info(
                    "[Model_Eval] epoch: {}; local_step: {}; loss: {}; eval_accuracy_step: {}; eval_fps_step: {}".format(
                        epoch,
                        local_step,
                        loss,
                        acc1,
                        eval_fps_step)
                )
            if args.eval_step_per_epoch > 0 and local_step >= args.eval_step_per_epoch:
                logger.info("only evaluate steps {} for one epoch.".format(args.eval_step_per_epoch))
                break
        logger.info(
                "[Model_Eval] epoch: {}; local_step: {}; eval_loss_epoch: {}; eval_acc1_epoch: {}; eval_acc5_epoch: {};".format(
                    epoch,
                    local_step,
                    loss_am.avg,
                    top1_am.avg,
                    top5_am.avg))
    return top1_am.avg, top5_am.avg


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, cache_dataset, distributed):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if distributed:
        if args.device == 'gcu':
            train_sampler = torch_gcu.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            test_sampler = torch_gcu.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    elif args.enable_horovod:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    if args.apex and amp is None:
        raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                           "to enable mixed-precision training.")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)

    if args.device=='gcu':
        device = torch_gcu.gcu_device(args.local_rank)
        if args.enable_horovod:
            device = torch_gcu.gcu_device(hvd.local_rank())
    else:
        device = torch.device(args.device)
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir,
                                                                   args.cache_dataset, args.distributed)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True, persistent_workers=True)

    print("Creating model")
    # model = torchvision.models.__dict__[args.model](pretrained=args.pretrained,num_classes=int(args.num_classes))
    model = vgg16(pretrained=args.pretrained, num_classes=int(args.num_classes),dropout_rate=args.dropout_rate)
    model.to(device)

    scaler = None
    if args.torch_amp:
        if args.device == 'gcu':
            scaler = torch_gcu.amp.GradScaler(
                init_scale=128,
                growth_factor=2,
                backoff_factor=0.5,
                growth_interval=1000000000,
                enabled=True,
                use_zero_grad=True,
            )
        elif args.device == 'cuda':
            scaler = torch.cuda.amp.GradScaler(
                init_scale=128,
                growth_factor=2,
                backoff_factor=0.5,
                growth_interval=1000000000,
                enabled=True,
            )
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()
    if args.enable_horovod:
      initial_learning_rate = args.lr * args.batch_size * hvd.size() / 256.
    else:
      initial_learning_rate = args.lr * args.batch_size * args.world_size / 256.
    optimizer = get_optimizer(list(model.named_parameters()), args.lr, args)
    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )

    if args.lr_schedule == "step":
        lr_policy = lr_step_policy(initial_learning_rate, [30, 60, 80], 0.1, args.warmup)
    elif args.lr_schedule == "cosine":
        lr_policy = lr_cosine_policy(
        initial_learning_rate, args.warmup, args.epochs, end_lr=args.end_lr
    )
    elif args.lr_schedule == "linear":
        lr_policy = lr_linear_policy(initial_learning_rate, args.warmup, args.epochs)

    # Horovod: broadcast parameters
    if args.enable_horovod:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    model_without_ddp = model
    if args.distributed:
        if args.device == 'gcu':
            model = torch_gcu.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
        var.train_global_step = checkpoint['train_global_step'] + 1

    if args.test_only:
        evaluate(model, criterion, data_loader, args.start_epoch, args.print_freq, device)
        return

    print("Start training")
    start_time = time.time()
    eval_total_acc1=[]
    eval_total_acc5 = []
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed or args.enable_horovod:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, lr_policy, tb_writer, scaler)
        if (epoch + 1) % args.eval_freq == 0:
            eval_accuracy_epoch, eval_accuracy_5_epoch=evaluate(model, criterion, data_loader_test, epoch, args.print_freq, device=device)
            tb_writer.add_scalar('eval_accuracy_epoch', eval_accuracy_epoch, var.train_global_step)
            eval_total_acc1.append(eval_accuracy_epoch)
            eval_total_acc5.append(eval_accuracy_5_epoch)
            if args.output_dir:
                checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'train_global_step': var.train_global_step,
                    'train_fps_mean': timers('interval time').calc_fps(),
                    'train_fps_min': timers('interval time').min_fps(),
                    'train_fps_max': timers('interval time').max_fps()}
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'checkpoint.pth'))
        ## time will reset in end of traversal epoch
        timers('interval time').reset()
    acc1_best = max(eval_total_acc1)
    acc5_best = max(eval_total_acc5)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    runtime_info = OrderedDict(
        [('model', args.model),
         ('local_rank', args.local_rank),
         ('batch_size', args.batch_size),
         ('epochs', args.epochs),
         ('training_step_per_epoch', args.training_step_per_epoch),
         ('eval_step_per_epoch', args.eval_step_per_epoch),
         ('acc1', acc1_best),
         ('acc5', acc5_best),
         ('device', args.device),
         ('skip_steps', args.skip_steps),
         ('train_fps_mean', timers('interval time').calc_fps()),
         ('train_fps_min', timers('interval time').min_fps()),
         ('train_fps_max', timers('interval time').max_fps()),
         ('training_time', total_time_str)
        ])

    json_report = json.dumps(runtime_info, indent=4, sort_keys=False)
    logger.info("Final report:\n{}".format(json_report))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', help='dataset')
    parser.add_argument('--num_classes', default='1000', help='the number of class')
    parser.add_argument('--model', default='vgg16', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument(
        "--optimizer", default="sgd", type=str, choices=("sgd", "rmsprop")
    )
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate in network')
    parser.add_argument(
        "--nesterov",
        action="store_true",
        help="use nesterov momentum, (default: false)",
    )
    parser.add_argument(
        "--bn-weight-decay",
        action="store_true",
        help="use weight_decay on batch normalization learnable parameters, (default: false)",
    )
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--eval-freq', default=1, type=int, help='save model frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--lr-schedule",
        default="step",
        type=str,
        metavar="SCHEDULE",
        choices=["step", "linear", "cosine"],
        help="Type of LR schedule: {}, {}, {}".format("step", "linear", "cosine"),
    )
    parser.add_argument("--end-lr", default=0, type=float, help="set end learning rate when use cosine lr-schedule")
    parser.add_argument(
        "--warmup", default=5, type=int, metavar="E", help="number of warmup epochs"
    )
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--broadcast-buffers",
        dest="broadcast_buffers",
        help="broadcast batch norm moving mean and variance",
        action="store_true",
    )
    parser.add_argument(
        "--fix-seed",
        dest="fix_seed",
        help="fix seed",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--torch-amp', action='store_true',
                        help='Use torch_gcu.amp or torch.cuda.amp to do mixed precision')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--training_step_per_epoch', default=-1, type=int, help='Number of training steps for each epoch')
    parser.add_argument('--eval_step_per_epoch', default=-1, type=int,help='Number of evaluate steps for each epoch')
    parser.add_argument('--skip_steps', default=10, type=int,help='steps to skip while computing fps')
    parser.add_argument('--enable-horovod', type=bool, default=False,
                help='True to enable horovod distributed training',)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    if args.fix_seed:
        torch.manual_seed(1 << 20)
    utils.init_distributed_mode(args)
    # try to import torch_gcu before SummaryWriter to enable
    # DTU_UMD_FLAGS="cqm_executor_enable=true cluster_as_device=true" which
    # runtime 2.0 needed on LEO.
    if args.device=='gcu':
        import torch_gcu

    if args.enable_horovod:
        import horovod.torch as hvd
        # Horovod: initialize library.
        hvd.init()
    logger = tops_logger()
    if args.enable_horovod:
        tb_writer = SummaryWriter('runs/tensorboard/gcu_{}'.format(hvd.rank()))
    else:
        tb_writer = SummaryWriter('runs/tensorboard/gcu_{}'.format(args.local_rank))
    var = global_var()
    timers = Timers()
    bind_cpu(args.local_rank, -1)
    timers('interval time').init(args.batch_size, args.skip_steps, args.print_freq)
    main(args)
