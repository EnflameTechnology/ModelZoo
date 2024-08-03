# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
#-------------------------------------------------------------------------
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Train a network across multiple GPUs.
"""

from collections import defaultdict, OrderedDict
import contextlib
from itertools import chain
from pickle import TRUE
from time import time

import torch
import torch.nn.functional as F
HAVE_APEX = False
if torch.cuda.is_available():
    import torch.distributed as dist
    from torch.cuda.amp import autocast as autocast, GradScaler
    scaler = GradScaler()
    try:
        from apex import amp
        from apex.parallel import DistributedDataParallel as DDP
        HAVE_APEX = True
    except (ModuleNotFoundError,ImportError):
        print("warning: have not install apex")
        from torch.nn.parallel import DistributedDataParallel as DDP
else:
    import torch_gcu

    import torch_gcu.distributed as dist
    from torch_gcu.nn.parallel import DistributedDataParallel as DDP
    scaler = torch_gcu.amp.GradScaler(use_zero_grad=True)

from fairseq import distributed_utils, optim, utils
from fairseq.optim import lr_scheduler
from fairseq.meters import TimeMeter
from fairseq.criterions import CRITERION_REGISTRY
import tops_models.dllogger as DLLogger
import math


class DDPTrainer(object):
    """Main class for data parallel training.

    This class supports data parallel training, where multiple workers each
    have a full model replica and gradients are accumulated synchronously via
    dist.all_reduce.
    """

    def __init__(self, args, model):
        if args.device_type == "cuda":
            if not torch.cuda.is_available():
                raise NotImplementedError('have no GPU Training on CPU is not supported')

        self.args = args
        if args.device_type == "cpu":
            self.model = model
            self.criterion = CRITERION_REGISTRY[args.criterion](args)
        elif args.device_type == "gcu":
            device = self.args.device
            self.model = model.to(device)
            self.criterion = CRITERION_REGISTRY[args.criterion](args).to(device)
        else:
            self.model = model.cuda()
            self.criterion = CRITERION_REGISTRY[args.criterion](args).cuda()
        self.optimizer = optim.build_optimizer(self.args, self.model.parameters())
        self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)

        if args.amp and HAVE_APEX and args.amp_type == "apex":
            model, optimizer = amp.initialize(
                    self.model,
                    self.optimizer._optimizer,
                    opt_level=self.args.amp_level if self.args.amp_level else 'O2',
                    max_loss_scale=2**15,
                    cast_model_outputs=torch.float16
                    )

        if self.args.distributed:
            if HAVE_APEX:
                self.model = DDP(model)
            else:
                self.model = DDP(model, device_ids=[self.args.local_rank])

        self._buffered_stats = defaultdict(lambda: [])
        self._flat_grads = None
        self._num_updates = 0
        self._num_val_iterations = 0
        self._optim_history = None
        self.throughput_meter = TimeMeter()


    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        if self.args.amp and HAVE_APEX and self.args.amp_type == "apex":
            extra_state['amp_state_dict'] = amp.state_dict()
            extra_state['amp_master_params'] = list(amp.master_params(self.optimizer.optimizer))
        if distributed_utils.is_master(self.args):  # only save one checkpoint
            utils.save_state(
                filename, self.args, self.get_model(), self.criterion, self.optimizer,
                self.lr_scheduler, self._num_updates, self._optim_history, extra_state,
            )

    def load_checkpoint(self, filename, load_optim=True):
        """Load all training state from a checkpoint file."""
        extra_state, optim_history, last_optim_state = \
            utils.load_model_state(filename, self.get_model())

        if last_optim_state is not None:
            # rebuild optimizer after loading model, since params may have changed
            #self.optimizer = optim.build_optimizer(self.args, self.model.parameters())
            self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)

            if load_optim:
                self._optim_history = optim_history
                # only reload optimizer and lr_scheduler if they match
                last_optim = self._optim_history[-1]
                if last_optim['criterion_name'] == self.criterion.__class__.__name__:
                    self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
                    if last_optim['optimizer_name'] == self.optimizer.__class__.__name__:
                        self.optimizer.load_state_dict(last_optim_state)

                self._num_updates = last_optim['num_updates']

        if self.args.amp and HAVE_APEX and self.args.amp_type == "apex" \
            and extra_state is not None and 'amp_state_dict' in extra_state:
            self.optimizer.optimizer._lazy_init_maybe_master_weights()
            self.optimizer.optimizer._amp_stash.lazy_init_called = True
            self.optimizer.optimizer.load_state_dict(last_optim_state)
            for param, saved_param in zip(amp.master_params(self.optimizer.optimizer), extra_state['amp_master_params']):
                param.data.copy_(saved_param.data)

            amp.load_state_dict(extra_state['amp_state_dict'])

        return extra_state

    def train_step(self, sample, update_params=True, last_step=False,get_log=False,timers=None):
        """Do forward, backward and parameter update."""
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        if self.args.device_type == "cuda":
            seed = self.args.seed + self.get_num_updates()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        self.model.train()
        if isinstance(self.model, DDP) and HAVE_APEX:
            if last_step:
                self.model.disable_allreduce()
            else:
                self.model.enable_allreduce()

        # forward and backward pass
        if self.args.device_type == "cuda":
            sample, sample_size = self._prepare_sample(sample)
        elif self.args.device_type == "cpu":
            sample, sample_size = self._prepare_sample_cpu(sample)
        else:
            sample, sample_size = self._prepare_sample_to_device(sample)
        ntokens = sample["total_ntokens"] if self.args.distributed else sample_size
        if self.args.amp:
            if torch.cuda.is_available() and self.args.amp_type == "torch":
                with autocast():
                    loss, oom_fwd = self._forward(sample)
            else:
                with torch_gcu.amp.autocast(enabled=(self.args.amp_type == "torch")):
                    loss, oom_fwd = self._forward(sample)
        else:
            loss, oom_fwd = self._forward(sample)

        self.zero_grad()
        oom_bwd = self._backward(loss)
        if not self.args.use_aot  and self.args.device_type == "gcu":
            if self.args.amp and self.args.amp_type == "torch":
                scaler.unscale_(self.optimizer._optimizer)
                scaler.step(self.optimizer._optimizer)
                scaler.update()
                torch_gcu.unlazy(torch_gcu.collect_amp_training_params(self.model, self.optimizer._optimizer, scaler)+[loss])
            else:
                torch_gcu.optimizer_step(self.optimizer._optimizer, loss,
                                         mode=torch_gcu.JitRunMode.ASYNC if self.args.distributed else torch_gcu.JitRunMode.SAFEASYNC,
                                         model=self.model)
        else:
            if torch.cuda.is_available() and self.args.amp and self.args.amp_type == "torch":
                scaler.unscale_(self.optimizer._optimizer)
                scaler.step(self.optimizer._optimizer)
                scaler.update()
            else:
                self.optimizer.step()
        self._buffered_stats['ntokens'].append(ntokens)
        self._buffered_stats['ooms_fwd'].append(oom_fwd)
        self._buffered_stats['ooms_bwd'].append(oom_bwd)

        # update parameters
        if update_params and not last_step:
            ntokens_s = self._buffered_stats['ntokens']
            ooms_fwd = self._buffered_stats['ooms_fwd']
            ooms_bwd = self._buffered_stats['ooms_bwd']
            ooms_fwd = sum(ooms_fwd)
            ooms_bwd = sum(ooms_bwd)
            ooms = ooms_fwd + ooms_bwd #this is always <= distributed_world_size

            if ooms > 0 and ooms == self.args.distributed_world_size:
                print('| WARNING: OOM in all workers, skipping batch')
                self.zero_grad()
                return

            # aggregate stats and logging outputs
            grad_denom = sum(ntokens_s)
            for p in self.model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad /= grad_denom

            self._num_updates += 1
            # update learning rate
            self.lr_scheduler.step_update(self._num_updates)

            if get_log:
                loss = loss.cpu().item()
                timers('interval time').elapsed()
                # Handle logging
                if self.args.static_batch:
                    nsentences = sample['target'].size(0)
                    self.throughput_meter.update(nsentences)
                    info_log_data = {
                                'sentences/s':self.throughput_meter.avg,
                                'nsentences':nsentences,
                                'loss':loss / sample_size / math.log(2)
                                }
                    debug_log_data = {}
                else:
                    self.throughput_meter.update(grad_denom)
                    info_log_data = {
                                'tokens/s':self.throughput_meter.avg,
                                'tokens':grad_denom,
                                'loss':loss / sample_size / math.log(2)
                                }
                    debug_log_data = {
                                'batch_size':sample['target'].size(0),
                                'lr':self.get_lr(),
                                'grad_denom':grad_denom,
                                'updates':1
                                }

                DLLogger.log(step=self._num_updates, data=info_log_data, verbosity=0)
                DLLogger.log(step=self._num_updates, data=debug_log_data, verbosity=1)

            self.clear_buffered_stats()

    def _forward(self, sample):
        loss = None
        oom = 0
        try:
            if sample is not None:
                # calculate loss and sample size
                logits, _ = self.model(**sample['net_input'])
                target = sample['target']
                if not self.args.adaptive_softmax_cutoff:
                    probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                else:
                    #TODO: trainig crashes after couple hundred iterations because of unknown
                    #error in the PyTorch's autograd
                    probs, target = self.get_model().decoder.adaptive_softmax(logits, target.view(-1))
                loss = self.criterion(probs, target)
        except RuntimeError as e:
            if not eval and 'out of memory' in str(e):
                print('| WARNING: ran out of memory in worker {}, skipping batch'.format(self.args.distributed_rank), force=True)
                oom = 1
                loss = None
            else:
                raise e
        return loss, oom

    def _backward(self, loss):
        oom = 0
        if loss is not None:
            try:
                if self.args.amp:
                    if HAVE_APEX and self.args.amp_type == "apex":
                        with amp.scale_loss(loss, self.optimizer._optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        scaler.scale(loss).backward()
                else:
                    loss.backward()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory in worker {}, skipping batch'.format(self.args.distributed_rank), force=True)
                    oom = 1
                    self.zero_grad()
                else:
                    raise e
        return oom

    def valid_step(self, sample):
        """Do forward pass in evaluation mode."""
        self.model.eval()
        self._num_val_iterations += 1
        # forward pass
        if self.args.device_type == "cuda":
            sample, ntokens = self._prepare_sample(sample)
        elif self.args.device_type == "cpu":
            sample, ntokens = self._prepare_sample_cpu(sample)
        else:
            sample, ntokens = self._prepare_sample_to_device(sample)
        with torch.no_grad():
            loss, oom_fwd = self._forward(sample)
        loss = loss.item() if loss is not None else 0
        assert not oom_fwd, 'Ran out of memory during validation'
        if self.args.distributed:
            losses, ntokens_s = zip(*distributed_utils.all_gather_list(
                (loss, ntokens), 16384, self.args.device
            ))
        else:
            losses = [loss]
            ntokens_s = [ntokens]
        weight = sum(ntokens_s)
        scaled_loss = sum(losses) / weight / math.log(2)
        return scaled_loss

    def dummy_train_step(self, dummy_batch):
        """Dummy training step for warming caching allocator."""
        self.train_step(dummy_batch, update_params=False)
        self.zero_grad()
        self.clear_buffered_stats()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def clear_buffered_stats(self):
        self._buffered_stats.clear()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate based on the validation loss."""
        return self.lr_scheduler.step(epoch, val_loss)

    def lr_step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.lr_scheduler.step_update(num_updates)

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_throughput_meter(self):
        """Get the throughput meter"""
        return self.throughput_meter

    def get_model(self):
        """Get the model replica."""
        return self.model.module if isinstance(self.model, DDP) else self.model

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None, 0
        return utils.move_to_cuda(sample), sample['ntokens']

    def _prepare_sample_to_device(self, sample):
        if sample is None or len(sample) == 0:
            return None, 0

        return utils.move_to_device(sample,self.args.device), sample['ntokens']

    def _prepare_sample_cpu(self, sample):
        def _move_to_cpu(maybe_tensor):
            if len(sample) == 0:
                return {}
            if torch.is_tensor(maybe_tensor):
                return maybe_tensor
            elif isinstance(maybe_tensor, dict):
                return {
                    key: _move_to_cpu(value)
                    for key, value in maybe_tensor.items()
                }
            elif isinstance(maybe_tensor, list):
                return [_move_to_cpu(x) for x in maybe_tensor]
            else:
                return maybe_tensor
        return _move_to_cpu(sample), sample['ntokens']
