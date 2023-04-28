#
# Copyright 2021-2022 Enflame. All Rights Reserved.
#

import collections
import os
import torch
import time
import ctypes
from copy import deepcopy
from fairseq import data, distributed_utils, options, utils, tokenizer
from fairseq.ddp_trainer import DDPTrainer
from fairseq.meters import StopwatchMeter
from fairseq.sequence_generator import SequenceGenerator
from fairseq.data import dictionary, data_utils, load_dataset_splits
from fairseq.models import build_model
import sacrebleu
import tops_models.dllogger as DLLogger
from fairseq.log_helper import setup_logger

from tops_models.logger import tops_logger
from tops_models.timers import Timers
import subprocess
import socket

if torch.cuda.is_available():
    import torch.distributed as dist
else:
    import torch_gcu.distributed as dist
    import torch_gcu

def distributed_main(args):
    if args.distributed_init_method is None and args.distributed_port > 0:
        # We can determine the init method automatically for Slurm.
        node_list = os.environ.get('SLURM_JOB_NODELIST')
        if node_list is not None:
            try:
                hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', node_list])
                args.distributed_init_method = 'tcp://{host}:{port}'.format(
                    host=hostnames.split()[0].decode('utf-8'),
                    port=args.distributed_port)
                args.distributed_rank = int(os.environ.get('SLURM_PROCID'))
                args.device_id = int(os.environ.get('SLURM_LOCALID'))
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError as e:  # Slurm is not installed
                pass
    if args.distributed_init_method is None:
        raise ValueError('--distributed-init-method or --distributed-port '
                         'must be specified for distributed training')
    args.distributed_rank = distributed_utils.distributed_init(args)
    args.device_id = int(os.environ.get('LOCAL_RANK', args.local_rank))
    print('| initialized host {} as rank {} and device id {}'.format(socket.gethostname(), args.distributed_rank, args.device_id))
    main(args)


def main(args):
    print(args)
    setup_logger(args)

    if args.device_type == "cuda":
        if not torch.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')
        torch.cuda.set_device(args.device_id)
        if args.distributed_world_size > 1:
            assert(dist.is_initialized())
            dist.broadcast(torch.tensor([1], device="cuda"), 0)
            torch.cuda.synchronize()
        if args.max_tokens is None:
            args.max_tokens = 6000
        pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
        ctypes.CDLL('libcudart.so').cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
        ctypes.CDLL('libcudart.so').cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    elif args.device_type == "gcu":
        args.device = torch_gcu.gcu_device(args.local_rank)
    else:
        args.device = "cpu"

    torch.manual_seed(args.seed)
    src_dict, tgt_dict = data_utils.load_dictionaries(args)
    add_extra_items_to_checkpoint({'src_dict': src_dict, 'tgt_dict': tgt_dict})
    datasets = load_dataset_splits(args, ['train', 'valid', 'test'], src_dict, tgt_dict)
    model = build_model(args)
    print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))
    # Build trainer
    if args.device_type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 7 and not args.amp:
        print('| NOTICE: your device may support faster training with --amp')
    trainer = DDPTrainer(args, model)
    print('| model {}, criterion {}'.format(args.arch, trainer.criterion.__class__.__name__))

    args.remove_bpe='@@ '
    if args.device_type == "cuda":
        print('| training on {} GPUs'.format(args.distributed_world_size))
        print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
            args.max_tokens,
            args.max_sentences,
        ))

    epoch_itr = data.EpochBatchIterator(
        dataset=datasets[args.train_subset],
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=args.max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=1,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        static_batch=args.static_batch,
    )

    all_files = os.listdir(os.path.join(args.save_dir, 'checkpoints'))
    all_files = [i for i in all_files if ".pt" in i]
    all_files.sort(reverse=True)
    valid_subsets = args.valid_subset.split(',')
    print("eval all_files:",all_files)
    for file in all_files:
        args.restore_file = file
        # Load the latest checkpoint if one is available
        # Load gcu model
        args.device_type = "gcu"
        args.device = torch_gcu.gcu_device(args.local_rank)
        load_checkpoint(args, trainer, epoch_itr)
        args.device_type = "gcu"
        args.device = "cpu"

        trainer.model.to("cpu")
        valid_losses =[0.0]
        print('Performing sanity check...')
        # valid_losses = validate(args, trainer, datasets, valid_subsets)
        # print("valid_losses:",valid_losses)
        valid_bleu = score(args, trainer, datasets[valid_subsets[0]], src_dict, tgt_dict, 'valid.raw.de')
        DLLogger.log(step=trainer.get_num_updates(), data={'val_loss': valid_losses[0], 'val_bleu': valid_bleu}, verbosity=1)
        # sanity_score = score(args, trainer, datasets['test'], src_dict, tgt_dict, 'test.raw.de')
        # DLLogger.log(step='SANITY_CHECK', data={'sanity_check_score': sanity_score}, verbosity=1)


def validate(args, trainer, datasets, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    # Reset value iterations counter
    trainer._num_val_iterations = 0

    valid_losses = []
    for subset in subsets:

        if len(subsets) > 1:
            print('Validating on \'{}\' subset'.format(subset))

        # Initialize data iterator
        itr = data.EpochBatchIterator(
            dataset=datasets[subset],
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=args.max_positions,
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=1,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            static_batch=args.static_batch,
        ).next_epoch_itr(shuffle=False)

        # reset validation loss meters
        DLLogger.flush()
        subset_losses = []
        local_step = 0
        for sample in itr:
            loss = trainer.valid_step(sample)
            subset_losses.append(loss)
            local_step += 1
            if args.eval_step_per_epoch > 0 and local_step >= args.eval_step_per_epoch:
                logger.info("only evaluate steps {} for one epoch.".format(args.eval_step_per_epoch))
                break

        subset_loss = sum(subset_losses)/len(subset_losses)

        DLLogger.flush()

        valid_losses.append(subset_loss)
        print(f'Validation loss on subset {subset}: {subset_loss}')

    return valid_losses


def score(args, trainer, dataset, src_dict, tgt_dict, ref_file):

    begin = time.time()
    src_dict = deepcopy(src_dict) # This is necessary, generation of translations
    tgt_dict = deepcopy(tgt_dict) # alters target dictionary messing up with the rest of training

    model = trainer.get_model()

    # Initialize data iterator
    datatmp = data.EpochBatchIterator(
        dataset=dataset,
        max_tokens=None,
        max_sentences=args.max_sentences_valid,
        max_positions=args.max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=1,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        static_batch=args.static_batch,
        test_data = True,
    )
    itr = datatmp.next_epoch_itr(shuffle=False)


    # Initialize generator
    gen_timer = StopwatchMeter()
    translator = SequenceGenerator(
	[model],
        tgt_dict.get_metadata(),
        maxlen=args.max_target_positions - 1, #do not include EOS token
        beam_size=args.beam,
	stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
	len_penalty=args.lenpen, unk_penalty=args.unkpen,
	sampling=args.sampling, sampling_topk=args.sampling_topk, minlen=args.min_len,
        )
    # Generate and compute BLEU
    dict = dictionary.Dictionary()
    num_sentences = 0
    predictions = []

    translations = translator.generate_batched_itr(
            itr, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b,
            cuda=(args.device_type == "cuda"), timer=gen_timer, prefix_size=args.prefix_size, device_type=args.device_type, device=args.device
            )

    for sample_id, src_tokens, target_tokens, hypos in translations:
        # Process input and grount truth
        target_tokens = target_tokens.int().cpu()

        src_str = src_dict.string(src_tokens, args.remove_bpe)
        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

        # Process top predictions
        for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict = None,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe
                    )
            # Score only the top hypothesis
            if i==0:
                if args.sentencepiece:
                    hypo_str = hypo_str.replace(' ', '').replace('▁', ' ')
                    target_str = target_str.replace(' ', '').replace('▁', ' ')
                sys_tok = tokenizer.Tokenizer.tokenize((hypo_str.lower() if not args.test_cased_bleu else hypo_str), dict)
                ref_tok = tokenizer.Tokenizer.tokenize((target_str.lower() if not args.test_cased_bleu else target_str), dict)
                if not args.sentencepiece:
                    hypo_str = tokenizer.Tokenizer.detokenize(hypo_str, 'de')
                predictions.append('{}\t{}'.format(sample_id, hypo_str))

        num_sentences += 1

    if args.distributed_world_size > 1:
        predictions = _all_gather_predictions(predictions)

    with open(os.path.join(args.data, ref_file), 'r') as reference:
        tmplines = []
        for i,line in enumerate(reference.readlines()):
            if i in datatmp.ignore_index: # remove ignore index
                continue
            tmplines.append(line)
        refs = [tmplines]

    #reducing indexed predictions as strings is more memory efficient than reducing tuples
    predictions = [tuple(item.split('\t')) for item in predictions]
    predictions = [(int(item[0]), item[1]) for item in predictions]
    predictions.sort(key=lambda tup: tup[0])
    predictions = [hypo[1] + ('\n' if hypo[1][-1] != '\n' else '')  for hypo in predictions]
    sacrebleu_score = sacrebleu.corpus_bleu(predictions, refs, lowercase=not args.test_cased_bleu).score
    if args.save_predictions:
        os.makedirs(os.path.join(args.save_dir, 'predictions'), exist_ok=True)
        with open(os.path.join(args.save_dir, 'predictions', ref_file + '.pred.update_{}'.format(trainer._num_updates)), 'w') as f:
            f.write(''.join(predictions))

    DLLogger.log(step=trainer.get_num_updates(),
            data={
                'inference tokens/s': float(args.distributed_world_size)/gen_timer.avg
                },
            verbosity=0)
    DLLogger.flush()
    if gen_timer.sum != 0:
        print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
            len(predictions), gen_timer.n, gen_timer.sum, len(predictions) / gen_timer.sum, float(args.distributed_world_size)/gen_timer.avg))

    print('| Eval completed in: {:.2f}s | {}CASED BLEU {:.2f}'.format(time.time()-begin, '' if args.test_cased_bleu else 'UN', sacrebleu_score))

    return sacrebleu_score


def _all_gather_predictions(predictions):
    ready = False
    all_ready = False
    reduced_predictions = []
    max_size = 65000
    while not all_ready:
        lst_len = len(predictions)
        size = 2000     #some extra space for python stuff
        n = 0
        while n < lst_len:
            str_len = len(predictions[n].encode('utf8')) + 8 # per string pickle overhead
            if size + str_len >= max_size:
                break
            size += str_len
            n += 1
        chunk = predictions[:n]
        predictions = predictions[n:]
        if not predictions:
            ready = True
        chunk = (ready, chunk)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gathered = distributed_utils.all_gather_list(chunk, max_size=65000)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        reduced_predictions += [t[1] for t in gathered]
        all_ready = all([t[0] for t in gathered])

    reduced_predictions = [item for sublist in reduced_predictions for item in sublist]

    return reduced_predictions


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
            not end_of_epoch and args.save_interval_updates > 0 and
            updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'best': None if not hasattr(save_checkpoint, 'best') else save_checkpoint.best,
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }
    extra_state.update(save_checkpoint.extra_items)

    checkpoints = [os.path.join(args.save_dir, 'checkpoints', fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(os.path.join(args.save_dir, 'checkpoints'), pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            os.remove(old_chk)


def add_extra_items_to_checkpoint(dict):
    if not hasattr(save_checkpoint, 'extra_items'):
        save_checkpoint.extra_items = {}
    save_checkpoint.extra_items.update(dict)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, 'checkpoints', args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path,load_optim=False)
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']


class global_var:
    def __init__(self):
        self.train_global_step = 0


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    parser = options.get_training_parser()
    ARGS = options.parse_args_and_arch(parser)
    ARGS.local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
    ARGS.rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
    ARGS.distributed_world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    ARGS.distributed = ARGS.distributed_world_size > 1
    logger = tops_logger()
    timers = Timers()
    timers('interval time').init(ARGS.max_sentences, ARGS.skip_steps, ARGS.log_get_freq)
    if ARGS.distributed:
        distributed_main(ARGS)
    else:
        main(ARGS)
