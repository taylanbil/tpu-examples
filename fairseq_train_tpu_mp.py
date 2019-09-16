"""Python script to run FAIRSEQ models on TPU

This file mimics pytorch/fairseq/train.py, but contains some changes that work
  well with TPUs. Example bash script:


```bash
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

python fairseq_train_tpu.py \
  $data_path \
  --arch=transformer_vaswani_wmt_en_de_big \
  --max-target-positions=64 \
  --no-save \
  --attention-dropout=0.1 \
  --no-progress-bar \
  --criterion=label_smoothed_cross_entropy \
  --source-lang=en \
  --lr-scheduler=inverse_sqrt \
  --min-lr 1e-09 \
  --skip-invalid-size-inputs-valid-test \
  --target-lang=de \
  --label-smoothing=0.1 \
  --update-freq=1 \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --warmup-init-lr 1e-07 \
  --lr 0.0005 \
  --warmup-updates 4000 \
  --share-all-embeddings \
  --dropout 0.3 \
  --weight-decay 0.0 \
  --valid-subset=valid \
  --max-epoch=50 \
    --input_shapes 1024x16 512x32 256x64 \
    --num_cores=8 \
    --metrics_debug \
    --log_steps=100
```

Here, TPU specific flags are:

    --input_shapes 1024x16 512x32 256x64 \
    --num_cores=8 \
    --metrics_debug \
    --log_steps=100

"""

import argparse
import sys
import os
import math
import collections

import utils as utils_tpu

utils_tpu.initialize_path('fairseq')

import torch
import torch_xla
import torch_xla_py.data_parallel as dp
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm
import torch_xla_py.xla_multiprocessing as xmp

from fairseq.data import data_utils
# Overwriting collate_tokens to guarantee constant size input tensors
# This is reducing the number of graph recompiles
collate_tokens_gpu = data_utils.collate_tokens
batch_by_size_gpu = data_utils.batch_by_size
import train as fairseq_train


def batch_by_size_tpu(
    indices,
    num_tokens_fn,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
):
  batches = [[] for _ in FLAGS.input_shapes]
  for idx in indices:
    sample_len = num_tokens_fn(idx)
    for j, (batch_size, padlen) in enumerate(FLAGS.input_shapes):
      if padlen < sample_len:
        continue
      batches[j].append(idx)
      if len(batches[j]) == batch_size:
        yield batches[j]
        batches[j] = []
      break


def collate_tokens_tpu(values,
                       pad_idx,
                       eos_idx=None,
                       left_pad=False,
                       move_eos_to_beginning=False):
  size = max(v.size(0) for v in values)
  for batch_size, padlen in FLAGS.input_shapes:
    if padlen < size:
      continue
    size = padlen
    break
  res = values[0].new(len(values), size).fill_(pad_idx)

  def copy_tensor(src, dst):
    assert dst.numel() == src.numel()
    if move_eos_to_beginning:
      assert src[-1] == eos_idx
      dst[0] = eos_idx
      dst[1:] = src[:-1]
    else:
      dst.copy_(src)

  for i, v in enumerate(values):
    copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
  return res


data_utils.collate_tokens = collate_tokens_tpu
data_utils.batch_by_size = batch_by_size_tpu

from fairseq import options, tasks, checkpoint_utils, progress_bar, utils
from fairseq.trainer import Trainer
from fairseq.data import iterators
from fairseq.meters import StopwatchMeter, AverageMeter


def parse_args():
  # We need to control certain flags here.
  # e.g. parallelization needs to be suppressed and deferred to torch_xla flags
  # e.g. input tensor shapes need to be controlled via --input_shapes
  parser = options.get_training_parser()
  parser.add_argument(
      '--input_shapes',
      nargs='*',
      default=None,
      help=(
          'This is used to specify batches and pad lengths. Ex: '
          '`--input_shapes 256x32 512x16` will produce batches w/ 256 '
          'sentences padded to length 32, or 512 sentences padded to length '
          '16. Including too many input shapes will cause graph recompiles and'
          ' degrade performance. On the other extreme, including 1 shape may '
          'waste a ton of flops, since batches may contain a lot of pad '
          'indices on average. Note that the max pad length in this arg will '
          'be used as `--max-source-positions`'))
  parser.add_argument('--log_steps', type=int, default=20)
  parser.add_argument('--num_cores', type=int, default=8)
  parser.add_argument('--use_gpu', action='store_true')
  parser.add_argument('--metrics_debug', action='store_true')
  FLAGS = options.parse_args_and_arch(parser)
  if not FLAGS.use_gpu:
    if FLAGS.fp16:
      raise RuntimeError(
          '--fp16 was provided, this is controlled by env var XLA_USE_BF16')
    xu.eprint('suppressing "distributed_world_size"')
    FLAGS.distributed_world_size = xm.xrt_world_size()
    if FLAGS.distributed_init_method is not None:
      xu.eprint('suppressing "distributed_init_method"')
      FLAGS.distributed_init_method = None
    if FLAGS.input_shapes is None:
      raise RuntimeError('Please specify batches and pad lengths using '
                         '--input_shapes. Ex: `--input_shapes 256x32 512x16` .'
                         'Please refer to the description of the --input_shape'
                         ' arg in --help')
    gpu_input_shape_args = [
        'max_sentences', 'max_sentences_valid', 'max_tokens'
    ]
    nonnull_gpu_input_shape_args = [
        arg for arg in gpu_input_shape_args if getattr(FLAGS, arg) is not None
    ]
    if nonnull_gpu_input_shape_args:
      errmsg = ('On TPUs, please control input shapes '
                'using `--input_shapes`. Any non-null arg in {} will trigger'
                ' this error.').format(gpu_input_shape_args)
      raise RuntimeError(errmsg)

    FLAGS.input_shapes = parse_input_shapes(FLAGS.input_shapes)
    # XXX (taylanbil): do we ever have more than 2 dimensions in fairseq?
    FLAGS.max_source_positions = FLAGS.input_shapes[-1][1]
    if xu.getenv_as('XLA_USE_BF16', bool, False):
      xu.eprint(
          'WARNING: bfloat16 is enabled. Note that fairseq meters such as '
          'loss will accumulate the numerator, and increment the denominator. '
          'Due to lack of precision in higher numbers in bfloat16, these '
          'meters will report invalid values after a while.')

  return FLAGS


def parse_input_shapes(input_shapes_arg):
  input_shapes = (
      shape.replace('*', 'x').split('x') for shape in input_shapes_arg)
  input_shapes = [list(map(int, shape)) for shape in input_shapes]
  input_shapes.sort(key=lambda shape: shape[1])
  return input_shapes


def load_checkpoint_tpu(args, trainer, device_preloaded):

  def meter_to_device(meter, device):
    for key, val in vars(meter).items():
      if isinstance(val, torch.Tensor):
        newval = val.to(device=torch.device(device))
        setattr(meter, key, newval)

  def trainer_meters_to_device(trainer, device):
    for meter in trainer.meters.values():
      meter_to_device(meter, device)

  for device, trainer in trainers.items():
    if device != device_preloaded:
      _ = trainer.load_checkpoint(
          checkpoint_utils.get_checkpoint_path(args),
          reset_optimizer=args.reset_optimizer,
          reset_lr_scheduler=args.reset_lr_scheduler,
          optimizer_overrides=eval(args.optimizer_overrides),
          reset_meters=args.reset_meters,
      )
    trainer_meters_to_device(trainer, device)




def main_tpu(args):

  def log_step(step_type, device, step, log_output=None, tracker=None):
    msg = '{}/ {}, device {}, step {}'.format(step_type, utils_tpu.now(),
                                              device, step)
    if tracker:
      rates = tracker.rate(), tracker.global_rate()
      msg += ', Rate={:.2f}, GlobalRate={:.2f}'.format(*rates)
    if log_output:
      msg += ', loss={:.4f}, nll_loss={:.4f}'.format(
          log_output['loss'].item(), log_output['nll_loss'].item())
    return msg

  def prepare_task(args, xla_device):
    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
      task.load_dataset(valid_sub_split, combine=True, epoch=0)

    # Build models and criteria to print some metadata
    model, criterion = task.build_model(args), task.build_criterion(args)
    print(model)
    print('| model {}, criterion {}'.format(args.arch,
                                            criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    model = model.to(xla_device)
    trainer = Trainer(args, task, model, criterion, xla=True)
    lr = trainer.get_lr()

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)
    # FIXME: load checkpoint re-do
    if extra_state is not None:
      # checkpoint detected, load saved model weights
      xu.eprint(
          'checkpoint detected, device 0 meters need to be re-loaded to device')
      raise
      #load_checkpoint_tpu(args, trainers, devices[0])
    valid_subsets = args.valid_subset.split(',')
    ordinal = xm.get_ordinal(defval=-1)
    device_str = str(xla_device) if ordinal < 0 else '{}/{}'.format(xla_device, ordinal)
    return task, trainer, model, epoch_itr, lr, valid_subsets, device_str

  def train_loop_fn(device, trainer, loader):
    stats, log_output, tracker = None, None, xm.RateTracker()
    for i, samples in loader:
      if i and not (i % args.log_steps):
        print(
            log_step(
                'training', device, i, log_output=log_output, tracker=tracker))
      log_output = trainer.train_step(samples)
      xm.optimizer_step(trainer.optimizer)
      tracker.add(sum(sample['nsentences'] for sample in samples))
    return tracker

  def valid_loop_fn(device, trainer, loader):
    # reset validation loss meters
    for k in ['valid_loss', 'valid_nll_loss']:
      meter = trainer.get_meter(k)
      if meter is not None:
        meter.reset()
    extra_meters = collections.defaultdict(lambda: AverageMeter())
    for i, sample in loader:
      if not (i % args.log_steps):
        print(log_step('validation', device, i, tracker=None))
      log_output = trainer.valid_step(sample)
      for k, v in log_output.items():
        if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
          continue
        extra_meters[k].update(v)
    stats = fairseq_train.get_valid_stats(trainer)
    for k, meter in extra_meters.items():
      stats[k] = meter.avg
    return stats

  def validate_subset(args, device, trainer, task, epoch_itr, subset):
    print('Validating the subset "{}"'.format(subset))
    # Initialize data iterator
    itr = task.get_batch_iterator(
        dataset=task.dataset(subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            list(trainer.get_model().max_positions()),
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_workers=args.num_workers).next_epoch_itr(shuffle=False)
    progress = progress_bar.build_progress_bar(
        args,
        itr,
        epoch_itr.epoch,
        prefix='valid on {} \'{}\' subset'.format(device, subset),
        no_progress_bar='simple')
    para_loader = dp.ParallelLoader(progress, [device])
    stats = valid_loop_fn(device, trainer, para_loader.per_device_loader(device))
    progress.print(stats, tag=subset, step=trainer.get_num_updates())
    return stats['loss'].avg

  def validate(args, device, trainer, task, epoch_itr, subsets):
    valid_losses = {
        subset: validate_subset(args, device, trainer, task, epoch_itr, subset)
        for subset in subsets
    }
    return valid_losses

  def initialize_loader_for_epoch(args, epoch_itr, device):
    if epoch_itr.epoch <= len(args.update_freq):
      update_freq = args.update_freq[epoch_itr.epoch - 1]
    else:
      update_freq = args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=False, shuffle=(epoch_itr.epoch >= args.curriculum))
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, prefix='training on {}'.format(device), no_progress_bar='simple')
    para_loader = dp.ParallelLoader(progress, [device])
    return progress, para_loader

  def keep_training(lr, epoch_itr, trainer):
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    n_updates = trainer.get_num_updates()
    return ((lr > FLAGS.min_lr) and (epoch_itr.epoch < max_epoch) and
            (n_updates < max_update))

  # `xla_device` is `torch.device` and `device` is `str`
  xla_device = xm.xla_device()
  task, trainer, model, epoch_itr, lr, valid_subsets, device = prepare_task(
      args, xla_device)
  train_meter = StopwatchMeter()
  train_meter.start()
  while keep_training(lr, epoch_itr, trainer):
    # TRAINING
    print('Device {} Epoch {} begin {}'.format(device, epoch_itr.epoch + 1, utils_tpu.now()))
    progress, para_loader = initialize_loader_for_epoch(args, epoch_itr, device)
    tracker = train_loop_fn(device, trainer, para_loader.per_device_loader(device))
    print('Device {} Epoch {} Training stats:'.format(device, epoch_itr.epoch))
    stats = fairseq_train.get_training_stats(trainer)
    progress.print(stats, tag=device)
    print('Device {} Epoch {} Tracker Rate={:.2f}, GlobalRate={:.2f}'.format(device, epoch_itr.epoch, tracker.rate(), tracker.global_rate()))
    print('Device {} Epoch {} end {}'.format(device, epoch_itr.epoch, utils_tpu.now()))
    if args.metrics_debug:
      print(torch_xla._XLAC._xla_metrics_report())

    # VALIDATION
    if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
      valid_losses = validate(args, device, trainer, task, epoch_itr, valid_subsets)

      # only use average first validation loss from the first device
      # to update the learning rate
      # FIXME
      vloss = valid_losses[valid_subsets[0]].item()
      print('old learning rate: {}'.format(lr))
      lr = trainer.lr_step(epoch_itr.epoch, vloss)
      print('new learning rate: {}'.format(lr))

      # save checkpoint
      # FIXME: only save from first device
      from fairseq import distributed_utils
      print('device {} ISMASTER {}'.format(device, distributed_utils.is_master(args)))
      from fairseq import pdb
      pdb.set_trace()
      if epoch_itr.epoch % args.save_interval == 0:
        checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, vloss)

    if args.metrics_debug:
      print(torch_xla._XLAC._xla_metrics_report())

  train_meter.stop()
  print('| done training in {:.1f} seconds'.format(train_meter.sum))


def _mp_fn(index, flags):
  global FLAGS
  FLAGS = flags
  torch.set_default_tensor_type('torch.FloatTensor')
  main_tpu(flags)

if __name__ == '__main__':
  FLAGS = parse_args()
  if FLAGS.use_gpu:
    # undo batch preparation function assignments TPU -> GPU
    data_utils.collate_tokens = collate_tokens_gpu
    data_utils.batch_by_size = batch_by_size_gpu
    fairseq_train.cli_main()
  else:
    #main_tpu(FLAGS)
    xu.eprint('Args')
    for key, val in FLAGS.__dict__.items():
      xu.eprint('\t{} {}'.format(key, val))
    xu.eprint('---------')
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)
