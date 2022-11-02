import os
import math
import json
import logging
import datetime

import torch

from utils.util import ensure_dir
from utils.visualization import WriterTensorboardX
from model.layer_utils import init_weights


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, config, generator, discriminator, loss, metrics, optimizer, lr_scheduler, resume, train_logger):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)  # creat a logger

        # setup GPU device if available, move model into configured device and init the weights
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.generator.apply(init_weights)
        self.discriminator.apply(init_weights)
        if len(device_ids) > 1:
            self.generator = torch.nn.DataParallel(generator, device_ids=device_ids)
            self.discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids)

        self.adversarial_loss = loss['adversarial']    # diffrent loss functions
        self.content_loss = loss['content']
        self.metrics = metrics
        self.generator_optimizer = optimizer['generator']       # different optimizers
        self.discriminator_optimizer = optimizer['discriminator']
        self.generator_lr_scheduler = lr_scheduler['generator']             # different learning rate schedulers
        self.discriminator_lr_scheduler = lr_scheduler['discriminator']
        self.train_logger = train_logger

        cfg_trainer = config['trainer']         # the input trianer configurations / some hyper-parameters
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']       # save the model parameter in a frequency
        self.verbosity = cfg_trainer['verbosity']
        self.monitor = cfg_trainer.get('monitor', 'off')        # use monitor if the monitor exist, or "off"

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']              # only have the "max" and "min" modes

            self.mnt_best = math.inf if self.mnt_mode == 'min' else -math.inf
            self.early_stop = cfg_trainer.get('early_stop', math.inf)

        self.start_epoch = 1                                    # set the start epoch

        # setup directory for checkpoint saving
        start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], config['name'], start_time)     # set different folders
        # setup visualization writer instance
        writer_dir = os.path.join(cfg_trainer['log_dir'], config['name'], start_time)
        self.writer = WriterTensorboardX(writer_dir, self.logger, cfg_trainer['tensorboardX'])      # set tensorboard datas for visualization

        # Save configuration file into checkpoint directory
        ensure_dir(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(config, handle, indent=4)                 # use json to record the checkpoint data

        if resume:
            self._resume_checkpoint(resume)                     # resume the check point

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There's no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU's configured to use is {}, but only {} are available "
                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        device_ids = list(range(n_gpu_use))
        return device, device_ids

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):              # train all the epochs
            result = self._train_epoch(epoch)                               # each epoch is trained in this function

            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            # print logged informations to the screen                       # print the information each epoch
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            is_best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] < self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    is_best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:                        # early stop if not improved for long time (overfitting)                     
                    self.logger.info("Validation performance didn't improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:                                   # save the model for checkpoint
                self._save_checkpoint(epoch, save_best=is_best)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            'epoch': epoch,
            'logger': self.train_logger,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
            'generator_lr_scheduler': self.generator_lr_scheduler.state_dict(),
            'discriminator_lr_scheduler': self.discriminator_lr_scheduler.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:                                                               # also save the best if the current checkpoint is the best
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...".format('model_best.pth'))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load params from checkpoint
        if checkpoint['config']['name'] != self.config['name']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
            self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])

        # load learning scheduler state from checkpoint only when learning scheduler type is not changed.
        if checkpoint['config']['lr_scheduler']['type'] != self.config['lr_scheduler']['type']:
            self.logger.warning(
                "Warning: Learning scheduler type given in config file is different from that of checkpoint. "
                "Learning scheduler parameters not being resumed.")
        else:
            self.generator_lr_scheduler.load_state_dict(checkpoint['generator_lr_scheduler'])
            self.discriminator_lr_scheduler.load_state_dict(checkpoint['discriminator_lr_scheduler'])

        self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
