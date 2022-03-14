import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
import logging
import os

from src.utils import utils


class BaseAgent(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")
        self.log_path = os.path.join(config.log_dir, "log.txt")
        self.shot_mode = self.config.dataset.train.shot_mode if isinstance(self.config.dataset.train.shot_mode, str) else None

        self._set_seed()  # set seed as early as possible

        self._load_datasets()
        self._load_loaders()

        self._choose_device()
        self._create_model()
        self._create_optimizer()

        self.current_epoch = 0
        self.current_iteration = 0
        self.current_val_iteration = 0

        # we need these to decide best loss
        self.current_loss = 0
        self.current_val_metric = 0
        self.best_val_metric = 0
        self.iter_with_no_improv = 0

    def _set_seed(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _choose_device(self):
        self.is_cuda = torch.cuda.is_available()
        # if self.is_cuda and not self.config.cuda:
        #     self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda
        self.manual_seed = self.config.seed
        if self.cuda: torch.cuda.manual_seed(self.manual_seed)

        if self.cuda:
            if not isinstance(self.config.gpu_device, list):
                self.config.gpu_device = [self.config.gpu_device]

            # NOTE: we do not support multi-gpu run for now
            gpu_device = self.config.gpu_device[0]
            # self.logger.info("User specified 1 GPU: {}".format(gpu_device))
            self.device = torch.device("cuda")
            torch.cuda.set_device(gpu_device)

            # self.logger.info("Program will run on *****GPU-CUDA***** ")
            # print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            # self.logger.info("Program will run on *****CPU*****\n")

    def _load_datasets(self):
        raise NotImplementedError

    def _load_loaders(self):
        self.train_loader, self.train_len = self._create_dataloader(
            self.train_dataset,
            self.config.optim.train_batch_size,
            shuffle=True,
        )
        self.val_loader, self.val_len = self._create_test_dataloader(
            self.val_dataset,
            self.config.optim.test_batch_size,
        )
        self.test_loader, self.test_len = self._create_test_dataloader(
            self.test_dataset,
            self.config.optim.test_batch_size,
        )

    def _create_dataloader(self, dataset, batch_size, shuffle=True):
        dataset_size = len(dataset)
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, pin_memory=True,
                            num_workers=self.config.data_loader_workers)

        return loader, dataset_size

    def _create_test_dataloader(self, dataset, batch_size):
        return self._create_dataloader(dataset, batch_size, shuffle=False)

    def _create_model(self):
        raise NotImplementedError

    def _create_optimizer(self):
        raise NotImplementedError

    def run_validation(self):
        self.validate()

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()
        except KeyboardInterrupt as e:
            # self.logger.info("Interrupt detected. Saving data...")
            self.backup()
            raise e

    def write_to_file(self, text):
        assert text != None
        f = open(self.log_path, "a")
        f.write(str(text) + "\n")
        f.close()
        print(f"writing (acc or epoch) to file: {self.log_path}")

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch, self.config.num_epochs):
            print(f"Epoch: {epoch}")
            self.current_epoch = epoch
            self.train_one_epoch()
            if (self.config.validate and
                epoch % self.config.optim.validate_freq == 0):
                self.validate()  # validate every now and then
                self.test()
            self.save_checkpoint()

            # check if we should quit early bc bad perf
            if self.iter_with_no_improv > self.config.optim.patience:
                # self.logger.info("Exceeded patience. Stop training...")
                break

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def test(self):
        """
        One cycle of model testing
        :return:
        """
        raise NotImplementedError

    def backup(self):
        """
        Backs up the model upon interrupt
        """
        # self.logger.info("Backing up current version of model...")
        self.save_checkpoint(filename='backup.pth.tar')

    def finalise(self):
        """
        Do appropriate saving after model is finished training
        """
        # self.logger.info("Saving final versions of model...")
        self.save_checkpoint(filename='final.pth.tar')

    def save_metrics(self):
        raise NotImplementedError

    def save_checkpoint(self, filename="checkpoint.pth.tar"):
        out_dict = self.save_metrics()
        # if we aren't validating, then every time we save is the
        # best new epoch!
        is_best = ((self.current_val_metric == self.best_val_metric) or
                   not self.config.validate)
        utils.save_checkpoint(out_dict, is_best, filename=filename,
                              folder=self.config.checkpoint_dir)
        self.copy_checkpoint()

    def copy_checkpoint(self, filename="checkpoint.pth.tar"):
        if self.current_epoch % self.config.copy_checkpoint_freq == 0:
            utils.copy_checkpoint(
                filename=filename, folder=self.config.checkpoint_dir,
                copyname='checkpoint_epoch{}.pth.tar'.format(self.current_epoch),
            )

    def load_checkpoint(self, filename):
        raise NotImplementedError
