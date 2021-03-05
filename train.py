import random
from datetime import datetime
from pathlib import Path
from typing import Tuple, Callable

import torch
from PIL import Image
from fire import Fire
from torch import Tensor
# noinspection PyPep8Naming
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import image_utils
from config import TrainerConfig
from model import ImageTransformerModel, calc_stats
from normalized_vgg import Vgg19Features


class Dataset(torch.utils.data.Dataset):
    def __init__(self, content_root: str, style_root: str, transform: Callable, is_train: bool):
        self._transform = transform
        sub_folder = 'train' if is_train else 'validation'
        content_dir = Path(content_root) / sub_folder
        style_dir = Path(style_root) / sub_folder
        self._content_paths = image_utils.list_images(content_dir)
        self._style_paths = image_utils.list_images(style_dir)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        while True:
            try:
                content_path = self._content_paths[index]
                content = image_utils.to_pil(image_utils.load(content_path))
                content_t = self._transform(content)

                style_path = self._style_paths[index % len(self._style_paths)]
                style = image_utils.to_pil(image_utils.load(style_path))
                style_t = self._transform(style)

                return content_t, style_t
            except:
                index = random.randint(0, len(self))
                continue

    def __len__(self):
        return len(self._content_paths)


class Trainer:
    def __init__(self, config: TrainerConfig, use_gpu: bool = True):
        self._config = config
        gpu_available = torch.cuda.is_available()
        self._device = 'cuda' if use_gpu and gpu_available else 'cpu'
        self._create_model()
        self._opt = torch.optim.Adam(self._model.decoder.parameters(), config.learning_rate)
        self._lr_scheduler = torch.optim.lr_scheduler.StepLR(self._opt, step_size=1, gamma=config.lr_gamma)
        self._create_data_loader()
        self._create_logdir()
        self._tensorboard = SummaryWriter(self._logdir)
        self._global_step = 0

    def train(self) -> None:
        c = self._config
        print(c)
        self._global_step = 0

        for epoch in range(c.epochs):
            self._train_epoch()
            self._validate(epoch)
            self._save_snapshot(self._global_step)
            self._lr_scheduler.step()

    def _train_epoch(self) -> None:
        c = self._config

        prog_bar = tqdm(self._train_data_loader)
        for i, (content, style) in enumerate(prog_bar):
            content = content.to(self._device)
            style = style.to(self._device)

            stylized, content_norm, style_enc = self._model(content, style)
            loss, content_loss, style_loss = self._calc_loss(stylized, content_norm, style_enc)
            self._step(loss)

            prog_bar.set_description(f'Train loss: {loss.item() :.2f}')
            self._tensorboard.add_scalar('train/loss', loss, self._global_step)
            self._tensorboard.add_scalar('train/content_loss', content_loss, self._global_step)
            self._tensorboard.add_scalar('train/style_loss', style_loss, self._global_step)

            if i % c.visualization_interval == 0:
                self._visualize_images(content, style, stylized, self._global_step, 'train')

            if i != 0 and i % c.snapshot_interval == 0:
                self._save_snapshot(self._global_step)

            self._global_step += 1

    def _validate(self, epoch: int) -> None:
        prog_bar = tqdm(self._validation_data_loader)
        losses = []

        for i, (content, style) in enumerate(prog_bar):
            content = content.to(self._device)
            style = style.to(self._device)

            with torch.no_grad():
                stylized, content_norm, style_enc = self._model(content, style)
                loss, content_loss, style_loss = self._calc_loss(stylized, content_norm, style_enc)

            losses.append(loss)
            prog_bar.set_description(f'Validation loss: {loss:.2f}')

        # noinspection PyUnresolvedReferences,PyTypeChecker
        mean_loss = torch.mean(losses).item()
        self._tensorboard.add_scalar('validation/loss', mean_loss, epoch)

        # noinspection PyUnboundLocalVariable
        self._visualize_images(content, style, stylized, epoch, 'validation')

    def _create_model(self) -> None:
        c = self._config
        self._model = ImageTransformerModel().to(self._device)
        if c.weights_snapshot_path:
            weights = torch.load(c.weights_snapshot_path)
            self._model.load_state_dict(weights, strict=False)

    def _create_data_loader(self) -> None:
        c = self._config

        transform = transforms.Compose([
            transforms.Resize(c.input_images_dim, interpolation=Image.ANTIALIAS),
            transforms.RandomCrop(c.input_images_dim),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        train_dataset = Dataset(c.content_dataset_path, c.style_dataset_path, transform, is_train=True)
        self._train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=c.batch_size,
            shuffle=True,
            num_workers=c.num_data_loader_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_dataset = Dataset(c.content_dataset_path, c.style_dataset_path, transform, is_train=False)
        self._validation_data_loader = DataLoader(
            dataset=val_dataset,
            batch_size=c.batch_size,
            num_workers=c.num_data_loader_workers,
            pin_memory=True,
            drop_last=True,
        )

    def _create_logdir(self) -> None:
        current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self._logdir = Path(self._config.root_logdir) / current_time
        self._logdir.mkdir(parents=True, exist_ok=True)

    def _step(self, loss:  Tensor) -> None:
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()

    def _calc_loss(self,
                   stylized: Tensor,
                   content_norm: Tensor,
                   style_enc:  Vgg19Features) -> Tuple[Tensor, Tensor, Tensor]:
        c = self._config
        stylized_enc = self._model.encode(stylized)

        loss = torch.zeros(1, device=self._device, dtype=torch.float32)
        if c.lambda_content > 0:
            content_loss = F.mse_loss(stylized_enc.relu4_1, content_norm)
            loss = loss + c.lambda_content * content_loss
        else:
            content_loss = 0.0

        def _style_loss(x: Tensor, y: Tensor) -> Tensor:
            x_mu, x_sigma = calc_stats(x)
            y_mu, y_sigma = calc_stats(y)
            _loss = F.mse_loss(x_mu, y_mu) + F.mse_loss(x_sigma, y_sigma)
            return _loss

        if c.lambda_style > 0:
            style_loss = (
                _style_loss(stylized_enc.relu1_1, style_enc.relu1_1) +
                _style_loss(stylized_enc.relu2_1, style_enc.relu2_1) +
                _style_loss(stylized_enc.relu3_1, style_enc.relu3_1) +
                _style_loss(stylized_enc.relu4_1, style_enc.relu4_1)
            )
            loss = loss + c.lambda_style * style_loss
        else:
            style_loss = 0.0

        if c.lambda_tv > 0:
            tv_loss = self._tv_loss(stylized)
            loss = loss + c.lambda_tv * tv_loss
        else:
            tv_loss = 0.0

        return loss, content_loss, style_loss

    @staticmethod
    def _tv_loss(image: Tensor) -> Tensor:
        tv_loss = (image[:, :, :, :-1] - image[:, :, :, 1:]).abs().mean() + \
                  (image[:, :, :-1, :] - image[:, :, 1:, :]).abs().mean()

        return tv_loss

    def _visualize_images(self, content: Tensor, style: Tensor, stylized: Tensor, step: int, tag: str) -> None:
        self._tensorboard.add_images(f'{tag}/content', content, step)
        self._tensorboard.add_images(f'{tag}/style', style, step)
        self._tensorboard.add_images(f'{tag}/stylized', stylized, step)

    def _save_snapshot(self, step: int) -> None:
        output_path = self._logdir / f'step_{step}.pt'
        state = {k: v for k, v in self._model.state_dict().items() if 'encoder' not in k}
        torch.save(state, output_path)


def train(**kwargs):
    config = TrainerConfig(**kwargs)
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    Fire(train)
