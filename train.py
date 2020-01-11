import os
from collections import defaultdict

import numpy
import pscp
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.utils import save_image

from models import PixelCNN


def nat2bit(val):
    return torch.log2(torch.exp(val))


def loop(ctx, phase):
    device = ctx['device']
    dataloader = ctx['dataloaders'][phase]
    optimizer = ctx['optimizer']
    # scheduler = ctx['scheduler']
    model = ctx['model']
    # logger = ctx['logger']
    epoch = ctx['epoch']
    out_dir = ctx['out_dir']
    train_step = ctx.get('train_step', 0)
    print(f'{ctx["pscp"]}/Epoch {epoch}/{phase}')

    is_train = phase == 'train'
    model.train(is_train)

    if model.format == 'bit':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    nll_loss_criterion = nn.NLLLoss()
    metrics = defaultdict(lambda: 0.)
    num_metrics = 0

    def log_metrics():
        nonlocal num_metrics
        mean_metrics = {k: v / num_metrics for k, v in metrics.items()}
        # mean_metrics['loss'] = nat2bit(mean_metrics['loss'])
        # unit['loss'] = 'bpd'
        unit = defaultdict(str)

        for k, v in mean_metrics.items():
            print(f'[{epoch}/{train_step}] {k}: {v:.4f} {unit[k]}')
            # logger.add_scalar(f'{phase}/{k}', v, train_step)

        metrics.clear()
        num_metrics = 0
        return mean_metrics

    os.makedirs(out_dir, exist_ok=True)

    for (x, y), _ in dataloader:
        x = x.to(device, torch.long, non_blocking=True).permute(0, 3, 1, 2)
        y = y.to(device, torch.float, non_blocking=True).permute(0, 3, 4, 1, 2)
        n_size, c_size, num_bits, h_size, w_size = y.shape
        reshaped_y = y.view(n_size, c_size * num_bits, h_size, w_size)

        if model.format == 'bit':
            logits = model(reshaped_y)
            loss = criterion(logits, reshaped_y)
        else:
            logits = model(x)
            nll_loss = loss = criterion(logits, x)

        if is_train:
            optimizer.zero_grad()

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()

            optimizer.step()
            # scheduler.step()
            train_step += 1

        with torch.no_grad():
            if model.format == 'bit':
                log_probs = model.logprob_packbits(logits.flip(1))
                # log_probs = model.logprob_packbits(logits)
                nll_loss = nll_loss_criterion(log_probs.transpose(1, 2), x)
            else:
                log_probs = logits

            num_metrics += 1
            metrics['loss'] += nll_loss

            if is_train:
                metrics['bce_loss'] += loss

            if is_train and train_step % 100 == 99:
                loss = log_metrics()['loss']
                postfix = f'_{loss:.3f}_{ctx["pscp"]}'

                sample = [
                    x.float() / 127.5 - 1.,
                    log_probs.argmax(1) / 127.5 - 1.,
                ]

                if train_step % 10000 == 999 and False:
                    postfix += '_s'
                    sample_ = model.sample(1).view(-1, c_size, num_bits, h_size, w_size)
                    sample_ = sample_.cpu().numpy()[:, ::-1]
                    # sample_ = sample_.cpu().numpy()
                    sample_ = numpy.packbits(sample_ == 1., axis=2).squeeze(2)
                    sample.append(torch.from_numpy(sample_).to(device) / 127.5 - 1.)

                # if i % (m * 10) == 0:
                #     resultsample.append(vae.decode(z))

                sample = torch.cat(sample, dim=0)
                sample = (sample * 0.5 + 0.5).cpu()
                save_image(
                    sample,
                    f'{out_dir}/sample_{epoch:02d}_{train_step:06d}{postfix}.png',
                )

    newctx = {**ctx, 'train_step': train_step}

    if not is_train:
        newctx['val_metrics'] = log_metrics()

    return newctx


def transform(image):
    image = numpy.array(image)

    if len(image.shape) == 2:
        image = numpy.expand_dims(image, -1)

    binary_image = numpy.unpackbits(numpy.expand_dims(image, axis=-1), axis=-1)
    binary_image = numpy.ascontiguousarray(binary_image[..., ::-1])
    return image, binary_image


def main():
    ctx = {'pscp': pscp.create()}
    ctx['device'] = device = torch.device('cuda')
    datasets = {
        # phase: torchvision.datasets.MNIST(
        phase: torchvision.datasets.CIFAR10(
            'datasets',
            train=phase == 'train',
            download=True,
            transform=transform,
        )
        for phase in ('train', 'val')
    }
    ctx['out_dir'] = f'results_{datasets["train"].__class__.__name__}'
    ctx['dataloaders'] = {
        phase: torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        for phase, dataset in datasets.items()
    }
    model = PixelCNN(3, 32, 32, format='softmax')
    # model = PixelCNN(3 * 8, 32, 32)
    print(f'r = {model.r}')
    model.to(device)
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=1e-3,
    )
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    ctx['model'], ctx['optimizer'] = model, optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        verbose=True,
        min_lr=1e-4,
    )
    '''
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=5e-6,
        max_lr=5e-3,
        cycle_momentum=False,
        step_size_up=2500,
    )
    '''
    ctx['scheduler'] = scheduler

    for epoch in range(250):
        ctx['epoch'] = epoch
        print(f'lr: {optimizer.param_groups[0]["lr"]}')
        ctx = loop(ctx, 'train')

        with torch.no_grad():
            ctx = loop(ctx, 'val')
        """
        for _ in loop(ctx, 'train'):
            with torch.no_grad():
                list(loop(ctx, 'val'))

        """
        val_loss = ctx['val_metrics']['loss'].item()
        print(f'val_loss: {val_loss:.4f}')
        scheduler.step(val_loss)


if __name__ == '__main__':
    main()
