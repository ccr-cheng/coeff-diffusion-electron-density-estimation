import argparse
import os

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

torch.multiprocessing.set_sharing_strategy('file_system')

from datasets import get_dataset, DensityCollator
from models import get_model
from utils import load_config, seed_all, get_optimizer, get_scheduler, count_parameters
from visualize import draw_stack

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file')
    parser.add_argument('--mode', type=str, choices=['train', 'inf'], default='train', help='running mode')
    parser.add_argument('--device', type=str, default='cuda', help='running device')
    parser.add_argument('--logdir', type=str, default='./logs', help='log directory')
    parser.add_argument('--savename', type=str, default='test', help='running save name')
    parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint')
    parser.add_argument('--resume-predictor-only', action='store_true', help='only resume predictor but not diffnet')
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    seed_all(config.train.seed)
    print(config)
    logdir = os.path.join(args.logdir, args.savename)
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    # Data
    print('Loading datasets...')
    train_collator = DensityCollator(config.train.train_samples)
    val_collator = DensityCollator(config.train.val_samples)
    inf_collator = DensityCollator()
    train_set, test_set = get_dataset(config.datasets)
    train_loader = DataLoader(train_set, config.train.batch_size, shuffle=False,
                              num_workers=32, collate_fn=train_collator)
    val_loader = DataLoader(test_set, config.train.batch_size, shuffle=True,
                            num_workers=32, collate_fn=val_collator)
    inf_loader = DataLoader(test_set, 2, shuffle=True, num_workers=2, collate_fn=inf_collator)

    # Model
    print('Building model...')
    predictor, diffnet, orbital = get_model(config.model, args.device)
    print(f'Number of parameters in predictor: {count_parameters(predictor)}')
    print(f'Number of parameters in diffnet: {count_parameters(diffnet)}')
    freeze_predictor = config.train.freeze_predictor
    if freeze_predictor:
        predictor.eval()

    # Optimizer & Scheduler
    optimizer_pred = get_optimizer(config.train.optimizer.predictor, predictor)
    scheduler_pred = get_scheduler(config.train.scheduler.predictor, optimizer_pred)
    criterion_pred = nn.MSELoss().to(args.device)
    optimizer_pred.zero_grad()
    optimizer_diff = get_optimizer(config.train.optimizer.diffusion, diffnet)
    scheduler_diff = get_scheduler(config.train.scheduler.diffusion, optimizer_diff)
    optimizer_diff.zero_grad()

    # Resume
    if args.resume is not None:
        print(f'Resuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=args.device)
        predictor.load_state_dict(ckpt['predictor'])
        if not args.resume_predictor_only:
            diffnet.load_state_dict(ckpt['diffnet'])
        if 'optimizer' in ckpt:
            print('Resuming optimizer states...')
            optimizer_pred.load_state_dict(ckpt['optimizer']['predictor'])
            if not args.resume_predictor_only:
                optimizer_diff.load_state_dict(ckpt['optimizer']['diffusion'])
        if 'scheduler' in ckpt:
            print('Resuming scheduler states...')
            scheduler_pred.load_state_dict(ckpt['scheduler']['predictor'])
            if not args.resume_predictor_only:
                scheduler_diff.load_state_dict(ckpt['scheduler']['diffusion'])

    global_step = 0


    def scheduler_step(scheduler, loss):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss)
        else:
            scheduler.step()


    def train():
        global global_step, freeze_predictor

        epoch = 0
        while True:
            if not freeze_predictor:
                predictor.train()
            diffnet.train()
            epoch_loss_pred, epoch_loss_diff = [], []
            for g, density, grid_coord, infos in train_loader:
                g = g.to(args.device)
                density, grid_coord = density.to(args.device), grid_coord.to(args.device)

                # predict density
                if freeze_predictor:
                    torch.set_grad_enabled(False)
                coeff = predictor(g.x, g.pos, g.batch)
                pred = orbital(coeff, g.pos, grid_coord, g.batch)

                # optimize predictor
                loss_pred = criterion_pred(pred, density)
                mae = torch.abs(pred.detach() - density).sum() / density.sum()
                epoch_loss_pred.append(loss_pred.item())
                if not freeze_predictor:
                    loss_pred.backward()
                    grad_norm_pred = clip_grad_norm_(predictor.parameters(), config.train.max_grad_norm)
                    writer.add_scalar('train/grad_pred', grad_norm_pred.item(), global_step)
                    optimizer_pred.step()
                    optimizer_pred.zero_grad()

                # optimize diffusion
                coeff = coeff.detach()
                torch.set_grad_enabled(True)
                loss_diff = diffnet(coeff, g.x, g.pos, g.batch)
                epoch_loss_diff.append(loss_diff.item())
                loss_diff.backward()
                grad_norm_diff = clip_grad_norm_(diffnet.parameters(), config.train.max_grad_norm)
                optimizer_diff.step()
                optimizer_diff.zero_grad()

                # Logging
                writer.add_scalar('train/loss_pred', loss_pred.item(), global_step)
                writer.add_scalar('train/mae_pred', mae.item(), global_step)
                writer.add_scalar('train/lr_pred', optimizer_pred.param_groups[0]['lr'], global_step)
                writer.add_scalar('train/loss_diff', loss_diff.item(), global_step)
                writer.add_scalar('train/grad_diff', grad_norm_diff.item(), global_step)
                writer.add_scalar('train/lr_diff', optimizer_diff.param_groups[0]['lr'], global_step)
                if global_step % config.train.log_freq == 0:
                    print(f'Epoch {epoch} Step {global_step} predict train loss {loss_pred.item():.6f},'
                          f' train mae {mae.item():.6f}, diffusion train loss {loss_diff.item():.6f}')

                global_step += 1
                if global_step % config.train.val_freq == 0:
                    val_loss_pred, val_loss_diff = validate(val_loader)
                    inference(inf_loader, 1, config.test.num_vis, config.test.inf_samples)

                    if not freeze_predictor:
                        scheduler_step(scheduler_pred, val_loss_pred)
                        predictor.train()
                    scheduler_step(scheduler_diff, val_loss_diff)
                    diffnet.train()
                    torch.save({
                        'predictor': predictor.state_dict(),
                        'diffnet': diffnet.state_dict(),
                        'step': global_step,
                        'config': config,
                    }, os.path.join(logdir, 'latest.pt'))
                    if global_step % config.train.save_freq == 0:
                        ckpt_path = os.path.join(logdir, f'{global_step}.pt')
                        torch.save({
                            'config': config,
                            'predictor': predictor.state_dict(),
                            'diffnet': diffnet.state_dict(),
                            'optimizer': {
                                'predictor': optimizer_pred.state_dict(),
                                'diffusion': optimizer_diff.state_dict(),
                            },
                            'scheduler': {
                                'predictor': scheduler_pred.state_dict(),
                                'diffusion': scheduler_diff.state_dict(),
                            },
                            'val_loss_pred': val_loss_pred,
                            'val_loss_diff': val_loss_diff,
                        }, ckpt_path)
                if global_step >= config.train.max_iter:
                    return
                if global_step >= config.train.max_predictor_iter:
                    freeze_predictor = True

            # epoch_loss_pred = sum(epoch_loss_pred) / len(epoch_loss_pred)
            # epoch_loss_diff = sum(epoch_loss_diff) / len(epoch_loss_diff)
            # print(f'Epoch {epoch} predict train loss {epoch_loss_pred:.6f},'
            #       f' diffusion train loss {epoch_loss_diff:.6f}')
            epoch += 1


    @torch.no_grad()
    def validate(dataloader, split='val'):
        predictor.eval()
        diffnet.eval()

        val_loss_pred, val_loss_diff = [], []
        val_mae, val_cnt = 0., 0.
        for g, density, grid_coord, infos in tqdm(dataloader, total=len(dataloader)):
            g = g.to(args.device)
            density, grid_coord = density.to(args.device), grid_coord.to(args.device)

            coeff = predictor(g.x, g.pos, g.batch)
            pred = orbital(coeff, g.pos, grid_coord, g.batch)
            loss_pred = criterion_pred(pred, density)
            val_loss_pred.append(loss_pred.item())
            val_mae += torch.abs(pred - density).sum().item()
            val_cnt += density.sum().item()

            coeff = coeff.detach()
            loss_diff = diffnet(coeff, g.x, g.pos, g.batch)
            val_loss_diff.append(loss_diff.item())
        val_loss_pred = sum(val_loss_pred) / len(val_loss_pred)
        val_loss_diff = sum(val_loss_diff) / len(val_loss_diff)
        val_mae = val_mae / val_cnt

        writer.add_scalar(f'{split}/loss_pred', val_loss_pred, global_step)
        writer.add_scalar(f'{split}/mae_pred', val_mae, global_step)
        writer.add_scalar(f'{split}/loss_diff', val_loss_diff, global_step)
        print(f'Step {global_step} {split} predict loss {val_loss_pred:.6f}, {split} mae {val_mae:.6f},'
              f' {split} diffusion loss {val_loss_diff:.6f}')
        return val_loss_pred, val_loss_diff


    @torch.no_grad()
    def orbital_expansion(coeff, g, density, grid_coord, grid_batch_size=None):
        if grid_batch_size is None:
            preds = orbital(coeff, g.pos, grid_coord, g.batch)
        else:
            preds = []
            for grid in grid_coord.split(grid_batch_size, dim=1):
                preds.append(orbital(coeff, g.pos, grid, g.batch))
            preds = torch.cat(preds, dim=1)
        mask = (density > 0).float()
        preds = preds * mask
        density = density * mask
        diff = torch.abs(preds - density)
        loss = diff.pow(2).sum(-1) / mask.sum(-1)
        mae = diff.sum(-1) / density.sum(-1)
        return preds, loss, mae


    @torch.no_grad()
    def inference(dataloader, num_infer=None, num_vis=2, samples=None):
        predictor.eval()
        diffnet.eval()
        inf_loss_pred, inf_mae_pred = [], []
        inf_loss_diff, inf_mae_diff = [], []
        num_infer = num_infer or len(dataloader)
        for idx, (g, density, grid_coord, infos) in enumerate(dataloader):
            if idx >= num_infer:
                break

            g = g.to(args.device)
            density, grid_coord = density.to(args.device), grid_coord.to(args.device)
            coeff = predictor(g.x, g.pos, g.batch)
            preds, loss_pred, mae_pred = orbital_expansion(
                coeff, g, density, grid_coord, grid_batch_size=samples
            )

            traj = diffnet.sample(g.x, g.pos, g.batch)
            coeff_diff = traj[:, -1]
            preds_diff, loss_diff, mae_diff = orbital_expansion(
                coeff_diff, g, density, grid_coord, grid_batch_size=samples
            )

            inf_loss_pred.append(loss_pred.detach().cpu().numpy())
            inf_mae_pred.append(mae_pred.detach().cpu().numpy())
            inf_loss_diff.append(loss_diff.detach().cpu().numpy())
            inf_mae_diff.append(mae_diff.detach().cpu().numpy())

            if idx == 0:
                for vis_idx, (p, pd, d, info) in enumerate(zip(preds, preds_diff, density, infos)):
                    if vis_idx >= num_vis:
                        break

                    shape = info['shape']
                    mask = g.batch == vis_idx
                    atom_type, coord = g.x[mask], g.pos[mask]
                    grid_cell = (info['cell'] / torch.FloatTensor(shape).view(3, 1)).to(args.device)
                    coord = coord @ torch.linalg.inv(grid_cell)
                    num_voxel = shape[0] * shape[1] * shape[2]
                    d, p, pd = d[:num_voxel].view(*shape), p[:num_voxel].view(*shape), pd[:num_voxel].view(*shape)
                    writer.add_image(f'inf/d_pred_{vis_idx}', draw_stack(d - p, atom_type, coord), global_step)
                    writer.add_image(f'inf/d_diff_{vis_idx}', draw_stack(d - pd, atom_type, coord), global_step)
        inf_loss_pred = np.concatenate(inf_loss_pred, axis=0).mean()
        inf_mae_pred = np.concatenate(inf_mae_pred, axis=0).mean()
        inf_loss_diff = np.concatenate(inf_loss_diff, axis=0).mean()
        inf_mae_diff = np.concatenate(inf_mae_diff, axis=0).mean()
        writer.add_scalar('inf/loss_pred', inf_loss_pred, global_step)
        writer.add_scalar('inf/mae_pred', inf_mae_pred, global_step)
        writer.add_scalar('inf/loss_diff', inf_loss_diff, global_step)
        writer.add_scalar('inf/mae_diff', inf_mae_diff, global_step)
        print(f'Step {global_step} inf predict loss {inf_loss_pred:.6f}, predict mae {inf_mae_pred:.6f},'
              f' diffusion loss {inf_loss_diff:.6f}, diffusion mae {inf_mae_diff:.6f}')


    try:
        if args.mode == 'train':
            # inference(inf_loader, 1, config.test.num_vis, config.test.inf_samples)
            train()
            print('Training finished!')

        if args.mode == 'inf' and args.resume is None:
            print('[WARNING]: inference mode without loading a pretrained model')
        test_loader = DataLoader(test_set, config.test.batch_size, shuffle=False,
                                 num_workers=16, collate_fn=inf_collator)
        inference(test_loader, config.test.num_infer, config.test.num_vis, config.test.inf_samples)
    except KeyboardInterrupt:
        print('Terminating...')
