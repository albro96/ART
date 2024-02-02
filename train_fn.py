import numpy as np
import torch
import torch.nn.functional as F
import os, time
import matplotlib.pyplot as plt
from torch import nn, optim
from models import pointnet
from utils import random_rotate_batch
import sys
import wandb

sys.path.append('/storage/share/code/01_scripts/modules/')
from general_tools.format import format_duration

# , random_rotate_y_batch
# from visualize import output_meshes

def unfold_rotenc(data, rotenc, iters=5):
    R_cum = torch.eye(3).unsqueeze(0).repeat(data.size(0), 1, 1).to(data.device)
    for _ in range(iters):
        R = rotenc(data.transpose(1, 2).contiguous())
        data = torch.matmul(data, R).detach()
        R_cum = torch.matmul(R_cum, R)
    return R_cum

# ----------------------------- Train Single Model ----------------------------- #
                
def calc_loss_rotenc(model, losses, data, opt):
    # Define the rotation function
    random_rotate = random_rotate_batch

    # see chapter 3.2 EQ 4 and 5
    # Apply random rotations to the data (R_tilde)
    data_rot_1, rotmat_1 = random_rotate(data)
    data_rot_2, rotmat_2 = random_rotate(data)
    data_rot_3, rotmat_3 = random_rotate(data)
    
    # Unfold the rotation encodings (R2)
    R_1 = unfold_rotenc(data_rot_1, model, opt.iters)
    R_2 = unfold_rotenc(data_rot_2, model, opt.iters)
    R_3 = unfold_rotenc(data_rot_3, model, opt.iters)

    # Unfold the rotation encoding of the original data (R1)
    R = unfold_rotenc(data, model, opt.iters)

    # Compute the product of the rotation matrices EQ = R2^T * R1
    rotprod_1 = torch.matmul(R, R_1.transpose(1, 2))
    rotprod_2 = torch.matmul(R, R_2.transpose(1, 2))
    rotprod_3 = torch.matmul(R, R_3.transpose(1, 2))

    # Compute the mean squared error loss between the rotation matrices and their products
    rot_loss_mse = (F.mse_loss(rotmat_1, rotprod_1) + \
                    F.mse_loss(rotmat_2, rotprod_2) + \
                    F.mse_loss(rotmat_3, rotprod_3)) / 3

    # Compute the chamfer loss between the rotated data and the product of the original data and the rotation matrices
    with torch.cuda.device(data.device):
        rot_loss_chamfer = (losses['chamfer'](torch.matmul(data, rotprod_1), data_rot_1) + \
                            losses['chamfer'](torch.matmul(data, rotprod_2), data_rot_2) + \
                            losses['chamfer'](torch.matmul(data, rotprod_3), data_rot_3)) / 3

    # return chamfer_dist, rot_loss_mse, rot_loss_chamfer
    return rot_loss_mse, rot_loss_chamfer


def train_single_model(model, losses, optimizer, train_loader, val_loader, device, opt, save_path=None):
    
    end_epoch = opt.end_epoch
    start_epoch = 0
    log_step = 1
    best_loss = 1000

    ckpt_files = sorted(os.listdir(save_path))

    if opt.resume and len(ckpt_files) > 0:
        ckpt_file = ckpt_files[-1]
        ckpt = torch.load(os.path.join(save_path, ckpt_file), map_location=device)

        model.load_state_dict(ckpt)
        optimizer.load_state_dict(ckpt)

        start_epoch = int(ckpt_file.split('.')[0])

    print('Training started')
    epoch_times = []

    for epoch in range(start_epoch, end_epoch+1):
        epoch_t0 = time.time()

        # load val and train into cache if specified
        if train_loader.dataset.cache_data and not train_loader.dataset.use_cache:
            print(f'[Train] Filling cache...')
            t0 = time.time()
            for i, data in enumerate(train_loader):
                pass
            print(f'[Train] Filling cache took {format_duration(time.time()-t0)}',)

        if val_loader.dataset.cache_data and not val_loader.dataset.use_cache:
            print(f'[Val] Filling cache...')
            t0 = time.time()
            for i, data in enumerate(val_loader):
                pass
            print(f'[Val] Filling cache took {format_duration(time.time()-t0)}')


        if epoch==0:
            train_loader.dataset.set_use_cache(True)
            val_loader.dataset.set_use_cache(True)

    
        t1 = time.time()
        print('\nEpoch {}/{}: Training'.format(epoch, end_epoch))

        model.train()

        train_loss_dict = {'total_loss': 0, 'rot_loss_mse': 0, 'rot_loss_chamfer': 0}
        val_loss_dict = {'total_loss': 0, 'rot_loss_mse': 0, 'rot_loss_chamfer': 0}

        for i, data in enumerate(train_loader): 
            print('\rBatch {}/{}'.format(i+1, len(train_loader)), end='', flush=True)

            data = data.to(device)

            # Zero the gradients of the optimizer
            optimizer.zero_grad()

            rot_loss_mse, rot_loss_chamfer = calc_loss_rotenc(model, losses, data, opt)

            # Compute the total loss
            # loss =  rot_loss_mse * 0.02 + rot_loss_chamfer * opt.lambda2
            loss =  rot_loss_mse + rot_loss_chamfer*opt.lambda_cd

            # Backpropagate the loss
            loss.backward()

            # Update the model parameters
            optimizer.step()

            train_loss_dict['total_loss'] += loss * data.size(0) 
            train_loss_dict['rot_loss_mse'] += rot_loss_mse.item() * data.size(0)
            train_loss_dict['rot_loss_chamfer'] += rot_loss_chamfer.item() * data.size(0)

        wandb.log(
            {
                "train/epoch": epoch,
                "train/total_loss": train_loss_dict['total_loss'] / len(train_loader.dataset),
                "train/rot_loss_mse": train_loss_dict['rot_loss_mse'] / len(train_loader.dataset),
                "train/rot_loss_chamfer": train_loss_dict['rot_loss_chamfer'] / len(train_loader.dataset), 
            })
        print()
        for term in train_loss_dict:
            train_loss = train_loss_dict[term] / len(train_loader.dataset)
            
            print('\t{}: {:.5f}'.format(term, train_loss))
          
          
        if epoch % opt.val_step == 0:
            print('\nEpoch {}/{}: Validation'.format(epoch, end_epoch))

            model.eval()

            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    print('\rBatch {}/{}'.format(i+1, len(val_loader)), end='', flush=True)
                    data = data.to(device)

                    rot_loss_mse, rot_loss_chamfer = calc_loss_rotenc(model, losses, data, opt)
                    loss =  rot_loss_mse + rot_loss_chamfer*opt.lambda_cd

                    val_loss_dict['total_loss'] += loss * data.size(0)
                    val_loss_dict['rot_loss_mse'] += rot_loss_mse.item() * data.size(0)
                    val_loss_dict['rot_loss_chamfer'] += rot_loss_chamfer.item() * data.size(0)

    
            print('Epoch {}/{}: Validation'.format(epoch, end_epoch), flush=True)
            wandb.log(
            {
                "val/epoch": epoch,
                "val/total_loss": val_loss_dict['total_loss'] / len(val_loader.dataset),
                "val/rot_loss_mse": val_loss_dict['rot_loss_mse'] / len(val_loader.dataset),
                "val/rot_loss_chamfer": val_loss_dict['rot_loss_chamfer'] / len(val_loader.dataset), 
            })
            print()
            for term in val_loss_dict:
                val_loss = val_loss_dict[term] / len(val_loader.dataset)
                print('\t{}: {:.5f}'.format(term, val_loss))


        checkpoint = {
                'm': model.state_dict() if model else None,
                'o': optimizer.state_dict(),
                'torch_rnd': torch.get_rng_state(),
                'numpy_rnd': np.random.get_state()
            }

        val_total_loss = val_loss_dict['total_loss'] / len(val_loader.dataset)

        if val_total_loss < best_loss:
            best_loss = val_total_loss
            torch.save(checkpoint, os.path.join(save_path, 'best.pth'))

        if epoch % opt.epoch_save == 0:
            torch.save(checkpoint, os.path.join(save_path, '{}.pth'.format(epoch)))

        epoch_times.append(time.time() - epoch_t0)
        mean_epoch_time = np.mean(epoch_times)

        print(f'Mean epoch time: {format_duration(mean_epoch_time)}.')
        print(f'Estimated remaining time: {format_duration(mean_epoch_time*(end_epoch-epoch))}')


# def train_rotenc(model, losses, optimizer, data, epoch, opt):
#     # Zero the gradients of the optimizer
#     optimizer.zero_grad()

#     # Define the rotation function
#     random_rotate = random_rotate_batch

#     # see chapter 3.2 EQ 4 and 5
#     # Apply random rotations to the data (R_tilde)
#     data_rot_1, rotmat_1 = random_rotate(data)
#     data_rot_2, rotmat_2 = random_rotate(data)
#     data_rot_3, rotmat_3 = random_rotate(data)
    
#     # Unfold the rotation encodings (R2)
#     R_1 = unfold_rotenc(data_rot_1, model, opt.iters)
#     R_2 = unfold_rotenc(data_rot_2, model, opt.iters)
#     R_3 = unfold_rotenc(data_rot_3, model, opt.iters)

#     # Unfold the rotation encoding of the original data (R1)
#     R = unfold_rotenc(data, model, opt.iters)

#     # Compute the product of the rotation matrices EQ = R2^T * R1
#     rotprod_1 = torch.matmul(R, R_1.transpose(1, 2))
#     rotprod_2 = torch.matmul(R, R_2.transpose(1, 2))
#     rotprod_3 = torch.matmul(R, R_3.transpose(1, 2))

#     # Compute the mean squared error loss between the rotation matrices and their products
#     rot_loss_mse = (F.mse_loss(rotmat_1, rotprod_1) + \
#                     F.mse_loss(rotmat_2, rotprod_2) + \
#                     F.mse_loss(rotmat_3, rotprod_3)) / 3

#     # Compute the chamfer loss between the rotated data and the product of the original data and the rotation matrices
#     with torch.cuda.device(data.device):
#         rot_loss_chamfer = (losses['chamfer'](torch.matmul(data, rotprod_1), data_rot_1) + \
#                             losses['chamfer'](torch.matmul(data, rotprod_2), data_rot_2) + \
#                             losses['chamfer'](torch.matmul(data, rotprod_3), data_rot_3)) / 3


#     # # Compute the chamfer distance between the original data and the target data
#     # with torch.cuda.device(data.device):
#     #     chamfer_dist = losses['chamfer'](data, y)

#     # Compute the total loss
#     # loss =  rot_loss_mse * 0.02 + rot_loss_chamfer * opt.lambda2
#     loss =  rot_loss_mse+ rot_loss_chamfer

#     # loss = chamfer_dist + rot_loss_mse * 0.02 + rot_loss_chamfer * opt.lambda2

#     # Backpropagate the loss
#     loss.backward()

#     # Update the model parameters
#     optimizer.step()

#     # return chamfer_dist, rot_loss_mse, rot_loss_chamfer
#     return rot_loss_mse, rot_loss_chamfer

def train_autoencoder(models, losses, optimizers, data, epoch, opt):
    optimizers['opt'].zero_grad()

    # if opt.azimuthal:
    #     random_rotate = random_rotate_y_batch
    # else:
    random_rotate = random_rotate_batch

    if opt.art:
        data_rot_1, rotmat_1 = random_rotate(data)
        data_rot_2, rotmat_2 = random_rotate(data)
        data_rot_3, rotmat_3 = random_rotate(data)
        R_1 = unfold_rotenc(data_rot_1, models['rot_enc'], opt.iters)
        R_2 = unfold_rotenc(data_rot_2, models['rot_enc'], opt.iters)
        R_3 = unfold_rotenc(data_rot_3, models['rot_enc'], opt.iters)

        R = unfold_rotenc(data, models['rot_enc'], opt.iters)

        rotprod_1 = torch.matmul(R, R_1.transpose(1, 2))
        rotprod_2 = torch.matmul(R, R_2.transpose(1, 2))
        rotprod_3 = torch.matmul(R, R_3.transpose(1, 2))

        rot_loss_mse = (F.mse_loss(rotmat_1, rotprod_1) + \
                        F.mse_loss(rotmat_2, rotprod_2) + \
                        F.mse_loss(rotmat_3, rotprod_3)) / 3
        

        z = models['enc'](data, R)
        y = models['dec'](z, R)

        with torch.cuda.device(data.device):
            rot_loss_chamfer = (losses['chamfer'](torch.matmul(data, rotprod_1), data_rot_1) + \
                                losses['chamfer'](torch.matmul(data, rotprod_2), data_rot_2) + \
                                losses['chamfer'](torch.matmul(data, rotprod_3), data_rot_3)) / 3
    elif opt.itn:
        R = unfold_rotenc(data, models['rot_enc'], opt.iters)
        z = models['enc'](data, R)
        y = models['dec'](z, R)
    elif opt.tnet:
        R = models['rot_enc'](data.transpose(1, 2).contiguous())
        z = models['enc'](data, R)
        y = models['dec'](z, torch.inverse(R))
    else:
        data, _ = random_rotate(data)

        z = models['enc'](data)
        y = models['dec'](z)        

    with torch.cuda.device(data.device):
        chamfer_dist = losses['chamfer'](data, y)

    if opt.art:
        loss = chamfer_dist + rot_loss_mse * 0.02 + rot_loss_chamfer * opt.lambda2

    else:
        loss = chamfer_dist
        rot_loss_mse = torch.tensor(0)
        rot_loss_chamfer = torch.tensor(0)

    loss.backward()

    optimizers['opt'].step()

    return chamfer_dist, rot_loss_mse, rot_loss_chamfer

def train_model(models, losses, optimizers, train_loader, val_loader, device, opt, save_path=None):
    num_epochs = 500
    start_epoch = 1
    vis_step = 500
    log_step = 1
    best_loss = 1000

    ckpt_files = sorted(os.listdir(save_path))
    if opt.resume and len(ckpt_files) > 0:
        ckpt_file = ckpt_files[-1]
        ckpt = torch.load(os.path.join(save_path, ckpt_file), map_location=device)
        for k in models:
            if models[k]:
                models[k].load_state_dict(ckpt['m_'+k])
        for k in optimizers:
            optimizers[k].load_state_dict(ckpt['o_'+k])
        start_epoch = int(ckpt_file.split('.')[0]) + 1

    print('Training started')
    # print('azimuthal?', opt.azimuthal)

    for epoch in range(start_epoch, 1+num_epochs):
        t1 = time.time()
        print ('Epoch {}/{}'.format(epoch, num_epochs))

        models['enc'].train()
        models['dec'].train()
        if opt.art:
            models['rot_enc'].train()

        # train_loader.dataset.resample()

        train_loss_dict = {'chamfer_dist': 0, 'rot_loss_mse': 0, 'rot_loss_chamfer': 0}
        vald_loss_dict = {'chamfer_dist': 0}

        for i, data in enumerate(train_loader):
            print('\tBatch {}/{}'.format(i+1, len(train_loader)), end='\r', flush=True)

            data = data.to(device)

            recon_loss, rot_loss_mse, rot_loss_chamfer = train_autoencoder(models, losses, optimizers, data, epoch, opt)

            train_loss_dict['chamfer_dist'] += recon_loss.item() * data.size(0)
            train_loss_dict['rot_loss_mse'] += rot_loss_mse.item() * data.size(0)
            train_loss_dict['rot_loss_chamfer'] += rot_loss_chamfer.item() * data.size(0)

        t2 = time.time()
        print(t2-t1)

        if epoch > 0:
            models['enc'].eval()
            models['dec'].eval()
            if opt.art:
                models['rot_enc'].eval()

            with torch.no_grad():
                for batch_idx, x in enumerate(val_loader):
                    x = x.to(device)
                    if opt.art:
                        R = unfold_rotenc(x, models['rot_enc'], opt.iters)
                        z = models['enc'](x, R)
                        y = models['dec'](z, R)                    
                    else:
                        z = models['enc'](x)
                        y = models['dec'](z)

                    with torch.cuda.device(device):
                        recon_loss = losses['chamfer'](x, y)

                    vald_loss_dict['chamfer_dist'] += recon_loss.item() * x.size(0)

                    # if epoch % vis_step == 0 and batch_idx == 0:
                    #     x = x.cpu().numpy().reshape(x.shape[0], 1, -1, 3)
                    #     y = y.cpu().numpy().reshape(y.shape[0], 1, -1, 3)
                    #     meshes = np.concatenate([x, y], axis=1)
                    #     output_meshes(meshes, epoch)
                

            if epoch % log_step == 0:
                print('Epoch {}/{}: Training'.format(epoch, num_epochs), flush=True)
                
                for term in train_loss_dict:
                    print('\t{} {:.5f}'.format(term, train_loss_dict[term] / len(train_loader.dataset)), flush=True)
                
                vald_loss = vald_loss_dict['chamfer_dist'] / len(val_loader.dataset)
                print('Epoch {}/{}: Validation'.format(epoch, num_epochs), flush=True)
                for term in vald_loss_dict:
                    print('\t{} {:.5f}'.format(term, vald_loss), flush=True)

            if vald_loss < best_loss:
                best_loss = vald_loss
                checkpoint = dict([('m_'+t, models[t].state_dict() if models[t] else None) for t in models])
                checkpoint.update(dict([('o_'+t, optimizers[t].state_dict()) for t in optimizers]))
                checkpoint.update({'torch_rnd': torch.get_rng_state(), 'numpy_rnd': np.random.get_state()})
                torch.save(checkpoint, os.path.join(save_path, '{}.pth'.format(epoch)))

