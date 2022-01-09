import pathhack
import pickle
import time
from os.path import join, isdir
from os import makedirs
import torch
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from src.mgnn.utils import arg_parse, load_dataset, average_offset_error, final_offset_error, \
    negative_log_likelihood, random_rotate_graph, args2writername
from src.gumbel_social_transformer.temperature_scheduler import Temp_Scheduler


def main(args):
    print('\n\n')
    print('-'*50)
    print('arguments: ', args)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    loader_train = load_dataset(args, pathhack.pkg_path, subfolder='train')
    loader_val = load_dataset(args, pathhack.pkg_path, subfolder='val')
    train_data_loaders = [loader_train, loader_val]
    print('dataset: ', args.dataset)
    writername = args2writername(args)
    print('Config: ', writername)
    logdir = join(pathhack.pkg_path, 'results', writername, args.dataset)
    if isdir(logdir):
        print('Error: The result directory was already created and used.')
        print('-'*50)
        print('\n\n')
        return
    writer = SummaryWriter(logdir=logdir)
    print('-'*50)
    print('\n\n')
    train(args, train_data_loaders, writer, logdir, device=device)
    writer.close()


def train(args, data_loaders, writer, logdir, device='cuda:0'):
    print('-'*50)
    print('Training Phase')
    print('-'*50, '\n')
    loader_train, loader_val = data_loaders
    model = st_model(args, device=device).to(device)
    temperature_scheduler = Temp_Scheduler(args.num_epochs, args.init_temp, args.init_temp, temp_min=0.03)      
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, step_size=50, gamma=0.3)
    print('Model is initialized.')
    print('learning rate: ', args.lr)
    checkpoint_dir = join(logdir, 'checkpoint')
    if not isdir(checkpoint_dir):
        makedirs(checkpoint_dir)
    with open(join(checkpoint_dir, 'args.pickle'), 'wb') as f:
        pickle.dump(args, f)
    print('EPOCHS: ', args.num_epochs)
    print('Training started.\n')
    train_loss_task, train_aoe_task, train_foe_task = [], [], []
    val_loss_task, val_aoe_task, val_foe_task = [], [], []
    for epoch in range(1, args.num_epochs+1):
        model.train()
        epoch_start_time = time.time()
        tau = temperature_scheduler.step()
        train_loss_epoch, train_aoe_epoch, train_foe_epoch, train_loss_mask_epoch = [], [], [], []
        for batch_idx, batch in enumerate(loader_train):
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, loss_mask_rel, loss_mask, \
            v_obs, A_obs, v_pred_gt, A_pred_gt, attn_mask_obs, attn_mask_pred_gt = batch
            if args.rotation_pattern is not None:
                (v_obs, A_obs, v_pred_gt, A_pred_gt), _ = \
                    random_rotate_graph(args, v_obs, A_obs, v_pred_gt, A_pred_gt)
            v_obs, A_obs, v_pred_gt, attn_mask_obs, loss_mask_rel = \
                v_obs.to(device), A_obs.to(device), v_pred_gt.to(device), \
                attn_mask_obs.to(device), loss_mask_rel.to(device)
            if args.deterministic:
                sampling = False
            else:
                sampling = True
            results = model(v_obs, A_obs, attn_mask_obs, loss_mask_rel, tau=tau, hard=False, sampling=sampling, device=device)
            gaussian_params_pred, x_sample_pred, info = results
            loss_mask_per_pedestrian = info['loss_mask_per_pedestrian']
            loss_mask_rel_full_partial = info['loss_mask_rel_full_partial']
            if args.deterministic:
                loss_mask_rel_pred = loss_mask_rel[:,:,-args.pred_seq_len:]
                offset_error_sq, eventual_loss_mask = offset_error_square_full_partial(x_sample_pred, v_pred_gt, loss_mask_rel_full_partial, loss_mask_rel_pred)
                loss = offset_error_sq.sum()/eventual_loss_mask.sum()
            else:
                loss = negative_log_likelihood(gaussian_params_pred, v_pred_gt, loss_mask=loss_mask_per_pedestrian)  
            train_loss_epoch.append(loss.detach().to('cpu').item())
            loss = loss / args.batch_size
            loss.backward()
            aoe = average_offset_error(x_sample_pred, v_pred_gt, loss_mask=loss_mask_per_pedestrian)
            foe = final_offset_error(x_sample_pred, v_pred_gt, loss_mask=loss_mask_per_pedestrian)
            train_aoe_epoch.append(aoe.detach().to('cpu').numpy())
            train_foe_epoch.append(foe.detach().to('cpu').numpy())
            train_loss_mask_epoch.append(loss_mask_per_pedestrian[0].detach().to('cpu').numpy())

            if (batch_idx+1) % args.batch_size == 0 or batch_idx+1 == len(loader_train):
                if args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_grad)
                optimizer.step()
                optimizer.zero_grad()

        lr_scheduler.step()
        train_aoe_epoch, train_foe_epoch, train_loss_mask_epoch = \
            np.concatenate(train_aoe_epoch, axis=0), \
            np.concatenate(train_foe_epoch, axis=0), \
            np.concatenate(train_loss_mask_epoch, axis=0)
        train_loss_epoch, train_aoe_epoch, train_foe_epoch = \
            np.mean(train_loss_epoch), \
            train_aoe_epoch.sum()/train_loss_mask_epoch.sum(), \
            train_foe_epoch.sum()/train_loss_mask_epoch.sum()
        train_loss_task.append(train_loss_epoch)
        train_aoe_task.append(train_aoe_epoch)
        train_foe_task.append(train_foe_epoch)
        training_epoch_period = time.time() - epoch_start_time
        training_epoch_period_per_sample = training_epoch_period/len(loader_train)

        val_loss_epoch, val_aoe_epoch, val_foe_epoch = inference(loader_val, model, args, mode='val', tau=tau, device=device)
        
        print('Epoch: {0} | train loss: {1:.4f} | val loss: {2:.4f} | train aoe: {3:.4f} | val aoe: {4:.4f} | train foe: {5:.4f} | val foe: {6:.4f} | period: {7:.2f} sec | time per sample: {8:.4f} sec'\
                        .format(epoch, train_loss_epoch, val_loss_epoch,\
                        train_aoe_epoch, val_aoe_epoch,\
                        train_foe_epoch, val_foe_epoch,\
                        training_epoch_period, training_epoch_period_per_sample))
        if epoch % 10 == 0:
            model_filename = join(checkpoint_dir, 'epoch_'+str(epoch)+'.pt')
            torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss_epoch,
                    'val_loss': val_loss_epoch,
                    'train_aoe': train_aoe_epoch,
                    'val_aoe': val_aoe_epoch, 
                    'train_foe': train_foe_epoch,
                    'val_foe': val_foe_epoch,   
                    }, model_filename)
            print('epoch_'+str(epoch)+'.pt is saved.')
        
        val_loss_task.append(val_loss_epoch)
        val_aoe_task.append(val_aoe_epoch)
        val_foe_task.append(val_foe_epoch)
        writer.add_scalars('loss', {'train': train_loss_task[-1], 'val': val_loss_task[-1]}, epoch)
        writer.add_scalars('aoe', {'train': train_aoe_task[-1], 'val': val_aoe_task[-1]}, epoch)
        writer.add_scalars('foe', {'train': train_foe_task[-1], 'val': val_foe_task[-1]}, epoch)

    hist = {}
    hist['train_loss'], hist['val_loss'] = train_loss_task, val_loss_task
    hist['train_aoe'], hist['val_aoe'] = train_aoe_task, val_aoe_task
    hist['train_foe'], hist['val_foe'] = train_foe_task, val_foe_task
    with open(join(checkpoint_dir, 'train_hist.pickle'), 'wb') as f:
        pickle.dump(hist, f)
        print(join(checkpoint_dir, 'train_hist.pickle')+' is saved.')
    return

    


def inference(loader, model, args, mode='val', tau=1., device='cuda:0'):
    with torch.no_grad():
        model.eval()
        epoch_start_time = time.time()
        loss_epoch, aoe_epoch, foe_epoch, loss_mask_epoch = [], [], [], []

        for batch_idx, batch in enumerate(loader):
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, loss_mask_rel, loss_mask, \
            v_obs, A_obs, v_pred_gt, A_pred_gt, attn_mask_obs, attn_mask_pred_gt = batch
            v_obs, A_obs, v_pred_gt, attn_mask_obs, loss_mask_rel = \
                v_obs.to(device), A_obs.to(device), v_pred_gt.to(device), \
                attn_mask_obs.to(device), loss_mask_rel.to(device)
            if mode == 'val':
                if args.deterministic:
                    sampling = False
                else:
                    sampling = True
                results = model(v_obs, A_obs, attn_mask_obs, loss_mask_rel, tau=tau, hard=False, sampling=sampling, device=device)
                gaussian_params_pred, x_sample_pred, info = results
                loss_mask_per_pedestrian = info['loss_mask_per_pedestrian']
                loss_mask_rel_full_partial = info['loss_mask_rel_full_partial']
                if args.deterministic:
                    loss_mask_rel_pred = loss_mask_rel[:,:,-args.pred_seq_len:]
                    offset_error_sq, eventual_loss_mask = offset_error_square_full_partial(x_sample_pred, v_pred_gt, loss_mask_rel_full_partial, loss_mask_rel_pred)
                    loss = offset_error_sq.sum()/eventual_loss_mask.sum()
                else:
                    loss = negative_log_likelihood(gaussian_params_pred, v_pred_gt, loss_mask=loss_mask_per_pedestrian)
                aoe = average_offset_error(x_sample_pred, v_pred_gt, loss_mask=loss_mask_per_pedestrian)
                foe = final_offset_error(x_sample_pred, v_pred_gt, loss_mask=loss_mask_per_pedestrian)
            else:
                raise RuntimeError('We now only support val mode.')
            loss_epoch.append(loss.detach().to('cpu').item())
            aoe_epoch.append(aoe.detach().to('cpu').numpy())
            foe_epoch.append(foe.detach().to('cpu').numpy())
            loss_mask_epoch.append(loss_mask_per_pedestrian[0].detach().to('cpu').numpy())

        aoe_epoch, foe_epoch, loss_mask_epoch = \
            np.concatenate(aoe_epoch, axis=0), \
            np.concatenate(foe_epoch, axis=0), \
            np.concatenate(loss_mask_epoch, axis=0)

        loss_epoch, aoe_epoch, foe_epoch = \
            np.mean(loss_epoch), \
            aoe_epoch.sum()/loss_mask_epoch.sum(), \
            foe_epoch.sum()/loss_mask_epoch.sum()
        return loss_epoch, aoe_epoch, foe_epoch


if __name__ == "__main__":
    args = arg_parse()
    if args.temporal == "lstm":
        from src.gumbel_social_transformer.st_model import st_model, offset_error_square_full_partial
    elif args.temporal == "temporal_convolution_net":
        from src.gumbel_social_transformer.st_model_tcn import st_model, offset_error_square_full_partial
    else:
        raise RuntimeError('The temporal component is not lstm nor tcn.')
    main(args)