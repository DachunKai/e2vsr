from cmath import isnan
import os
import torch
from collections import Counter
from collections import OrderedDict
from os import path as osp
from torch import distributed as dist
from tqdm import tqdm

from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.data.transforms import mod_crop
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel
from torchvision import utils as vutils

def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)


@MODEL_REGISTRY.register()
class E2VSRModel(VideoBaseModel):

    def __init__(self, opt):
        super(E2VSRModel, self).__init__(opt)
        self.scale = opt['scale']
        if self.is_train:
            self.fix_flow_iter = opt['train'].get('fix_flow')

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'key' in data:
            self.key = data['key']
            # self.neighbor_list = data['neighbor_list']
            # rank, _ = get_dist_info()
            # if rank == 0:
            #     print("current key: ", self.key, ' ', "neighbor_list: ", self.neighbor_list)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'event_lq' in data:
            self.event_lq = data['event_lq'].to(self.device)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate flow params and normal params for different lr
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                if 'spynet' in name:
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': flow_params,
                    'lr': train_opt['optim_g']['lr'] * flow_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        if self.fix_flow_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'edvr' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.event_lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            if torch.isnan(l_pix).any():
                print("self.key: ", self.key)
                print("self.output.shape: ", self.output.shape, '\t', "self.gt.shape: ", self.gt.shape)
                print("torch.isnan(self.output).any(): ", torch.isnan(self.output).any())
                print("torch.isnan(self.gt).any(): ", torch.isnan(self.gt).any())
                raise ValueError(f"l_pix is nan. Please check.")
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None
        # initialize self.metric_results
        # It is a dict: {
        #    'folder1': tensor (num_frame x len(metrics)),
        #    'folder2': tensor (num_frame x len(metrics))
        # }
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {}
                num_frame_each_folder = Counter(dataset.data_info['folder'])
                for folder, num_frame in num_frame_each_folder.items():
                    self.metric_results[folder] = torch.zeros(
                        num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
            # initialize the best metric results
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()

        metric_data = dict()
        num_folders = len(dataset)
        num_pad = (world_size - (num_folders % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='folder')
        # Will evaluate (num_folders + num_pad) times, but only the first num_folders results will be recorded.
        # (To avoid wait-dead)
        for i in range(rank, num_folders + num_pad, world_size):
            idx = min(i, num_folders - 1)
            val_data = dataset[idx]
            folder = val_data['folder']

            # compute outputs
            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)
            val_data['event_lq'].unsqueeze_(0)

            self.feed_data(val_data)
            val_data['lq'].squeeze_(0)
            val_data['gt'].squeeze_(0)
            val_data['event_lq'].squeeze_(0)

            self.test(folder)
            visuals = self.get_current_visuals()

            # tentative for out of GPU memory
            del self.lq
            del self.event_lq
            if 'gt' in visuals:
                del self.gt
            torch.cuda.empty_cache()

            if self.center_frame_only:
                visuals['result'] = visuals['result'].unsqueeze(1)
                if 'gt' in visuals:
                    visuals['gt'] = visuals['gt'].unsqueeze(1)

            # evaluate
            if i < num_folders:
                for idx in range(visuals['result'].size(1)):
                    result = visuals['result'][0, idx, :, :, :]
                    result_img = tensor2img([result])  # uint8, bgr
                    metric_data['img'] = result_img
                    if 'gt' in visuals:
                        gt = visuals['gt'][0, idx, :, :, :]
                        gt_img = tensor2img([gt])  # uint8, bgr
                        gt_img = mod_crop(gt_img, self.scale)
                        metric_data['img2'] = gt_img

                    if save_img:
                        if self.opt['is_train']:
                            raise NotImplementedError('saving image is not supported during training.')
                        else:
                            if self.center_frame_only:  # vimeo-90k
                                clip_ = val_data['lq_path'].split('/')[-3]
                                seq_ = val_data['lq_path'].split('/')[-2]
                                name_ = f'{clip_}_{seq_}'
                                img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                    f"{name_}_{self.opt['name']}.png")
                            else:  # others
                                img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                    f"{idx:06d}_{self.opt['name']}.png")
                            # image name only for REDS dataset
                        imwrite(result_img, img_path)

                    # calculate metrics
                    if with_metrics:
                        for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                            result = calculate_metric(metric_data, opt_)
                            self.metric_results[folder][idx, metric_idx] += result

                # progress bar
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Folder: {folder}')

        if rank == 0:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.barrier()

            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def test(self, folder):
        n = self.lq.size(1)
        self.net_g.eval()

        flip_seq = self.opt['val'].get('flip_seq', False)
        self.center_frame_only = self.opt['val'].get('center_frame_only', False)

        if flip_seq:
            self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)

        with torch.no_grad():
            assert self.lq.shape[1] == self.event_lq.shape[1] + 1, f"{folder} shape error, lq.shape is: {self.lq.shape}, event_lq.shape is: {self.event_lq.shape}"
            # print("self.event_lq.shape: ", self.event_lq.shape)
            self.output = self.net_g(self.lq, self.event_lq)

        if flip_seq:
            output_1 = self.output[:, :n, :, :, :]
            output_2 = self.output[:, n:, :, :, :].flip(1)
            self.output = 0.5 * (output_1 + output_2)

        if self.center_frame_only:
            self.output = self.output[:, n // 2, :, :, :]

        self.net_g.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        # ----------------- calculate the average values for each folder, and for each metric  ----------------- #
        # average all frames for each sub-folder
        # metric_results_avg is a dict:{
        #    folder example: 'people_dynamic_wave/split0'
        #    'folder1': tensor (len(metrics)),
        #    'folder2': tensor (len(metrics))
        # }
        metric_results_avg = {
            folder: torch.mean(tensor, dim=0).cpu()
            for (folder, tensor) in self.metric_results.items()
        }

        if dataset_name.lower() == 'ced11':
            # folder_len_dict is a dict: {
            #    count each folder frames number
            #    folder example is: 'people_dynamic_wave/split0'
            #    'folder1': int
            #    'folder2': int
            # }
            # cnt_folder_split is a dict: {
            #    count each root_folder split number
            #    root_folder example: indoors_foosball_2
            #    'root_folder1': int(759)
            #    'root_folder2': int
            # }
            folder_len_dict = {}
            cnt_folder_split = {}
            for (folder, tensor) in self.metric_results.items():
                folder_len_dict[folder] = tensor.size(0)
                cnt_folder_split[osp.dirname(folder)] = 0
            # print("folder_len_dict: ", folder_len_dict)

            for folder, length in folder_len_dict.items():
                cnt_folder_split[osp.dirname(folder)] += length

            # total_avg_results is a dict: {
            #    'metric1': float,
            #    'metric2': float
            # }

            total_avg_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            for folder, tensor in metric_results_avg.items():
                for metric_idx, metric in enumerate(total_avg_results.keys()):
                    total_avg_results[metric] += tensor[metric_idx].item() * folder_len_dict[folder]

            total_samples_length = 0
            for folder, length in folder_len_dict.items():
                total_samples_length += length
        else:
            # for vid4 datasets
            # cnt_folder is a dict:{
            #    'folder1': int (len(folder1))
            #    'folder2': int (len(folder2))
            # }
            cnt_folder = {
                folder: tensor.size(0)
                for (folder, tensor) in self.metric_results.items()
            }
            # total_avg_results is a dict: {
            #    'metric1': float,
            #    'metric2': float
            # }
            total_avg_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            for folder, tensor in metric_results_avg.items():
                for metric_idx, metric in enumerate(total_avg_results.keys()):
                    total_avg_results[metric] += tensor[metric_idx].item() * cnt_folder[folder]

            total_samples_length = 0
            for folder, length in cnt_folder.items():
                total_samples_length += length

        # average among folders
        for metric in total_avg_results.keys():
            total_avg_results[metric] /= total_samples_length
            # update the best metric result
            self._update_best_metric_result(dataset_name, metric, total_avg_results[metric], current_iter)

        # ------------------------------------------ log the metric ------------------------------------------ #
        log_str = f'Validation {dataset_name}\n'

        if dataset_name.lower() == 'ced11':
            # name_turple is a turple:(
            #    root_folder example: indoors_foosball_2
            #    'root_folder1': str,
            #    'root_folder2': str
            # )
            name_turple = set(osp.dirname(temp) for temp in list(self.metric_results.keys()))
            # print("name_turple: ", name_turple, " ", "len(name_turple)", len(name_turple))
            # avg_metric_dict and sum_folder_split is a dict: {
            #   'metric1': dict{
            #          'root_folder1': float
            #          'root_folder2': float
            #       }
            #   'metric2': dict{
            #          'root_folder1': float
            #          'root_folder2': float
            #       }i
            # }
            avg_metric_dict = {metric: {} for metric in self.opt['val']['metrics'].keys()}
            sum_folder_split = {metric: {} for metric in self.opt['val']['metrics'].keys()}
            for _, folder_dict in avg_metric_dict.items():
                for name in name_turple:
                    folder_dict[name] = 0.0
            for _, folder_dict in sum_folder_split.items():
                for name in name_turple:
                    folder_dict[name] = 0

            for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
                log_str += f'\t # {metric}: {value:.4f}'
                for folder, tensor in metric_results_avg.items():
                    # sum_folder_split[metric][osp.dirname(folder)] += tensor[metric_idx].item() * folder_len_dict[folder]
                    sum_folder_split[metric][osp.dirname(folder)] += tensor[metric_idx].item() * folder_len_dict[folder]

            # print("sum_folder_split: ", sum_folder_split)
            for metric, values in sum_folder_split.items():
                for name in name_turple:
                    # print(f"value[{name}]: ", values[name], " ", f"cnt_folder_split[{name}]: ", cnt_folder_split[name])
                    avg_metric_dict[metric][name] = values[name] / cnt_folder_split[name]
                    log_str += f'\n\t # {osp.splitext(name)[0]}: {avg_metric_dict[metric][name]:.4f}'
                if hasattr(self, 'best_metric_results'):
                    log_str += (f'\n\t Best: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')

        else:
            for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
                log_str += f'\t # {metric}: {value:.4f}'
                for folder, tensor in metric_results_avg.items():
                    log_str += f'\t # {folder}: {tensor[metric_idx].item():.4f}'
                if hasattr(self, 'best_metric_results'):
                    log_str += (f'\n\t    Best: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                                f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
                log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
                if dataset_name.lower() == 'ced11':
                    for metric, root_result in avg_metric_dict.items():
                        for name in name_turple:
                            tb_logger.add_scalar(f'metrics/{metric}/{osp.splitext(name)[0]}', root_result[f'{name}'], current_iter)
                else:
                    for folder, tensor in metric_results_avg.items():
                        tb_logger.add_scalar(f'metrics/{metric}/{folder}', tensor[metric_idx].item(), current_iter)
