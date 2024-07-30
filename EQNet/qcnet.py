# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from itertools import chain
from itertools import compress
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from losses import MixtureNLLLoss
from losses import NLLLoss
from metrics import Brier
from metrics import MR
from metrics import minADE
from metrics import minAHE
from metrics import minFDE
from metrics import minFHE
from EQNet.qcnet_decoder import QCNetDecoder
from EQNet.qcnet_encoder import QCNetEncoder


try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object

from torch_geometric.data.collate import collate
from shared_libs.oom_recover_lightning_module import OOMRecoverable

class QCNet(OOMRecoverable):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_dec_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 submission_dir: str,
                 submission_file_name: str,
                 experiment_id: str = None,
                 grad_acc: int = 1,
                 freeze_encoder: bool = False,
                 freeze_proposal: bool = False,
                 load_encoder: str = None,
                 load_spec: str = None,
                 load_spec_ckpt: str = None,
                 load_ckpt: str = None,
                 detach_m: bool = False,
                 brier_loss: bool = False,
                 refine: bool = False,
                 **kwargs) -> None:
        super().__init__(
            grad_acc=grad_acc,
            memory_safe_margin=3 * 1024**3,  # 3 GB
        )
        self.save_hyperparameters()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_freq_bands = num_freq_bands
        self.num_map_layers = num_map_layers
        self.num_agent_layers = num_agent_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.pl2pl_radius = pl2pl_radius
        self.time_span = time_span
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_t2m_steps = num_t2m_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.submission_dir = submission_dir
        self.submission_file_name = submission_file_name
        self.experiment_id = experiment_id
        self.freeze_encoder = freeze_encoder
        self.freeze_proposal = freeze_proposal
        self.load_encoder = load_encoder
        self.load_spec = load_spec
        self.load_spec_ckpt = load_spec_ckpt
        self.load_ckpt = load_ckpt
        self.brier_loss_enabled = brier_loss
        self.refine = refine

        self.encoder = QCNetEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers,
            num_agent_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.decoder = QCNetDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            output_head=output_head,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            num_modes=num_modes,
            num_recurrent_steps=num_recurrent_steps,
            num_t2m_steps=num_t2m_steps,
            pl2m_radius=pl2m_radius,
            a2m_radius=a2m_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_dec_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            detach_m=detach_m,
            refine=refine,
        )

        self.reg_loss = NLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                reduction='none')
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                       reduction='none')

        self.Brier = Brier(max_guesses=6)
        self.minADE = minADE(max_guesses=6)
        self.minAHE = minAHE(max_guesses=6)
        self.minFDE = minFDE(max_guesses=6)
        self.minFHE = minFHE(max_guesses=6)
        self.MR = MR(max_guesses=6)

        self.test_predictions = dict()

    def forward(self, data: HeteroData):
        if self.freeze_encoder:
            with torch.no_grad():
                scene_enc = self.encoder(data)
        else:
            scene_enc = self.encoder(data)
        pred = self.decoder(data, scene_enc)
        return pred

    def on_train_start(self) -> None:
        pass

    def modify_data_on_oom(self, data_list):
        data_list.sort(key=lambda x:x['agent']['num_nodes'])
        data_list.pop()
        return data_list

    def compute_loss(self, data_list):

        data, _, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
        )

        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        pred = self(data)
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            if self.refine:
                traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                         pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
            else:
                traj_refine = traj_propose
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        self.log('train_reg_loss', reg_loss_propose, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        if self.refine:
            traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
            reg_loss_refine = self.reg_loss(traj_refine_best,
                                            gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
            reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
            reg_loss_refine = reg_loss_refine.mean()
            self.log('train_reg_loss_refine', reg_loss_refine, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        else:
            reg_loss_refine = 0.

        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                target=gt[:, -1:, :self.output_dim + self.output_head],
                                prob=pi,
                                mask=reg_mask[:, -1:])
        cls_loss = cls_loss * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        loss = reg_loss_propose + reg_loss_refine + cls_loss
        self.log('train_cls_loss', cls_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)

        if self.brier_loss_enabled:
            pred=traj_refine[:, :, -1:, :self.output_dim].detach()  # [A, K, 1, 2]
            target=gt[:, None, -1:, :self.output_dim]  # [A, 1, 1, 2]
            fde = torch.norm(pred-target, dim=-1).squeeze(-1)  # [A, K]
            min_fde_idx = torch.argmin(fde, dim=1)
            prob = F.softmax(pi, dim=-1)  # [A, K]
            min_fde_prob = prob.gather(1, min_fde_idx.unsqueeze(-1)).squeeze(1) # [A]
            brier_loss = torch.square(1 - min_fde_prob) * cls_mask
            brier_loss = brier_loss.sum() / cls_mask.sum().clamp_(min=1)
            loss = loss + brier_loss

        # learning rate warmup steps
        if self.global_step < 500:
            opt:torch.optim.Optimizer = self.optimizers()
            lr_scale = min(1., float(self.global_step + 1) / 500.)
            for pg in opt.optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr

        return loss

    def validation_step(self,
                        data,
                        batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        pred = self(data)
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            # traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
            #                          pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = traj_propose
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        self.log('val_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_cls_loss', cls_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        if self.dataset in ['argoverse_v2', 'argoverse_v1']:
            eval_mask = data['agent']['category'] == 3
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        valid_mask_eval = reg_mask[eval_mask]
        traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]
        if not self.output_head:
            traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                     traj_eval[..., :2]], dim=-2)
            motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
            head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
            traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
        pi_eval = F.softmax(pi[eval_mask], dim=-1)
        gt_eval = gt[eval_mask]

        self.Brier.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                          valid_mask=valid_mask_eval)
        self.minADE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.minAHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.minFDE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.minFHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.MR.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                       valid_mask=valid_mask_eval)
        self.log('val_Brier', self.Brier, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minAHE', self.minAHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFHE', self.minFHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))

    def test_step(self,
                  data,
                  batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        pred = self(data)
        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] == 3
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2],
                                 rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        pi_eval = F.softmax(pi[eval_mask], dim=-1)

        traj_eval = traj_eval.cpu().numpy()
        pi_eval = pi_eval.cpu().numpy()
        if self.dataset == 'argoverse_v2':
            eval_id = list(compress(list(chain(*data['agent']['id'])), eval_mask))
            if isinstance(data, Batch):
                for i in range(data.num_graphs):
                    self.test_predictions[data['scenario_id'][i]] = (pi_eval[i], {eval_id[i]: traj_eval[i]})
            else:
                self.test_predictions[data['scenario_id']] = (pi_eval[0], {eval_id[0]: traj_eval[0]})
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def on_test_end(self):
        if self.dataset == 'argoverse_v2':
            ChallengeSubmission(self.test_predictions).to_parquet(
                Path(self.submission_dir) / f'{self.submission_file_name}.parquet')
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)

        if self.load_ckpt:
            self.load_state_dict(torch.load(self.load_ckpt)['state_dict'], strict=False)

        if self.load_encoder:
            state_dict = torch.load(self.load_encoder)['state_dict']
            state_dict = {key: value for key, value in state_dict.items() if key.startswith("encoder.")}
            self.load_state_dict(state_dict, strict=False)

        if self.load_spec and self.load_spec_ckpt:
            with open(self.load_spec) as f:
                load_spec = f.readlines()
            state_dict = torch.load(self.load_spec_ckpt)["state_dict"]
            parsed_state_dict = dict()
            for load_spec_line in load_spec:
                if ":" not in load_spec_line: continue
                dst, src = load_spec_line.strip().split(":")
                parsed_state_dict[dst] = state_dict[src]
            self.load_state_dict(parsed_state_dict, strict=False)

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            model = self.decoder
        elif self.freeze_proposal:
            for name, param in self.named_parameters():
                for key in ["refine", "y_emb", "traj_emb"]:
                    if key in name:
                        break
                else:
                    param.requires_grad = False
            model = self
        else:
            model = self

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('QCNet')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--output_head', action='store_true')
        parser.add_argument('--num_historical_steps', type=int, required=True)
        parser.add_argument('--num_future_steps', type=int, required=True)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_recurrent_steps', type=int, required=True)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_map_layers', type=int, default=1)
        parser.add_argument('--num_agent_layers', type=int, default=2)
        parser.add_argument('--num_dec_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--pl2pl_radius', type=float, required=True)
        parser.add_argument('--time_span', type=int, default=None)
        parser.add_argument('--pl2a_radius', type=float, required=True)
        parser.add_argument('--a2a_radius', type=float, required=True)
        parser.add_argument('--num_t2m_steps', type=int, default=None)
        parser.add_argument('--pl2m_radius', type=float, required=True)
        parser.add_argument('--a2m_radius', type=float, required=True)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        parser.add_argument('--submission_dir', type=str, default='./submissions')
        parser.add_argument('--submission_file_name', type=str, default='submission')
        parser.add_argument('--freeze_encoder', action='store_true')
        parser.add_argument('--freeze_proposal', action='store_true')
        parser.add_argument('--load_encoder', type=str, default=None)
        parser.add_argument('--load_spec', type=str, default=None)
        parser.add_argument('--load_spec_ckpt', type=str, default=None)
        parser.add_argument('--load_ckpt', type=str, default=None)
        parser.add_argument('--detach_m', action="store_true")
        parser.add_argument('--brier_loss', action="store_true")
        parser.add_argument('--refine', action="store_true")
        return parent_parser
