import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from copy import deepcopy
from torch import nn
import os
from torch.optim import SGD
from backbone.MNISTMLP import SparseMNISTMLP
from backbone.SparseResNet18_global import sparse_resnet18

num_classes_dict = {
    'seq-cifar10': 10,
    'seq-cifar100': 100,
    'seq-tinyimg': 200,
    'gcil-cifar100': 100
}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Complementary Learning Systems Based Experience Replay')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # Consistency Regularization Weight
    parser.add_argument('--reg_weight', type=float, default=0.1)
    # Sparsity param
    parser.add_argument('--kw', type=float, nargs='*', default=[0.9, 0.9, 0.9, 0.8])
    parser.add_argument('--kw_criterion', type=str, default='abs_sum')
    parser.add_argument('--kw_relu', type=int, default=1)
    # Stable Model parameters
    parser.add_argument('--ema_update_freq', type=float, default=0.70)
    parser.add_argument('--ema_alpha', type=float, default=0.999)
    # Heterogeneous Dropout
    parser.add_argument('--apply_heterogeneous_dropout', type=int, default=1)
    parser.add_argument('--layerwise_dropout', type=int, nargs='*', default=[0, 0, 0, 1])
    parser.add_argument('--dropout_alpha', type=float, nargs='*', default=(0.5, 0.5, 5, 5))
    parser.add_argument('--classwise_dropout_alpha', type=float, nargs='*', default=(2, 2, 2, 2))
    parser.add_argument('--dropout_warmup', type=int, default=30)
    # Initialize heterogeneous dropout probabilities
    parser.add_argument('--init_dropout', type=int, default=1)
    parser.add_argument('--init_active_factor', type=float, nargs='*', default=[1.1, 1.1, 1.1, 1.1])
    # Experimental Args
    parser.add_argument('--save_interim', type=int, default=1)
    return parser


# =============================================================================
# Mean-ER
# =============================================================================
class SCoMMER(ContinualModel):
    NAME = 'scommer'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(SCoMMER, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        # Initialize plastic and stable model
        if 'mnist' in self.args.dataset:
            self.net = SparseMNISTMLP(28 * 28, 10, kw_percent_on=args.kw).to(self.device)
            self.ema_model = SparseMNISTMLP(28 * 28, 10, kw_percent_on=args.stable_kw).to(self.device)
        else:
            self.net = sparse_resnet18(nclasses=num_classes_dict[args.dataset], kw_percent_on=args.kw, kw_criterion=args.kw_criterion, kw_relu=args.kw_relu).to(self.device)
            self.ema_model = sparse_resnet18(nclasses=num_classes_dict[args.dataset], kw_percent_on=args.kw, kw_criterion=args.kw_criterion, kw_relu=args.kw_relu).to(self.device)

        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        self.ema_model.load_state_dict(self.net.state_dict())

        self.net_before_mc = deepcopy(self.net)

        # Set Dropout Params
        self.net.apply_heterogeneous_dropout = args.apply_heterogeneous_dropout
        self.ema_model.apply_heterogeneous_dropout = args.apply_heterogeneous_dropout
        self.net.layerwise_dropout = self.args.layerwise_dropout
        self.ema_model.layerwise_dropout = self.args.layerwise_dropout

        # set regularization weight
        self.reg_weight = args.reg_weight
        # set parameters for stable model
        self.ema_update_freq = args.ema_update_freq
        self.ema_alpha = args.ema_alpha

        self.consistency_loss = nn.MSELoss(reduction='none')
        self.current_task = 0
        self.global_step = 0
        self.dropout_warmup = args.dropout_warmup
        self.lst_models = ['net', 'ema_model']

        # Initialize dropout params
        if self.args.init_dropout:
            for model_id in self.lst_models:
                if 'mnist' in self.args.dataset:
                    model = getattr(self, model_id)
                    print('Initializing Dropout')
                    model._keep_probs['layer_1'] = torch.ones(100)
                    model._keep_probs['layer_2'] = torch.ones(100)
                    model._keep_probs['layer_1_classwise'] = torch.zeros(10, 100)
                    model._keep_probs['layer_2_classwise'] = torch.zeros(10, 100)

                else:
                    model = getattr(self, model_id)
                    print('Initializing Dropout')
                    model._keep_probs['layer_1'] = torch.zeros(model.nf * 1)
                    model._keep_probs['layer_2'] = torch.zeros(model.nf * 2)
                    model._keep_probs['layer_3'] = torch.zeros(model.nf * 4)
                    model._keep_probs['layer_4'] = torch.zeros(model.nf * 8)
                    model._keep_probs['layer_1'][:min(int(self.args.kw[0] * self.args.init_active_factor[0] * model.nf), model.nf)] = 1
                    model._keep_probs['layer_2'][:min(int(self.args.kw[1] * self.args.init_active_factor[1] * model.nf * 2), model.nf * 2)] = 1
                    model._keep_probs['layer_3'][:min(int(self.args.kw[2] * self.args.init_active_factor[2] * model.nf * 4), model.nf * 4)] = 1
                    model._keep_probs['layer_4'][:min(int(self.args.kw[3] * self.args.init_active_factor[3] * model.nf * 8), model.nf * 8)] = 1
                    model._keep_probs['layer_1_classwise'] = torch.zeros(num_classes_dict[args.dataset], model.nf * 1)
                    model._keep_probs['layer_2_classwise'] = torch.zeros(num_classes_dict[args.dataset], model.nf * 2)
                    model._keep_probs['layer_3_classwise'] = torch.zeros(num_classes_dict[args.dataset], model.nf * 4)
                    model._keep_probs['layer_4_classwise'] = torch.zeros(num_classes_dict[args.dataset], model.nf * 8)

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        self.net.train()
        self.ema_model.train()

        loss = 0

        if not self.buffer.is_empty():

            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)

            stable_model_logits = self.ema_model(
                buf_inputs,
                y=buf_labels,
                disable_dropout=True,
                update_act_counts=False
            )

            buff_out = self.net(
                buf_inputs,
                y=buf_labels,
                disable_dropout=True,
                update_act_counts=False
            )

            buff_ce = self.loss(buff_out, buf_labels)
            l_cons = torch.mean(self.consistency_loss(buff_out, stable_model_logits.detach()))
            l_reg = self.args.reg_weight * l_cons
            loss += buff_ce + l_reg

            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/l_cons', l_cons.item(), self.iteration)
                self.writer.add_scalar(f'Task {self.current_task}/l_reg', l_reg.item(), self.iteration)

            # Log values
            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/l_reg', l_reg.item(), self.iteration)

        outputs = self.net(inputs, y=labels)

        ce_loss = self.loss(outputs, labels)
        loss += ce_loss

        if torch.isnan(loss):
            raise ValueError('NAN Loss')

        # Log values
        if hasattr(self, 'writer'):
            self.writer.add_scalar(f'Task {self.current_task}/ce_loss', ce_loss.item(), self.iteration)
            self.writer.add_scalar(f'Task {self.current_task}/loss', loss.item(), self.iteration)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels[:real_batch_size],
        )

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.ema_update_freq:
            self.update_ema_weights()

        return loss.item()

    def update_ema_weights(self):
        # print('Using EMA on Working model')
        alpha = min(1 - 1 / (self.global_step + 1), self.ema_alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    @staticmethod
    def return_param(model, param_name):
        name_tokens = param_name.split('.')
        if len(name_tokens) == 4:
            layer, idx, bn, _ = name_tokens
            param = getattr(getattr(model, layer)[int(idx)], bn)

        else:
            bn, _ = name_tokens
            param = getattr(model, bn)

        return param

    def end_task(self, dataset) -> None:
        self.current_task += 1

        # Heterogeneous Dropout
        if self.args.apply_heterogeneous_dropout:
            # Calculate the Dropout keep Probabilities
            for model_id in self.lst_models:
                model = getattr(self, model_id)
                for layer_idx in range(1, len(model._layers) + 1):
                    # Heterogeneous Dropout
                    activation_counts = model._activation_counts[f'layer_{layer_idx}']
                    max_act = torch.max(activation_counts)
                    model._keep_probs[f'layer_{layer_idx}'] = torch.exp(-activation_counts * self.args.dropout_alpha[layer_idx - 1] / max_act)
                    # Classwise Dropout
                    activation_counts = model._activation_counts[f'layer_{layer_idx}_classwise']
                    max_act = torch.max(activation_counts, dim=1)[0]
                    model._keep_probs[f'layer_{layer_idx}_classwise'] = 1 - torch.exp(-activation_counts * self.args.classwise_dropout_alpha[layer_idx - 1] / (max_act[:, None] + 1e-16))

        if self.args.save_interim:
            model_dir = os.path.join(self.args.output_dir, "task_models", dataset.NAME, self.args.experiment_id)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(self.net, os.path.join(model_dir, f'model.ph'))
            torch.save(self.ema_model, os.path.join(model_dir, f'stable_model.ph'))

            # Activation Counts
            torch.save(self.net._activation_counts, os.path.join(model_dir, f'activation_count.ph'))
            torch.save(self.net._keep_probs, os.path.join(model_dir, f'keep_prob.ph'))

            # buffer samples
            torch.save(self.buffer.get_all_data(), os.path.join(model_dir, f'buffer.ph'))

    def end_epoch(self, epoch) -> None:
        if epoch > self.dropout_warmup:
            print(f'Updating classwise probabilities at epoch {epoch}')
            for model_id in self.lst_models:
                model = getattr(self, model_id)
                for layer_idx in range(1, len(model._layers) + 1):
                    activation_counts = model._activation_counts[f'layer_{layer_idx}_classwise']
                    max_act = torch.max(activation_counts, dim=1)[0]
                    model._keep_probs[f'layer_{layer_idx}_classwise'] = 1 - torch.exp(-activation_counts * self.args.classwise_dropout_alpha[layer_idx - 1] / (max_act[:, None] + 1e-16))
