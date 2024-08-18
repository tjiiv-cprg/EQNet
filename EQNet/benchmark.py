from shared_libs.eec_parse import extend_extra_sys_path
extend_extra_sys_path("EQNet/EEC.md")

import argparse
import time
import torch
import numpy as np
import json
import os
from tqdm import tqdm
from EQNet.data import get_argoverse_v2_dataset_for_qcnet
from EQNet.eqnet import EQNet
from qcnet_55cacb4.predictors.qcnet import QCNet
from qcnet_55cacb4.modules.qcnet_agent_encoder import *

DEVICE = "cuda"
NUMDAT = 24000


def get_eqnet():
    argp = argparse.ArgumentParser()
    EQNet.add_model_specific_args(argp)
    args, _ = argp.parse_known_args([
        '--train_batch_size', '8', '--val_batch_size', '8', '--test_batch_size', '8',
        '--devices', '1', '--max_epochs', '32', '--T_max', '32', '--lr', '5e-4',
        '--num_workers', '0', '--dataset', 'argoverse_v2', '--num_historical_steps', '50',
        '--num_future_steps', '60', '--num_recurrent_steps', '3', '--pl2pl_radius', '150',
        '--time_span', '10', '--pl2a_radius', '50', '--a2a_radius', '50', '--num_t2m_steps',
        '30', '--pl2m_radius', '150', '--a2m_radius', '150', '--grad_acc', '1',
    ])
    eqnet = EQNet(**vars(args))
    eqnet.to(DEVICE)
    eqnet.eval()
    return eqnet


def get_qcnet():
    argp = argparse.ArgumentParser()
    QCNet.add_model_specific_args(argp)
    args, _ = argp.parse_known_args([
        '--train_batch_size', '8', '--val_batch_size', '8', '--test_batch_size', '8',
        '--devices', '1', '--max_epochs', '32', '--T_max', '32', '--lr', '5e-4',
        '--num_workers', '0', '--dataset', 'argoverse_v2', '--num_historical_steps', '50',
        '--num_future_steps', '60', '--num_recurrent_steps', '3', '--pl2pl_radius', '150',
        '--time_span', '10', '--pl2a_radius', '50', '--a2a_radius', '50', '--num_t2m_steps',
        '30', '--pl2m_radius', '150', '--a2m_radius', '150',
    ])
    qcnet = QCNet(**vars(args))
    qcnet.to(DEVICE)
    qcnet.eval()
    return qcnet


def num_agents(data):
    return data['agent']['num_nodes']


def find_most_busy_scene(dataset):
    from functools import reduce
    imax, nmax = reduce(
        lambda kv1, kv2: kv1 if kv1[1] > kv2[1] else kv2,
        tqdm(((i, num_agents(d)) for i, d in enumerate(dataset)), total=len(dataset)),
        (-1, -1))
    print(
        f"I believe scene {dataset.raw_file_names[imax]} the most busy scene, which has {nmax} agents."
    ) # "7ec7f490-fba6-463b-997e-9e55bd5d8e0f"


def forward(config):

    def eqnet_runtime(m: EQNet, data, decode=True):
        map_enc = m.encoder.map_encoder(data)

        states = []
        x_a_seq = []
        for t in range(50):
            if t == 49:
                torch.cuda.synchronize()
                T0 = time.time()
            agent_enc_seq_t = m.encoder.agent_encoder.forward_one_step(data, map_enc, t, states=states)
            x_a_seq.append(agent_enc_seq_t)
        
        x_a_seq = torch.cat(x_a_seq, dim=1)

        # agent_enc_par = model.encoder.agent_encoder(data, map_enc)
        # torch.testing.assert_close(agent_enc_seq[:, 0], agent_enc_par['x_a'][:, 0])

        if decode:
            pred = m.decoder(data, {"x_a": x_a_seq, **map_enc})

        torch.cuda.synchronize()
        T1 = time.time()

        return T1 - T0

    def qcnet_runtime(qcnet:QCNet, data, decode=True):
        map_enc = qcnet.encoder.map_encoder(data)
        torch.cuda.synchronize()
        T0 = time.time()
        aenc = qcnet.encoder.agent_encoder
        mask = data['agent']['valid_mask'][:, :aenc.num_historical_steps].contiguous()
        pos_a = data['agent']['position'][:, :aenc.num_historical_steps, :aenc.input_dim].contiguous()
        motion_vector_a = torch.cat([pos_a.new_zeros(data['agent']['num_nodes'], 1, aenc.input_dim),
                                     pos_a[:, 1:] - pos_a[:, :-1]], dim=1)
        head_a = data['agent']['heading'][:, :aenc.num_historical_steps].contiguous()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        pos_pl = data['map_polygon']['position'][:, :aenc.input_dim].contiguous()
        orient_pl = data['map_polygon']['orientation'].contiguous()
        if aenc.dataset == 'argoverse_v2':
            length = width = height = None
            categorical_embs = [
                aenc.type_a_emb(data['agent']['type'].long()).repeat_interleave(repeats=aenc.num_historical_steps,
                                                                                dim=0),
            ]
        elif aenc.dataset == 'argoverse_v1':
            categorical_embs = None
        else:
            raise ValueError('{} is not a valid dataset'.format(aenc.dataset))

        if aenc.dataset == 'argoverse_v2':
            vel = data['agent']['velocity'][:, :aenc.num_historical_steps, :aenc.input_dim].contiguous()
            x_a = torch.stack(
                [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]),
                 torch.norm(vel[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel[:, :, :2])], dim=-1)
        elif aenc.dataset == "argoverse_v1":
            x_a = torch.stack(
                [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]),
                ], dim=-1)
        else:
            raise ValueError('{} is not a valid dataset'.format(aenc.dataset))
        x_a = aenc.x_a_emb(continuous_inputs=x_a.view(-1, x_a.size(-1)), categorical_embs=categorical_embs)
        x_a = x_a.view(-1, aenc.num_historical_steps, aenc.hidden_dim)

        pos_t = pos_a.reshape(-1, aenc.input_dim)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)
        mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)
        now_mask = torch.zeros_like(mask_t) #!
        now_mask[:, :, -1] = True #!
        mask_t = torch.logical_and(mask_t, now_mask) #!
        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[:, edge_index_t[1] - edge_index_t[0] <= aenc.time_span]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
             rel_head_t,
             edge_index_t[0] - edge_index_t[1]], dim=-1)
        r_t = aenc.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        # print(r_t.shape)

        pos_s = pos_a.transpose(0, 1).reshape(-1, aenc.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        now_mask_s = torch.zeros_like(mask) #!
        now_mask_s[:, -1] = 1 #!
        mask_s = torch.logical_and(mask, now_mask_s).transpose(0, 1).reshape(-1) #!
        pos_pl = pos_pl.repeat(aenc.num_historical_steps, 1)
        orient_pl = orient_pl.repeat(aenc.num_historical_steps)
        if isinstance(data, Batch):
            batch_s = torch.cat([data['agent']['batch'] + data.num_graphs * t
                                 for t in range(aenc.num_historical_steps)], dim=0)
            batch_pl = torch.cat([data['map_polygon']['batch'] + data.num_graphs * t
                                  for t in range(aenc.num_historical_steps)], dim=0)
        else:
            batch_s = torch.arange(aenc.num_historical_steps,
                                   device=pos_a.device).repeat_interleave(data['agent']['num_nodes'])
            batch_pl = torch.arange(aenc.num_historical_steps,
                                    device=pos_pl.device).repeat_interleave(data['map_polygon']['num_nodes'])
        edge_index_pl2a = radius(x=pos_s[:, :2], y=pos_pl[:, :2], r=aenc.pl2a_radius, batch_x=batch_s, batch_y=batch_pl,
                                 max_num_neighbors=300)
        edge_index_pl2a = edge_index_pl2a[:, mask_s[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])
        r_pl2a = torch.stack(
            [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]),
             rel_orient_pl2a], dim=-1)
        r_pl2a = aenc.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
        edge_index_a2a = radius_graph(x=pos_s[:, :2], r=aenc.a2a_radius, batch=batch_s, loop=False,
                                      max_num_neighbors=300)
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
             rel_head_a2a], dim=-1)
        r_a2a = aenc.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

        for i in range(aenc.num_layers):
            x_a = x_a.reshape(-1, aenc.hidden_dim)
            x_a = aenc.t_attn_layers[i](x_a, r_t, edge_index_t)
            x_a = x_a.reshape(-1, aenc.num_historical_steps,
                              aenc.hidden_dim).transpose(0, 1).reshape(-1, aenc.hidden_dim)
            x_a = aenc.pl2a_attn_layers[i]((map_enc['x_pl'].transpose(0, 1).reshape(-1, aenc.hidden_dim), x_a), r_pl2a,
                                           edge_index_pl2a)
            x_a = aenc.a2a_attn_layers[i](x_a, r_a2a, edge_index_a2a)
            x_a = x_a.reshape(aenc.num_historical_steps, -1, aenc.hidden_dim).transpose(0, 1)

        scene_enc = {'x_a': x_a, **map_enc}
        
        if decode:
            pred = qcnet.decoder(data, scene_enc)

        torch.cuda.synchronize()
        T1 = time.time()
        return T1 - T0

    return locals()[config]


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("mode", choices=["bench", "display"])
    argp.add_argument("logjsonfile")
    argp.add_argument("--saveimg", default="results/DIRTY/bench.png")
    argp.add_argument("--machine", default="3090-C")
    args = argp.parse_args()

    if args.mode == "bench":

        # dataset
        dataset = get_argoverse_v2_dataset_for_qcnet(args.machine, "val")
        try:
            busy_scene = [dataset.raw_file_names.index("7ec7f490-fba6-463b-997e-9e55bd5d8e0f")]
        except:
            busy_scene = []
        subset = np.random.choice(range(len(dataset)), size=NUMDAT-1, replace=False).tolist() + busy_scene
        dataset = dataset[subset]

        # model
        models = {
            "eqnet": get_eqnet(),
            "qcnet": get_qcnet(),
        }

        # results
        configs = [
            ("eqnet", "runtime", dict(decode=True)),
            ("eqnet", "runtime", dict(decode=False)),
            ("qcnet", "runtime", dict(decode=True)),
            ("qcnet", "runtime", dict(decode=False)),
        ]
        results = {}

        for config in configs:
            model, runtime, kwargs = config
            forward_fn = forward(f"{model}_{runtime}")
            config_s = str(config)
            na_tm = []
            for data in tqdm(dataset, desc=config_s):
                if 0 == len(na_tm):
                    # warming up
                    time_elapsed = forward_fn(models[model], data.to(DEVICE), **kwargs)
                    time_elapsed = forward_fn(models[model], data.to(DEVICE), **kwargs)
                    time_elapsed = forward_fn(models[model], data.to(DEVICE), **kwargs)

                time_elapsed = forward_fn(models[model], data.to(DEVICE), **kwargs)
                na_tm.append((num_agents(data), time_elapsed))
            results[config_s] = na_tm

        print(results)
        
        with open(args.logjsonfile, "w") as fp:
            json.dump(results, fp)

    elif args.mode == "display":
        assert os.path.isfile(args.logjsonfile)
        with open(args.logjsonfile, 'r') as fp:
            results = json.load(fp)
        
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        plt.grid(True)
        
        legend_map = {
            "('eqnet', 'runtime', {'decode': True})": "EQNet(Full)",
            "('eqnet', 'runtime', {'decode': False})": "EQNet(Encoder Only)",
            "('qcnet', 'runtime', {'decode': True})": "QCNet(Full)",
            "('qcnet', 'runtime', {'decode': False})": "QCNet(Encoder Only)",
        }
        markers = "+oxd"
        linecolors = [
            "#13496e", # deep blue
            "#a44d00", # deep orange
            "#195b19", # deep green
            "#791617", # deep red
        ]
        legend = []
        threshold = 0.1 # unit: second
        for i, (k, v) in enumerate(results.items()):
            legend.append(legend_map[k])
            v = np.array(v)
            v = v[v[:, 1] < threshold]
            plt.scatter(v[:, 0], v[:, 1] * 1000, marker=markers[i], linewidths=1)

        plt.legend(legend)
        plt.ylabel("runtime latency (ms)")
        plt.xlabel("number of agents")
        
        for i, (k, v) in enumerate(results.items()):
            v = np.array(v)
            v = v[v[:, 1] < threshold]
            poly_coeff = np.polyfit(v[:, 0], v[:, 1] * 1000, 2)
            poly_fn = np.poly1d(poly_coeff)
            x = np.linspace(min(v[:, 0]), max(v[:, 0]), 200)
            y = poly_fn(x)
            plt.plot(x, y, color=linecolors[i])

        # plt.show()
        plt.savefig(args.saveimg, dpi=300)


def debug():
    argp = argparse.ArgumentParser()
    argp.add_argument("--machine", default="3090-C")
    args = argp.parse_args()

    # dataset
    dataset = get_argoverse_v2_dataset_for_qcnet(args.machine, "val")
    print(len(dataset))
    return
    try:
        busy_scene = [dataset.raw_file_names.index("7ec7f490-fba6-463b-997e-9e55bd5d8e0f")]
    except:
        busy_scene = []
    subset = np.random.choice(range(len(dataset)), size=NUMDAT-1, replace=False).tolist() + busy_scene
    dataset = dataset[subset]


    # model
    qcnet = get_qcnet()

    data = dataset[NUMDAT-1].to(DEVICE)

    forward_fn = forward('qcnet_runtime')
    t = forward_fn(qcnet, data, decode=False)
    t = forward_fn(qcnet, data, decode=False)
    t = forward_fn(qcnet, data, decode=False)
    print(t)
    ...


if __name__ == "__main__":
    # find_most_busy_scene(get_argoverse_v2_dataset_for_qcnet("3090-C", "val"))
    main()
