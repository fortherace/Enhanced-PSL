"""
Runing the proposed Enhanced Pareto Set Learning method on 16 test problems,
with MOEA/D early stopping and full-archive non-dominated selection.
"""
import os
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from torch.utils.data import DataLoader, Dataset
from re_problem import obtain_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.decomposition.pbi import PBI
from pymoo.decomposition.tchebicheff import Tchebicheff

from problem import get_problem
from model import ParetoSetModel
import schedulefree
import math
from typing import List, Tuple, Optional
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV

import timeit

def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)


def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)

def mc_hypervolume(data, ref_point, sample_num):
    """
    Monte Carlo HV for high-dim RE problems (e.g., re61/re91 where exact HV is heavy).
    """
    data = np.array(data)
    if len(data) == 0:
        return 0.0
    n, m = data.shape
    min_value = np.min(data, axis=0)

    # uniform samples in [min_value, ref_point]^m
    samples = np.random.uniform(low=min_value, high=ref_point, size=(sample_num, m))

    # remove dominated samples incrementally
    for obj in data:
        dominated = np.all(samples >= obj, axis=1)
        samples = samples[~dominated]

    volume = np.prod(ref_point - min_value)
    hv = volume * (1 - len(samples) / sample_num)
    return hv


def select_nondominated_from_archive(all_X: List[np.ndarray], all_F: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stack all generations, run non-dominated sorting on the union, and return rank-1 set.
    """
    if len(all_X) == 0:
        return np.empty((0,)), np.empty((0,))
    X_all = np.vstack(all_X)
    F_all = np.vstack(all_F)
    nds = NonDominatedSorting()
    I = nds.do(F_all, only_non_dominated_front=True)
    return X_all[I], F_all[I]


def _normalize_F(F: np.ndarray, ideal: Optional[np.ndarray], nadir: Optional[np.ndarray]) -> np.ndarray:
    if ideal is None or nadir is None:
        return F
    ideal = np.asarray(ideal)
    nadir = np.asarray(nadir)
    denom = np.maximum(nadir - ideal, 1e-12)
    return (F - ideal) / denom


# ------------------------- Early Stop + Archive -------------------------
class EarlyStopWithArchive:
    """
    A callback for pymoo that:
    1) Archives population (X, F) at each generation.
    2) Early-stops when the ideal point in the archive does not improve for `patience` generations.
    """

    def __init__(self,
                 min_gen: int = 50,
                 patience: int = 3,
                 tol: float = 1e-6
                 ):
        self.min_gen = int(min_gen)
        self.patience = int(patience)
        self.tol = float(tol)

        self.archive_X: List[np.ndarray] = []
        self.archive_F: List[np.ndarray] = []

        self.ideal_hist: List[np.ndarray] = []
        self.trig = 0

    def __call__(self, algorithm):
        # collect current population
        pop = algorithm.pop
        if pop is None or len(pop) == 0:
            return

        X = pop.get("X")
        F = pop.get("F")

        # store into archive
        if X is not None and F is not None:
            self.archive_X.append(np.array(X))
            self.archive_F.append(np.array(F))

        # compute current ideal point from ALL archive solutions
        F_all = np.vstack(self.archive_F)
        ideal = F_all.min(axis=0)  # ideal point

        self.ideal_hist.append(ideal)

        gen = algorithm.n_gen
        if gen < self.min_gen:
            return

        # check improvement
        if len(self.ideal_hist) > 1:
            prev = self.ideal_hist[-2]
            curr = self.ideal_hist[-1]

            range_scale = np.max(F, axis=0) - np.min(F, axis=0) + 1e-12
            diff = np.linalg.norm((curr - prev) / range_scale, ord=np.inf)
            if diff < self.tol:
                self.trig += 1
            else:
                self.trig = 0

            if self.trig >= self.patience:
                algorithm.termination.force_termination = True

ins_list = ['re21','re22','re23','re24','re25','re31','re32','re33','re34','re35', 're36', 're37', 're41', 're42', 're61', 're91']

# number of independent runs
n_run = 31

# EPSL core hyperparams
n_steps = 1000
n_pref_update = 5
n_sample = 5

# pretraining
pretrain_epochs = 30
sampling_method = 'Bernoulli'

# device
device = 'cpu'

# ------------------------- Main -------------------------
hv_list = {}

for test_ins in ins_list:
    print(test_ins)
    small_Hyper = []
    large_Hyper = []

    if test_ins in ['re21', 're22', 're23', 're24', 're25', 're31', 're32', 're33', 're34', 're35', 're36', 're37',
                    're41', 're42', 're61', 're91']:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'data/RE/ParetoFront/{test_ins}.dat')
        pf = np.loadtxt(file_path)
        True_PF = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'data/RE/ParetoFront/{test_ins}.dat'))
        ideal_point = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),f'data/RE/ideal_nadir_points/ideal_point_{test_ins}.dat'))
        nadir_point = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),f'data/RE/ideal_nadir_points/nadir_point_{test_ins}.dat'))
    else:
        ideal_point = np.zeros(2)
        nadir_point = np.ones(2)

    problem = get_problem(test_ins)
    problem_nsga = obtain_problem(test_ins)
    n_dim = problem.n_dim
    n_obj = problem.n_obj

    if n_obj == 2:
        n_pref_update = 5
    else:
        n_pref_update = 8

    ref_point = problem.nadir_point
    ref_point = [1.1 * x for x in ref_point]
    ref_point = torch.Tensor(ref_point).to(device)

    # repeatedly run the algorithm n_run times
    for run_iter in range(n_run):

        print(run_iter)

        start = timeit.default_timer()

        if n_obj == 2:
            pref = np.stack([np.linspace(0, 1, 100), 1 - np.linspace(0, 1, 100)]).T

        if n_obj == 3:
            pref_size = 105
            pref = das_dennis(13, 3)

        if n_obj == 4:
            pref_size = 120
            pref = das_dennis(7, 4)

        if n_obj == 6:
            pref_size = 182
            pref = get_reference_directions("multi-layer",
                                                         get_reference_directions("das-dennis", 6, n_partitions=4,
                                                                                  scaling=1.0),
                                                         get_reference_directions("das-dennis", 6, n_partitions=3,
                                                                                  scaling=0.5))

        if n_obj == 9:
            pref_size = 90
            pref = get_reference_directions("multi-layer",
                                                         get_reference_directions("das-dennis", 9, n_partitions=2,
                                                                                  scaling=1.0),
                                                         get_reference_directions("das-dennis", 9, n_partitions=2,
                                                                                  scaling=0.5))

        if n_obj == 2:
            pop_size = 100
            generations = 250
        elif n_obj == 3:
            pop_size = 105
            generations = 380
        elif n_obj == 4:
            pop_size = 120
            generations = 333
        elif n_obj == 6:
            pop_size = 182
            generations = 246
        else:
            pop_size = 90
            generations = 444

        decomposition = Tchebicheff()
        # Early stop hyperparams 
        early_cb = EarlyStopWithArchive(
            min_gen=0,
            patience=20,
            tol=1e-4
        )
        algorithm = MOEAD(
            ref_dirs=pref,
            decomposition=decomposition,
        )
        termination = ('n_gen', generations)
        res = minimize(
            problem_nsga,
            algorithm,
            termination,
            verbose=False,
            callback=early_cb,
        )

        arch_X_nd, arch_F_nd = select_nondominated_from_archive(early_cb.archive_X, early_cb.archive_F)

        if arch_X_nd.size == 0:
            approx_pareto_front = res.F
            approx_pareto_set = res.X
        else:
            approx_pareto_front = arch_F_nd
            approx_pareto_set = arch_X_nd

        approx_pareto_front = approx_pareto_front + 0.1


        print("number of non-dominated solutions：", len(approx_pareto_front))

        prefs_nsga2 = approx_pareto_front / approx_pareto_front.sum(axis=1, keepdims=True)

        X_tensor = torch.tensor(approx_pareto_set, dtype=torch.float32, device=device)
        prefs_tensor = torch.tensor(prefs_nsga2, dtype=torch.float32, device=device)

        class PreTrainDataset(Dataset):
            def __init__(self, prefs, x):
                self.prefs = prefs
                self.x = x

            def __len__(self):
                return len(self.prefs)

            def __getitem__(self, idx):
                return self.prefs[idx], self.x[idx]

        # ----------------------function evaluation allocation------------------------------
        n_gen = res.algorithm.n_gen
        if n_obj == 2:
            n_steps = math.floor((25000 - n_gen * 100) / 25)
        elif n_obj == 3:
            n_steps = math.floor((40000 - n_gen * 105) / 40)
        elif n_obj == 4:
            n_steps = math.floor((40000 - n_gen * 120) / 40)
        elif n_obj == 6:
            n_steps = math.floor((40000 - n_gen * 182) / 40)
        else:
            n_steps = math.floor((40000 - n_gen * 90) / 40)
        print(n_gen, n_steps)

        pretrain_dataset = PreTrainDataset(prefs_tensor, X_tensor)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True)

        # intitialize the model and optimizer
        psmodel = ParetoSetModel(n_dim, n_obj)
        psmodel.to(device)

        # optimizer
        optimizer_p = schedulefree.AdamWScheduleFree(psmodel.parameters(), lr=0.0025, warmup_steps=10)
        optimizer = schedulefree.AdamWScheduleFree(psmodel.parameters(), lr=0.0025, warmup_steps=10)

        # Pretrain steps
        for pretrain_epoch in range(pretrain_epochs):
            psmodel.train()
            optimizer_p.train()
            for batch_prefs, batch_X in pretrain_loader:
                pred_X = psmodel(batch_prefs)
                loss = torch.mean((pred_X - batch_X) ** 2)
                optimizer_p.zero_grad()
                loss.backward()
                optimizer_p.step()

        z = torch.ones(n_obj).to(device)

        # EPSL steps
        for t_step in range(n_steps):
            psmodel.train()
            optimizer.train()

            sigma = 0.01

            # sample n_pref_update preferences
            alpha = np.ones(n_obj)
            pref = np.random.dirichlet(alpha, n_pref_update)
            pref_vec = torch.tensor(pref).to(device).float()

            # get the current coressponding solutions
            x = psmodel(pref_vec)

            grad_es_list = []
            for k in range(pref_vec.shape[0]):

                if sampling_method == 'Gaussian':
                    delta = torch.randn(n_sample, n_dim).to(device).double()

                if sampling_method == 'Bernoulli':
                    delta = (torch.bernoulli(0.5 * torch.ones(n_sample, n_dim)) - 0.5) / 0.5
                    delta = delta.to(device).double()

                if sampling_method == 'Bernoulli-Shrinkage':
                    m = np.sqrt((n_sample + n_dim - 1) / (4 * n_sample))
                    delta = (torch.bernoulli(0.5 * torch.ones(n_sample, n_dim)) - 0.5) / m
                    delta = delta.to(device).double()

                x_plus_delta = x[k] + sigma * delta
                delta_plus_fixed = delta
                x_plus_delta[x_plus_delta > 1] = 1
                x_plus_delta[x_plus_delta < 0] = 0

                value_plus_delta = problem.evaluate(x_plus_delta)

                ideal_point_tensor = torch.tensor(ideal_point).to(device)
                nadir_point_tensor = torch.tensor(nadir_point).to(device)
                value_plus_delta = (value_plus_delta - ideal_point_tensor) / (nadir_point_tensor - ideal_point_tensor)

                z = torch.full((n_obj,), -0.1).to(device)

                # STCH Scalarization
                u = 0.1

                tch_value = u * torch.logsumexp((1 / pref_vec[k]) * torch.abs(value_plus_delta - z) / u, axis=1)
                tch_value = tch_value.detach()

                rank_idx = torch.argsort(tch_value)
                tch_value_rank = torch.ones(len(tch_value)).to(device)
                tch_value_rank[rank_idx] = torch.linspace(-0.5, 0.5, len(tch_value)).to(device)

                grad_es_k = 1.0 / (n_sample * sigma) * torch.sum(
                    tch_value_rank.reshape(len(tch_value), 1) * delta_plus_fixed, axis=0)
                grad_es_list.append(grad_es_k)

            grad_es = torch.stack(grad_es_list)

            # gradient-based pareto set model update
            optimizer.zero_grad()
            psmodel(pref_vec).backward(grad_es)
            optimizer.step()

        stop = timeit.default_timer()

        psmodel.eval()
        optimizer.eval()

        # calculate and report the hypervolume value
        with torch.no_grad():

            psmodel.eval()

            generated_pf = []
            generated_ps = []

            # ----------------small---------------------
            if n_obj == 2:
                pref = np.stack([np.linspace(0, 1, 100), 1 - np.linspace(0, 1, 100)]).T
                pref = torch.tensor(pref).to(device).float()

            if n_obj == 3:
                pref_size = 105
                pref = torch.tensor(das_dennis(13, 3)).to(device).float()

            if n_obj == 4:
                pref_size = 120
                pref = torch.tensor(das_dennis(7, 4)).to(device).float()

            if n_obj == 6:
                pref_size = 182
                pref = torch.tensor(get_reference_directions("multi-layer",
                                                             get_reference_directions("das-dennis", 6, n_partitions=4,
                                                                                      scaling=1.0),
                                                             get_reference_directions("das-dennis", 6, n_partitions=3,
                                                                                      scaling=0.5))).to(device).float()

            if n_obj == 9:
                pref_size = 90
                pref = torch.tensor(get_reference_directions("multi-layer",
                get_reference_directions("das-dennis", 9, n_partitions=2, scaling=1.0),
                get_reference_directions("das-dennis", 9, n_partitions=2, scaling=0.5))).to(device).float()

            sol = psmodel(pref)
            obj = problem.evaluate(sol)
            generated_ps = sol.cpu().numpy()
            generated_pf = obj.cpu().numpy()

            results_F_norm = (generated_pf - ideal_point) / (nadir_point - ideal_point)
            T = (True_PF - ideal_point) / (nadir_point - ideal_point)

            hv = HV(ref_point=np.array([1.1] * n_obj))
            if test_ins == 're91':
                hv_value = mc_hypervolume(results_F_norm, ref_point=np.array([1.1] * n_obj), sample_num=100000)
            else:
                hv_value = hv(results_F_norm)

            if test_ins == 're41':
                test = 0.8213476947317269
            elif test_ins == 're42':
                test = 0.9223397435426084
            elif test_ins == 're61':
                test = 1.2225904937002783
            elif test_ins == 're91':
                test = 0.15311592166668858
            else:
                test = hv(T)


            print('Time: ', stop - start)
            print("hv_gap", "{:.4e}".format(np.mean(hv_value)))
            print("true_hv_gap", "{:.4e}".format(np.mean(test)))
            a = np.mean(test) - np.mean(hv_value)
            print("Δhv", a)
            small_Hyper.append(a)

            # ------------------------large----------------------
            if n_obj == 2:
                pref = np.stack([np.linspace(0, 1, 1000), 1 - np.linspace(0, 1, 1000)]).T
                pref = torch.tensor(pref).to(device).float()

            if n_obj == 3:
                pref_size = 990
                pref = torch.tensor(das_dennis(43, 3)).to(device).float()

            if n_obj == 4:
                pref_size = 969
                pref = torch.tensor(das_dennis(16, 4)).to(device).float()

            if n_obj == 6:
                pref_size = 923
                pref = torch.tensor(get_reference_directions("multi-layer",
                            get_reference_directions("das-dennis", 6, n_partitions=6, scaling=1.0),
                            get_reference_directions("das-dennis", 6, n_partitions=6, scaling=0.5))).to(device).float()

            if n_obj == 9:
                pref_size = 990
                pref = torch.tensor(
                    get_reference_directions(
                        "multi-layer",
                        get_reference_directions("das-dennis", 9, n_partitions=4, scaling=1.0),
                        get_reference_directions("das-dennis", 9, n_partitions=4, scaling=0.5)
                    )
                ).to(device).float()

            sol = psmodel(pref)
            obj = problem.evaluate(sol)
            generated_ps = sol.cpu().numpy()
            generated_pf = obj.cpu().numpy()

            results_F_norm = (generated_pf - ideal_point) / (nadir_point - ideal_point)
            T = (True_PF - ideal_point) / (nadir_point - ideal_point)

            hv = HV(ref_point=np.array([1.1] * n_obj))
            if test_ins == 're61':
                hv_value = mc_hypervolume(results_F_norm, ref_point=np.array([1.1] * n_obj), sample_num=100000)
            elif test_ins == 're91':
                hv_value = mc_hypervolume(results_F_norm, ref_point=np.array([1.1] * n_obj), sample_num=100000)
            else:
                hv_value = hv(results_F_norm)

            if test_ins == 're41':
                test = 0.8213476947317269
            elif test_ins == 're42':
                test = 0.9223397435426084
            elif test_ins == 're61':
                test = 1.2225904937002783
            elif test_ins == 're91':
                test = 0.15311592166668858
            else:
                test = hv(T)

            print('Time: ', stop - start)
            print("hv_gap", "{:.4e}".format(np.mean(hv_value)))
            print("true_hv_gap", "{:.4e}".format(np.mean(test)))
            a = np.mean(test) - np.mean(hv_value)
            print("Δhv", a)
            large_Hyper.append(a)

    print(f"{test_ins} Δsmall HyperVolume：", small_Hyper)
    print(f"{test_ins} Δlarge HyperVolume：", large_Hyper)








