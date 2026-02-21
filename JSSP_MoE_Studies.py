import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from ft06_env import FT06JobShopEnv  # Assumes ft06_env.py is in the same directory
import matplotlib.pyplot as plt
import random
import time
import pandas as pd

# Reproducibility
random.seed(90)
np.random.seed(90)
torch.manual_seed(90)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cpu")

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98
entropy_coefficient_start = 0.01
entropy_coefficient_end = 0.001
entropy_decay_episode = 500
step_size = 20
episode = 5000

#   2: [makespan, total_tardiness]
#   3: [makespan, total_tardiness, idle_penalty]
#   4: [makespan, total_tardiness, idle_penalty, num_tardy_jobs]
num_objectives = 3

### ======================================================================================
### MODEL DEFINITIONS
### ======================================================================================

class A2C_FC(nn.Module):
    """
    Standard Actor-Critic with a simple Fully-Connected (FC) critic.
    The critic has a single body that outputs a vector for all objective values.
    """

    def __init__(self, state_dim, action_dim, num_objectives=num_objectives):
        super().__init__()
        self.data = []
        self.num_objectives = num_objectives
        h1, h2 = 128, 128

        # Actor Network for policy
        self.actor = nn.Sequential(
            nn.Linear(state_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, action_dim)
        )

        # Critic Network for value (vector-valued)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, self.num_objectives)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, mask: torch.Tensor | None = None):
        logits = self.actor(x)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(logits, dim=-1)

    def v(self, x):
        return self.critic(x)

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst, mask_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, truncate, done, mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done or truncate else 1.0
            done_lst.append([done_mask])
            mask_lst.append(mask)
        self.data = []

        return (
            torch.tensor(np.array(s_lst), dtype=torch.float32, device=device),
            torch.tensor(a_lst, device=device),
            torch.tensor(np.array(r_lst), dtype=torch.float32, device=device),
            torch.tensor(np.array(s_prime_lst), dtype=torch.float32, device=device),
            torch.tensor(done_lst, dtype=torch.float32, device=device),
            torch.tensor(np.array(mask_lst), dtype=torch.bool, device=device)
        )

    def train_net(self, reward_weights, entropy_coeff):
        s, a, r, s_prime, done, masks = self.make_batch()

        # Critic update (vector TD)
        v_s = self.v(s)
        v_s_prime = self.v(s_prime)
        td_target = r + gamma * v_s_prime * done
        critic_loss = F.mse_loss(v_s, td_target.detach())

        # Actor update with scalarized advantage
        delta = td_target - v_s
        scalar_advantage = (delta.detach() * reward_weights).sum(dim=1)

        pi = self.pi(s, mask=masks)
        m = Categorical(pi)
        log_probs = m.log_prob(a.squeeze())
        entropy = m.entropy()

        actor_loss = -(log_probs * scalar_advantage).mean()
        entropy_loss = -entropy_coeff * entropy.mean()

        total_loss = actor_loss + critic_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item()


class A2C_MultiHead(nn.Module):
    """
    An Actor-Critic with a shared-bottom, multi-head critic.
    """

    def __init__(self, state_dim, action_dim, num_objectives=num_objectives):
        super().__init__()
        self.data = []
        self.num_objectives = num_objectives
        h1, h2 = 128, 128

        # Actor Network for policy
        self.actor = nn.Sequential(
            nn.Linear(state_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, action_dim)
        )

        # Critic: Shared body
        self.critic_body = nn.Sequential(
            nn.Linear(state_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU()
        )
        # Critic: Separate heads for each objective
        self.critic_heads = nn.ModuleList(
            [nn.Linear(h2, 1) for _ in range(self.num_objectives)]
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, mask: torch.Tensor | None = None):
        logits = self.actor(x)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(logits, dim=-1)

    def v(self, x):
        features = self.critic_body(x)
        return torch.cat([head(features) for head in self.critic_heads], dim=-1)

    # put_data and make_batch
    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst, mask_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, truncate, done, mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done or truncate else 1.0
            done_lst.append([done_mask])
            mask_lst.append(mask)
        self.data = []
        return (
            torch.tensor(np.array(s_lst), dtype=torch.float32, device=device),
            torch.tensor(a_lst, device=device),
            torch.tensor(np.array(r_lst), dtype=torch.float32, device=device),
            torch.tensor(np.array(s_prime_lst), dtype=torch.float32, device=device),
            torch.tensor(done_lst, dtype=torch.float32, device=device),
            torch.tensor(np.array(mask_lst), dtype=torch.bool, device=device)
        )

    def train_net(self, reward_weights, entropy_coeff):
        s, a, r, s_prime, done, masks = self.make_batch()

        v_s = self.v(s)
        v_s_prime = self.v(s_prime)
        td_target = r + gamma * v_s_prime * done
        critic_loss = F.mse_loss(v_s, td_target.detach())

        delta = td_target - v_s
        scalar_advantage = (delta.detach() * reward_weights).sum(dim=1)

        pi = self.pi(s, mask=masks)
        m = Categorical(pi)
        log_probs = m.log_prob(a.squeeze())
        entropy = m.entropy()

        actor_loss = -(log_probs * scalar_advantage).mean()
        entropy_loss = -entropy_coeff * entropy.mean()

        total_loss = actor_loss + critic_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item()


class A2C_MoE(nn.Module):
    """
    Vector-valued MoE critic for multi-objective RL.
    - K experts; each outputs an m-dim value vector (m=num_objectives)
    """

    def __init__(self, state_dim, action_dim, num_objectives=num_objectives, num_experts=num_objectives,
                 gate_use_w=True, load_balance_coef=1e-3, gate_temp=1.0):
        super().__init__()
        self.data = []
        self.num_objectives = num_objectives
        self.num_experts = num_experts
        self.gate_use_w = gate_use_w
        self.load_balance_coef = load_balance_coef
        self.gate_temp = gate_temp

        h1, h2 = 128, 128

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, action_dim)
        )

        # Experts: each maps state -> m-dim value vector
        def make_expert():
            return nn.Sequential(
                nn.Linear(state_dim, h1), nn.ReLU(),
                nn.Linear(h1, h2), nn.ReLU(),
                nn.Linear(h2, self.num_objectives)
            )

        self.experts = nn.ModuleList([make_expert() for _ in range(num_experts)])

        # Gate: maps [state,(optional) w] -> K logits
        gate_in = state_dim + (self.num_objectives if self.gate_use_w else 0)
        self.gate = nn.Sequential(
            nn.Linear(gate_in, h1), nn.ReLU(),
            nn.Linear(h1, num_experts)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, mask: torch.Tensor | None = None):
        logits = self.actor(x)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(logits, dim=-1)

    def v_components(self, x):
        # Stack expert outputs: [B,K,m]
        expert_vals = torch.stack([E(x) for E in self.experts], dim=1)
        return expert_vals

    def gate_probs(self, x, w: torch.Tensor | None = None):
        if self.gate_use_w:
            assert w is not None and w.dim() == 2 and w.size(1) == self.num_objectives
            g_in = torch.cat([x, w], dim=-1)
        else:
            g_in = x
        logits = self.gate(g_in) / self.gate_temp
        return F.softmax(logits, dim=-1)  # [B,K]

    def v(self, x, w: torch.Tensor | None = None):
        expert_vals = self.v_components(x)  # [B,K,m]
        probs = self.gate_probs(x, w)       # [B,K]
        v_mix = (expert_vals * probs.unsqueeze(-1)).sum(dim=1)  # [B,m]
        return v_mix, probs, expert_vals

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst, mask_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, truncate, done, mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done or truncate else 1.0
            done_lst.append([done_mask])
            mask_lst.append(mask)
        self.data = []
        return (
            torch.tensor(np.array(s_lst), dtype=torch.float32, device=device),
            torch.tensor(a_lst, device=device),
            torch.tensor(np.array(r_lst), dtype=torch.float32, device=device),
            torch.tensor(np.array(s_prime_lst), dtype=torch.float32, device=device),
            torch.tensor(done_lst, dtype=torch.float32, device=device),
            torch.tensor(np.array(mask_lst), dtype=torch.bool, device=device)
        )

    def train_net(self, reward_weights, entropy_coeff):
        # reward_weights: [1,m]
        s, a, r, s_prime, done, masks = self.make_batch()

        # Critic targets (vector)
        v_s_prime, _, _ = self.v(s_prime, w=reward_weights.expand(s_prime.size(0), -1))
        td_target = r + gamma * v_s_prime * done  # [B,m]

        v_s, gate_p, expert_vals = self.v(s, w=reward_weights.expand(s.size(0), -1))
        critic_mse = F.mse_loss(v_s, td_target.detach())

        # Load balancing loss
        p_mean = gate_p.mean(dim=0)  # [K]
        uniform = torch.full_like(p_mean, 1.0 / self.num_experts)
        lb_loss = F.kl_div((p_mean + 1e-8).log(), uniform, reduction='batchmean')

        critic_loss = critic_mse + self.load_balance_coef * lb_loss

        # Actor loss (scalarized advantage)
        delta = (td_target - v_s).detach()  # [B,m]
        scalar_advantage = (delta * reward_weights).sum(dim=1)  # [B]

        pi = self.pi(s, mask=masks)
        m = Categorical(pi)
        log_probs = m.log_prob(a.squeeze())
        entropy = m.entropy()

        actor_loss = -(log_probs * scalar_advantage).mean()
        entropy_loss = -entropy_coeff * entropy.mean()

        total_loss = actor_loss + critic_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_mse.item()


class HeadsThenMoE(nn.Module):
    """
    Shared trunk -> per-objective base heads h(s) in R^m -> MoE coupling (R^m->R^m) gated by state and optional w.
    """

    def __init__(self, state_dim, action_dim, num_objectives=num_objectives, K=num_objectives,
                 gate_use_w=True, gate_temp=1.0, lb_coef=1e-3, couple_type="affine",
                 clip_grad=0.5):
        super().__init__()
        self.m = num_objectives
        self.K = K
        self.gate_use_w = gate_use_w
        self.gate_temp = gate_temp
        self.lb_coef = lb_coef
        self.clip_grad = clip_grad

        h1, h2 = 128, 128

        # Trunk shared by actor & critic
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU()
        )

        # Actor head
        self.actor_head = nn.Linear(h2, action_dim)

        # Multi-head critic base: z -> h(s) in R^m
        self.base_heads = nn.ModuleList([nn.Linear(h2, 1) for _ in range(self.m)])

        # Experts: map R^m -> R^m
        self.couple_type = couple_type
        if couple_type == "affine":
            self.A = nn.Parameter(torch.stack([torch.eye(self.m) for _ in range(K)], dim=0))  # [K,m,m]
            self.b = nn.Parameter(torch.zeros(K, self.m))  # [K,m]
            self.experts = None
            with torch.no_grad():
                for k in range(K):
                    self.A[k].copy_(torch.eye(self.m))
                    self.b[k].zero_()
        else:
            hidden = 64
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.m, hidden), nn.ReLU(),
                    nn.Linear(hidden, self.m)
                ) for _ in range(K)
            ])
            self.A = self.b = None

        # Gate over experts, conditioned on trunk features and optional w
        gate_in = h2 + (self.m if gate_use_w else 0)
        self.gate = nn.Sequential(
            nn.Linear(gate_in, 64), nn.ReLU(),
            nn.Linear(64, K)
        )

        # Optimizer and buffer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.data = []

    # --------- Policy ----------
    def pi(self, x, mask: torch.Tensor | None = None):
        z = self.trunk(x)
        logits = self.actor_head(z)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))
        return F.softmax(logits, dim=-1)

    # --------- Critic helpers ----------
    def _base_h(self, z):
        return torch.cat([head(z) for head in self.base_heads], dim=-1)  # [B,m]

    def _experts_apply(self, h):
        # h: [B,m] -> [B,K,m]
        if self.experts is not None:
            return torch.stack([E(h) for E in self.experts], dim=1)
        # affine: out[b,k,:] = A[k] @ h[b,:] + b[k]
        return torch.einsum('kmn,bn->bkm', self.A, h) + self.b.unsqueeze(0)

    def _gate_probs(self, z, w_row=None):
        g_in = torch.cat([z, w_row], dim=-1) if (self.gate_use_w and w_row is not None) else z
        logits = self.gate(g_in) / self.gate_temp
        return F.softmax(logits, dim=-1)  # [B,K]

    def v(self, x, w_row=None):
        """
        Returns:
          V_mix: [B,m]  final value vector
          gate_p: [B,K] gate probs
          h: [B,m]      base head outputs
        """
        z = self.trunk(x)
        h = self._base_h(z)  # [B,m]
        E = self._experts_apply(h)  # [B,K,m]
        gate_p = self._gate_probs(z, w_row)  # [B,K]
        V_mix = (E * gate_p.unsqueeze(-1)).sum(dim=1)  # [B,m]
        return V_mix, gate_p, h

    # --------- Buffer I/O ----------
    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        dev = next(self.parameters()).device
        s_lst, a_lst, r_lst, s_prime_lst, done_lst, mask_lst = [], [], [], [], [], []
        for (s, a, r, s_prime, truncate, done, mask) in self.data:
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append(r)  # vector reward
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if (done or truncate) else 1.0
            done_lst.append([done_mask])
            mask_lst.append(mask)
        self.data = []
        return (
            torch.tensor(np.array(s_lst), dtype=torch.float32, device=dev),
            torch.tensor(a_lst, device=dev),
            torch.tensor(np.array(r_lst), dtype=torch.float32, device=dev),
            torch.tensor(np.array(s_prime_lst), dtype=torch.float32, device=dev),
            torch.tensor(done_lst, dtype=torch.float32, device=dev),
            torch.tensor(np.array(mask_lst), dtype=torch.bool, device=dev),
        )

    # --------- A2C update ----------
    def train_net(self, reward_weights: torch.Tensor, entropy_coeff: float):
        """
        reward_weights: [1,m] row vector; expanded to [B,m] inside.
        """
        s, a, r, s_prime, done, masks = self.make_batch()
        B, m = s.size(0), r.size(1)
        wB = reward_weights.expand(B, -1)  # [B,m]

        # Critic (vector TD)
        v_sp, _, _ = self.v(s_prime, w_row=wB)  # [B,m]
        td_target = r + gamma * v_sp * done  # [B,m]
        v_s, gate_p, _ = self.v(s, w_row=wB)  # [B,m]
        critic_mse = F.mse_loss(v_s, td_target.detach())  # avg over B*m

        # Gate load-balance KL to uniform
        p_mean = gate_p.mean(dim=0)  # [K]
        uniform = torch.full_like(p_mean, 1.0 / self.K)
        lb_loss = F.kl_div((p_mean + 1e-8).log(), uniform, reduction='batchmean')

        # Optional identity reg for affine experts
        id_reg = torch.tensor(0.0, device=s.device)
        if getattr(self, "A", None) is not None and self.A is not None:
            I = torch.eye(m, device=s.device)
            id_reg = sum((Ak - I).pow(2).sum() for Ak in self.A) / self.K
            id_reg = 1e-4 * id_reg

        # Actor (weight-scalarized advantage)
        delta = (td_target - v_s).detach()  # [B,m]
        scalar_adv = (delta * wB).sum(dim=1)  # [B]
        pi = self.pi(s, mask=masks)
        dist = Categorical(pi)
        logp = dist.log_prob(a.squeeze())
        entropy = dist.entropy()

        actor_loss = -(logp * scalar_adv).mean()
        entropy_loss = -entropy_coeff * entropy.mean()

        total_loss = actor_loss + critic_mse + self.lb_coef * lb_loss + entropy_loss + id_reg

        self.optimizer.zero_grad()
        total_loss.backward()
        if self.clip_grad is not None and self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad)
        self.optimizer.step()

        return actor_loss.item(), critic_mse.item()

def _train_m2_single_model(model_class, model_name: str, model_path: str,
                           reward_weights=(0.7, 0.3),
                           total_episodes: int = 2000,
                           base_seed: int = 12345):
    """
    Helper: train a single MoE-type model with 2 objectives (first two env rewards)
    and save state_dict to `model_path`.

    model_class: HeadsThenMoE or A2C_MoE
    reward_weights: length-2 tuple of scalarization weights
    """
    if len(reward_weights) != 2:
        raise ValueError(f"{model_name}: reward_weights must have length 2 for 2-objective training.")

    env = FT06JobShopEnv()  # env returns a 3-dim reward; we use only first two for m=2
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 2-objective critic
    model = model_class(state_dim, action_dim, num_objectives=2).to(device)
    w_vec = np.array(reward_weights, dtype=np.float32)
    w_tensor = torch.tensor(w_vec, dtype=torch.float32, device=device).view(1, -1)

    print(f"[2-objective pretrain:{model_name}] w={reward_weights}, episodes={total_episodes}")

    current_time = time.time()

    for epi in range(total_episodes):
        raw = entropy_coefficient_start + (entropy_coefficient_end - entropy_coefficient_start) * (
                epi / entropy_decay_episode)
        entropy_coeff = float(np.clip(raw, entropy_coefficient_end, entropy_coefficient_start))

        s, _ = env.reset(seed=base_seed + epi)
        done, truncate = False, False

        while not done and not truncate:
            for _ in range(step_size):
                state_tensor = torch.from_numpy(s).float().unsqueeze(0).to(device)

                valid_actions = env.action_mask()
                if not valid_actions.any():
                    done = True
                    break

                mask_tensor = torch.from_numpy(valid_actions).bool().unsqueeze(0).to(device)

                with torch.no_grad():
                    prob = model.pi(state_tensor, mask=mask_tensor)
                    action = Categorical(prob).sample().item()

                s_prime, r_full, done, truncate, _ = env.step(action)

                # use only first two objectives for 2-objective training
                r_vec = np.asarray(r_full, dtype=np.float32)[:2]
                model.put_data((s, action, r_vec, s_prime, truncate, done, valid_actions))
                s = s_prime

                if done or truncate:
                    break

            if len(model.data) > 0:
                model.train_net(w_tensor, entropy_coeff)

        if (epi + 1) % 100 == 0:
            print(f"[2-obj {model_name}] episode {epi + 1}/{total_episodes}")

    env.close()
    cpu = time.time() - current_time
    print("CPU time =",cpu,"sec")

    torch.save(model.state_dict(), model_path)
    print(f"[2-objective pretrain:{model_name}] saved model to {model_path}")
    return model_path


def pretrain_m2_and_save(
    heads_path: str = "heads_m2_pretrained.pt",
    moe_path: str = "a2cmoe_m2_pretrained.pt",
    reward_weights=(0.7, 0.3),
    total_episodes: int = 2000,
):
    """
    Train HeadsThenMoE and A2C_MoE with 2 objectives (first two env rewards),
    then save them to `heads_path` and `moe_path`.
    """
    _train_m2_single_model(HeadsThenMoE, "HeadsThenMoE", heads_path,
                           reward_weights=reward_weights,
                           total_episodes=total_episodes,
                           base_seed=12345)
    _train_m2_single_model(A2C_MoE, "A2C_MoE", moe_path,
                           reward_weights=reward_weights,
                           total_episodes=total_episodes,
                           base_seed=22345)

# =====================================================================
# 2) Build 3-objective models from saved 2-objective models
# =====================================================================

def build_heads_m3_from_heads_m2(
    pretrained_path: str,
    state_dim: int,
    action_dim: int,
    K: int = 2,
) -> HeadsThenMoE:
    """
    Load a 2-objective HeadsThenMoE, then create a 3-objective HeadsThenMoE
    whose trunk, actor_head, and first two base heads are initialized from the
    2-objective model. Gate & experts & 3rd head are left at their random init.
    """
    # load m=2 model
    m2 = HeadsThenMoE(state_dim, action_dim, num_objectives=2, K=K).to(device)
    state_dict_2 = torch.load(pretrained_path, map_location=device)
    m2.load_state_dict(state_dict_2)

    # create m=3 model
    m3 = HeadsThenMoE(state_dim, action_dim, num_objectives=3, K=K).to(device)

    # copy trunk & actor
    m3.trunk.load_state_dict(m2.trunk.state_dict())
    m3.actor_head.load_state_dict(m2.actor_head.state_dict())

    # copy first two base heads (for old objectives)
    for i in range(2):
        m3.base_heads[i].load_state_dict(m2.base_heads[i].state_dict())

    print("[HeadsThenMoE adapt-init] copied trunk, actor_head, first 2 base_heads from 2-objective model.")
    return m3


def build_a2cmoe_m3_from_m2(
    pretrained_path: str,
    state_dim: int,
    action_dim: int,
    num_experts: int = 2,
) -> A2C_MoE:
    """
    Load a 2-objective A2C_MoE, then create a 3-objective A2C_MoE whose actor
    and the shared parts of the experts are initialized from the old model.
    The final expert output layer (h2 -> 3) and gate remain at random init.
    """
    m2 = A2C_MoE(state_dim, action_dim, num_objectives=2, num_experts=num_experts).to(device)
    state_dict_2 = torch.load(pretrained_path, map_location=device)
    m2.load_state_dict(state_dict_2)

    m3 = A2C_MoE(state_dim, action_dim, num_objectives=3, num_experts=num_experts).to(device)

    m3.actor.load_state_dict(m2.actor.state_dict())

    for e3, e2 in zip(m3.experts, m2.experts):
        sd2 = e2.state_dict()
        sd3 = e3.state_dict()

        for name, param in e3.named_parameters():
            if name in sd2 and sd2[name].shape == param.shape:
                with torch.no_grad():
                    param.copy_(sd2[name])

    print("[A2C_MoE adapt-init] copied actor and early expert layers from 2-objective model.")
    return m3

# =====================================================================
# 3) Adaptive training for 3 objectives using the pretrained models
# =====================================================================

def adaptive_train_heads_m3(
    pretrained_heads_path: str = "heads_m2_pretrained.pt",
    reward_weights=(0.8, 0.1, 0.1),
    total_episodes: int = 1000,
    freeze_pretrained: bool = True,
    K: int = 2,
):
    """
    Adaptive training of a 3-objective HeadsThenMoE, initialized from
    a 2-objective checkpoint.

    - If freeze_pretrained=True, the trunk, actor_head, and the first two
      base_heads are frozen; only the 3rd head + MoE parameters adapt.
    """
    if len(reward_weights) != 3:
        raise ValueError("HeadsThenMoE (m=3): reward_weights must have length 3.")

    env = FT06JobShopEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = build_heads_m3_from_heads_m2(pretrained_heads_path, state_dim, action_dim, K=K)

    if freeze_pretrained:
        for p in model.trunk.parameters():
            p.requires_grad = False
        for p in model.actor_head.parameters():
            p.requires_grad = False
        # freeze first two objectives' heads
        for i in range(2):
            for p in model.base_heads[i].parameters():
                p.requires_grad = False
        print("[HeadsThenMoE adapt-train] froze trunk, actor_head, and first 2 base_heads.")

    # optimizer over trainable parameters only
    model.optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    w_vec = np.array(reward_weights, dtype=np.float32)
    w_tensor = torch.tensor(w_vec, dtype=torch.float32, device=device).view(1, -1)

    print(f"[HeadsThenMoE adaptive m=3] w={reward_weights}, episodes={total_episodes}")
    current_time = time.time()

    for epi in range(total_episodes):
        raw = entropy_coefficient_start + (entropy_coefficient_end - entropy_coefficient_start) * (
                epi / entropy_decay_episode)
        entropy_coeff = float(np.clip(raw, entropy_coefficient_end, entropy_coefficient_start))

        s, _ = env.reset(seed=30000 + epi)
        done, truncate = False, False

        while not done and not truncate:
            for _ in range(step_size):
                state_tensor = torch.from_numpy(s).float().unsqueeze(0).to(device)

                valid_actions = env.action_mask()
                if not valid_actions.any():
                    done = True
                    break

                mask_tensor = torch.from_numpy(valid_actions).bool().unsqueeze(0).to(device)

                with torch.no_grad():
                    prob = model.pi(state_tensor, mask=mask_tensor)
                    action = Categorical(prob).sample().item()

                s_prime, r_full, done, truncate, _ = env.step(action)

                # use the full 3 objectives for m=3
                r_vec = np.asarray(r_full, dtype=np.float32)[:3]
                model.put_data((s, action, r_vec, s_prime, truncate, done, valid_actions))
                s = s_prime

                if done or truncate:
                    break

            if len(model.data) > 0:
                model.train_net(w_tensor, entropy_coeff)

        if (epi + 1) % 100 == 0:
            print(f"[HeadsThenMoE m=3 adaptive] episode {epi + 1}/{total_episodes}")

    env.close()
    cpu = time.time() - current_time
    print("CPU time =",cpu,"sec")

    torch.save(model.state_dict(), "heads_m3_adaptive.pt")
    print("[HeadsThenMoE adaptive] saved 3-objective model to heads_m3_adaptive.pt")


def adaptive_train_a2cmoe_m3(
    pretrained_moe_path: str = "a2cmoe_m2_pretrained.pt",
    reward_weights=(0.8, 0.1, 0.1),
    total_episodes: int = 1000,
    freeze_pretrained: bool = False,
    num_experts: int = 2,
    log_path: str = "a2cmoe_m3_adaptive_logs.csv",
    model_save_path: str = "a2cmoe_m3_adaptive.pt",
):
    """
    Adaptive training of a 3-objective A2C_MoE, initialized from
    a 2-objective checkpoint.

    - If freeze_pretrained=True, actor and the expert layers that came from
      m=2 can be frozen, leaving only the new output dims & gate to adapt.
    """
    if len(reward_weights) != 3:
        raise ValueError("A2C_MoE (m=3): reward_weights must have length 3.")

    env = FT06JobShopEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = build_a2cmoe_m3_from_m2(pretrained_moe_path, state_dim, action_dim, num_experts=num_experts)

    if freeze_pretrained:
        # freeze actor and the early expert layers;
        # note: the last linear layer in each expert was never loaded from m=2.
        for p in model.actor.parameters():
            p.requires_grad = False
        for expert in model.experts:
            # all layers were (partially) copied from m=2; you could selectively
            # freeze only some of them, but here we freeze whole experts.
            for p in expert.parameters():
                p.requires_grad = False
        print("[A2C_MoE adapt-train] froze actor and expert parameters from 2-objective pretrain.")

    model.optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    w_vec = np.array(reward_weights, dtype=np.float32)
    w_tensor = torch.tensor(w_vec, dtype=torch.float32, device=device).view(1, -1)

    rewards_log, actors_log, critics_log = [], [], []
    final_reward_lst, objectives = [], []

    print(f"[A2C_MoE adaptive m=3] w={reward_weights}, episodes={total_episodes}")

    current_time = time.time()

    for epi in range(total_episodes):
        raw = entropy_coefficient_start + (entropy_coefficient_end - entropy_coefficient_start) * (
            epi / entropy_decay_episode
        )
        entropy_coeff = float(np.clip(raw, entropy_coefficient_end, entropy_coefficient_start))

        s, _ = env.reset(seed=50000 + epi)
        done, truncate = False, False
        episode_reward = 0.0
        actor_loss = critic_loss = 0.0
        last_r_vec = np.zeros(4, dtype=np.float32)

        while not done and not truncate:
            for _ in range(step_size):
                state_tensor = torch.from_numpy(s).float().unsqueeze(0).to(device)

                valid_actions = env.action_mask()
                if not valid_actions.any():
                    done = True
                    break

                mask_tensor = torch.from_numpy(valid_actions).bool().unsqueeze(0).to(device)

                with torch.no_grad():
                    prob = model.pi(state_tensor, mask=mask_tensor)
                    action = torch.distributions.Categorical(prob).sample().item()

                s_prime, r_full, done, truncate, _ = env.step(action)

                r_vec = np.asarray(r_full, dtype=np.float32)[:3]
                last_r_vec = r_vec.copy()

                model.put_data((s, action, r_vec, s_prime, truncate, done, valid_actions))

                episode_reward += float((r_vec * w_vec).sum())
                s = s_prime

                if done or truncate:
                    break

            if len(model.data) > 0:
                actor_loss, critic_loss = model.train_net(w_tensor, entropy_coeff)

        final_reward = float((last_r_vec * w_vec).sum())
        rewards_log.append(episode_reward)
        actors_log.append(actor_loss)
        critics_log.append(critic_loss)
        final_reward_lst.append(final_reward)
        objectives.append(last_r_vec)

        if (epi + 1) % 100 == 0:
            print(f"[A2C_MoE m=3 adaptive] episode {epi + 1}/{total_episodes}")

    env.close()
    cpu = time.time() - current_time
    print("CPU time =",cpu,"sec")

    # ==============================
    # Log CSV
    # ==============================
    obj_array = np.stack(objectives, axis=0)  # [episode, 4]

    data = {
        "Episode": np.arange(total_episodes),
        "Reward": rewards_log,
        "ActorLoss": actors_log,
        "CriticLoss": critics_log,
        "FinalReward": final_reward_lst,
    }
    for i in range(obj_array.shape[1]):
        data[f"Objective {i + 1}"] = obj_array[:, i]

    df = pd.DataFrame(data)

    if log_path:
        if not os.path.exists(log_path):
            df.to_csv(log_path, index=False)
        else:
            df.to_csv(log_path, mode='a', header=False, index=False)

    # ==============================
    # Save Model
    # ==============================
    torch.save(model.state_dict(), model_save_path)
    print(f"[A2C_MoE adaptive m=3] saved logs to {log_path} and model to {model_save_path}")

    return df, model_save_path

### ======================================================================================
### TRAINING AND EVALUATION
### ======================================================================================


def run_experiment(
    model_class,
    reward_weights,
    base_seed,
    save_path=None,
    rep_id=0,
    model_name="FC",
    model_save_dir=None,
):
    """
    Consolidated function to run a training experiment for a given model class.
    In addition to the training logs (CSV), this also optionally saves
    the trained model parameters for each replication.
    """
    if len(reward_weights) != num_objectives:
        raise ValueError(
            f"len(reward_weights)={len(reward_weights)} and num_objectives={num_objectives} are different."
        )

    env = FT06JobShopEnv(num_objectives=num_objectives)
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    weights_tensor = torch.tensor(reward_weights, dtype=torch.float32, device=device).view(1, -1)

    model = model_class(state_dim, action_dim, num_objectives=num_objectives).to(device)
    rewards_log, actors_log, critics_log = [], [], []
    final_reward_lst = []
    objectives = []

    if model_save_dir is not None:
        os.makedirs(model_save_dir, exist_ok=True)

    current_time = time.time()

    for epi in range(episode):
        raw = entropy_coefficient_start + (entropy_coefficient_end - entropy_coefficient_start) * (
            epi / entropy_decay_episode
        )
        entropy_coeff = float(np.clip(raw, entropy_coefficient_end, entropy_coefficient_start))

        s, _ = env.reset(seed=base_seed + epi)
        done, truncate = False, False
        episode_reward = 0.0

        actor_loss = critic_loss = 0.0
        r = np.zeros(num_objectives, dtype=np.float32)

        while not done and not truncate:
            for _ in range(step_size):
                state_tensor = torch.from_numpy(s).float().unsqueeze(0).to(device)

                valid_actions = env.action_mask()
                if not valid_actions.any():
                    done = True
                    break

                mask_tensor = torch.from_numpy(valid_actions).bool().unsqueeze(0).to(device)

                with torch.no_grad():
                    prob = model.pi(state_tensor, mask=mask_tensor)
                    action = torch.distributions.Categorical(prob).sample().item()

                s_prime, r_env, done, truncate, _ = env.step(action)

                r = np.asarray(r_env, dtype=np.float32)[:num_objectives]

                model.put_data((s, action, r, s_prime, truncate, done, valid_actions))

                episode_reward += float((r * np.array(reward_weights, dtype=np.float32)).sum())
                s = s_prime

                if done or truncate:
                    break

            if len(model.data) > 0:
                actor_loss, critic_loss = model.train_net(weights_tensor, entropy_coeff)

        final_reward = float((r * np.array(reward_weights, dtype=np.float32)).sum())

        rewards_log.append(episode_reward)
        actors_log.append(actor_loss)
        critics_log.append(critic_loss)
        final_reward_lst.append(final_reward)
        objectives.append(r)

    env.close()

    cpu = time.time() - current_time
    print("CPU time =", cpu, "sec")
    Objective_array = np.array(objectives)

    if save_path:
        data = {
            "Replication": rep_id,
            "Model": model_name,
            "Episode": np.arange(len(rewards_log)),
            "Reward": rewards_log,
            "ActorLoss": actors_log,
            "CriticLoss": critics_log,
            "FinalReward": final_reward_lst,
        }
        for i in range(Objective_array.shape[1]):
            data[f"Objective {i + 1}"] = Objective_array[:, i]

        df = pd.DataFrame(data)

        if not os.path.exists(save_path):
            df.to_csv(save_path, index=False)
        else:
            df.to_csv(save_path, mode='a', header=False, index=False)

    if model_save_dir is not None:
        model_filename = f"{model_name}_rep{rep_id:03d}.pt"
        save_full_path = os.path.join(model_save_dir, model_filename)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_class": model_class.__name__,
                "reward_weights": reward_weights,
                "num_objectives": num_objectives,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "base_seed": base_seed,
            },
            save_full_path,
        )
        print(f"[{model_name}] Rep {rep_id} Model save -> {save_full_path}")

    return rewards_log, actors_log, critics_log, final_reward_lst, objectives


def replicate_and_compare(*weights, replications=10):
    """
    weights: scalar weight
        num_objectives=2  -> replicate_and_compare(0.7, 0.3, replications=...)
        num_objectives=3  -> replicate_and_compare(0.8, 0.1, 0.1, replications=...)
        num_objectives=4  -> replicate_and_compare(0.4, 0.3, 0.2, 0.1, replications=...)
    """
    reward_weights = list(weights)
    if len(reward_weights) != num_objectives:
        raise ValueError(
            f"num_objectives={num_objectives}, weight ={len(reward_weights)}."
        )

    weight_tag = "_".join(f"{w:.2f}" for w in reward_weights)
    base_model_dir = os.path.join("saved_models", f"m{num_objectives}_w_{weight_tag}")

    all_fc_rewards, all_mh_rewards, all_moe_rewards, all_moe2_rewards = [], [], [], []
    all_fc_critics, all_mh_critics, all_moe_critics, all_moe2_critics = [], [], [], []
    all_fc_actor, all_mh_actor, all_moe_actor, all_moe2_actor = [], [], [], []

    for rep in range(replications):
        print(f"Replication {rep + 1}/{replications} for weights {tuple(reward_weights)}")
        base_seed = 12345 + rep * 1000

        fc_dir = os.path.join(base_model_dir, "FC")
        mh_dir = os.path.join(base_model_dir, "MultiHead")
        moe_dir = os.path.join(base_model_dir, "MoE")
        mmoe_dir = os.path.join(base_model_dir, "MMoE")

        # Run FC Model
        print("...Training FC Model")
        fc_rewards, fc_actor, fc_critics, fc_final_rewards, _ = run_experiment(
            A2C_FC,
            reward_weights,
            base_seed,
            save_path=f"fc_m{num_objectives}_logs.csv",
            rep_id=rep,
            model_name="FC",
            model_save_dir=fc_dir,
        )
        all_fc_rewards.append(fc_rewards)
        all_fc_critics.append(fc_critics)
        all_fc_actor.append(fc_actor)
        print("FC_reward:", fc_final_rewards[-1])

        # Run Multi-Head Model
        print("...Training Multi-Head Model")
        mh_rewards, mh_actor, mh_critics, mh_final_rewards, _ = run_experiment(
            A2C_MultiHead,
            reward_weights,
            base_seed,
            save_path=f"multihead_m{num_objectives}_logs.csv",
            rep_id=rep,
            model_name="MH",
            model_save_dir=mh_dir,
        )
        all_mh_rewards.append(mh_rewards)
        all_mh_critics.append(mh_critics)
        all_mh_actor.append(mh_actor)
        print("MH_reward:", mh_final_rewards[-1])

        # Run MoE Model
        print("...Training MoE Model")
        moe_rewards, moe_actor, moe_critics, moe_final_rewards, _ = run_experiment(
            A2C_MoE,
            reward_weights,
            base_seed,
            save_path=f"moe_m{num_objectives}_logs.csv",
            rep_id=rep,
            model_name="A2C_MoE",
            model_save_dir=moe_dir,
        )
        all_moe_rewards.append(moe_rewards)
        all_moe_critics.append(moe_critics)
        all_moe_actor.append(moe_actor)
        print("MoE_reward:", moe_final_rewards[-1])

        # Run Multi-head+MoE Model
        print("...Training MMoE Model")
        moe2_rewards, moe2_actor, moe2_critics, moe2_final_rewards, _ = run_experiment(
            HeadsThenMoE,
            reward_weights,
            base_seed,
            save_path=f"mmoe_m{num_objectives}_logs.csv",
            rep_id=rep,
            model_name="HeadsThenMoE",
            model_save_dir=mmoe_dir,
        )
        all_moe2_rewards.append(moe2_rewards)
        all_moe2_critics.append(moe2_critics)
        all_moe2_actor.append(moe2_actor)
        print("MMoE_reward:", moe2_final_rewards[-1])

    # Process and Plot Results
    fc_rewards_mean = np.mean(all_fc_rewards, axis=0)
    fc_rewards_std = np.std(all_fc_rewards, axis=0)
    mh_rewards_mean = np.mean(all_mh_rewards, axis=0)
    mh_rewards_std = np.std(all_mh_rewards, axis=0)
    moe_rewards_mean = np.mean(all_moe_rewards, axis=0)
    moe_rewards_std = np.std(all_moe_rewards, axis=0)
    moe2_rewards_mean = np.mean(all_moe2_rewards, axis=0)
    moe2_rewards_std = np.std(all_moe2_rewards, axis=0)

    x_line = np.arange(episode)

    plt.figure(figsize=(10, 6))
    plt.plot(x_line, fc_rewards_mean, label="FC")
    plt.fill_between(x_line, fc_rewards_mean - fc_rewards_std, fc_rewards_mean + fc_rewards_std, alpha=0.2)

    plt.plot(x_line, mh_rewards_mean, label="Multi-Head")
    plt.fill_between(x_line, mh_rewards_mean - mh_rewards_std, mh_rewards_mean + mh_rewards_std, alpha=0.2)

    plt.plot(x_line, moe_rewards_mean, label="MoE")
    plt.fill_between(x_line, moe_rewards_mean - moe_rewards_std, moe_rewards_mean + moe_rewards_std, alpha=0.2)

    plt.plot(x_line, moe2_rewards_mean, label="Multi-Head+MoE")
    plt.fill_between(x_line, moe2_rewards_mean - moe2_rewards_std, moe2_rewards_mean + moe2_rewards_std, alpha=0.2)

    plt.legend()
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title(f"m={num_objectives}, weights={tuple(reward_weights)}")
    plt.grid(True)
    plt.show()

def build_a2cmoe_m4_from_m3(
    pretrained_path: str,
    state_dim: int,
    action_dim: int,
    num_experts: int = 2,
) -> A2C_MoE:
    # 1) m=3
    m3 = A2C_MoE(state_dim, action_dim, num_objectives=3, num_experts=num_experts).to(device)
    state_dict_3 = torch.load(pretrained_path, map_location=device)
    m3.load_state_dict(state_dict_3)

    # 2) m=4
    m4 = A2C_MoE(state_dim, action_dim, num_objectives=4, num_experts=num_experts).to(device)

    # 3) actor
    m4.actor.load_state_dict(m3.actor.state_dict())

    # 4)
    for e4, e3 in zip(m4.experts, m3.experts):
        sd3 = e3.state_dict()
        for name, param in e4.named_parameters():
            if name in sd3 and sd3[name].shape == param.shape:
                with torch.no_grad():
                    param.copy_(sd3[name])

    print("[A2C_MoE adapt-init m=4] copied actor and early expert layers from 3-objective model.")
    return m4


def adaptive_train_a2cmoe_m4(
    pretrained_m3_path: str = "a2cmoe_m3_adaptive.pt",
    reward_weights=(0.7, 0.1, 0.1, 0.1),
    total_episodes: int = 1000,
    freeze_pretrained: bool = False,
    num_experts: int = 2,
    log_path: str = "a2cmoe_m4_adaptive_logs.csv",
    model_save_path: str = "a2cmoe_m4_adaptive.pt",
):

    if len(reward_weights) != 4:
        raise ValueError("A2C_MoE (m=4): reward_weights must have length 4.")

    # 4-objective
    env = FT06JobShopEnv(num_objectives=4)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 3-objective → 4-objective MoE
    model = build_a2cmoe_m4_from_m3(
        pretrained_path=pretrained_m3_path,
        state_dim=state_dim,
        action_dim=action_dim,
        num_experts=num_experts,
    )

    if freeze_pretrained:
        for p in model.actor.parameters():
            p.requires_grad = False
        for expert in model.experts:
            for p in expert.parameters():
                p.requires_grad = False
        print("[A2C_MoE m=4 adaptive] froze actor and expert parameters from 3-objective model.")

    model.optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    w_vec = np.array(reward_weights, dtype=np.float32)
    w_tensor = torch.tensor(w_vec, dtype=torch.float32, device=device).view(1, -1)

    rewards_log, actors_log, critics_log = [], [], []
    final_reward_lst, objectives = [], []

    print(f"[A2C_MoE adaptive m=4] w={reward_weights}, episodes={total_episodes}")
    current_time = time.time()

    for epi in range(total_episodes):
        raw = entropy_coefficient_start + (entropy_coefficient_end - entropy_coefficient_start) * (
            epi / entropy_decay_episode
        )
        entropy_coeff = float(np.clip(raw, entropy_coefficient_end, entropy_coefficient_start))

        s, _ = env.reset(seed=50000 + epi)
        done, truncate = False, False
        episode_reward = 0.0
        actor_loss = critic_loss = 0.0
        last_r_vec = np.zeros(4, dtype=np.float32)

        while not done and not truncate:
            for _ in range(step_size):
                state_tensor = torch.from_numpy(s).float().unsqueeze(0).to(device)

                valid_actions = env.action_mask()
                if not valid_actions.any():
                    done = True
                    break

                mask_tensor = torch.from_numpy(valid_actions).bool().unsqueeze(0).to(device)

                with torch.no_grad():
                    prob = model.pi(state_tensor, mask=mask_tensor)
                    action = torch.distributions.Categorical(prob).sample().item()

                s_prime, r_full, done, truncate, _ = env.step(action)

                r_vec = np.asarray(r_full, dtype=np.float32)[:4]
                last_r_vec = r_vec.copy()

                model.put_data((s, action, r_vec, s_prime, truncate, done, valid_actions))

                episode_reward += float((r_vec * w_vec).sum())
                s = s_prime

                if done or truncate:
                    break

            if len(model.data) > 0:
                actor_loss, critic_loss = model.train_net(w_tensor, entropy_coeff)

        final_reward = float((last_r_vec * w_vec).sum())
        rewards_log.append(episode_reward)
        actors_log.append(actor_loss)
        critics_log.append(critic_loss)
        final_reward_lst.append(final_reward)
        objectives.append(last_r_vec)

        if (epi + 1) % 100 == 0:
            print(f"[A2C_MoE m=4 adaptive] episode {epi + 1}/{total_episodes}")

    env.close()
    cpu = time.time() - current_time
    print("CPU time =", cpu, "sec")

    # ==============================
    # Log csv saving
    # ==============================
    obj_array = np.stack(objectives, axis=0)  # [episode, 4]

    data = {
        "Episode": np.arange(total_episodes),
        "Reward": rewards_log,
        "ActorLoss": actors_log,
        "CriticLoss": critics_log,
        "FinalReward": final_reward_lst,
    }
    for i in range(obj_array.shape[1]):
        data[f"Objective {i + 1}"] = obj_array[:, i]

    df = pd.DataFrame(data)

    if log_path:
        if not os.path.exists(log_path):
            df.to_csv(log_path, index=False)
        else:
            df.to_csv(log_path, mode='a', header=False, index=False)

    # ==============================
    # Mdoel saving
    # ==============================
    torch.save(model.state_dict(), model_save_path)
    print(f"[A2C_MoE adaptive m=4] saved logs to {log_path} and model to {model_save_path}")

    return df, model_save_path


if __name__ == "__main__":
    print("Starting training...")

    print("Example: 3-objective adaptive MoE → 4-objective adaptive MoE with logging")

    #
    # pretrain_m2_and_save(
    #     heads_path="heads_m2_pretrained.pt",
    #     moe_path="a2cmoe_m2_pretrained.pt",
    #     reward_weights=(0.7, 0.3),   # 2개 objective 가중치 (makespan, tardiness)
    #     total_episodes=5000,
    # )
    #
    # adaptive_train_heads_m3(
    #     pretrained_heads_path="heads_m2_pretrained.pt",
    #     reward_weights=(0.8, 0.1, 0.1),
    #     total_episodes=5000,
    #     freeze_pretrained=True,
    # )
    #
    adaptive_train_a2cmoe_m3(
        pretrained_moe_path="a2cmoe_m2_pretrained.pt",
        reward_weights=(0.8, 0.1, 0.1),
        total_episodes=5000,
        freeze_pretrained=False,
    )

    # -----------------------------
    # 4-objective adaptive training
    # -----------------------------
    # adaptive_train_a2cmoe_m4(
    #     pretrained_m3_path="a2cmoe_m3_adaptive.pt",  # 3-objective adaptive
    #     reward_weights=(0.7, 0.1, 0.1, 0.1),          # 4개 objective
    #     total_episodes=5000,
    #     freeze_pretrained=False,
    #     num_experts=2,
    #     log_path="a2cmoe_m4_adaptive_logs.csv",
    #     model_save_path="a2cmoe_m4_adaptive.pt",
    # )

    # weight_configs = [(0.7, 0.1, 0.1, 0.1), (0.4, 0.2, 0.2, 0.2), (0.25, 0.25, 0.25, 0.25)]
    # for w_cfg in weight_configs:
    #     replicate_and_compare(*w_cfg, replications=1)
