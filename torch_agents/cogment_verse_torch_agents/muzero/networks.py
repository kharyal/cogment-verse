# Copyright 2021 AI Redefined Inc. <dev+cogment@ai-r.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import torch

from .mcts import MCTS


@torch.no_grad()
def parameters_l2(model):
    l2 = 0
    for p in model.parameters():
        l2 += torch.sum(torch.square(p))
    return l2


def relative_error(u, v, eps=1e-6):
    # clamp for NaN issue??
    return torch.clamp(torch.abs(u - v) / torch.sqrt(u ** 2 + v ** 2 + eps), 0, 1)


def cosine_similarity_loss(u, v, dim=1, weights=None, eps=1e-6):
    if weights is None:
        weights = 1.0
    uv = torch.sum(u * v, dim=dim)
    u2 = torch.sum(u * u, dim=dim)
    v2 = torch.sum(v * v, dim=dim)
    cosine = uv / torch.sqrt(u2 * v2 + eps)
    return torch.mean((1.0 - cosine) * weights)


def mean_error_loss(u, v, dim=1, weights=None, eps=1e-6):
    if weights is None:
        weights = 1.0
    return torch.mean(weights * torch.mean((u - v) ** 2, dim=dim))  # , dim=dim)


def lin_bn_act(num_in, num_out, bn=True, act=None):
    layers = [torch.nn.Linear(num_in, num_out)]
    if bn:
        layers.append(torch.nn.BatchNorm1d(num_out))
    if act:
        layers.append(act)
    return torch.nn.Sequential(*layers)


def mlp(
    num_in,
    num_hidden,
    num_out,
    hidden_layers=1,
    bn=True,
    act=torch.nn.ReLU(),
    final_bn=False,
    final_act=None,
):
    act = torch.nn.ReLU()
    stem = lin_bn_act(num_in, num_hidden, bn, act)
    hiddens = [lin_bn_act(num_hidden, num_hidden, bn, act) for _ in range(hidden_layers - 1)]
    output = lin_bn_act(num_hidden, num_out, final_bn, final_act)
    layers = [stem] + hiddens + [output]
    return torch.nn.Sequential(*layers)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self._fc1 = torch.nn.Linear(channels, channels)
        self._fc2 = torch.nn.Linear(channels, channels)
        self._bn1 = torch.nn.BatchNorm1d(channels)
        self._bn2 = torch.nn.BatchNorm1d(channels)
        self._act = torch.nn.ReLU()

    def forward(self, x):
        y = self._fc1(x)
        y = self._bn1(y)
        y = self._act(y)
        y = self._fc2(y)
        y = self._bn2(y)
        return self._act(x + y)


class DynamicsAdapter(torch.nn.Module):
    def __init__(self, net, act_dim, num_input, num_latent, reward_dist):
        super().__init__()
        self._net = net
        self._act_dim = act_dim
        self._num_latent = num_latent
        self._reward_dist = reward_dist
        self._state_pred = torch.nn.Linear(num_input, num_latent)

    def forward(self, state, action):
        """
        Returns tuple (next_state, reward)
        """
        action = torch.nn.functional.one_hot(action, self._act_dim)
        intermediate = self._net(torch.cat([state, action], dim=1))
        next_state = self._state_pred(intermediate)
        reward_probs, reward = self._reward_dist(intermediate)

        return next_state, reward_probs, reward


def resnet(num_in, num_hidden, num_out, hidden_layers=1):
    layers = []
    if num_in != num_hidden:
        layers.append(lin_bn_act(num_in, num_hidden, bn=True, act=torch.nn.ReLU()))

    layers.extend([ResidualBlock(num_hidden) for _ in range(hidden_layers)])
    if num_hidden != num_out:
        layers.append(lin_bn_act(num_hidden, num_out, bn=True, act=torch.nn.ReLU()))
    return torch.nn.Sequential(*layers)


def reward_transform(reward, eps=0.001):
    # 1911.08265 appendix F (NB: there is a typo in the formula!)
    # See the original paper 1805.11593 for the correct formula
    return torch.sign(reward) * (torch.sqrt(1 + torch.abs(reward)) - 1) + eps * reward


def reward_transform_inverse(transformed_reward, eps=0.001):
    s = torch.sign(transformed_reward)
    y = torch.abs(transformed_reward)

    a = eps ** 2
    b = -2 * eps * (y + 1) - 1
    c = y ** 2 + 2 * y

    d = torch.sqrt(b ** 2 - 4 * a * c)
    e = 2 * a

    x = torch.abs((-b - d) / e)
    # 1 step of Newton's method for numerical stability
    x = -c / (a * x + b)

    return s * x


class Distributional(torch.nn.Module):
    def __init__(self, vmin, vmax, num_input, count, transform=None, inverse_transform=None):
        super().__init__()
        assert count >= 2
        self._transform = transform or torch.nn.Identity()
        self._inverse_transform = inverse_transform or torch.nn.Identity()
        self._act = torch.nn.Softmax(dim=1)
        self._vmin = self._transform(torch.tensor(vmin)).detach()
        self._vmax = self._transform(torch.tensor(vmax)).detach()
        self._bins = torch.nn.parameter.Parameter(
            torch.from_numpy(np.linspace(self._vmin, self._vmax, num=count, dtype=np.float32)), requires_grad=False
        )
        self._fc = torch.nn.Linear(num_input, count)

    def forward(self, x):
        probs = self._act(self._fc(x))
        val = torch.sum(probs * self._bins.unsqueeze(0), dim=1)
        return probs, self._inverse_transform(val)

    def compute_value(self, probs):
        assert probs.shape[-1] == self._bins.shape[0]
        bin_shape = [1 for _ in range(len(probs.shape))]
        bin_shape[-1] = self._bins.shape[0]
        val = torch.sum(probs * self._bins.view(*bin_shape), dim=-1)
        return self._inverse_transform(val)

    @torch.no_grad()
    def compute_target(self, vals):
        count = self._bins.shape[-1]
        shape = vals.shape
        target_shape = list(shape) + [count]

        # print("DEBUG vals", vals.shape)
        vals = vals.view(-1)
        target = torch.zeros(target_shape, dtype=torch.float32).view(-1, count)

        vals = self._transform(vals)
        vals = torch.clamp(vals, self._vmin, self._vmax)
        vals = (vals - self._vmin) / (self._vmax - self._vmin) * (count - 1)
        bin_low = torch.clamp(vals.type((torch.long)), 0, count - 1)
        bin_high = torch.clamp(bin_low + 1, 0, count - 1)
        t = bin_high - vals

        for i in range(len(vals)):
            target[i, bin_low[i]] = t[i]
            target[i, bin_high[i]] = 1 - t[i]

        return target.view(*target_shape)


def cross_entropy(pred, target, weights=None, dim=1):
    return torch.mean(weights * torch.sum(-target * torch.log(pred + 1e-12), dim=dim))


class LambdaModule(torch.nn.Module):
    def __init__(self, fn, context=None):
        super().__init__()
        self._fn = fn
        self._context = context

    def forward(self, *args, **kwargs):
        return self._fn(*args, context=self._context, **kwargs)


def normalize_scale(state):
    max_state = torch.max(state, dim=1, keepdim=True)[0]
    min_state = torch.min(state, dim=1, keepdim=True)[0]
    return (state - min_state) / (max_state - min_state)


class RepresentationNetwork(torch.nn.Module):
    def __init__(self, stem, num_hidden, num_hidden_layers):
        super().__init__()
        self.stem = stem
        self.blocks = torch.nn.ModuleList([ResidualBlock(num_hidden) for _ in range(num_hidden_layers)])

    def forward(self, x):
        features = self.stem(x)
        for block in self.blocks:
            features = block(features)
        return normalize_scale(features)


class PolicyNetwork(torch.nn.Module):
    def __init__(self, num_hidden, num_hidden_layers, num_act):
        super().__init__()
        self.blocks = torch.nn.ModuleList([ResidualBlock(num_hidden) for _ in range(num_hidden_layers)])
        self.fc = torch.nn.Linear(num_hidden, num_act)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.fc(x)
        return torch.nn.functional.softmax(x, dim=1)


class ValueNetwork(torch.nn.Module):
    def __init__(self, num_hidden, num_hidden_layers, vmin, vmax, vbins):
        super().__init__()
        self.blocks = torch.nn.ModuleList([ResidualBlock(num_hidden) for _ in range(num_hidden_layers)])
        self.distribution = Distributional(vmin, vmax, num_hidden, vbins)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.distribution(x)


class QNetwork(torch.nn.Module):
    def __init__(self, num_act, num_hidden, num_hidden_layers, vmin, vmax, vbins):
        super().__init__()
        self.blocks = torch.nn.ModuleList([ResidualBlock(num_hidden) for _ in range(num_hidden_layers)])
        self.distributions = torch.nn.ModuleList(
            [Distributional(vmin, vmax, num_hidden, vbins) for _ in range(num_act)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        probs, vals = zip(*[dist(x) for dist in self.distributions])
        return torch.stack(probs, dim=1), torch.stack(vals, dim=1)


class DynamicsNetwork(torch.nn.Module):
    def __init__(self, num_action, num_hidden, num_hidden_layers, rmin, rmax, rbins):
        super().__init__()
        self.num_action = num_action

        self.blocks = torch.nn.ModuleList([ResidualBlock(num_hidden + num_action) for _ in range(num_hidden_layers)])
        self.distribution = Distributional(rmin, rmax, num_hidden + num_action, rbins)
        self.state_predictor = torch.nn.Linear(num_hidden + num_action, num_hidden)

    def forward(self, state, action):
        action_one_hot = torch.nn.functional.one_hot(action, self.num_action)
        x = torch.cat((state, action_one_hot), dim=1)

        for block in self.blocks:
            x = block(x)

        reward_probs, reward = self.distribution(x)
        next_state = self.state_predictor(x)
        return normalize_scale(next_state), reward_probs, reward


class MuZero(torch.nn.Module):
    def __init__(
        self,
        representation,
        dynamics,
        policy,
        value,
        projector,
        predictor,
        reward_distribution,
        value_distribution,
        dqn,
        similarity_loss=cosine_similarity_loss,
        # similarity_loss=mean_error_loss,
    ):
        super().__init__()
        self._representation = representation
        self._dynamics = dynamics
        self._policy = policy
        self._value = value
        self._projector = projector
        self._predictor = predictor
        self._reward_distribution = reward_distribution
        self._value_distribution = value_distribution
        self._similarity_loss = similarity_loss
        self._dqn = dqn

    @torch.no_grad()
    def act(self, observation, epsilon, alpha, temperature, discount_rate, mcts_depth, mcts_count, ucb_c1, ucb_c2):
        self.eval()

        if False:
            policy, q, value = self.reanalyze(
                observation, epsilon, alpha, discount_rate, mcts_depth, mcts_count, ucb_c1, ucb_c2
            )
            policy = torch.pow(policy, 1 / temperature)
            policy /= torch.sum(policy, dim=1)
            action = torch.distributions.Categorical(policy).sample()
            action = action.cpu().numpy().item()

        representation = self._representation(observation.unsqueeze(0))
        # testing/debug
        q_probs, q = self._dqn(representation)
        qmin = torch.min(q)
        qmax = torch.max(q)
        policy = torch.clamp((q - qmin) / (qmax - qmin), 0.1, 1.0)
        policy /= torch.sum(policy)
        # action = torch.distributions.Categorical(policy).sample().detach().item()

        if np.random.rand() < 0.1:
            action = np.random.randint(0, q.shape[1])
        else:
            action = torch.argmax(q).detach().cpu().item()

        value = q[:, action]

        return action, policy, q, value

    def train_step(
        self,
        optimizer,
        observation,
        action,
        reward,
        next_observation,
        done,
        target_policy,
        target_value,
        importance_weight,
        max_norm,
        target_label_smoothing_factor,
        s_weight,
        v_weight,
        discount_factor,
        target_muzero,
    ):
        """ """
        self.train()
        target_muzero.eval()

        batch_size, rollout_length = observation.shape[:2]
        current_representation = self._representation(observation[:, 0])
        device = current_representation.device
        priority = torch.zeros((batch_size, rollout_length), dtype=current_representation.dtype, device=device)

        target_value = target_value.view(batch_size, rollout_length)
        reward = reward.view(batch_size, rollout_length)

        loss_kl = 0
        loss_v = 0
        loss_r = 0
        loss_s = 0
        loss_q = 0

        max_value_error = 0
        max_reward_error = 0

        for k in range(rollout_length):
            pred_next_state, pred_reward_probs, pred_reward = self._dynamics(current_representation, action[:, k])
            # todo: check this against other implementations
            # next_representation.register_hook(lambda grad: grad * 0.5)

            max_reward_error = max(max_reward_error, torch.max(torch.abs(pred_reward - reward[:, k])))

            pred_policy = self._policy(current_representation)
            pred_value_probs, pred_value = self._value(current_representation)
            pred_projection = self._predictor(self._projector(pred_next_state))

            # anooying squeeze/unsqueeze nonsense (check replay/episode buffer sampling?)
            target_policy_k = target_policy[:, k].reshape(*pred_policy.shape)

            pred_q_probs, pred_q_vals = self._dqn(current_representation)

            with torch.no_grad():
                next_state = target_muzero._representation(next_observation[:, k])
                target_projection = target_muzero._projector(next_state)

                _, pred_next_val = target_muzero._value(next_state)

                next_q_probs, next_q_vals = target_muzero._dqn(next_state)
                next_action = torch.argmax(next_q_vals, dim=1)
                # print("JJJJJJJJJJJJJJJ", next_q_vals.shape, next_action.shape)

                target_q = torch.zeros_like(pred_q_vals)
                for b in range(pred_q_vals.shape[0]):
                    target_q[b] = reward[b, k] + discount_factor * (1 - done[b, k]) * next_q_vals[b, next_action[b]]

                target_q_probs = self._value_distribution.compute_target(target_q)
                # td_target = reward[:, k] + discount_factor * pred_next_val
                # target_value_probs = self._value_distribution.compute_target(td_target)

                target_value_probs = self._value_distribution.compute_target(target_value[:, k]).to(device)
                target_reward_probs = self._reward_distribution.compute_target(reward[:, k]).to(device)
                target_value_clamped = self._value_distribution.compute_value(target_value_probs)

                max_value_error = max(max_value_error, torch.max(torch.abs(pred_value - target_value_clamped)))

                # debug
                # print("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
                # print(importance_weight.shape, target_policy.shape)

                entropy_target = cross_entropy(target_policy_k, target_policy_k, importance_weight.view(-1))
                entropy_pred = cross_entropy(pred_policy, pred_policy, importance_weight.view(-1))
                priority[:, k] = torch.abs(pred_value.view(-1) - target_value_clamped.view(-1))

            loss_kl += cross_entropy(pred_policy, target_policy_k, importance_weight) - entropy_target
            loss_r += cross_entropy(pred_reward_probs, target_reward_probs, importance_weight)
            loss_v += cross_entropy(pred_value_probs, target_value_probs, importance_weight)
            loss_s += self._similarity_loss(pred_projection, target_projection, weights=importance_weight)
            # print("GGGGGGGGGGGGGGGGGGGGGG", pred_q_probs.shape, target_q_probs.shape, importance_weight.shape)
            coords = [(i, j) for i, j in enumerate(action[:, k].view(-1))]
            next_coords = [(i, j) for i, j in enumerate(next_action.view(-1))]
            # debug)
            for b in range(pred_q_probs.shape[0]):
                loss_q += cross_entropy(
                    pred_q_probs[b, action[b, k]].view(1, -1),
                    target_q_probs[b, next_action[b]].view(1, -1),
                    weights=importance_weight[b].view(1, 1),
                )

            # end of rollout loop body
            current_representation = pred_next_state

        # testing; disable priority replay
        priority = torch.ones_like(priority)

        priority = priority.detach().cpu().numpy()

        loss_q /= rollout_length * batch_size
        loss_kl /= rollout_length
        loss_s /= rollout_length
        loss_r /= rollout_length
        loss_v /= rollout_length

        optimizer.zero_grad()

        # total_loss = loss_kl + loss_r + v_weight * loss_v + s_weight * loss_s
        # total_loss = loss_kl + loss_r + v_weight * loss_v + s_weight * loss_s + loss_q
        total_loss = loss_q
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm)
        optimizer.step()

        # debug
        # for n in range(pred_policy.shape[0]):
        #    print(k, pred_policy[n], target_policy[n])

        info = dict(
            loss_r=loss_r,
            loss_v=loss_v,
            loss_s=loss_s,
            loss_q=loss_q,
            loss_kl=loss_kl,
            total_loss=total_loss,
            entropy_target=entropy_target,
            entropy_pred=entropy_pred,
            # reward_error=reward_delta,
            # value_error=value_delta,
            representation_l2=parameters_l2(self._representation),
            dynamics_l2=parameters_l2(self._dynamics),
            policy_l2=parameters_l2(self._policy),
            value_l2=parameters_l2(self._value),
            projector_l2=parameters_l2(self._projector),
            predictor_l2=parameters_l2(self._predictor),
            # mean_value_target=torch.mean(importance_weight.view(-1, 1) * target_value),
            # mean_reward_target=torch.mean(importance_weight.view(-1, 1) * reward),
            # mean_value_pred=torch.mean(importance_weight.view(-1) * pred_value.view(-1)),
            # mean_reward_pred=torch.mean(importance_weight.view(-1) * pred_reward.view(-1)),
            # min_reward=torch.min(reward),
            ##max_reward=torch.max(reward),
            # min_value=torch.min(target_value),
            # max_value=torch.max(target_value),
            priority_mean=priority.mean(),
            max_value_error=max_value_error,
            max_reward_error=max_reward_error,
        )
        return priority, info

    @torch.no_grad()
    def reanalyze(self, state, epsilon, alpha, discount_rate, mcts_depth, mcts_count, ucb_c1, ucb_c2):
        self.eval()

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)

        state = state.unsqueeze(0)

        def _dynamics_deterministic(rep, action, *, context):
            distributional_dynamics = context
            rep, probs, reward = distributional_dynamics(rep, action)
            return rep, reward

        def _value_deterministic(rep, *, context):
            distributional_value = context
            probs, val = distributional_value(rep)
            return val

        dynamics = LambdaModule(_dynamics_deterministic, context=self._dynamics)
        value = LambdaModule(_value_deterministic, context=self._value)

        # get policy/val training targets via MCTS improvement
        representation = self._representation(state)
        mcts = MCTS(
            policy=self._policy,
            value=value,
            dynamics=dynamics,
            representation=representation,
            max_depth=mcts_depth,
            epsilon=epsilon,
            alpha=alpha,
            c1=ucb_c1,
            c2=ucb_c2,
            discount=discount_rate,
        )

        mcts.build_search_tree(mcts_count)
        return mcts.improved_targets()
