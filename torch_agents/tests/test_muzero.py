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

import os
from tempfile import TemporaryDirectory
import pytest
import torch

from cogment_verse_torch_agents.muzero.networks import (
    MuZero,
    Distributional,
    DynamicsAdapter,
    reward_transform,
    reward_tansform_inverse,
    resnet,
    mlp,
)

from cogment_verse_torch_agents.muzero.agent import MuZeroAgent

from cogment_verse_environment.factory import make_environment
from cogment_verse_torch_agents.third_party.hive.utils.schedule import LinearSchedule

# pylint doesn't like test fixtures
# pylint: disable=redefined-outer-name


@pytest.fixture
def rollout_length():
    return 3


@pytest.fixture
def batch_size():
    return 32


@pytest.fixture
def env():
    return make_environment("gym", "LunarLander-v2")


@pytest.fixture
def obs_dim():
    return 8


@pytest.fixture
def act_dim():
    return 4


@pytest.fixture
def num_hidden():
    return 8


@pytest.fixture
def num_latent():
    return 4


@pytest.fixture
def value_distribution(num_hidden):
    return Distributional(
        -10.0,
        10.0,
        num_hidden,
        10,
        reward_transform,
        reward_tansform_inverse,
    )


@pytest.fixture
def reward_distribution(num_hidden):
    return Distributional(
        -2.0,
        2.0,
        num_hidden,
        10,
        reward_transform,
        reward_tansform_inverse,
    )


@pytest.fixture
def representation(obs_dim, num_hidden, num_latent):
    return resnet(
        obs_dim,
        num_hidden,
        num_latent,
        2,
        # final_act=torch.nn.BatchNorm1d(self._params["num_latent"]),  # normalize for input to subsequent networks
    )


@pytest.fixture
def dynamics(num_latent, act_dim, num_hidden, reward_distribution):
    # debugging
    # self._representation = torch.nn.Identity()

    return DynamicsAdapter(
        resnet(
            num_latent + act_dim,
            num_hidden,
            num_hidden,
            2,
            final_act=torch.nn.LeakyReLU(),
        ),
        act_dim,
        num_hidden,
        num_latent,
        reward_dist=reward_distribution,
    )


@pytest.fixture
def policy(num_latent, num_hidden, act_dim):
    return resnet(
        num_latent,
        num_hidden,
        act_dim,
        2,
        final_act=torch.nn.Softmax(dim=1),
    )


@pytest.fixture
def value(num_latent, num_hidden, value_distribution):
    return resnet(
        num_latent,
        num_hidden,
        num_hidden,
        2,
        final_act=value_distribution,
    )


@pytest.fixture
def projector_dim():
    return 2


@pytest.fixture
def projector(num_latent, projector_dim):
    return mlp(num_latent, 16, projector_dim)


@pytest.fixture
def predictor(projector_dim):
    return mlp(
        projector_dim,
        16,
        projector_dim,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_create(
    representation, dynamics, policy, value, projector, predictor, device, reward_distribution, value_distribution
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip()

    model = MuZero(
        representation, dynamics, policy, value, projector, predictor, reward_distribution, value_distribution
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_act(
    representation, dynamics, policy, value, projector, predictor, device, env, reward_distribution, value_distribution
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip()

    model = MuZero(
        representation, dynamics, policy, value, projector, predictor, reward_distribution, value_distribution
    )
    model.eval()
    state = env.reset()

    for i in range(100):
        observation = torch.from_numpy(state.observation).to(device).float()
        action, policy, value = model.act(observation, 0.1, 0.3, 0.75)
        state = env.step(action)
        if state.done:
            state = env.reset()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("reanalyze_fraction", [0.0])  # , 1.0])
def test_learn(
    obs_dim,
    act_dim,
    representation,
    dynamics,
    policy,
    value,
    projector,
    predictor,
    env,
    batch_size,
    device,
    reanalyze_fraction,
    reward_distribution,
    value_distribution,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip()

    model = MuZero(
        representation, dynamics, policy, value, projector, predictor, reward_distribution, value_distribution
    )
    optimizer = torch.optim.Adam(model.parameters())
    observation = torch.rand((4, 3, obs_dim))
    next_observation = torch.rand((4, 3, obs_dim))
    reward = torch.rand((4, 3))
    target_policy = torch.rand((4, 3, act_dim))
    target_value = torch.rand((4, 3))
    action = torch.randint(low=0, high=act_dim, size=(4, 3))
    importance_weight = 1 / (1 + torch.rand(4) ** 2)
    priority, info = model.train_step(
        optimizer, observation, action, reward, next_observation, target_policy, target_value, importance_weight
    )


def test_distributional():
    dist = Distributional(-2.0, 3.0, 8, 11)
    v = torch.tensor(1.738).to(torch.float32)
    t = dist.compute_target(v)
    assert torch.allclose(torch.sum(t * dist._bins), v)
    assert torch.sum(t != 0) == 2

    t = dist.compute_target(torch.tensor(-3.0, dtype=torch.float32))
    assert t[0] == 1
    assert t[1] == 0

    t = dist.compute_target(torch.tensor(4.0, dtype=torch.float32))
    assert t[-1] == 1
    assert t[-2] == 0


def test_agent(env):
    agent = MuZeroAgent(id="blah", obs_dim=8, act_dim=4, device="cpu")
    obs = env.reset()
    done = False
    step = 0
    while not done:
        action = step % 4
        step += 1
        next_obs = env.step(action)
        policy = torch.ones(4)
        value = 0.0
        agent.consume_training_sample(
            obs.observation, action, next_obs.rewards[0], next_obs.observation, next_obs.done, policy, value
        )
        done = next_obs.done
        obs = next_obs

    batch = agent.sample_training_batch(8)
    info = agent.learn(batch)