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

from audioop import mul
import logging
from data_pb2 import (
    AgentConfig,
    ActorParams,
    TrialConfig,
)
from cogment_verse import MlflowExperimentTracker


# pylint: disable=protected-access
# pylint: disable=W0612
log = logging.getLogger(__name__)


def create_training_run(agent_adapter):
    async def training_run(run_session):
        run_id = run_session.run_id
        config = run_session.config

        run_xp_tracker = MlflowExperimentTracker(run_session.params_name, run_id)
        try:
            # Initialize Alice Agent
            num_alice = config.training.num_teachers
            alice_ids = []
            for alice_idx in range(num_alice):
                alice_ids.append(f"{run_id}_alice_{alice_idx}")
            alice_version_number = 1

            # alice_kwargs = MessageToDict(config.model_kwargs, preserving_proto_field_name=True)
            alice = []
            for alice_idx in range(num_alice):
                alice.append((await agent_adapter.create_and_publish_initial_version(
                    alice_ids[alice_idx],
                    **{
                        "obs_dim1": config.actor.config.num_input,
                        "obs_dim2": config.actor.config.num_input_2,
                        "act_dim": config.actor.config.num_action,
                        "action_scale": config.actor.config.action_scale,
                        "action_bias": config.actor.config.action_bias,
                        "max_action": config.actor.config.max_action,
                        "grid_shape": config.actor.config.alice_grid_shape,
                        "SIGMA": config.training.SIGMA,
                        "max_buffer_size": config.replaybuffer.max_replay_buffer_size,
                        "min_buffer_size": config.replaybuffer.min_replay_buffer_size,
                        "batch_size": config.training.batch_size,
                        "num_training_steps": config.training.num_training_steps,
                        "discount_factor": config.training.discount_factor,
                        "tau": config.training.tau,
                        "policy_noise": config.training.policy_noise,
                        "noise_clip": config.training.noise_clip,
                        "learning_rate": config.training.learning_rate,
                        "policy_freq": config.training.policy_freq,
                        "beta": config.training.beta,
                    },
                ))[0]
                )

            # Initialize Bob Agent
            bob_id = f"{run_id}_bob"
            bob_version_number = 1

            # bob_kwargs = MessageToDict(config.model_kwargs, preserving_proto_field_name=True)
            bob, _ = await agent_adapter.create_and_publish_initial_version(
                bob_id,
                **{
                    "obs_dim1": config.actor.config.num_input,
                    "obs_dim2": config.actor.config.num_input_2,
                    "act_dim": config.actor.config.num_action,
                    "action_scale": config.actor.config.action_scale,
                    "action_bias": config.actor.config.action_bias,
                    "max_action": config.actor.config.max_action,
                    "grid_shape": config.actor.config.bob_grid_shape,
                    "SIGMA": config.training.SIGMA,
                    "max_buffer_size": config.replaybuffer.max_replay_buffer_size,
                    "min_buffer_size": config.replaybuffer.min_replay_buffer_size,
                    "batch_size": config.training.batch_size,
                    "num_training_steps": config.training.num_training_steps,
                    "discount_factor": config.training.discount_factor,
                    "tau": config.training.tau,
                    "policy_noise": config.training.policy_noise,
                    "noise_clip": config.training.noise_clip,
                    "learning_rate": config.training.learning_rate,
                    "policy_freq": config.training.policy_freq,
                    "beta": config.training.beta,
                    "num_alice": config.training.num_teachers,
                },
            )

            # create Alice_config
            alice_configs = []
            for alice_idx in range(num_alice):
                alice_configs.append(
                    ActorParams(
                        name="selfplayRL_Alice"+str(alice_idx),
                        actor_class="agent",
                        implementation=config.actor.implementation,
                        agent_config=AgentConfig(
                            model_id=alice_ids[alice_idx],
                            model_version=alice_version_number,
                            run_id=run_id,
                            environment_specs=config.environment.specs,
                        ),
                    )
                )

            # create Bob_config
            bob_configs = [
                ActorParams(
                    name="selfplayRL_Bob",
                    actor_class="agent",
                    implementation=config.actor.implementation,
                    agent_config=AgentConfig(
                        model_id=bob_id,
                        model_version=bob_version_number,
                        run_id=run_id,
                        environment_specs=config.environment.specs,
                    ),
                )
            ]

            config.environment.config.mode = "train"
            train_trial_configs = [
                TrialConfig(
                    run_id=run_id,
                    environment=config.environment,
                    actors=bob_configs + alice_configs,
                )
                for _ in range(config.rollout.epoch_train_trial_count)
            ]

            config.environment.config.mode = "test"
            test_trial_configs = [
                TrialConfig(
                    run_id=run_id,
                    environment=config.environment,
                    actors=bob_configs + alice_configs,
                )
                for _ in range(config.rollout.epoch_test_trial_count)
            ]

            total_number_trials = 0
            alice_rewards = []
            bob_rewards = []
            test_success = []
            multiplier = config.training.rollout_multiplier

            # Rollout trials
            for epoch in range(config.rollout.epoch_count):

                alice_total_reward = 0
                bob_total_reward = 0
                alice_samples = [[] for i in range(num_alice)]
                bob_samples = [[] for i in range(num_alice)]
                for rollout in range(1):
                    # Training
                    async for (
                        step_idx,
                        step_timestamp,
                        _trial_id,
                        _tick_id,
                        sample,
                    ) in run_session.start_trials_and_wait_for_termination(
                        trial_configs=train_trial_configs,
                        max_parallel_trials=config.rollout.max_parallel_trials,
                    ):

                        if sample.current_player == 0:  # bob's sample
                            bob_samples[alice_index].append(sample)
                            # penalize/reward alice if bob does/doesn't achieve goal
                            if sample.player_done:
                                if int(sample.reward) > 0:
                                    alice_reward = config.training.alice_penalty
                                    bob_reward = config.training.bob_reward
                                else:
                                    alice_reward = config.training.alice_reward
                                    bob_reward = config.training.bob_penalty

                                alice_total_reward += alice_reward
                                bob_total_reward += bob_reward

                                alice_samples[alice_index][-1] = alice_samples[alice_index][-1]._replace(reward=alice_reward)
                                bob_samples[alice_index][-1] = bob_samples[alice_index][-1]._replace(reward=bob_reward)
                                alice_rewards.append(alice_reward)
                                bob_rewards.append(bob_reward)
                        else:  # alice's sample
                            alice_index = sample.current_player-1
                            alice_samples[alice_index].append(sample)

                    for i in range(num_alice):
                        alice[i].consume_samples(alice_samples[i])
                        bob.consume_samples(bob_samples[i], i)

                print("episode:", epoch, "alice:", alice_rewards, "bob:", bob_rewards)
                run_xp_tracker.log_metrics(
                    step_timestamp,
                    step_idx,
                    alice_rewards=alice_total_reward,
                    bob_rewards=bob_total_reward,
                    difference_bob_alice_rewards=bob_total_reward - alice_total_reward,
                )
                total_number_trials += config.rollout.epoch_train_trial_count

                # Train Alice and Bob
                for alice_index in range(num_alice):
                    alice_hyperparams = alice[alice_index].learn()
                bob_hyperparams = bob.learn(alice)

                alice_version_info = await agent_adapter.publish_version(alice_ids[0], alice[0]) # not sure what this does, alice_version_info isn't used anywhere
                bob_version_info = await agent_adapter.publish_version(bob_id, bob)

                # Test bob's performance
                if epoch and epoch % config.rollout.test_freq == 0:
                    async for (
                        step_idx,
                        step_timestamp,
                        _trial_id,
                        _tick_id,
                        sample,
                    ) in run_session.start_trials_and_wait_for_termination(
                        trial_configs=test_trial_configs,
                        max_parallel_trials=config.rollout.max_parallel_trials,
                    ):
                        if sample.current_player == 0:  # bob' sample
                            if sample.player_done:
                                test_success.append(0)
                                if int(sample.reward) > 0:
                                    test_success[-1] = 1

                                run_xp_tracker.log_metrics(
                                    step_timestamp,
                                    step_idx,
                                    bob_success=test_success[-1],
                                )

            run_xp_tracker.terminate_success()

        except Exception as exception:
            logging.error(f"An exception occurred: {exception}")
            run_xp_tracker.terminate_failure()
            raise

    return training_run
