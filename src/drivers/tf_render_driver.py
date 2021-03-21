"""A Driver that steps a TF environment using a TF policy and rendering the steps."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.environments import tf_environment
from tf_agents.policies import tf_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types
from tf_agents.drivers.tf_driver import TFDriver


class TFRenderDriver(TFDriver):

    def __init__(self,
        env: tf_environment.TFEnvironment,
        policy: tf_policy.TFPolicy, observers: Sequence[Callable[[trajectory.Trajectory], Any]],
        transition_observers: Optional[Sequence[Callable[[trajectory.Transition], Any]]] = None,
        max_steps: Optional[types.Int] = None,
        max_episodes: Optional[types.Int] = None):

        super(TFRenderDriver, self).__init__(env,
            policy,
            observers,
            transition_observers,
            max_steps,
            max_episodes,
            True)

    
    def run(self,
        time_step: ts.TimeStep,
        policy_state: types.NestedTensor = ()) -> Tuple[ts.TimeStep, types.NestedTensor]:

        num_steps = tf.constant(0.0)
        num_episodes = tf.constant(0.0)

        while num_steps < self._max_steps and num_episodes < self._max_episodes:
            action_step = self.policy.action(time_step, policy_state)
            next_time_step = self.env.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            for observer in self._transition_observers:
                observer((time_step, action_step, next_time_step))
            for observer in self.observers:
                observer(traj)

            num_episodes += tf.math.reduce_sum(
                tf.cast(traj.is_boundary(), tf.float32))
            num_steps += tf.math.reduce_sum(tf.cast(~traj.is_boundary(), tf.float32))

            time_step = next_time_step
            policy_state = action_step.state
            self.env.render(mode="human")

        return time_step, policy_state

