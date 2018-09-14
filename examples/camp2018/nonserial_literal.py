#!/usr/bin/env python3

import flor
import gym
import pong_py
import ray
import cloudpickle

from ray.tune.registry import register_env
from ray.rllib.agents import ppo

@flor.func
def instantiate_agent(**kwargs):
    try:
        ray.get([])
    except:
        ray.init()

    register_env("my_env", lambda ec: pong_py.PongJSEnv())
    trainer = ppo.PPOAgent(env="my_env", config={"env_config": {}})
    return trainer

@flor.func
def write_literal(literal, artifact_path, **kwargs):
    with open(artifact_path, 'wb') as f:
        cloudpickle.dump(literal, f)

with flor.Experiment('nonserial_literal') as ex:
    do_instantiate_agent = ex.action(instantiate_agent)
    trainer = ex.literal(name="literal", parent=do_instantiate_agent)

    do_write_literal = ex.action(write_literal, [trainer,])
    serialized_trainer = ex.artifact("trainer.pkl", 'artifact_path', do_write_literal)

serialized_trainer.plot()
serialized_trainer.pull()