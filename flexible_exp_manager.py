import argparse
import os
from pprint import pprint
from typing import Any, Dict, Optional, Tuple, Union, Type

import optuna
import torch as th
from sb3_contrib.common.vec_env import AsyncEval

# For using HER with GoalEnv
from stable_baselines3 import HerReplayBuffer  # noqa: F401
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

# For custom activation fn
from torch import nn as nn  # noqa: F401

# Register custom envs
from rl_zoo3.callbacks import TrialEvalCallback
from rl_zoo3.hyperparams_opt import HYPERPARAMS_SAMPLER
from rl_zoo3.utils import get_callback_list
from rl_zoo3.exp_manager import ExperimentManager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

class FlexibleExperimentManager(ExperimentManager):
    """
    Experiment Manager, but with the capability to easily add new algorithms to the list of available algorithms.
    Input FULL_ALGO_LIST with a new entry in the dictionary to add an algorithm.
    """
    def __init__(
        self,
        args: argparse.Namespace,
        algo: str,
        FULL_ALGO_LIST: Dict[str, Type[BaseAlgorithm]],
        env_id: str,
        log_folder: str,
        tensorboard_log: str = "",
        n_timesteps: int = 0,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_freq: int = -1,
        hyperparams: Optional[Dict[str, Any]] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        trained_agent: str = "",
        optimize_hyperparameters: bool = False,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
        n_trials: int = 1,
        max_total_trials: Optional[int] = None,
        n_jobs: int = 1,
        sampler: str = "tpe",
        pruner: str = "median",
        optimization_log_path: Optional[str] = None,
        n_startup_trials: int = 0,
        n_evaluations: int = 1,
        truncate_last_trajectory: bool = False,
        uuid_str: str = "",
        seed: int = 0,
        log_interval: int = 0,
        save_replay_buffer: bool = False,
        verbose: int = 1,
        vec_env_type: str = "dummy",
        n_eval_envs: int = 1,
        no_optim_plots: bool = False,
        device: Union[th.device, str] = "auto",
        config: Optional[str] = None,
        show_progress: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(args,
            algo,
            env_id,
            log_folder,
            tensorboard_log,
            n_timesteps,
            eval_freq,
            n_eval_episodes,
            save_freq,
            hyperparams,
            env_kwargs,
            trained_agent,
            optimize_hyperparameters,
            storage,
            study_name,
            n_trials,
            max_total_trials,
            n_jobs,
            sampler,
            pruner,
            optimization_log_path,
            n_startup_trials,
            n_evaluations,
            truncate_last_trajectory,
            uuid_str,
            seed,
            log_interval,
            save_replay_buffer,
            verbose,
            vec_env_type,
            n_eval_envs,
            no_optim_plots,
            device,
            config,
            show_progress)
        self.FULL_ALGO_LIST = FULL_ALGO_LIST
        self.policy_kwargs = policy_kwargs


    def setup_experiment(self) -> Optional[Tuple[BaseAlgorithm, Dict[str, Any]]]:
        """
        Read hyperparameters, pre-process them (create schedules, wrappers, callbacks, action noise objects)
        create the environment and possibly the model.

        :return: the initialized RL model
        """
        hyperparams, saved_hyperparams = self.read_hyperparameters()
        hyperparams, self.env_wrapper, self.callbacks, self.vec_env_wrapper = self._preprocess_hyperparams(hyperparams)

        self.create_log_folder()
        self.create_callbacks()

        # Create env to have access to action space for action noise
        n_envs = 1 if self.algo == "ars" or self.optimize_hyperparameters else self.n_envs
        env = self.create_envs(n_envs, no_log=False)

        self._hyperparams = self._preprocess_action_noise(hyperparams, saved_hyperparams, env)

        if self.continue_training:
            model = self._load_pretrained_agent(self._hyperparams, env)
        elif self.optimize_hyperparameters:
            env.close()
            return None
        else:
            #Merge policy_kwargs classified as hyperparameters, and those not classified as hyperparameters
            if "policy_kwargs" in hyperparams.keys():
                policy_kwargs = {**hyperparams["policy_kwargs"], **self.policy_kwargs}
                del hyperparams["policy_kwargs"]

            # Train an agent from scratch
            model = self.FULL_ALGO_LIST[self.algo](
                env=env,
                tensorboard_log=self.tensorboard_log,
                seed=self.seed,
                verbose=self.verbose,
                device=self.device,
                policy_kwargs=policy_kwargs,
                **self._hyperparams,
            )

        self._save_config(saved_hyperparams)
        return model, saved_hyperparams
    
    def _load_pretrained_agent(self, hyperparams: Dict[str, Any], env: VecEnv) -> BaseAlgorithm:
        # Continue training
        print("Loading pretrained agent")
        # Policy should not be changed
        del hyperparams["policy"]

        if "policy_kwargs" in hyperparams.keys():
            del hyperparams["policy_kwargs"]

        model = self.FULL_ALGO_LIST[self.algo].load(
            self.trained_agent,
            env=env,
            seed=self.seed,
            tensorboard_log=self.tensorboard_log,
            verbose=self.verbose,
            device=self.device,
            **hyperparams,
        )

        replay_buffer_path = os.path.join(os.path.dirname(self.trained_agent), "replay_buffer.pkl")

        if os.path.exists(replay_buffer_path):
            print("Loading replay buffer")
            # `truncate_last_traj` will be taken into account only if we use HER replay buffer
            model.load_replay_buffer(replay_buffer_path, truncate_last_traj=self.truncate_last_trajectory)
        return model
    
    def objective(self, trial: optuna.Trial) -> float:
        kwargs = self._hyperparams.copy()

        # Hack to use DDPG/TD3 noise sampler
        trial.n_actions = self.n_actions
        # Hack when using HerReplayBuffer
        trial.using_her_replay_buffer = kwargs.get("replay_buffer_class") == HerReplayBuffer
        if trial.using_her_replay_buffer:
            trial.her_kwargs = kwargs.get("replay_buffer_kwargs", {})
        # Sample candidate hyperparameters
        sampled_hyperparams = HYPERPARAMS_SAMPLER[self.algo](trial)
        kwargs.update(sampled_hyperparams)

        n_envs = 1 if self.algo == "ars" else self.n_envs
        env = self.create_envs(n_envs, no_log=True)

        # By default, do not activate verbose output to keep
        # stdout clean with only the trials results
        trial_verbosity = 0
        # Activate verbose mode for the trial in debug mode
        # See PR #214
        if self.verbose >= 2:
            trial_verbosity = self.verbose

        model = self.FULL_ALGO_LIST[self.algo](
            env=env,
            tensorboard_log=None,
            # We do not seed the trial
            seed=None,
            verbose=trial_verbosity,
            device=self.device,
            policy_kwargs=self.policy_kwargs, #?? do we need or want this?
            **kwargs,
        )

        eval_env = self.create_envs(n_envs=self.n_eval_envs, eval_env=True)

        optuna_eval_freq = int(self.n_timesteps / self.n_evaluations)
        # Account for parallel envs
        optuna_eval_freq = max(optuna_eval_freq // self.n_envs, 1)
        # Use non-deterministic eval for Atari
        path = None
        if self.optimization_log_path is not None:
            path = os.path.join(self.optimization_log_path, f"trial_{str(trial.number)}")
        callbacks = get_callback_list({"callback": self.specified_callbacks})
        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            best_model_save_path=path,
            log_path=path,
            n_eval_episodes=self.n_eval_episodes,
            eval_freq=optuna_eval_freq,
            deterministic=self.deterministic_eval,
        )
        callbacks.append(eval_callback)

        learn_kwargs = {}
        # Special case for ARS
        if self.algo == "ars" and self.n_envs > 1:
            learn_kwargs["async_eval"] = AsyncEval(
                [lambda: self.create_envs(n_envs=1, no_log=True) for _ in range(self.n_envs)], model.policy
            )

        try:
            model.learn(self.n_timesteps, callback=callbacks, **learn_kwargs)
            # Free memory
            model.env.close()
            eval_env.close()
        except (AssertionError, ValueError) as e:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            eval_env.close()
            # Prune hyperparams that generate NaNs
            print(e)
            print("============")
            print("Sampled hyperparams:")
            pprint(sampled_hyperparams)
            raise optuna.exceptions.TrialPruned()
        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_mean_reward

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward