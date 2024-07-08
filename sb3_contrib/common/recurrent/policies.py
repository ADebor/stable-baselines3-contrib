from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import zip_strict
from torch import nn

from sb3_contrib.common.recurrent.type_aliases import RNNStates

# from nmn_continuous.src.nmn import GRU

class RecurrentActorCriticPolicy(ActorCriticPolicy):
    """
    Recurrent policy class for actor-critic algorithms (has both policy and value prediction).
    To be used with A2C, PPO and the likes.
    It assumes that both the actor and the critic RNN
    have the same architecture.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param rnn_hidden_size: Number of hidden units for each RNN layer.
    :param n_rnn_layers: Number of RNN layers.
    :param shared_rnn: Whether the RNN is shared between the actor and the critic
        (in that case, only the actor gradient is used)
        By default, the actor and the critic have two separate RNN.
    :param enable_critic_rnn: Use a seperate RNN for the critic.
    :param rnn_kwargs: Additional keyword arguments to pass the the RNN
        constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        use_beta: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        # rnn_type: Optional[str] = "lstm",
        rnn_type: Optional[Type[nn.Module]] = nn.LSTM,
        rnn_hidden_size: int = 256,
        n_rnn_layers: int = 1,
        shared_rnn: bool = False,
        enable_critic_rnn: bool = True,
        rnn_kwargs: Optional[Dict[str, Any]] = None,
        rnn_input_dim: Optional[int] = None,
        *args, **kwargs,
    ):
        
        self.rnn_output_dim = rnn_hidden_size
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            use_beta,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        # RNNs creation
        self.rnn_kwargs = rnn_kwargs or {}
        self.shared_rnn = shared_rnn
        self.enable_critic_rnn = enable_critic_rnn

        if not issubclass(rnn_type, (nn.LSTM, nn.GRU)):
            print("Invalid RNN type. Falling off to LSTM.")
            rnn_type = nn.LSTM

        if rnn_input_dim is not None:
            self.rnn_input_dim = rnn_input_dim
        else:
            self.rnn_input_dim = self.features_dim
        
        self.rnn_type = rnn_type
        self.rnn_actor = rnn_type(
            self.rnn_input_dim,
            rnn_hidden_size,
            num_layers=n_rnn_layers,
            **self.rnn_kwargs,
        )

        # if rnn_type == "lstm":
        #     self.rnn_actor = nn.LSTM(
        #         self.rnn_input_dim,
        #         rnn_hidden_size,
        #         num_layers=n_rnn_layers,
        #         **self.rnn_kwargs,
        #     )
        # elif rnn_type == "gru":
        #     self.rnn_actor = GRU(
        #         self.rnn_input_dim,
        #         rnn_hidden_size,
        #         num_layers=n_rnn_layers,
        #         **self.rnn_kwargs,
        #     )

        # For the predict() method, to initialize hidden states
        # (n_rnn_layers, batch_size, rnn_hidden_size)
        self.rnn_hidden_state_shape = (n_rnn_layers, 1, rnn_hidden_size)
        self.critic = None
        self.rnn_critic = None
        assert not (
            self.shared_rnn and self.enable_critic_rnn
        ), "You must choose between shared RNN, seperate or no RNN for the critic."

        assert not (
            self.shared_rnn and not self.share_features_extractor
        ), "If the features extractor is not shared, the RNN cannot be shared."

        # No RNN for the critic, we still need to convert
        # output of features extractor to the correct size
        # (size of the output of the actor rnn)
        if not (self.shared_rnn or self.enable_critic_rnn):
            self.critic = nn.Linear(self.features_dim, rnn_hidden_size) # this will fail as features_dim should be replaced by rnn_output_dim

        # Use a separate RNN for the critic
        if self.enable_critic_rnn:
            self.rnn_critic = rnn_type(
                self.rnn_input_dim,
                rnn_hidden_size,
                num_layers=n_rnn_layers,
                **self.rnn_kwargs,
            )
        #     if rnn_type == "lstm":
        #         self.rnn_critic = nn.LSTM(
        #             self.rnn_input_dim,
        #             rnn_hidden_size,
        #             num_layers=n_rnn_layers,
        #             **self.rnn_kwargs,
        #         )
        #     elif rnn_type == "gru":
        #         self.rnn_critic = GRU(
        #             self.rnn_input_dim,
        #             rnn_hidden_size,
        #             num_layers=n_rnn_layers,
        #             **self.rnn_kwargs,
        #         )

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = MlpExtractor(
            self.rnn_output_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    @staticmethod
    def _process_sequence(
        features: th.Tensor,
        rnn_states: Union[Tuple[th.Tensor, th.Tensor], Tuple[th.Tensor]],
        episode_starts: th.Tensor,
        rnn: Union[nn.LSTM, nn.GRU],
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Do a forward pass in the RNN network.

        :param features: Input tensor
        :param rnn_states: previous hidden and cell states of the RNN, respectively
        :param episode_starts: Indicates when a new episode starts,
            in that case, we need to reset RNN states.
        :param rnn: RNN object.
        :return: RNN output and updated RNN states.
        """
        # RNN logic
        # (sequence length, batch size, features dim)
        # (batch size = n_envs for data collection or n_seq when doing gradient update)
        n_seq = rnn_states[0].shape[1]

        # Batch to sequence
        # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
        # note: max length (max sequence length) is always 1 during data collection
        features_sequence = features.reshape((n_seq, -1, rnn.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        # If we don't have to reset the state in the middle of a sequence
        # we can avoid the for loop, which speeds up things
        if th.all(episode_starts == 0.0):
            rnn_output, rnn_states = rnn(features_sequence, rnn_states)
            rnn_output = th.flatten(rnn_output.transpose(0, 1), start_dim=0, end_dim=1)
            return rnn_output, rnn_states

        rnn_output = []

        # Iterate over the sequence
        for features, episode_start in zip_strict(features_sequence, episode_starts):
            if len(rnn_states) == 2:
                rnn_states = (
                    # Reset the states at the beginning of a new episode
                    (1.0 - episode_start).view(1, n_seq, 1) * rnn_states[0],
                    (1.0 - episode_start).view(1, n_seq, 1) * rnn_states[1],
                )
            else:
                rnn_states = (
                    # Reset the states at the beginning of a new episode
                    (1.0 - episode_start).view(1, n_seq, 1)
                    * rnn_states[0],
                )
            hidden, rnn_states = rnn(
                features.unsqueeze(dim=0),
                rnn_states,
            )
            rnn_output += [hidden]

        # Sequence to batch
        # (sequence length, n_seq, rnn_out_dim) -> (batch_size, rnn_out_dim)
        rnn_output = th.flatten(th.cat(rnn_output).transpose(0, 1), start_dim=0, end_dim=1)
        return rnn_output, rnn_states

    def forward(
        self,
        obs: th.Tensor,
        rnn_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation. Observation
        :param rnn_states: The last hidden and memory states for the RNN.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the rnn states in that case).
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features  # alis
        else:
            pi_features, vf_features = features

        latent_pi, rnn_states_pi = self._process_sequence(pi_features, rnn_states.pi, episode_starts, self.rnn_actor)

        if self.rnn_critic is not None:
            latent_vf, rnn_states_vf = self._process_sequence(vf_features, rnn_states.vf, episode_starts, self.rnn_critic)
        elif self.shared_rnn:
            # Re-use RNN features but do not backpropagate
            latent_vf = latent_pi.detach()
            if self.rnn_type == nn.LSTM:
                rnn_states_vf = (rnn_states_pi[0].detach(), rnn_states_pi[1].detach())
            else:
                rnn_states_vf = rnn_states_pi[0].detach()
        else:
            # Critic only has a feedforward network
            latent_vf = self.critic(vf_features)
            rnn_states_vf = rnn_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, RNNStates(rnn_states_pi, rnn_states_vf)

    def get_distribution(
        self,
        obs: th.Tensor,
        rnn_states: Union[Tuple[th.Tensor, th.Tensor], Tuple[th.Tensor]],
        episode_starts: th.Tensor,
    ) -> Tuple[Distribution, Tuple[th.Tensor, ...]]:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation.
        :param rnn_states: The last hidden and memory states for the RNN.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the rnn states in that case).
        :return: the action distribution and new hidden states.
        """
        # Call the method from the parent of the parent class
        features = super(ActorCriticPolicy, self).extract_features(obs, self.pi_features_extractor)
        latent_pi, rnn_states = self._process_sequence(features, rnn_states, episode_starts, self.rnn_actor)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), rnn_states

    def predict_values(
        self,
        obs: th.Tensor,
        rnn_states: Union[Tuple[th.Tensor, th.Tensor], Tuple[th.Tensor]],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation.
        :param rnn_states: The last hidden and memory states for the RNN.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the rnn states in that case).
        :return: the estimated values.
        """
        # NOTE: This is the same logic as in the forward method.

        # Call the method from the parent of the parent class
        features = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)

        if self.rnn_critic is not None:
            latent_vf, rnn_states_vf = self._process_sequence(features, rnn_states, episode_starts, self.rnn_critic)
        elif self.shared_rnn:
            # Use RNN from the actor
            latent_pi, _ = self._process_sequence(features, rnn_states, episode_starts, self.rnn_actor)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor, rnn_states: RNNStates, episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation.
        :param actions:
        :param rnn_states: The last hidden and memory states for the RNN.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the rnn states in that case).
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features  # alias
        else:
            pi_features, vf_features = features

        # process the sequence
        latent_pi, _ = self._process_sequence(pi_features, rnn_states.pi, episode_starts, self.rnn_actor)
        if self.rnn_critic is not None:
            latent_vf, _ = self._process_sequence(vf_features, rnn_states.vf, episode_starts, self.rnn_critic)
        elif self.shared_rnn:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def _predict(
        self,
        observation: th.Tensor,
        rnn_states: Union[Tuple[th.Tensor, th.Tensor], Tuple[th.Tensor]],
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, ...]]:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param rnn_states: The last hidden and memory states for the RNN.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the rnn states in that case).
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy and hidden states of the RNN
        """
        distribution, rnn_states = self.get_distribution(observation, rnn_states, episode_starts)
        return distribution.get_actions(deterministic=deterministic), rnn_states

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param rnn_states: The last hidden and memory states for the RNN.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the rnn states in that case).
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[next(iter(observation.keys()))].shape[0]
        else:
            n_envs = observation.shape[0]
        # state : (n_layers, n_envs, dim)
        if state is None:
            # Initialize hidden states to zeros
            state = np.concatenate([np.zeros(self.rnn_hidden_state_shape) for _ in range(n_envs)], axis=1)
            state = (state, state)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            # Convert to PyTorch tensors
            if self.rnn_type == nn.LSTM:
                states = th.tensor(state[0], dtype=th.float32, device=self.device), th.tensor(
                    state[1], dtype=th.float32, device=self.device
                )
            else:
                states = (th.tensor(state[0], dtype=th.float32, device=self.device),)
            episode_starts = th.tensor(episode_start, dtype=th.float32, device=self.device)
            actions, states = self._predict(
                observation, rnn_states=states, episode_starts=episode_starts, deterministic=deterministic
            )
            if self.rnn_type == nn.LSTM:
                states = (states[0].cpu().numpy(), states[1].cpu().numpy())
            else:
                states = (states[0].cpu().numpy(),)

        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, states


class RecurrentActorCriticCnnPolicy(RecurrentActorCriticPolicy):
    """
    CNN recurrent policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic.
        By default, only the actor has a recurrent network.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            lstm_kwargs,
        )


class RecurrentMultiInputActorCriticPolicy(RecurrentActorCriticPolicy):
    """
    MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic.
        By default, only the actor has a recurrent network.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            lstm_kwargs,
        )
