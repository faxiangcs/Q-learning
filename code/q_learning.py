"""Tabular Q-learning with per-episode Q-table snapshots."""

from dataclasses import dataclass
from math import isfinite
from random import Random

from environment import Environment


@dataclass(frozen=True)
class TrainingSettings:
    q_initial_value: float
    learning_rate: float
    discount_factor: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_episodes: int
    episodes: int
    random_seed: int

    @classmethod
    def from_config(cls, config: dict) -> "TrainingSettings":
        values = config["training"]
        settings = cls(
            q_initial_value=float(values["q_initial_value"]),
            learning_rate=float(values["learning_rate"]),
            discount_factor=float(values["discount_factor"]),
            epsilon_start=float(values["epsilon_start"]),
            epsilon_end=float(values["epsilon_end"]),
            epsilon_decay_episodes=values["epsilon_decay_episodes"],
            episodes=values["episodes"],
            random_seed=values["random_seed"],
        )
        if not all(isfinite(value) for value in (
            settings.q_initial_value, settings.learning_rate,
            settings.discount_factor, settings.epsilon_start, settings.epsilon_end
        )):
            raise ValueError("training values must be finite")
        if not 0 < settings.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 <= settings.discount_factor < 1:
            raise ValueError("discount_factor must be in [0, 1)")
        if not 0 <= settings.epsilon_end <= settings.epsilon_start <= 1:
            raise ValueError("exploration must satisfy 0 <= epsilon_end <= epsilon_start <= 1")
        if (type(settings.episodes) is not int or settings.episodes < 1
                or type(settings.epsilon_decay_episodes) is not int
                or settings.epsilon_decay_episodes < 1
                or type(settings.random_seed) is not int):
            raise ValueError("episodes, decay length, and random seed must be integers")
        return settings

    def epsilon(self, episode_index: int) -> float:
        fraction = min(episode_index / max(1, self.epsilon_decay_episodes - 1), 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


@dataclass(frozen=True)
class EpisodeRecord:
    episode: int
    epsilon: float
    total_reward: float
    steps: int
    reached_target: bool
    max_abs_q_change: float
    q_values: dict[tuple[str, str], float]


@dataclass(frozen=True)
class TrainingResult:
    initial_q: dict[tuple[str, str], float]
    final_q: dict[tuple[str, str], float]
    records: list[EpisodeRecord]
    visits: dict[tuple[str, str], int]


def q_learning_update(
    old_value: float, reward: float, next_max: float,
    learning_rate: float, discount_factor: float, terminated: bool
) -> float:
    target = reward if terminated else reward + discount_factor * next_max
    return old_value + learning_rate * (target - old_value)


def _choose_action(
    state: str, actions: tuple[str, ...], q: dict[tuple[str, str], float],
    epsilon: float, rng: Random
) -> str:
    if rng.random() < epsilon:
        return rng.choice(actions)
    best_value = max(q[state, action] for action in actions)
    best_actions = [
        action for action in actions
        if abs(q[state, action] - best_value) <= 1e-12
    ]
    return rng.choice(best_actions)


def train(environment: Environment, settings: TrainingSettings) -> TrainingResult:
    rng = Random(settings.random_seed)
    q = {
        (state, action): settings.q_initial_value
        for state in environment.states
        for action in environment.available_actions(state)
    }
    initial_q = q.copy()
    visits = {key: 0 for key in q}
    records = []

    for episode_index in range(settings.episodes):
        before_episode = q.copy()
        epsilon = settings.epsilon(episode_index)
        state = environment.start_state
        total_reward = 0.0
        reached_target = False

        for step_number in range(1, environment.max_steps + 1):
            actions = environment.available_actions(state)
            action = _choose_action(state, actions, q, epsilon, rng)
            next_state, reward, terminated = environment.step(state, action)
            next_actions = environment.available_actions(next_state)
            next_max = max((q[next_state, a] for a in next_actions), default=0.0)
            q[state, action] = q_learning_update(
                q[state, action], reward, next_max,
                settings.learning_rate, settings.discount_factor, terminated
            )
            visits[state, action] += 1
            total_reward += reward
            state = next_state
            if terminated:
                reached_target = True
                break

        records.append(EpisodeRecord(
            episode=episode_index + 1,
            epsilon=epsilon,
            total_reward=total_reward,
            steps=step_number,
            reached_target=reached_target,
            max_abs_q_change=max(abs(q[key] - before_episode[key]) for key in q),
            q_values=q.copy(),
        ))

    return TrainingResult(initial_q, q.copy(), records, visits)
