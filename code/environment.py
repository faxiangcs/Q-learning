"""Deterministic region environment loaded from an experiment config."""

from collections import deque
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Environment:
    states: tuple[str, ...]
    adjacency: dict[str, tuple[str, ...]]
    start_state: str
    target_state: str
    non_target_reward: float
    target_reward: float
    max_steps: int

    @classmethod
    def from_config(cls, config: dict) -> "Environment":
        states = tuple(config["states"])
        if not states or len(set(states)) != len(states):
            raise ValueError("states must be a nonempty list of unique names")
        if not all(isinstance(state, str) and state for state in states):
            raise ValueError("every state must have a nonempty string name")

        raw_adjacency = config["adjacency"]
        if set(raw_adjacency) != set(states):
            raise ValueError("adjacency must contain exactly the declared states")
        adjacency = {state: tuple(raw_adjacency[state]) for state in states}
        for state, neighbors in adjacency.items():
            if len(neighbors) != len(set(neighbors)) or state in neighbors:
                raise ValueError(f"invalid or repeated neighbor in {state}")
            for neighbor in neighbors:
                if neighbor not in adjacency or state not in adjacency[neighbor]:
                    raise ValueError(f"adjacency must be bidirectional: {state} -> {neighbor}")

        start = config["start_state"]
        target = config["target_state"]
        if start not in states or target not in states or start == target:
            raise ValueError("start and target must be distinct declared states")
        if config["action_semantics"] != "move_to_adjacent_state":
            raise ValueError("only adjacent-state actions are supported")
        if config["transition"] != {
            "type": "deterministic",
            "next_state": "selected_action",
            "invalid_actions": "excluded",
        }:
            raise ValueError("unsupported transition rule")

        rewards = config["rewards"]
        normal_reward = float(rewards["non_target_move"])
        target_reward = float(rewards["enter_target"])
        if not isfinite(normal_reward) or not isfinite(target_reward):
            raise ValueError("rewards must be finite numbers")
        termination = config["termination"]
        max_steps = termination["max_steps_per_episode"]
        if (termination["on_enter_target"] is not True
                or termination["on_step_limit"] != "truncated_without_goal_reward"
                or type(max_steps) is not int or max_steps < 1):
            raise ValueError("unsupported termination rule")

        environment = cls(
            states, adjacency, start, target, normal_reward, target_reward, max_steps
        )
        if environment.shortest_path() is None:
            raise ValueError("target is unreachable from start")
        return environment

    def available_actions(self, state: str) -> tuple[str, ...]:
        if state not in self.adjacency:
            raise ValueError(f"unknown state: {state}")
        return () if state == self.target_state else self.adjacency[state]

    def step(self, state: str, action: str) -> tuple[str, float, bool]:
        if action not in self.available_actions(state):
            raise ValueError(f"illegal action {action!r} from state {state!r}")
        terminated = action == self.target_state
        reward = self.target_reward if terminated else self.non_target_reward
        return action, reward, terminated

    def shortest_path(self) -> list[str] | None:
        queue = deque([(self.start_state, [self.start_state])])
        visited = {self.start_state}
        while queue:
            state, path = queue.popleft()
            if state == self.target_state:
                return path
            for next_state in self.adjacency[state]:
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, path + [next_state]))
        return None
