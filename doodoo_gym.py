"""
DooDoo vs Physics — Same collapse core, different world.
Zero training. Zero examples. Just the algebra.

Usage:
    python doodoo_gym.py              # Run all environments
    python doodoo_gym.py CartPole     # Run one specific env
"""

import sys
import numpy as np
from aoi_collapse import aoi_collapse

import gymnasium


def encode_state(obs: np.ndarray, env_name: str, history: list) -> np.ndarray:
    """
    Encode gym observation + history into dense 24D state vector.
    6 feature layers x up to 4 obs dims = 24D of real signal.
    Same dimensionality as market data — the collapse doesn't know the difference.
    """
    state = np.zeros(24)
    obs = np.asarray(obs, dtype=np.float64)
    n = len(obs)
    hist = [np.asarray(h, dtype=np.float64) for h in history]

    # Spread obs features evenly across 24D with temporal depth
    # Each obs feature gets: raw, velocity, acceleration, mean, std, cross
    slots_per_feat = min(6, 24 // max(n, 1))
    for i in range(min(n, 4)):
        base = i * slots_per_feat

        # Raw value
        if base < 24:
            state[base] = obs[i]

        # Velocity (first derivative)
        if base + 1 < 24 and len(hist) >= 2:
            state[base + 1] = obs[i] - hist[-2][i]

        # Acceleration (second derivative)
        if base + 2 < 24 and len(hist) >= 3:
            v1 = obs[i] - hist[-2][i]
            v0 = hist[-2][i] - hist[-3][i]
            state[base + 2] = v1 - v0

        # Rolling mean
        if base + 3 < 24 and len(hist) >= 3:
            window = min(len(hist), 10)
            state[base + 3] = np.mean([h[i] for h in hist[-window:]])

        # Rolling std
        if base + 4 < 24 and len(hist) >= 5:
            window = min(len(hist), 10)
            state[base + 4] = np.std([h[i] for h in hist[-window:]])

        # Cross-feature: product with next feature
        if base + 5 < 24 and n >= 2:
            state[base + 5] = obs[i] * obs[(i + 1) % n]

    # Scale to unit-ish norm for consistent collapse behavior
    norm = np.linalg.norm(state)
    if norm > 0.01:
        state = state / norm * 3.0  # target norm ~3 (mid-range for collapse)

    return state


def collapse_to_discrete_action(result: dict, n_actions: int) -> int:
    """Map collapse outputs to discrete action."""
    control = result['control_vec']
    chaos = result['normalized_chaos']
    jordan = result['decomposition']['jordan']
    jordan_mean = float(np.mean(jordan.vec))

    if n_actions == 2:
        # CartPole: left(0) or right(1)
        # Use control[0] (steer) as primary signal
        # Jordan mean biases toward "stabilizing" direction
        signal = control[0] + jordan_mean * 0.5
        return 1 if signal > 0 else 0

    elif n_actions == 3:
        # MountainCar/Acrobot: left(0), nothing(1), right(2)
        signal = control[0] + jordan_mean * 0.3

        # High chaos → do nothing (uncertain)
        if chaos > 8 and abs(signal) < 0.5:
            return 1

        if signal > 0.3:
            return 2  # right/forward
        elif signal < -0.3:
            return 0  # left/backward
        else:
            return 1  # nothing

    else:
        # Generic: map control magnitude to action index
        idx = int((control[0] + 5) / 10 * n_actions)
        return max(0, min(n_actions - 1, idx))


def collapse_to_continuous_action(result: dict, action_space) -> np.ndarray:
    """Map collapse outputs to continuous action."""
    control = result['control_vec']
    chaos = result['normalized_chaos']

    # Scale control[0] to action range
    low = action_space.low[0]
    high = action_space.high[0]
    mid = (low + high) / 2
    rng = (high - low) / 2

    # Higher chaos → smaller actions (cautious)
    confidence = max(0.1, 1.0 - chaos * 0.08)
    action_val = mid + np.tanh(control[0]) * rng * confidence

    return np.array([np.clip(action_val, low, high)], dtype=np.float32)


def run_environment(env_name: str, max_episodes: int = 20, max_steps: int = 500):
    """Run DooDoo in a gymnasium environment."""
    env = gymnasium.make(env_name)
    n_actions = env.action_space.n if hasattr(env.action_space, 'n') else None
    is_continuous = n_actions is None

    print(f"\n{'=' * 60}")
    print(f"DooDoo vs {env_name}")
    print(f"  Obs dim:  {env.observation_space.shape[0]}")
    print(f"  Actions:  {'continuous' if is_continuous else f'{n_actions} discrete'}")
    print(f"  Episodes: {max_episodes}")
    print(f"  Zero training. Zero examples. Pure algebra.")
    print(f"{'=' * 60}")

    rewards_all = []
    best_reward = -float('inf')
    best_episode = -1

    for ep in range(max_episodes):
        obs, _ = env.reset()
        history = [obs.copy()]
        total_reward = 0
        chaos_sum = 0

        for step in range(max_steps):
            # Encode observation into 24D
            state = encode_state(obs, env_name, history)

            # Collapse
            result = aoi_collapse(state)
            chaos_sum += result['normalized_chaos']

            # Act
            if is_continuous:
                action = collapse_to_continuous_action(result, env.action_space)
            else:
                action = collapse_to_discrete_action(result, n_actions)

            obs, reward, terminated, truncated, info = env.step(action)
            history.append(obs.copy())
            total_reward += reward

            if terminated or truncated:
                break

        avg_chaos = chaos_sum / (step + 1)
        rewards_all.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = ep

        # Print every episode
        steps_survived = step + 1
        marker = " <-- BEST" if total_reward == best_reward else ""
        print(f"  Ep {ep:3d}: reward={total_reward:8.1f}  steps={steps_survived:4d}  avg_chaos={avg_chaos:.1f}{marker}")

    env.close()

    # Summary
    avg_reward = np.mean(rewards_all)
    std_reward = np.std(rewards_all)
    print(f"\n  --- SUMMARY ---")
    print(f"  Mean reward:  {avg_reward:.1f} +/- {std_reward:.1f}")
    print(f"  Best reward:  {best_reward:.1f} (episode {best_episode})")
    print(f"  Worst reward: {min(rewards_all):.1f}")

    # Benchmark comparison
    if env_name == 'CartPole-v1':
        print(f"\n  Benchmarks:")
        print(f"    Random agent:  ~20 steps")
        print(f"    Solved:        500 steps (reward 500)")
        print(f"    DooDoo:        {avg_reward:.0f} steps avg, {best_reward:.0f} best")
        if avg_reward > 100:
            print(f"    >> 5x better than random — NOT BAD for zero training")
        if best_reward >= 500:
            print(f"    >> SOLVED at least once — zero training, pure algebra")

    elif env_name == 'Acrobot-v1':
        print(f"\n  Benchmarks:")
        print(f"    Random agent:  ~-500 (never solves)")
        print(f"    Solved:        -100 or better")
        print(f"    DooDoo:        {avg_reward:.0f} avg, {best_reward:.0f} best")

    elif env_name == 'MountainCar-v0':
        print(f"\n  Benchmarks:")
        print(f"    Random agent:  -200 (never reaches flag)")
        print(f"    Solved:        -110 or better")
        print(f"    DooDoo:        {avg_reward:.0f} avg, {best_reward:.0f} best")

    elif env_name == 'Pendulum-v1':
        print(f"\n  Benchmarks:")
        print(f"    Random agent:  ~-1200")
        print(f"    Good:          -200 or better")
        print(f"    DooDoo:        {avg_reward:.0f} avg, {best_reward:.0f} best")

    return avg_reward, best_reward


if __name__ == '__main__':
    envs = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'Pendulum-v1']

    # Allow running specific env
    if len(sys.argv) > 1:
        target = sys.argv[1]
        envs = [e for e in envs if target.lower() in e.lower()]
        if not envs:
            print(f"No environment matching '{target}'")
            sys.exit(1)

    print("=" * 60)
    print("DooDoo vs Physics — Zero Training Challenge")
    print("Same aoi_collapse() that's trading live right now.")
    print("=" * 60)

    results = {}
    for env_name in envs:
        try:
            avg, best = run_environment(env_name)
            results[env_name] = (avg, best)
        except Exception as e:
            print(f"\n{env_name}: FAILED — {e}")

    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print(f"FINAL SCORECARD")
        print(f"{'=' * 60}")
        for name, (avg, best) in results.items():
            print(f"  {name:30s}  avg={avg:8.1f}  best={best:8.1f}")
        print(f"{'=' * 60}")
