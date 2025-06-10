# n_step.py
# ──────────────────────────────────────────────────────────────────────────────
# N-Step transition generator: batches exactly `num_steps` transitions per call.
# Each episode’s total reward is logged once (WARNING level). If env.reset() fails,
# raises RuntimeError to be caught by ai.py.
# Modified to count raw_actions per episode and print a summary once per episode.
# ──────────────────────────────────────────────────────────────────────────────

from collections import deque, namedtuple
import torch
import numpy as np
import logging
import random
import cv2  # for simple lane detection

# Configure logging so only warnings and above show
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

Step = namedtuple('Step', ['state', 'action', 'reward', 'done', 'lstm'])

def detect_lane_simple(frame: np.ndarray) -> int:
    """
    Very naive: assume `frame` is a 2D grayscale or single‐channel image (e.g. 128×128),
    and the runner is the brightest cluster of pixels. Compute its x‐centroid,
    then assign lane 0/1/2 depending on whether centroid < 128/3, < 2*128/3, else 2.
    """
    if frame.ndim == 3 and frame.shape[2] > 1:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame.copy()

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    ys, xs = np.nonzero(thresh)
    if len(xs) == 0:
        return 1

    x_center = np.mean(xs)
    if x_center < 128 / 3:
        return 0
    elif x_center < 2 * (128 / 3):
        return 1
    else:
        return 2

class NStepProgress:
    """
    - env:    instance of Env (with reset() and step(...)).
    - ai:     instance of AI (with __call__(state, (hx, cx)) -> (action, (hx, cx))).
    - n_step: integer ≥ 1.
    - gamma:  discount factor for computing n-step returns (default 0.99).
    """
    def __init__(self, env, ai, n_step, gamma=0.99):
        self.env = env
        self.ai = ai
        self.n_step = n_step
        self.gamma = gamma

        self.episode_count = 0
        self.total_steps = 0
        self.rewards = []  # list of completed‐episode total rewards

    def __iter__(self):
        while True:
            yield from self._run_episode()

    def _run_episode(self):
        """
        Runs exactly one episode. Yields up to `n_step` transitions at a time.
        At the very end, logs one WARNING with total steps and total reward.
        Tracks raw_action counts and prints a summary once per episode.
        """
        state = self.env.reset()
        if state is None:
            raise RuntimeError("n_step._run_episode: env.reset() returned None (play.png not found)")

        history = deque()  # holds up to (n_step + 1) Step tuples
        reward_sum = 0.0
        is_done = False
        step_count = 0

        # Initialize LSTM hidden states (required by AI network)
        cx = torch.zeros(1, 256)
        hx = torch.zeros(1, 256)

        # Track how many times each raw_action is chosen this episode
        action_counts = [0, 0, 0, 0, 0]

        while True:
            # (1) Detach LSTM hidden if continuing, or zero them if starting new episode
            if is_done:
                cx = torch.zeros(1, 256)
                hx = torch.zeros(1, 256)
            else:
                cx = cx.detach()
                hx = hx.detach()

            # (2) Ask AI for an action given current state & hidden (hx, cx)
            ai_output, (hx, cx) = self.ai(torch.from_numpy(np.array([state], dtype=np.float32)), (hx, cx))

            # Ensure we have a single integer action
            if isinstance(ai_output, np.ndarray):
                raw_action = int(ai_output.flatten()[0])
            else:
                raw_action = int(ai_output)

            # Count this raw_action
            action_counts[raw_action] += 1

            # (3) Detect lane_index from the image in `state`
            frame_img = state
            if isinstance(frame_img, torch.Tensor):
                frame_img = frame_img.cpu().numpy()
            if frame_img.ndim == 3 and frame_img.shape[0] == 1:
                frame_img = frame_img[0]
            current_lane = detect_lane_simple(frame_img)

            # (4) Apply mask logic
            final_action = raw_action
            if raw_action == 1 and current_lane == 0:  # move-left blocked in lane 0
                final_action = 0
            if raw_action == 2 and current_lane == 2:  # move-right blocked in lane 2
                final_action = 0

            # (5) Step the environment
            next_state, r, is_done, _ = self.env.step(final_action, step_count)

            reward_sum += r
            history.append(Step(state=state, action=final_action, reward=r, done=is_done, lstm=(hx, cx)))

            # Keep at most (n_step + 1) in history
            while len(history) > self.n_step + 1:
                history.popleft()

            # (6) If we have exactly (n_step + 1), yield that slice as one transition
            if len(history) == self.n_step + 1:
                R = 0.0
                for idx, step in enumerate(list(history)[:self.n_step]):
                    R += (self.gamma ** idx) * step.reward

                done_n = history[self.n_step].done
                s_n = None if done_n else next_state

                s_0 = history[0].state
                a_0 = history[0].action

                yield (s_0, a_0, R, s_n, done_n)

            # (7) If done, flush remaining partial sequences
            if is_done:
                while len(history) >= 1:
                    R = 0.0
                    for idx, step in enumerate(history):
                        R += (self.gamma ** idx) * step.reward

                    s_0 = history[0].state
                    a_0 = history[0].action
                    yield (s_0, a_0, R, None, True)
                    history.popleft()

                # Log episode end
                self.episode_count += 1
                self.total_steps += step_count
                self.rewards.append(reward_sum)
                logger.warning(f"Episode {self.episode_count} ended: steps={step_count}, total_reward={reward_sum:.2f}")

                # Print action count summary for this episode
                print(f"[Debug] Episode {self.episode_count} raw_action counts: {action_counts}")

                break

            # (8) Prepare for next iteration
            state = next_state
            step_count += 1

    def rewards_steps(self):
        """
        Return the list of episode rewards from the most recent call
        (and then clear them). Called from ai.py after memory.run_steps().
        """
        out = self.rewards[:]
        self.rewards.clear()
        return out

    def get_statistics(self):
        total_eps = self.episode_count
        avg_steps = self.total_steps / total_eps if total_eps > 0 else 0.0
        return {
            'total_episodes': total_eps,
            'total_steps': self.total_steps,
            'avg_steps_per_episode': avg_steps,
        }

    def reset_statistics(self):
        self.episode_count = 0
        self.total_steps = 0
        self.rewards.clear()
        logger.warning("Statistics have been reset")
