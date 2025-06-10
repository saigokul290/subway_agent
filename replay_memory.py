# replay_memory.py
# ──────────────────────────────────────────────────────────────────────────────
# A simple ring buffer that, on each call to run_steps(num_steps),
# creates a fresh iterator from NStepProgress and pulls exactly num_steps
# transitions. This prevents StopIteration after a RuntimeError closes
# the old generator.
# ──────────────────────────────────────────────────────────────────────────────

import random
from collections import deque

class ReplayMemory:
    """
    A ring buffer of capacity `capacity`. Each element is expected
    to be a 5‐tuple: (state, action, reward, next_state, done).
    """

    def __init__(self, n_steps, capacity=20000):
        """
        Arguments:
          n_steps  – an instance of NStepProgress (i.e. an iterable yielding transitions)
          capacity – maximum number of transitions to store
        """
        self.n_steps = n_steps
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        """
        Add a single n‐step transition to the buffer.
        `transition` should be a 5‐tuple: (state, action, reward, next_state, done).
        """
        self.buffer.append(transition)

    def run_steps(self, num_steps):
        """
        Each time we enter run_steps, create a brand‐new iterator:
            iterator = iter(self.n_steps)
        Then pull exactly `num_steps` transitions (via next(iterator)) 
        and push them into our replay buffer. If NStepProgress.run_steps(...)
        raises a RuntimeError (e.g. because env.reset() returned None), let
        it propagate to ai.py so that epoch can be skipped. Any StopIteration
        from a fresh iterator should not happen (since NStepProgress is infinite),
        but if it did, it would simply bubble up.
        """
        # Create a fresh iterator from the NStepProgress object
        iterator = iter(self.n_steps)

        for _ in range(num_steps):
            # May raise RuntimeError if env.reset() fails
            transition = next(iterator)
            self.push(transition)

        # If buffer exceeds capacity, older items are automatically dropped
        # (deque with maxlen handles that for us).

    def sample_batch(self, batch_size):
        """
        Randomly sample a batch of `batch_size` transitions from the buffer.
        Returns a list of 5‐tuples.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
