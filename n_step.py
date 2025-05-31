from collections import deque, namedtuple
import torch
from torch.autograd import Variable
import numpy as np

Step = namedtuple('Step', ['state', 'action', 'reward', 'done', 'lstm'])

class NStepProgress:
    def __init__(self, env, ai, n_step):
        self.ai = ai  # ai object
        self.rewards = []
        self.env = env  # Importing our manual subway surfers environment
        self.n_step = n_step  # Number of steps to look forward

    def __iter__(self):  # Function to play game and collect/return samples
        state = self.env.reset()  # Resetting the game
        history = deque()
        reward = 0.0  # Initial reward = 0
        is_done = True
        end_buffer = []  # To remove unwanted images

        while True:
            if is_done:
                cx = Variable(torch.zeros(1, 256))
                hx = Variable(torch.zeros(1, 256))
            else:
                cx = Variable(cx.data)
                hx = Variable(hx.data)

            action, (hx, cx) = self.ai(Variable(torch.from_numpy(np.array([state], dtype=np.float32))), (hx, cx))
            end_buffer.append((state, action))

            while len(end_buffer) > 3:
                del end_buffer[0]

            # Printing action output
            t = action[0][0]
            if t == 1:
                print("left")
            elif t == 2:
                print("right")
            elif t == 3:
                print("jump")
            elif t == 4:
                print("roll")
            elif t == 0:
                print("do nothing")

            # Taking Action
            next_state, r, is_done, _ = self.env.step(action)

            # If game over
            if is_done:
                print("\nGame Ended\n")
                if len(end_buffer) >= 3:
                    state, action = end_buffer[-3]
                    history.pop()  # Removing unwanted experience
                r = -10
            reward += r
            history.append(Step(state=state, action=action, reward=r, done=is_done, lstm=(hx, cx)))

            # Returning the experiences to the replay memory
            while len(history) > self.n_step + 1:
                history.popleft()

            if len(history) == self.n_step + 1:
                yield tuple(history)

            state = next_state
            if is_done:
                if len(history) > self.n_step + 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
                self.rewards.append(reward)
                reward = 0.0
                state = self.env.reset()
                end_buffer = []
                history.clear()

    def rewards_steps(self):
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps
