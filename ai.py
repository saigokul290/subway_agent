# ai.py
# ──────────────────────────────────────────────────────────────────────────────
# Full training loop with:
#   • Safe‐Start (5 forced “forward” frames after each reset)
#   • Debug prints showing raw_action vs. final_action
# ──────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.optim as optim
import random

from env import Env
import n_step
import replay_memory
import neural_net
from eligibility_trace import eligibility_trace
import moving_avg

# 1) Choose device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ai] Using device: {device}")

# 2) Initialize environment & AI
senv = Env()                                # Your Env from env.py
number_actions = senv.action_space          # e.g. 5 possible moves

# Create the CNN+LSTM “brain” and ε-greedy policy “body”
cnn = neural_net.CNN(number_actions).to(device)
epsilon_body = neural_net.EpsilonGreedyBody(
    initial_epsilon=1.0,
    min_epsilon=0.05,
    decay_steps=10000
)
ai = neural_net.AI(brain=cnn, body=epsilon_body)

# 3) Set up N-step (n=2) and replay memory (capacity=20 000)
n_steps = n_step.NStepProgress(env=senv, ai=ai, n_step=2, gamma=0.99)
memory = replay_memory.ReplayMemory(n_steps=n_steps, capacity=20000)

# 4) Moving‐average over the last 500 episode‐rewards
ma = moving_avg.MA(500)

# 5) Checkpoint function
def save_checkpoint():
    torch.save({
        'state_dict': cnn.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }, 'old_brain.pth')
    print("[ai] Checkpoint saved")

# 6) Hyperparameters
nb_epochs  = 200
optimizer  = optim.Adam(cnn.parameters(), lr=0.001)
loss_fn    = nn.MSELoss()
batch_size = 64

for epoch in range(1, nb_epochs + 1):
    # ──────────────────────────────────────────────────────────────────────────
    # (A) Monkey‐patch senv.reset() so that each time n_step calls reset(),
    #     we automatically do 5 “forward” steps first. This breaks the “spawn
    #     between two trains” deadlock at frame 0/1.
    # ──────────────────────────────────────────────────────────────────────────
    forced_frames = 5
    real_reset = senv.reset  # keep a reference to the original reset()

    def reset_with_safe_start():
        """
        Wraps the original reset() so that, immediately after env.reset(),
        we execute `forced_frames` of env.step(action_idx=0) to move the trains
        slightly apart. Returns the post‐forward state.
        """
        state0 = real_reset()
        if state0 is None:
            return None

        scount = 0
        done0  = False
        _state = state0

        # Force 5 forward frames:
        for _ in range(forced_frames):
            _state, _, done0, _ = senv.step(action_idx=0, step_count=scount)
            scount += 1
            if done0:
                break

        return _state

    # Temporarily override senv.reset
    senv.reset = reset_with_safe_start

    # ──────────────────────────────────────────────────────────────────────────
    # (B) Collect exactly 128 N-step transitions into replay memory
    # ──────────────────────────────────────────────────────────────────────────
    prev_eps = n_steps.episode_count
    try:
        memory.run_steps(128)
    except RuntimeError as e:
        print(f"[ai] RuntimeError in run_steps: {e} → skipping this epoch")
        # Restore the original reset() before continuing
        senv.reset = real_reset
        continue

    new_eps         = n_steps.episode_count
    eps_this_epoch  = new_eps - prev_eps
    print(f"[ai] Epoch {epoch}: {eps_this_epoch} episodes ended this epoch")

    # Restore senv.reset back to the original for all future code
    senv.reset = real_reset

    # ──────────────────────────────────────────────────────────────────────────
    # (C) TRAIN ON MINIBATCHES (only if we have ≥ batch_size transitions)
    # ──────────────────────────────────────────────────────────────────────────
    if len(memory) >= batch_size:
        # Sample _one_ batch of size batch_size (a list of 5‐tuples)
        batch = memory.sample_batch(batch_size)
        # eligibility_trace expects: List[ (state, action, R, next_state, done) ]
        inputs, targets = eligibility_trace(batch, cnn, gamma=0.99, device=device)

        # Move tensors to device
        inputs  = inputs.to(device)   # shape: (batch_size, 1, 128, 128)
        targets = targets.to(device)  # shape: (batch_size, num_actions)

        # Forward‐pass + compute loss + backprop
        preds, _ = cnn(inputs, None)  # (batch_size, num_actions)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        print(f"[ai]   Not enough samples to train (buffer size {len(memory)} < {batch_size})")

    # ──────────────────────────────────────────────────────────────────────────
    # (D) Retrieve & print each episode’s reward that ended in this run_steps()
    # ──────────────────────────────────────────────────────────────────────────
    rewards = n_steps.rewards_steps()
    for idx, r in enumerate(rewards, start=1):
        print(f"[ai]   Episode {idx} Reward: {r:.2f}")

    # ──────────────────────────────────────────────────────────────────────────
    # (E) Print the epoch’s average reward (over all episodes that ended)
    # ──────────────────────────────────────────────────────────────────────────
    if len(rewards) > 0:
        avg = sum(rewards) / len(rewards)
        print(f"[ai] Epoch {epoch} Average Reward: {avg:.2f}")
    else:
        print(f"[ai] Epoch {epoch} Average Reward: (no episodes ended)")

    # ──────────────────────────────────────────────────────────────────────────
    # (F) Update & print moving‐average (last 500 episodes)
    # ──────────────────────────────────────────────────────────────────────────
    ma.add(rewards)
    print(f"[ai]    Moving‐Average (last 500 eps): {ma.average():.2f}")

    # ──────────────────────────────────────────────────────────────────────────
    # (G) Save a checkpoint each epoch
    # ──────────────────────────────────────────────────────────────────────────
    save_checkpoint()

    # ──────────────────────────────────────────────────────────────────────────
    # (H) Early‐stop if moving‐average ≥ 100
    # ──────────────────────────────────────────────────────────────────────────
    if ma.average() >= 100:
        print("[ai] Moving‐average ≥ 100; stopping training.")
        save_checkpoint()
        break

print("[ai] Training loop has completed.")
