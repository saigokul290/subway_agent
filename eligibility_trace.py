# ──────────────────────────────────────────────────────────────────────────────
# eligibility_trace.py (UPDATED)
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def eligibility_trace(batch, cnn, gamma=0.99, device=None):
    """
    Simplified eligibility trace / 1-step Q-learning update for flat transitions.
    Each element in `batch` is a 5-tuple: (state, action, R, next_state, done).

    Returns:
      • inputs: a FloatTensor of shape (batch_size × 1 × 128 × 128)
      • targets: a FloatTensor of shape (batch_size × num_actions)
    """
    inputs  = []
    targets = []

    # If no device specified, infer from CNN’s parameters
    if device is None:
        device = next(cnn.parameters()).device

    for (s0, a0, R, s_n, done_flag) in batch:
        # 1) Build input tensor from s0 (shape: 1×128×128)
        #    We assume s0 is a NumPy array of shape (1,128,128), dtype=float32
        state_tensor = torch.from_numpy(s0).unsqueeze(0).to(device)  
        #                ^ shape: (1, 1, 128, 128)

        # 2) Run a forward pass on s0 to get current Q(s0,·)
        with torch.no_grad():
            qvals, _ = cnn(state_tensor, None)  
            # qvals has shape (1, num_actions)

        # 3) Clone the Q-vector so we can modify the action slot
        target_q = qvals.clone().detach()  # shape: (1, num_actions)

        if done_flag:
            # If this transition was terminal, no next‐state bootstrap:
            target_q[0, a0] = R
        else:
            # Otherwise, compute max_a Q(s_n,a) and set:
            #    target[a0] = R + γ · max_a Q(s_n, a)
            next_state_tensor = torch.from_numpy(s_n).unsqueeze(0).to(device)
            with torch.no_grad():
                next_qvals, _ = cnn(next_state_tensor, None)  # (1, num_actions)
                max_next_q = next_qvals.max(dim=1)[0].item()
            target_q[0, a0] = R + gamma * max_next_q

        # 4) Append to our lists (dropping the leading batch‐size=1 dim here):
        inputs.append(s0)                      # NumPy array shape: (1,128,128)
        targets.append(target_q.squeeze(0))   # Torch tensor shape: (num_actions,)

    # Stack everything into tensors:
    inputs  = torch.from_numpy(np.array(inputs, dtype=np.float32))      # shape: (N, 1, 128, 128)
    targets = torch.stack(targets, dim=0).float()                       # shape: (N, num_actions)

    return inputs, targets
