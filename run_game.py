import torch
from env import Env
from neural_net import CNN, AI, EpsilonGreedyBody
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env()

    # load model
    n_actions = env.action_space
    model = CNN(n_actions).to(device)
    checkpoint = torch.load('old_brain.pth', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # greedy policy
    body = EpsilonGreedyBody(0.0, 0.0, 1)
    agent = AI(brain=model, body=body)

    num_eps = 50
    all_rewards = []
    total_action_counts = np.zeros(n_actions, dtype=int)

    for ep in range(1, num_eps+1):
        state = env.reset()
        hidden = (torch.zeros(1,256), torch.zeros(1,256))
        ep_reward = 0
        done = False
        step = 0
        ep_action_counts = np.zeros(n_actions, dtype=int)

        while not done:
            st = torch.from_numpy(state).unsqueeze(0).to(device)
            # peek at Q-values before selecting
            with torch.no_grad():
                q_vals = model(st)
            if step % 20 == 0:  # print every 20 steps
                print(f"[Ep{ep} Step{step}] Q-values: {q_vals.cpu().numpy().flatten()}")

            action, hidden = agent(st, hidden)
            ep_action_counts[action] += 1
            total_action_counts[action] += 1

            nxt, r, done, _ = env.step(action, step)
            ep_reward += r
            step += 1
            if nxt is None:
                break
            state = nxt

        all_rewards.append(ep_reward)
        print(f"Episode {ep:2d} → reward {ep_reward:.1f}, action counts {ep_action_counts}")

    env.close()

    # summary
    print("\n=== Summary over", num_eps, "episodes ===")
    print("Rewards: mean {:.2f}, min {:.2f}, max {:.2f}, std {:.2f}"
          .format(np.mean(all_rewards), np.min(all_rewards),
                  np.max(all_rewards), np.std(all_rewards)))
    print("Total action distribution:", total_action_counts)
    print("Percentages:", (total_action_counts/total_action_counts.sum()*100).round(1))

if __name__ == "__main__":
    main()
