import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

from .model import RefLLENet
from .env import LowLightEnv
from .data import LOLDataset
from .a3c import train_step
from .config import PAPER_CONFIG, TINY_CONFIG
from .rewards import ProxyQualityScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="tiny", choices=["tiny", "paper"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = TINY_CONFIG if args.preset == "tiny" else PAPER_CONFIG
    device = torch.device(args.device)

    model = RefLLENet(input_channels=1, hidden_dim=config.hidden_dim, num_actions=31).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    dataset = LOLDataset()
    scorer = ProxyQualityScorer()

    print(f"Training with {config.name} config, {config.num_rounds} rounds")

    total_reward = 0
    for round_idx in tqdm(range(config.num_rounds)):
        idx = round_idx % len(dataset)
        low_image, normal_image = dataset[idx]
        low_image = low_image.to(device)

        env = LowLightEnv(low_image, scorer=scorer, max_steps=config.max_episode_steps, zfc_target=config.zfc_target)
        state = env.reset()

        episode_reward = 0
        for step in range(config.steps_per_episode):
            with torch.no_grad():
                policy_logits, _ = model(state.to(device))
                policy = torch.softmax(policy_logits, dim=-1)
                action_dist = torch.distributions.Categorical(policy)
                action = action_dist.sample()

            if state.ndim == 2:
                action_map = action.reshape(state.shape)
            else:
                action_map = action.reshape(state.shape[1:])

            next_state, reward, done = env.step(action_map)

            loss, p_loss, v_loss, ent = train_step(
                model, optimizer, state.to(device), action_map.to(device),
                torch.tensor(reward, dtype=torch.float32).to(device),
                config.gamma, config.entropy_coef
            )

            episode_reward += reward
            state = next_state.to(device)

            if done:
                break

        total_reward += episode_reward

        if (round_idx + 1) % 100 == 0:
            avg_reward = total_reward / 100
            print(f"Round {round_idx + 1}: avg reward = {avg_reward:.4f}")
            total_reward = 0


if __name__ == "__main__":
    main()
