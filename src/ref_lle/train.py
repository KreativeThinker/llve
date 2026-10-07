import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import RefLLENet
from .env import LowLightEnv
from .data import LOLDataset
from .a3c import train_step
from .config import PAPER_CONFIG, TINY_CONFIG
from .rewards import UniqueScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="tiny", choices=["tiny", "paper"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--low-light-path", default="data/lolv2-real/Train/Input", type=str)
    parser.add_argument("--normal-light-path", default="data/lolv2-real/Train/GT", type=str)
    args = parser.parse_args()

    config = TINY_CONFIG if args.preset == "tiny" else PAPER_CONFIG
    config.lol_train_path = args.low_light_path
    config.lol_test_path = args.normal_light_path
    device = torch.device(args.device)

    model = RefLLENet(input_channels=3, hidden_dim=config.hidden_dim, num_actions=31).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    dataset = LOLDataset(low_light_path=config.lol_train_path, normal_light_path=config.lol_test_path)
    # Use DataLoader to fetch batches
    batch_size = max(1, config.batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scorer = UniqueScorer(device=args.device)

    print(f"Training with {config.name} config, {config.num_rounds} rounds, batch size {batch_size}")

    total_reward = 0
    data_iter = iter(dataloader)
    
    for round_idx in tqdm(range(config.num_rounds)):
        try:
            low_image, normal_image = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            low_image, normal_image = next(data_iter)
            
        low_image = low_image.to(device)

        env = LowLightEnv(low_image, scorer=scorer, max_steps=config.max_episode_steps, zfc_target=config.zfc_target, device=device)
        state = env.reset().to(device)

        episode_reward = 0
        for step in range(config.steps_per_episode):
            with torch.no_grad():
                policy_logits, _ = model(state)
                policy = torch.softmax(policy_logits, dim=-1)
                action_dist = torch.distributions.Categorical(policy)
                action = action_dist.sample()

            b, c, h, w = state.shape
            action_map = action.reshape(b, h, w)

            next_state, reward, done = env.step(action_map)

            loss, p_loss, v_loss, ent = train_step(
                model, optimizer, state, action_map,
                reward,
                config.gamma, config.entropy_coef
            )

            episode_reward += reward.mean().item()
            state = next_state.to(device)

            if done:
                break

        total_reward += episode_reward

        if (round_idx + 1) % 100 == 0:
            avg_reward = total_reward / 100
            print(f"Round {round_idx + 1}: avg reward = {avg_reward:.4f}")
            total_reward = 0

    torch.save(model.state_dict(), "model.pt")
    print("Saved trained model to model.pt")


if __name__ == "__main__":
    main()
