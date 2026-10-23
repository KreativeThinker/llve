import argparse
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import TemporalDiffRefLLENet
from .video_env import TemporalVideoLowLightEnv
from .rewards import ProxyQualityScorer
from .video_data import SDSDVideoDataset


def train_step_temporal(model, optimizer, composite_state, action_map, reward, entropy_coef=0.01):
    b, h, w = action_map.shape
    policy_logits, value = model(composite_state)

    policy = torch.softmax(policy_logits, dim=-1)
    log_policy = torch.log_softmax(policy_logits, dim=-1)

    action_flat = action_map.reshape(-1, 1)
    log_prob = log_policy.gather(1, action_flat).squeeze(1)

    advantage = reward.reshape(-1) - value.squeeze(1).detach()
    policy_loss = -(log_prob * advantage).mean()
    value_loss = 0.5 * (reward.reshape(-1) - value.squeeze(1)).pow(2).mean()

    entropy = -(policy * log_policy).sum(dim=-1).mean()
    total_loss = policy_loss + value_loss - entropy_coef * entropy

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=40.0)
    optimizer.step()

    return total_loss.item(), policy_loss.item(), value_loss.item(), entropy.item()


def main():
    parser = argparse.ArgumentParser(description="Train Motion-Consistent Temporal ReF-LLE on SDSD Video")
    parser.add_argument("--data-root", default="data/sdsd/indoor_np/indoor_np", type=str)
    parser.add_argument("--pretrained-10k", default="model_paper_10k.pt", type=str)
    parser.add_argument("--save-path", default="model_temporal_sdsd.pt", type=str)
    parser.add_argument("--num-rounds", default=1000, type=int)
    parser.add_argument("--steps-per-frame", default=8, type=int)
    parser.add_argument("--warmup-steps", default=12, type=int)
    parser.add_argument("--zfc-target", default=5.8e4, type=float)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== Starting Motion-Consistent Temporal ReF-LLE Training on {device} ===")

    model = TemporalDiffRefLLENet(hidden_dim=32, num_actions=31).to(device)

    # Warm-start from 10k single-image base model
    if Path(args.pretrained_10k).exists():
        print(f"Initializing temporal network from: {args.pretrained_10k}")
        model.load_from_10k(args.pretrained_10k, device=device)
    else:
        print("Pretrained 10k model not found. Training from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    dataset = SDSDVideoDataset(root_dir=args.data_root, clip_len=8, downsample=2)
    print(f"Cached {len(dataset)} video clips in RAM.")

    scorer = ProxyQualityScorer()

    total_reward = 0.0
    progress_bar = tqdm(range(args.num_rounds), desc="Temporal Training")

    for round_idx in progress_bar:
        in_clip, gt_clip = dataset[round_idx % len(dataset)]

        env = TemporalVideoLowLightEnv(
            video_clip=in_clip,
            gt_clip=gt_clip,
            scorer=scorer,
            steps_per_frame=args.steps_per_frame,
            warmup_steps=args.warmup_steps,
            zfc_target=args.zfc_target,
            device=device,
        )

        composite_state = env.reset_video()
        video_done = False
        episode_reward = 0.0

        while not video_done:
            b, c, h, w = env.image_current.shape
            with torch.no_grad():
                policy_logits, _ = model(composite_state)
                policy = torch.softmax(policy_logits, dim=-1)
                action_dist = torch.distributions.Categorical(policy)
                action = action_dist.sample()

            action_map = action.reshape(b, h, w)
            next_state, reward, frame_done, video_done, info = env.step(action_map)

            loss, p_loss, v_loss, ent = train_step_temporal(
                model=model,
                optimizer=optimizer,
                composite_state=composite_state,
                action_map=action_map,
                reward=reward,
            )

            episode_reward += reward.mean().item()
            composite_state = next_state

        total_reward += episode_reward

        if (round_idx + 1) % 100 == 0:
            avg_rew = total_reward / 100.0
            total_reward = 0.0
            progress_bar.set_postfix({"AvgRew": f"{avg_rew:.2f}", "Loss": f"{loss:.4f}"})
            # Save periodic checkpoint
            torch.save(model.state_dict(), args.save_path)

    torch.save(model.state_dict(), args.save_path)
    print(f"\nTraining Complete. Saved model to: {args.save_path}")


if __name__ == "__main__":
    main()
