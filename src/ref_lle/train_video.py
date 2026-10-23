import argparse
from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm

from .model import TemporalDiffRefLLENet
from .video_env import TemporalVideoLowLightEnv
from .video_data import SDSDVideoDataset
from .rewards import ProxyQualityScorer
from .utils import compute_entropy


def train_step_temporal(model, optimizer, composite_state, action_map, reward, entropy_coef=0.01):
    optimizer.zero_grad()
    policy_logits, value = model(composite_state)

    action_flat = action_map.reshape(-1)
    log_softmax = torch.log_softmax(policy_logits, dim=-1)
    log_probs = log_softmax.gather(1, action_flat.unsqueeze(-1)).squeeze(-1)

    value_flat = value.squeeze(-1)
    if reward.ndim == 1 or reward.numel() == 1:
        advantage = reward.item() - value_flat.detach()
    else:
        advantage = reward.reshape(-1) - value_flat.detach()

    policy_loss = -(log_probs * advantage).mean()
    value_loss = 0.5 * advantage.pow(2).mean()
    entropy = compute_entropy(policy_logits)
    loss = policy_loss + value_loss - entropy_coef * entropy

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 40.0)
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()


def main():
    parser = argparse.ArgumentParser(description="Train Causal Temporal Difference ReF-LLE on SDSD Video Clips")
    parser.add_argument("--sdsd-input", default="data/sdsd/indoor_np/indoor_np/input", help="SDSD low-light input directory")
    parser.add_argument("--sdsd-gt", default="data/sdsd/indoor_np/indoor_np/GT", help="SDSD ground-truth directory")
    parser.add_argument("--init-ckpt", default="model_paper_10k.pt", help="Pretrained 10k checkpoint to initialize from")
    parser.add_argument("--save-path", default="model_temporal_sdsd.pt", help="Path to save trained video model")
    parser.add_argument("--num-rounds", type=int, default=10000, help="Number of training rounds")
    parser.add_argument("--clip-len", type=int, default=4, help="Frames per video clip")
    parser.add_argument("--downsample", type=int, default=4, help="Spatial downsample factor for fast training")
    parser.add_argument("--steps-per-frame", type=int, default=2, help="Refinement steps per frame")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--zfc-target", type=float, default=5.8e4, help="Target ZFC for SDSD indoor")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("==================================================================")
    print(" Training Causal Temporal ReF-LLE (10k Rounds) on SDSD Dataset")
    print("==================================================================")
    print(f"Device: {device}")
    print(f"Initializing from: {args.init_ckpt}")
    print(f"Output weights:    {args.save_path}")

    model = TemporalDiffRefLLENet(hidden_dim=32, num_actions=31).to(device)
    if Path(args.init_ckpt).exists():
        model.load_from_10k(args.init_ckpt, device=device)
    else:
        print(f"Warning: {args.init_ckpt} not found. Starting from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Loading and caching SDSD video clips in memory...")
    dataset = SDSDVideoDataset(
        input_dir=args.sdsd_input,
        gt_dir=args.sdsd_gt,
        clip_len=args.clip_len,
        downsample=args.downsample,
        preload_to_ram=True,
    )
    print(f"Cached {len(dataset)} video clips in RAM.")

    scorer = ProxyQualityScorer()

    total_reward = 0.0
    progress_bar = tqdm(range(args.num_rounds), desc="10k Training")

    for round_idx in progress_bar:
        in_clip, gt_clip = dataset[round_idx % len(dataset)]

        env = TemporalVideoLowLightEnv(
            video_clip=in_clip,
            gt_clip=gt_clip,
            scorer=scorer,
            steps_per_frame=args.steps_per_frame,
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

        if (round_idx + 1) % 500 == 0:
            avg_rew = total_reward / 500.0
            total_reward = 0.0
            progress_bar.set_postfix({"AvgRew": f"{avg_rew:.2f}", "Loss": f"{loss:.4f}"})
            # Save periodic checkpoint
            torch.save(model.state_dict(), args.save_path)

    torch.save(model.state_dict(), args.save_path)
    print(f"\n10,000-Round Training Complete. Saved model to: {args.save_path}")


if __name__ == "__main__":
    main()
