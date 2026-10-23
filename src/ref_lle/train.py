import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from .model import RefLLENet
from .env import LowLightEnv
from .data import LOLDataset
from .a3c import train_trajectory_step
from .config import PAPER_CONFIG, TINY_CONFIG
from .rewards import UniqueScorer, ProxyQualityScorer


def get_scorer(name='auto', device='cpu'):
    if name == 'unique' or name == 'auto':
        try:
            return UniqueScorer(device=device)
        except Exception as e:
            if name == 'unique':
                raise e
            print(f"Notice: UniqueScorer unavailable ({e}), using ProxyQualityScorer.")
            return ProxyQualityScorer()
    return ProxyQualityScorer()


class RunLogger:
    """Writes formatted progress logs to runs/latest.log and an archival timestamped log."""
    def __init__(self, log_dir="runs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.log_dir / "latest.log"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.archive_path = self.log_dir / f"train_{timestamp}.log"

        # Initialize latest.log
        with open(self.latest_path, "w", encoding="utf-8") as f:
            f.write(f"=== ReF-LLE Training Log Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}\n"
        with open(self.latest_path, "a", encoding="utf-8") as f:
            f.write(formatted)
        with open(self.archive_path, "a", encoding="utf-8") as f:
            f.write(formatted)


def main():
    parser = argparse.ArgumentParser(description="ReF-LLE Training via Deep RL (A3C Trajectory)")
    parser.add_argument("--preset", default="paper", choices=["tiny", "paper"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--low-light-path", default="data/lolv2-real/Train/Input", type=str)
    parser.add_argument("--normal-light-path", default="data/lolv2-real/Train/GT", type=str)
    parser.add_argument("--patch-size", default=0, type=int, help="Random patch crop size (0 for full resolution)")
    parser.add_argument("--scorer", default="unique", choices=["auto", "unique", "proxy"])
    parser.add_argument("--save-path", default="model.pt", type=str)
    parser.add_argument("--save-every", default=100, type=int, help="Save intermediate checkpoint every N rounds")
    parser.add_argument("--log-every", default=10, type=int, help="Log metrics to latest.log every N rounds")
    parser.add_argument("--rounds", default=None, type=int, help="Override number of training rounds")
    parser.add_argument("--log-dir", default="runs", type=str, help="Directory to save run logs")
    args = parser.parse_args()

    config = TINY_CONFIG if args.preset == "tiny" else PAPER_CONFIG
    if args.rounds is not None:
        config.num_rounds = args.rounds
    config.lol_train_path = args.low_light_path
    config.lol_test_path = args.normal_light_path
    device = torch.device(args.device)

    logger = RunLogger(log_dir=args.log_dir)

    model = RefLLENet(input_channels=3, hidden_dim=config.hidden_dim, num_actions=31).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    dataset = LOLDataset(
        low_light_path=config.lol_train_path,
        normal_light_path=config.lol_test_path,
        patch_size=args.patch_size if args.patch_size > 0 else None,
        is_train=True
    )
    batch_size = max(1, config.batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scorer = get_scorer(args.scorer, device=args.device)

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() and device.type == 'cuda' else 'CPU'
    patch_desc = f"{args.patch_size}x{args.patch_size}" if args.patch_size > 0 else "Full Resolution (400x600)"

    header = (
        f"Config: {config.name} | Total Rounds: {config.num_rounds} | Batch Size: {batch_size} | "
        f"Steps/Ep: {config.steps_per_episode} | Learning Rate: {config.learning_rate}\n"
        f"Device: {device} ({gpu_info}) | PyTorch: {torch.__version__} | Scorer: {scorer.__class__.__name__}\n"
        f"Resolution: {patch_desc} | Dataset: {len(dataset)} pairs from {config.lol_train_path}\n"
        f"Checkpoints: checkpoints/ (every {args.save_every} rounds) | Final Save: {args.save_path}\n"
        f"Logging: {logger.latest_path} (every {args.log_every} rounds)"
    )

    print(f"=== ReF-LLE Training Started ===")
    print(header)
    logger.log(header)

    total_reward = 0
    recent_rewards = []
    data_iter = iter(dataloader)

    pbar = tqdm(range(config.num_rounds), desc="Training")
    for round_idx in pbar:
        try:
            low_image, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            low_image, _ = next(data_iter)

        low_image = low_image.to(device)

        # Dynamically scale target ZFC by actual resolution (baseline: 400x600 = 240,000 px)
        h, w = low_image.shape[-2:]
        effective_zfc = config.zfc_target * (h * w / 240000.0)

        env = LowLightEnv(
            low_image,
            scorer=scorer,
            max_steps=config.max_episode_steps,
            zfc_target=effective_zfc,
            device=device,
            return_image_state=True
        )
        state = env.reset().to(device)

        ep_states = []
        ep_actions = []
        ep_rewards = []
        episode_reward = 0

        for step in range(config.steps_per_episode):
            with torch.no_grad():
                policy_logits, _ = model(state)
                policy = torch.softmax(policy_logits, dim=-1)
                action_dist = torch.distributions.Categorical(policy)
                action = action_dist.sample()

            b, c, h_s, w_s = state.shape
            action_map = action.reshape(b, h_s, w_s)

            next_state, reward, done = env.step(action_map)

            ep_states.append(state)
            ep_actions.append(action_map)
            ep_rewards.append(reward)

            episode_reward += reward.mean().item()
            state = next_state.to(device)

            if done:
                break

        # Trajectory-based Actor-Critic optimization
        loss, p_loss, v_loss, ent = train_trajectory_step(
            model, optimizer,
            states=ep_states,
            actions=ep_actions,
            rewards=ep_rewards,
            next_state=state if not done else None,
            gamma=config.gamma,
            entropy_coef=config.entropy_coef
        )

        total_reward += episode_reward
        recent_rewards.append(episode_reward)

        pbar.set_postfix({
            "ep_r": f"{episode_reward:.1f}",
            "loss": f"{loss:.3f}",
            "ent": f"{ent:.2f}"
        })

        # Periodic log file entry
        if (round_idx + 1) % args.log_every == 0 or (round_idx + 1) == config.num_rounds:
            avg_r = sum(recent_rewards) / len(recent_rewards)
            recent_rewards = []
            log_msg = (
                f"Round {round_idx + 1}/{config.num_rounds} | "
                f"Episode Reward: {episode_reward:+.1f} | Avg Reward (last {args.log_every}): {avg_r:+.1f} | "
                f"Loss: {loss:.3e} (policy: {p_loss:.3f}, value: {v_loss:.3e}) | Entropy: {ent:.3f}"
            )
            logger.log(log_msg)

        # Periodic intermediate checkpointing
        if (round_idx + 1) % args.save_every == 0 or (round_idx + 1) == config.num_rounds:
            ckpt_path = ckpt_dir / f"model_round_{round_idx + 1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), args.save_path)
            logger.log(f"--> Saved checkpoint: {ckpt_path} and {args.save_path}")

    done_msg = f"Training finished successfully across {config.num_rounds} rounds. Final weights saved to {args.save_path}"
    print(f"\n{done_msg}")
    logger.log(done_msg)


if __name__ == "__main__":
    main()
