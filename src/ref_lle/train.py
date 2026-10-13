import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
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


def main():
    parser = argparse.ArgumentParser(description="ReF-LLE Training via Deep RL (A3C Trajectory)")
    parser.add_argument("--preset", default="paper", choices=["tiny", "paper"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--low-light-path", default="data/lolv2-real/Train/Input", type=str)
    parser.add_argument("--normal-light-path", default="data/lolv2-real/Train/GT", type=str)
    parser.add_argument("--patch-size", default=256, type=int, help="Random patch crop size")
    parser.add_argument("--scorer", default="unique", choices=["auto", "unique", "proxy"])
    parser.add_argument("--save-path", default="model.pt", type=str)
    parser.add_argument("--save-every", default=100, type=int, help="Save intermediate checkpoint every N rounds")
    parser.add_argument("--rounds", default=None, type=int, help="Override number of training rounds")
    args = parser.parse_args()

    config = TINY_CONFIG if args.preset == "tiny" else PAPER_CONFIG
    if args.rounds is not None:
        config.num_rounds = args.rounds
    config.lol_train_path = args.low_light_path
    config.lol_test_path = args.normal_light_path
    device = torch.device(args.device)

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

    print(f"=== ReF-LLE Training Started ===")
    print(f"Config: {config.name} | Rounds: {config.num_rounds} | Steps/ep: {config.steps_per_episode} | Batch: {batch_size}")
    print(f"Scorer: {scorer.__class__.__name__} | Device: {device} | Patch: {args.patch_size}x{args.patch_size}")
    print(f"Dataset pairs: {len(dataset)} | Checkpoints saved to: checkpoints/ & {args.save_path}")

    total_reward = 0
    data_iter = iter(dataloader)

    pbar = tqdm(range(config.num_rounds), desc="Training")
    for round_idx in pbar:
        try:
            low_image, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            low_image, _ = next(data_iter)

        low_image = low_image.to(device)

        # Dynamically scale target ZFC by actual patch resolution (baseline: 400x600 = 240,000 px)
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
        pbar.set_postfix({
            "ep_r": f"{episode_reward:.1f}",
            "loss": f"{loss:.3f}",
            "ent": f"{ent:.2f}"
        })

        # Periodic intermediate checkpointing
        if (round_idx + 1) % args.save_every == 0 or (round_idx + 1) == config.num_rounds:
            ckpt_path = ckpt_dir / f"model_round_{round_idx + 1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), args.save_path)

    print(f"\nTraining finished successfully. Final weights saved to {args.save_path}")


if __name__ == "__main__":
    main()
