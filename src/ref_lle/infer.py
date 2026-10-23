import argparse
import torch
from PIL import Image
import numpy as np

from .model import RefLLENet
from .env import LowLightEnv
from .rewards import ProxyQualityScorer
from .fourier import zero_frequency_component, amplitude_phase_split


def enhance_image(model, input_tensor, zfc_target=2.5e5, max_iterations=10, device='cpu', deterministic=True):
    """
    Run personalized adaptive enhancement guided by ZFC convergence.
    input_tensor: [C, H, W] in [0, 1]
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # In inference, reward scoring is not needed; use fast proxy scorer
    env = LowLightEnv(
        input_tensor,
        scorer=ProxyQualityScorer(),
        max_steps=max_iterations,
        zfc_target=zfc_target,
        device=device,
        return_image_state=True
    )
    state = env.reset().to(device)

    for step in range(max_iterations):
        with torch.no_grad():
            policy_logits, _ = model(state)
            if deterministic:
                action = torch.argmax(policy_logits, dim=-1)
            else:
                policy = torch.softmax(policy_logits, dim=-1)
                action_dist = torch.distributions.Categorical(policy)
                action = action_dist.sample()

        b, c, h, w = state.shape
        action_map = action.reshape(b, h, w)

        next_state, _, done = env.step(action_map)
        zfc = env.get_zfc()
        zfc_val = zfc.mean().item()

        # Convergence condition from paper Section III-D: within 10% of target
        if done or abs(zfc_val - zfc_target) / (zfc_target + 1e-8) < 0.1:
            break

        state = next_state.to(device)

    enhanced = env.image_current.squeeze(0).cpu().numpy()
    if enhanced.ndim == 3:
        enhanced = np.transpose(enhanced, (1, 2, 0))

    enhanced = np.clip(enhanced * 255.0, 0, 255.0).astype(np.uint8)
    return enhanced


def main():
    parser = argparse.ArgumentParser(description="ReF-LLE Adaptive Personalized Inference")
    parser.add_argument("--input", required=True, help="Input low-light image path")
    parser.add_argument("--ref", default=None, help="Reference image path for ZFC target")
    parser.add_argument("--zfc-ref", type=float, default=2.5e5, help="Reference ZFC value (default: 2.5e5)")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max iterations (default: 10)")
    parser.add_argument("--output", default="enhanced.png", help="Output image path")
    parser.add_argument("--model", default=None, help="Model checkpoint path")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use argmax greedy action selection")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    input_img = Image.open(args.input).convert("RGB")
    input_array = np.array(input_img).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(np.transpose(input_array, (2, 0, 1)))

    zfc_target = args.zfc_ref
    if args.ref:
        ref_img = Image.open(args.ref).convert("RGB")
        ref_array = np.array(ref_img).astype(np.float32) / 255.0
        ref_tensor = torch.from_numpy(np.transpose(ref_array, (2, 0, 1)))
        ref_amp, _ = amplitude_phase_split(ref_tensor.unsqueeze(0))
        zfc_target = zero_frequency_component(ref_amp).mean().item()
        print(f"Extracted ZFC target from reference image: {zfc_target:.2e}")

    hidden_dim = 32
    ckpt = None
    if args.model:
        ckpt = torch.load(args.model, map_location=args.device)
        if "encoder.conv1.weight" in ckpt:
            hidden_dim = ckpt["encoder.conv1.weight"].shape[0]

    model = RefLLENet(input_channels=3, hidden_dim=hidden_dim, num_actions=31)
    if ckpt is not None:
        model.load_state_dict(ckpt)
        print(f"Loaded weights from {args.model} (hidden_dim={hidden_dim})")

    enhanced = enhance_image(
        model=model,
        input_tensor=input_tensor,
        zfc_target=zfc_target,
        max_iterations=args.max_iterations,
        device=args.device,
        deterministic=args.deterministic
    )

    Image.fromarray(enhanced).save(args.output)
    print(f"Enhanced image saved to {args.output}")


if __name__ == "__main__":
    main()
