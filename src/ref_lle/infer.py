import argparse
import torch
from PIL import Image
import numpy as np

from .model import RefLLENet
from .env import LowLightEnv
from .rewards import ProxyQualityScorer
from .fourier import zero_frequency_component, amplitude_phase_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input low-light image path")
    parser.add_argument("--ref", default=None, help="Reference image path for ZFC target")
    parser.add_argument("--zfc-ref", type=float, default=2.5e5, help="Reference ZFC value")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max iterations")
    parser.add_argument("--output", default="enhanced.png", help="Output image path")
    parser.add_argument("--model", default=None, help="Model checkpoint path")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    input_img = Image.open(args.input).convert("L")
    input_array = np.array(input_img).astype(np.float32) / 255.0

    zfc_target = args.zfc_ref
    if args.ref:
        ref_img = Image.open(args.ref).convert("L")
        ref_array = np.array(ref_img).astype(np.float32) / 255.0
        ref_tensor = torch.from_numpy(ref_array)
        ref_amp, _ = amplitude_phase_split(ref_tensor)
        zfc_target_tensor = zero_frequency_component(ref_amp)
        zfc_target = zfc_target_tensor.item() if isinstance(zfc_target_tensor, torch.Tensor) else zfc_target_tensor

    model = RefLLENet(input_channels=1, hidden_dim=32, num_actions=31).to(device)
    if args.model:
        model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    input_tensor = torch.from_numpy(input_array)
    env = LowLightEnv(input_tensor, scorer=ProxyQualityScorer(), max_steps=args.max_iterations, zfc_target=zfc_target, device=device)
    state = env.reset()

    for step in range(args.max_iterations):
        with torch.no_grad():
            policy_logits, _ = model(state)
            policy = torch.softmax(policy_logits, dim=-1)
            action_dist = torch.distributions.Categorical(policy)
            action = action_dist.sample()

        if state.ndim == 2:
            action_map = action.reshape(state.shape)
        else:
            action_map = action.reshape(state.shape[1:])

        next_state, reward, done = env.step(action_map)

        zfc = env.get_zfc()

        print(f"Step {step + 1}: ZFC = {zfc[0].item():.2e}, reward = {reward:.4f}")

        if done or torch.abs(zfc[0] - zfc_target) / zfc_target < 0.1:
            break

        state = next_state.to(device)

    enhanced = state.cpu().numpy()
    if enhanced.ndim == 3:
        enhanced = np.mean(enhanced, axis=0)

    enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
    Image.fromarray(enhanced).save(args.output)
    print(f"Enhanced image saved to {args.output}")


if __name__ == "__main__":
    main()
