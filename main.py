from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from ultralytics import YOLO

from modules.ddpg import DDPGAgent, DDPGConfig
from modules.distillation import DistillationManager
from modules.environment import PruningEnv, apply_pruning_plan, build_pruning_targets, remove_pruning_reparam


# Teacher와 Student의 중간 feature를 비교할 지점입니다.
# Pruning 대상 레이어와는 별개이며, YOLO neck의 3개 scale feature를 KD 신호로 사용합니다.
DEFAULT_KD_LAYERS = {
    "model.15": "model.15",
    "model.18": "model.18",
    "model.21": "model.21",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DDPG-based pruning search with YOLOv8 pose layer-wise distillation."
    )
    parser.add_argument("--weights", type=str, default="yolov8n-pose.pt", help="YOLO pose weights path.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of pruning-search episodes.")
    parser.add_argument("--imgsz", type=int, default=320, help="Calibration image size. Use 640 for final runs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Calibration batch size.")
    parser.add_argument("--calib-dir", type=str, default="", help="Optional image directory for KD calibration/recovery.")
    parser.add_argument("--max-targets", type=int, default=12, help="Max Conv2d targets. Use 0 for all backbone/neck convs.")
    parser.add_argument("--max-prune-ratio", type=float, default=0.6, help="Upper bound for each layer pruning ratio.")
    parser.add_argument("--exploration-noise", type=float, default=0.08, help="Gaussian exploration noise std.")
    parser.add_argument("--warmup-steps", type=int, default=8, help="Initial random actions before using the actor.")
    parser.add_argument("--batch-update", type=int, default=16, help="Replay batch size for DDPG updates.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Execution device.")
    parser.add_argument("--save-dir", type=str, default="runs/pruning_ddpg", help="Output directory.")
    parser.add_argument("--recovery-steps", type=int, default=0, help="Optional KD fine-tuning steps after best pruning plan.")
    parser.add_argument("--recovery-lr", type=float, default=1e-4, help="Learning rate for optional KD recovery.")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "list-targets"],
        help="Run training or only print candidate pruning layers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] device={device}, weights={args.weights}, imgsz={args.imgsz}")
    # Teacher는 고정된 기준 모델, student_base는 매 episode마다 복사될 원본 모델입니다.
    teacher_model = load_yolo_model(args.weights, device)
    student_base = load_yolo_model(args.weights, device)

    # 실제 데이터셋 경로가 없으면 random tensor로 전체 파이프라인을 먼저 검증할 수 있습니다.
    calibration_batch = make_calibration_batch(args.batch_size, args.imgsz, device, args.calib_dir)

    # Agent가 action을 줄 pruning 후보 Conv layer를 자동으로 수집합니다.
    # pose head는 출력 의미가 쉽게 깨지므로 기본적으로 제외합니다.
    targets = build_pruning_targets(
        student_base,
        calibration_batch,
        max_targets=args.max_targets,
        exclude_pose_head=True,
    )

    print(f"[targets] selected={len(targets)}")
    for target in targets:
        print(
            f"  - {target.index:02d} {target.name:<24} "
            f"in={target.in_channels:<3} out={target.out_channels:<3} "
            f"k={target.kernel_size} stride={target.stride} flops={target.flops / 1e6:.2f}M"
        )

    if args.mode == "list-targets":
        return

    # KD loss는 teacher/student의 중간 feature map 차이를 계산해서 reward에 반영됩니다.
    distiller = DistillationManager(
        teacher_model=teacher_model,
        student_model=student_base,
        layer_pairs=DEFAULT_KD_LAYERS,
    ).to(device)

    # PruningEnv는 DDPG 관점에서 state, action, reward, done을 제공하는 환경입니다.
    env = PruningEnv(
        teacher_model=teacher_model,
        student_model_base=student_base,
        prune_targets=targets,
        distiller=distiller,
        calibration_batch=calibration_batch,
        device=device,
        max_prune_ratio=args.max_prune_ratio,
    )

    agent = DDPGAgent(
        DDPGConfig(
            state_dim=env.state_dim,
            max_action=args.max_prune_ratio,
        ),
        device=device,
    )

    best_reward = float("-inf")
    best_actions: List[float] = []
    global_step = 0

    for episode in range(1, args.episodes + 1):
        # 매 episode는 깨끗한 student 복사본에서 시작합니다.
        state = env.reset()
        episode_reward = 0.0
        last_info = {}

        for _ in range(env.total_steps):
            # 초반에는 랜덤 행동으로 replay buffer를 채우고, 이후 actor 정책을 사용합니다.
            if global_step < args.warmup_steps:
                action = np.random.uniform(0.0, args.max_prune_ratio, size=(1,)).astype(np.float32)
            else:
                action = agent.select_action(state, noise_std=args.exploration_noise)

            # 한 layer를 pruning하고 KD loss와 FLOPs 감소량을 조합한 reward를 받습니다.
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            update_info = agent.update(args.batch_update)

            episode_reward += reward
            state = next_state
            global_step += 1
            last_info = info

            layer = info["layer"]
            print(
                f"[ep {episode:03d} step {env.current_step:02d}/{env.total_steps:02d}] "
                f"{layer:<24} prune={info['action']:.3f} "
                f"kd={info['kd_loss']:.6f} reward={reward:.4f} "
                f"reduced={info['reduced_flops_ratio']:.3f}"
            )
            if update_info:
                print(
                    f"    update critic={update_info['critic_loss']:.5f} "
                    f"actor={update_info['actor_loss']:.5f}"
                )
            if done:
                break

        # 총 reward가 가장 높은 pruning 비율 조합을 최종 plan으로 저장합니다.
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_actions = list(env.action_history)
            save_plan(save_dir / "best_pruning_plan.json", targets, best_actions, best_reward, args)

        print(
            f"[episode {episode:03d}] total_reward={episode_reward:.4f} "
            f"best={best_reward:.4f} final_kd={last_info.get('kd_loss', 0.0):.6f}"
        )

    agent.save(str(save_dir / "ddpg_agent.pt"))
    # best plan을 새 student에 적용하고, 옵션에 따라 KD recovery까지 수행합니다.
    save_pruned_student(args.weights, device, targets, best_actions, args, save_dir)
    distiller.close()
    print(f"[done] outputs saved to {save_dir}")


def load_yolo_model(weights: str, device: torch.device) -> torch.nn.Module:
    model = YOLO(weights).model.to(device)
    model.eval()
    return model


def make_calibration_batch(
    batch_size: int,
    imgsz: int,
    device: torch.device,
    calib_dir: str = "",
) -> torch.Tensor:
    image_paths = list_image_paths(calib_dir)
    if not image_paths:
        # 주제 검증 단계에서는 데이터셋 없이도 dummy batch로 코드를 실행할 수 있게 둡니다.
        return torch.rand(batch_size, 3, imgsz, imgsz, device=device)

    chosen = random.choices(image_paths, k=batch_size)
    images = [load_image_tensor(path, imgsz) for path in chosen]
    return torch.stack(images, dim=0).to(device)


def list_image_paths(calib_dir: str) -> List[Path]:
    if not calib_dir:
        return []
    root = Path(calib_dir)
    if not root.exists():
        print(f"[warn] calibration directory does not exist: {root}. Falling back to random tensors.")
        return []
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [path for path in root.rglob("*") if path.suffix.lower() in suffixes]


def load_image_tensor(path: Path, imgsz: int) -> torch.Tensor:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    image = image.resize((imgsz, imgsz), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_plan(path: Path, targets, actions: List[float], reward: float, args: argparse.Namespace) -> None:
    payload = {
        "reward": reward,
        "max_prune_ratio": args.max_prune_ratio,
        "imgsz": args.imgsz,
        "targets": [
            {
                **target.to_dict(),
                "prune_ratio": float(action),
            }
            for target, action in zip(targets, actions)
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_pruned_student(
    weights: str,
    device: torch.device,
    targets,
    actions: List[float],
    args: argparse.Namespace,
    save_dir: Path,
) -> None:
    if not actions:
        return
    student = load_yolo_model(weights, device)

    # recovery를 할 때는 pruning mask를 유지한 상태로 학습하고, 끝난 뒤 weight에 반영합니다.
    apply_pruning_plan(
        student,
        targets,
        actions,
        max_prune_ratio=args.max_prune_ratio,
        make_permanent=args.recovery_steps <= 0,
    )

    if args.recovery_steps > 0:
        run_recovery_kd(student, weights, device, args)
        remove_pruning_reparam(student, targets)

    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "targets": [target.to_dict() for target in targets],
            "actions": [float(action) for action in actions],
            "note": "Mask-based structured pruning is baked into the weights; architecture channel counts are unchanged.",
        },
        save_dir / "best_student_pruned_state.pt",
    )


def run_recovery_kd(student: torch.nn.Module, weights: str, device: torch.device, args: argparse.Namespace) -> None:
    teacher = load_yolo_model(weights, device)
    for param in student.parameters():
        param.requires_grad = True

    # Label 없이 teacher feature를 따라가도록 하는 간단한 KD fine-tuning 단계입니다.
    distiller = DistillationManager(
        teacher_model=teacher,
        student_model=student,
        layer_pairs=DEFAULT_KD_LAYERS,
    ).to(device)
    optimizer = torch.optim.Adam((param for param in student.parameters() if param.requires_grad), lr=args.recovery_lr)
    student.eval()

    for step in range(1, args.recovery_steps + 1):
        batch = make_calibration_batch(args.batch_size, args.imgsz, device, args.calib_dir)
        optimizer.zero_grad(set_to_none=True)
        distiller(batch)
        loss = distiller.calculate_kd_loss()
        loss.backward()
        optimizer.step()
        print(f"[recovery {step:03d}/{args.recovery_steps:03d}] kd_loss={loss.item():.6f}")

    distiller.close()


if __name__ == "__main__":
    main()
