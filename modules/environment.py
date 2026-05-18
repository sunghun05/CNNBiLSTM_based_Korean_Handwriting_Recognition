from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from modules.distillation import DistillationManager, get_module_by_name


@dataclass
class PruningTarget:
    name: str
    index: int
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    input_h: int
    input_w: int
    output_h: int
    output_w: int
    params: int
    flops: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def conv2d_flops(module: nn.Conv2d, output_h: int, output_w: int) -> float:
    kernel_h, kernel_w = module.kernel_size
    channels_per_group = module.in_channels / module.groups
    return float(output_h * output_w * module.out_channels * channels_per_group * kernel_h * kernel_w)


def build_pruning_targets(
    model: nn.Module,
    sample_batch: torch.Tensor,
    max_targets: int = 12,
    exclude_pose_head: bool = True,
) -> List[PruningTarget]:
    """Pruning 후보 Conv2d 레이어와 state 계산에 필요한 메타데이터를 모읍니다.

    pose head는 bbox/keypoint 출력 의미가 직접 들어있는 부분이라 기본적으로 제외합니다.
    """

    modules = dict(model.named_modules())
    conv_names = [
        name
        for name, module in modules.items()
        if isinstance(module, nn.Conv2d) and _is_prunable_conv_name(name, exclude_pose_head)
    ]

    shapes: Dict[str, Dict[str, Sequence[int]]] = {}
    handles = []

    # 실제 forward를 한 번 흘려서 각 Conv의 입력/출력 feature map 크기를 기록합니다.
    def save_input_shape(name: str):
        def hook(_module, inputs):
            if inputs and isinstance(inputs[0], torch.Tensor):
                shapes.setdefault(name, {})["input"] = tuple(inputs[0].shape)

        return hook

    def save_output_shape(name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                shapes.setdefault(name, {})["output"] = tuple(output.shape)

        return hook

    for name in conv_names:
        module = modules[name]
        handles.append(module.register_forward_pre_hook(save_input_shape(name)))
        handles.append(module.register_forward_hook(save_output_shape(name)))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(sample_batch)
    model.train(was_training)

    for handle in handles:
        handle.remove()

    targets = []
    for index, name in enumerate(conv_names):
        module = modules[name]
        input_shape = shapes.get(name, {}).get("input")
        output_shape = shapes.get(name, {}).get("output")
        if input_shape is None or output_shape is None:
            continue

        params = int(module.weight.numel())
        if module.bias is not None:
            params += int(module.bias.numel())

        targets.append(
            PruningTarget(
                name=name,
                index=index,
                in_channels=int(module.in_channels),
                out_channels=int(module.out_channels),
                kernel_size=int(module.kernel_size[0]),
                stride=int(module.stride[0]),
                input_h=int(input_shape[-2]),
                input_w=int(input_shape[-1]),
                output_h=int(output_shape[-2]),
                output_w=int(output_shape[-1]),
                params=params,
                flops=conv2d_flops(module, int(output_shape[-2]), int(output_shape[-1])),
            )
        )

    if max_targets > 0:
        targets = _select_representative_targets(targets, max_targets)

    for index, target in enumerate(targets):
        target.index = index
    return targets


def apply_pruning_plan(
    model: nn.Module,
    targets: Iterable[PruningTarget],
    actions: Sequence[float],
    max_prune_ratio: float,
    make_permanent: bool = True,
) -> nn.Module:
    # DDPG가 찾은 layer별 pruning ratio를 실제 모델에 다시 적용합니다.
    for target, action in zip(targets, actions):
        layer = get_module_by_name(model, target.name)
        amount = float(np.clip(action, 0.0, max_prune_ratio))
        if amount <= 0.0:
            continue
        prune.ln_structured(layer, name="weight", amount=amount, n=1, dim=0)
        if make_permanent:
            prune.remove(layer, "weight")
    return model


def remove_pruning_reparam(model: nn.Module, targets: Iterable[PruningTarget]) -> nn.Module:
    for target in targets:
        layer = get_module_by_name(model, target.name)
        if hasattr(layer, "weight_orig"):
            prune.remove(layer, "weight")
    return model


class PruningEnv:
    """DDPG가 pruning ratio를 연속 action으로 탐색할 수 있게 만든 간단한 환경입니다."""

    state_dim = 11

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model_base: nn.Module,
        prune_targets: Sequence[PruningTarget],
        distiller: DistillationManager,
        calibration_batch: torch.Tensor,
        device: torch.device,
        max_prune_ratio: float = 0.6,
        kd_weight: float = 1.0,
        compression_weight: float = 2.0,
    ) -> None:
        if not prune_targets:
            raise ValueError("At least one pruning target is required.")

        self.teacher = teacher_model
        self.student_base = student_model_base
        self.prune_targets = list(prune_targets)
        self.distiller = distiller
        self.calibration_batch = calibration_batch.to(device)
        self.device = device
        self.max_prune_ratio = float(max_prune_ratio)
        self.kd_weight = float(kd_weight)
        self.compression_weight = float(compression_weight)

        # state 정규화를 위해 전체 후보 레이어의 최대값/합계를 미리 계산합니다.
        self.total_steps = len(self.prune_targets)
        self.total_flops = max(sum(target.flops for target in self.prune_targets), 1.0)
        self.max_out_channels = max(target.out_channels for target in self.prune_targets)
        self.max_in_channels = max(target.in_channels for target in self.prune_targets)
        self.max_hw = max(max(target.input_h, target.input_w) for target in self.prune_targets)
        self.max_stride = max(target.stride for target in self.prune_targets)
        self.max_kernel = max(target.kernel_size for target in self.prune_targets)

        self.current_step = 0
        self.last_action = 0.0
        self.reduced_flops = 0.0
        self.action_history: List[float] = []
        self.student: nn.Module | None = None

    def reset(self) -> np.ndarray:
        # 한 episode 안에서는 pruning이 누적되지만, 다음 episode는 원본 student에서 다시 시작합니다.
        self.student = copy.deepcopy(self.student_base).to(self.device)
        self.student.eval()
        self.distiller.set_student_model(self.student)

        self.current_step = 0
        self.last_action = 0.0
        self.reduced_flops = 0.0
        self.action_history = []
        return self._get_state()

    def step(self, action) -> tuple[np.ndarray, float, bool, Dict[str, float]]:
        if self.student is None:
            raise RuntimeError("Call reset() before step().")
        if self.current_step >= self.total_steps:
            raise RuntimeError("Episode is already finished. Call reset() to start a new one.")

        action_value = float(np.asarray(action).reshape(-1)[0])
        action_value = float(np.clip(action_value, 0.0, self.max_prune_ratio))

        target = self.prune_targets[self.current_step]
        target_layer = get_module_by_name(self.student, target.name)

        # L1 norm이 작은 출력 채널부터 0으로 만드는 structured pruning입니다.
        if action_value > 0.0:
            prune.ln_structured(target_layer, name="weight", amount=action_value, n=1, dim=0)

        self.reduced_flops += target.flops * action_value
        self.action_history.append(action_value)
        kd_loss = self._measure_kd_loss()
        reduced_ratio = self.reduced_flops / self.total_flops

        # reward는 "많이 줄이면 가산점, teacher feature와 멀어지면 감점" 구조입니다.
        reward = self.compression_weight * reduced_ratio - self.kd_weight * math.log1p(kd_loss)

        self.last_action = action_value
        self.current_step += 1
        done = self.current_step >= self.total_steps

        info = {
            "layer": target.name,
            "action": action_value,
            "kd_loss": kd_loss,
            "reduced_flops_ratio": reduced_ratio,
            "reward": reward,
        }
        return self._get_state(), float(reward), done, info

    def _measure_kd_loss(self) -> float:
        assert self.student is not None
        self.teacher.eval()
        self.student.eval()
        with torch.no_grad():
            self.distiller(self.calibration_batch)
            loss = self.distiller.calculate_kd_loss()
        return float(loss.detach().cpu().item())

    def _get_state(self) -> np.ndarray:
        if self.current_step >= self.total_steps:
            return np.zeros(self.state_dim, dtype=np.float32)

        target = self.prune_targets[self.current_step]
        rest_flops = sum(t.flops for t in self.prune_targets[self.current_step + 1 :])
        denom_step = max(self.total_steps - 1, 1)

        # 문서의 state 개념을 0~1 범위로 정규화한 11차원 벡터입니다.
        return np.array(
            [
                target.index / denom_step,
                target.out_channels / self.max_out_channels,
                target.in_channels / self.max_in_channels,
                target.input_h / self.max_hw,
                target.input_w / self.max_hw,
                target.stride / self.max_stride,
                target.kernel_size / self.max_kernel,
                target.flops / self.total_flops,
                self.reduced_flops / self.total_flops,
                rest_flops / self.total_flops,
                self.last_action / max(self.max_prune_ratio, 1e-6),
            ],
            dtype=np.float32,
        )


def _is_prunable_conv_name(name: str, exclude_pose_head: bool) -> bool:
    if not name.startswith("model."):
        return False
    parts = name.split(".")
    if len(parts) < 3:
        return False
    try:
        top_index = int(parts[1])
    except ValueError:
        return False
    if exclude_pose_head and top_index >= 22:
        return False
    return True


def _select_representative_targets(targets: Sequence[PruningTarget], max_targets: int) -> List[PruningTarget]:
    if len(targets) <= max_targets:
        return list(targets)

    ordered = sorted(targets, key=lambda target: target.flops, reverse=True)
    selected = sorted(ordered[:max_targets], key=lambda target: target.index)
    return selected
