from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"Layer '{name}' was not found in the model.")
    return modules[name]


class DistillationManager(nn.Module):
    """Teacher/Student의 중간 feature map을 비교하는 KD helper입니다.

    YOLO pose의 최종 출력은 bbox/keypoint가 섞여 복잡하므로, 여기서는 neck feature를
    hook으로 잡아 MSE loss를 계산합니다.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        layer_pairs: Mapping[str, str],
        adapter_config: Optional[Mapping[str, Tuple[int, int]]] = None,
        feature_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.layer_pairs = dict(layer_pairs)
        self.feature_weights = dict(feature_weights or {})
        self.teacher_outputs: Dict[str, torch.Tensor] = {}
        self.student_outputs: Dict[str, torch.Tensor] = {}
        self._teacher_hook_handles = []
        self._student_hook_handles = []

        # Teacher는 기준 모델이므로 gradient를 끄고 eval 상태로 고정합니다.
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # 구조가 다른 student를 쓰는 실험을 위해 channel adapter를 선택적으로 둡니다.
        self.adapters = nn.ModuleDict()
        for student_layer, (student_channels, teacher_channels) in (adapter_config or {}).items():
            if student_channels == teacher_channels:
                continue
            self.adapters[self._safe_key(student_layer)] = nn.Conv2d(
                student_channels,
                teacher_channels,
                kernel_size=1,
                bias=False,
            )

        self._register_teacher_hooks()
        self._register_student_hooks()

    @staticmethod
    def _safe_key(layer_name: str) -> str:
        return layer_name.replace(".", "__")

    def set_student_model(self, student_model: nn.Module) -> None:
        # episode마다 student가 deepcopy되므로 기존 hook을 제거하고 새 모델에 다시 연결합니다.
        self._remove_hooks(self._student_hook_handles)
        self.student_outputs.clear()
        self.student = student_model
        self._register_student_hooks()

    def close(self) -> None:
        self._remove_hooks(self._teacher_hook_handles)
        self._remove_hooks(self._student_hook_handles)

    def forward(self, x: torch.Tensor):
        self.teacher_outputs.clear()
        self.student_outputs.clear()

        # forward hook이 실행되면서 선택한 layer의 feature map이 dict에 저장됩니다.
        student_output = self.student(x)
        with torch.no_grad():
            teacher_output = self.teacher(x)
        return student_output, teacher_output

    def calculate_kd_loss(self) -> torch.Tensor:
        losses = []
        for teacher_layer, student_layer in self.layer_pairs.items():
            if teacher_layer not in self.teacher_outputs:
                raise RuntimeError(f"Teacher hook '{teacher_layer}' did not capture an output.")
            if student_layer not in self.student_outputs:
                raise RuntimeError(f"Student hook '{student_layer}' did not capture an output.")

            teacher_feature = self._as_tensor(self.teacher_outputs[teacher_layer]).detach()
            student_feature = self._as_tensor(self.student_outputs[student_layer])
            student_feature = self._match_shape(student_feature, teacher_feature, student_layer)

            # 중간 feature는 확률분포가 아니므로 KL 대신 MSE를 사용합니다.
            weight = self.feature_weights.get(student_layer, 1.0)
            losses.append(weight * F.mse_loss(student_feature, teacher_feature))

        if not losses:
            device = next(self.student.parameters()).device
            return torch.zeros((), device=device)
        return torch.stack(losses).sum()

    def _register_teacher_hooks(self) -> None:
        self._remove_hooks(self._teacher_hook_handles)
        for teacher_layer in self.layer_pairs:
            module = get_module_by_name(self.teacher, teacher_layer)
            handle = module.register_forward_hook(self._capture(teacher_layer, self.teacher_outputs))
            self._teacher_hook_handles.append(handle)

    def _register_student_hooks(self) -> None:
        self._remove_hooks(self._student_hook_handles)
        for student_layer in self.layer_pairs.values():
            module = get_module_by_name(self.student, student_layer)
            handle = module.register_forward_hook(self._capture(student_layer, self.student_outputs))
            self._student_hook_handles.append(handle)

    @staticmethod
    def _remove_hooks(handles: Iterable[torch.utils.hooks.RemovableHandle]) -> None:
        for handle in handles:
            handle.remove()
        if hasattr(handles, "clear"):
            handles.clear()

    @staticmethod
    def _capture(name: str, storage: Dict[str, torch.Tensor]):
        def hook(_module, _inputs, output):
            storage[name] = output

        return hook

    @staticmethod
    def _as_tensor(output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (list, tuple)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    return item
        raise TypeError(f"Expected a tensor-like hook output, got {type(output)!r}.")

    def _match_shape(
        self,
        student_feature: torch.Tensor,
        teacher_feature: torch.Tensor,
        student_layer: str,
    ) -> torch.Tensor:
        adapter_key = self._safe_key(student_layer)
        if adapter_key in self.adapters:
            student_feature = self.adapters[adapter_key](student_feature)

        if student_feature.shape[-2:] != teacher_feature.shape[-2:]:
            student_feature = F.interpolate(
                student_feature,
                size=teacher_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if student_feature.shape[1] != teacher_feature.shape[1]:
            raise RuntimeError(
                f"Channel mismatch at '{student_layer}': "
                f"student={student_feature.shape[1]}, teacher={teacher_feature.shape[1]}. "
                "Pass adapter_config if the student architecture has fewer channels."
            )

        return student_feature
