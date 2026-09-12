"""데이터 다운로드 없이 원본 모델의 최소 동작을 확인한다.

원본 연구 파일을 수정하거나 전체 학습을 실행하지 않는다. 모든 Python 파일의 문법을 검사하고,
2019년 모델 정의를 직접 불러와 작은 무작위 tensor로 forward pass를 수행한다.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import torch
from torchvision.utils import save_image


ROOT = Path(__file__).resolve().parents[1]
EXP_2019 = ROOT / "experiments" / "2019_baselines" / "original"
EXP_2023 = ROOT / "experiments" / "2023_model_exploration" / "original"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_python_syntax() -> int:
    files = sorted(EXP_2019.rglob("*.py")) + sorted(EXP_2023.rglob("*.py"))
    for path in files:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    print(f"[통과] 원본 Python 파일 문법 검사: {len(files)}개")
    return len(files)


def check_vanilla_models() -> None:
    models = load_module("legacy_2019_models", EXP_2019 / "model.py")
    with torch.no_grad():
        mnist_images = models.Generator()(torch.randn(4, 100))
        mnist_scores = models.Discriminator()(mnist_images)
        cifar_images = models.cifar_Generator()(torch.randn(4, 100))
        cifar_scores = models.cifar_Discriminator()(cifar_images)

    assert mnist_images.shape == (4, 1, 28, 28)
    assert mnist_scores.shape == (4, 1)
    assert cifar_images.shape == (4, 3, 32, 32)
    assert cifar_scores.shape == (4, 1)
    output_dir = ROOT / "results" / "toy_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(mnist_images, output_dir / "untrained_mnist_generator.png", normalize=True)
    print("[통과] 2019 Vanilla GAN: MNIST/CIFAR-10 forward pass")
    print("[생성] results/toy_samples/untrained_mnist_generator.png")


def check_acgan_models() -> None:
    models = load_module("legacy_2019_acgan_models", EXP_2019 / "AC_GAN" / "model.py")
    generator = models.netG(nz=100, ngf=8, nc=1).eval()
    discriminator = models.netD(ndf=8, nc=1, nb_label=10).eval()
    with torch.no_grad():
        images = generator(torch.randn(4, 100, 1, 1))
        source_scores, class_scores = discriminator(images)

    assert images.shape == (4, 1, 64, 64)
    assert source_scores.shape == (4,)
    assert class_scores.shape == (4, 10)
    print("[통과] 2019 ACGAN: Generator/Discriminator forward pass")


def main() -> None:
    torch.manual_seed(42)
    check_python_syntax()
    check_vanilla_models()
    check_acgan_models()
    print("[완료] 데이터 없는 toy smoke test를 모두 통과했습니다.")


if __name__ == "__main__":
    main()
