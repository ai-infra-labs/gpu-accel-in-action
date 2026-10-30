"""使用固定的MONAI脾脏bundle执行单病例推理，不接收参考标签。"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import monai
import torch
from monai.bundle import run


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")  # 下载好的0.6.0 bundle目录。
    parser.add_argument("image")  # 只输入CT，参考标签不参与推理。
    parser.add_argument("--out", default="case_output")
    parser.add_argument("--device", choices=["cuda:0", "cpu"], default="cuda:0")
    args = parser.parse_args()
    bundle = Path(args.bundle).resolve()
    image = Path(args.image).resolve()
    output = Path(args.out).resolve()
    source = bundle / "configs/inference.json"
    weights = bundle / "models/model.pt"
    for path in (image, source, weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("指定了CUDA设备，但PyTorch未检测到可用CUDA。")

    cfg = json.loads(source.read_text(encoding="utf-8"))
    transforms = cfg["postprocessing"]["transforms"]
    expected = ["Activationsd", "Invertd", "AsDiscreted", "SaveImaged"]
    if [item["_target_"] for item in transforms] != expected:
        raise ValueError("后处理结构与本文核对的0.6.0配置不符，请先检查版本。")
    # 保留官方网络、空间预处理、融合及逆变换顺序，只调整单病例执行和保存设置。
    cfg.update(bundle_root=str(bundle), datalist=[str(image)], output_dir=str(output),
               device=f"$torch.device('{args.device}')", load_pretrain=True,
               output_postfix="pred", output_dtype="$numpy.uint8", separate_folder=False)
    cfg["dataloader"]["num_workers"] = 0  # 单病例先减少进程配置的干扰。
    cfg["inferer"]["sw_batch_size"] = 1  # 每次仅送一个窗口，不改变窗口尺寸和重叠率。
    cfg["evaluator"]["amp"] = False  # 先建立不启用AMP的基线。
    transforms[-1]["resample"] = False  # Invertd已回原网格，保存时不再插值标签。
    output.mkdir(parents=True, exist_ok=True)
    case_config = output / "case_config.json"
    case_config.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    manifest = {"image": str(image), "weights_sha256": sha256(weights),
                "source_config_sha256": sha256(source), "torch": torch.__version__,
                "monai": monai.__version__, "device": args.device,
                "cuda_runtime": torch.version.cuda, "amp": False}

    # 禁用TF32，明确这一参考配置的计算设置；不宣称所有后端逐位一致。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    use_cuda = args.device.startswith("cuda")
    if use_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    run(config_file=str(case_config))  # 官方流程负责构建网络、加载权重、推理与保存。
    if use_cuda:
        torch.cuda.synchronize()
    manifest["first_workflow_seconds"] = time.perf_counter() - start
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("流程结束。请检查输出NIfTI、空间信息、叠加图和参考标签指标。")


if __name__ == "__main__":
    main()
