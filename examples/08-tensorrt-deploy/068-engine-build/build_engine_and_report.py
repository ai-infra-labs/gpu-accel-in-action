"""TensorRT 没改模型，为什么还能更快？（第8章 · TensorRT 部署）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：TensorRT。
"""
from pathlib import Path
import time

import tensorrt as trt


def main():
    print("TensorRT:", trt.__version__)
    if trt.__version__.split(".")[0] != "11":
        raise RuntimeError("本例面向 TensorRT 11.x，请先核对版本。")

    onnx_path = Path("onnx_demo/small_classifier.onnx")
    engine_path = onnx_path.with_suffix(".engine")
    info_path = onnx_path.with_suffix(".engine.json")

    # 详细日志用于观察解析、优化与构建过程。
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX 解析失败：\n" + "\n".join(errors))

    # 本篇只处理固定形状；动态输入需要额外的 optimization profile。
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        if any(dim < 0 for dim in tensor.shape):
            raise ValueError("发现动态输入，本篇脚本只演示固定形状。")
        print("输入:", tensor.name, tuple(tensor.shape), tensor.dtype)

    config = builder.create_builder_config()
    # 1 GiB 是教学配置，不是推荐值，也不是程序的显存总上限。
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    # 构建时记录详细信息，供后面的 Engine Inspector 查看。
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    start = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - start
    if plan is None:
        raise RuntimeError("引擎构建失败，请检查完整 TensorRT 日志。")

    engine_bytes = bytes(plan)
    engine_path.write_bytes(engine_bytes)

    # 加载刚刚在本机构建的引擎；这一步尚未执行模型推理。
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("引擎反序列化失败。")

    inspector = engine.create_engine_inspector()
    info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    info_path.write_text(info, encoding="utf-8")

    print(f"构建耗时: {build_seconds:.3f} 秒（不是推理耗时）")
    print("引擎文件:", engine_path)
    print("引擎信息:", info_path)
    print("完成构建与信息导出；输出正确性和性能仍待验证。")


if __name__ == "__main__":
    main()
