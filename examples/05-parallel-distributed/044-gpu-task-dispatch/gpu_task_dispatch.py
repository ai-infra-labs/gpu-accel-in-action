"""多加一张显卡，任务该交给谁？（第5章 · 并行与分布式）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：仅标准库。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    name: str  # 实例标识：模型版本/设备/实例编号。
    models: frozenset[str]
    queued: int
    available_ms: float
    input_ready_ms: float
    compute_ms: float
    ready: bool = True  # 简化表示健康、配置匹配与准入预算满足。

    def finish_ms(self):
        return max(self.available_ms, self.input_ready_ms) + self.compute_ms


def choose(candidates, model, policy):
    eligible = [c for c in candidates if c.ready and model in c.models]
    if not eligible:
        return None
    if policy == "shortest_queue":
        return min(eligible, key=lambda c: (c.queued, c.name))
    if policy == "earliest_finish":
        return min(eligible, key=lambda c: (c.finish_ms(), c.name))
    raise ValueError(policy)


def main():
    model = "identity@v1"  # 模型版本属于候选匹配条件。
    loaded = frozenset({model})
    # 候选用实例标识而非裸 GPU 名，强调路由对象是可执行实例。
    g0 = "identity@v1/GPU0/instance0"
    g1 = "identity@v1/GPU1/instance0"
    cases = {
        "空闲但没加载目标模型": [
            Candidate(g0, loaded, 1, 12, 0, 8),
            Candidate(g1, frozenset(), 0, 0, 0, 6),
        ],
        "空闲但数据准备更慢": [
            Candidate(g0, loaded, 1, 12, 0, 8),
            Candidate(g1, loaded, 0, 0, 25, 6),
        ],
        "队伍短但前面的工作更重": [
            Candidate(g0, loaded, 1, 80, 0, 8),
            Candidate(g1, loaded, 4, 24, 0, 8),
        ],
    }
    for title, candidates in cases.items():
        print(f"\n{title}")
        for policy in ["shortest_queue", "earliest_finish"]:
            selected = choose(candidates, model, policy)
            if selected is None:
                print(policy, "无可执行实例")
            else:
                print(f"{policy}: {selected.name}, "
                      f"预计完成 {selected.finish_ms():.0f} ms")


if __name__ == "__main__":
    main()
