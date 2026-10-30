"""一层矩阵太大，怎样让多张 GPU 合力算？（第5章 · 并行与分布式）。

本文件是对应博文代码块的实现/演示，直接运行可见效果；运行依赖：仅标准库。
"""
def linear(x, weight):
    # weight 按 [输入特征, 输出特征] 存放；先校验维度，避免 zip 静默截断。
    assert weight and weight[0], "权重不能为空"
    assert len(x) == len(weight), "输入特征数必须等于权重行数"
    assert all(len(row) == len(weight[0]) for row in weight), "权重各行列数必须一致"
    return [sum(a * b for a, b in zip(x, column))
            for column in zip(*weight)]


def add(a, b):
    assert len(a) == len(b), "相加的两份张量长度必须一致"
    return [x + y for x, y in zip(a, b)]


def relu(values):
    return [max(0, value) for value in values]


def main():
    x = [1, 2, 3, 4]
    wa = [[1, 0, 2, 0], [0, 1, 0, 2],
          [1, 1, 0, 0], [0, 0, 1, 1]]
    ba = [0, -6, 1, -10]
    wb = [[1, 0], [0, 1], [1, 1], [-1, 2]]
    bb = [1, -1]

    full = linear(x, wa)
    columns = [linear(x, [row[:2] for row in wa]),
               linear(x, [row[2:] for row in wa])]
    rows = [linear(x[:2], wa[:2]), linear(x[2:], wa[2:])]
    assert columns[0] + columns[1] == full
    assert add(rows[0], rows[1]) == full

    reference = add(linear(relu(add(full, ba)), wb), bb)
    h0 = relu(add(columns[0], ba[:2]))
    h1 = relu(add(columns[1], ba[2:]))
    p0, p1 = linear(h0, wb[:2]), linear(h1, wb[2:])
    tp_result = add(add(p0, p1), bb)
    assert tp_result == reference
    assert relu(add([-2], [3])) != add(relu([-2]), relu([3]))

    # B=2 对照：确认批次维与特征维没有混淆——每个样本独立过同一组权重。
    x2 = [[1, 2, 3, 4], [0, 1, 0, 1]]
    batch_ref = [add(linear(relu(add(linear(row, wa), ba)), wb), bb)
                 for row in x2]
    batch_tp = []
    for row in x2:
        c0, c1 = linear(row, [r[:2] for r in wa]), linear(row, [r[2:] for r in wa])
        q0, q1 = linear(relu(add(c0, ba[:2])), wb[:2]), linear(relu(add(c1, ba[2:])), wb[2:])
        batch_tp.append(add(add(q0, q1), bb))
    assert batch_tp == batch_ref

    print("单层完整结果:", full)
    print("按输出切，本地结果:", columns)
    print("按输入切，本地部分和:", rows)
    print("第一层激活分片:", [h0, h1])
    print("第二层部分和:", [p0, p1])
    print("完整两层结果:", reference)
    print("TP 两层结果:", tp_result)
    print("B=2 批次结果:", batch_tp)


if __name__ == "__main__":
    main()
