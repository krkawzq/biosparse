"""
Numba 稀疏矩阵使用示例
演示在 JIT 编译代码中使用 SCL 稀疏矩阵的各种场景
"""

import sys
sys.path.insert(0, 'src')

from python._binding._sparse import CSRF64
import scipy.sparse as sp
import numpy as np
from numba import njit
import time


# ============================================================
# 示例 1: 图算法 - PageRank
# ============================================================

@njit
def pagerank_iteration(adjacency, ranks, damping=0.85):
    """执行一次 PageRank 迭代 (纯 JIT 编译)

    Args:
        adjacency: CSR 邻接矩阵 (每列已归一化)
        ranks: 当前的 rank 值
        damping: 阻尼因子

    Returns:
        新的 rank 值
    """
    n = adjacency.nrows
    new_ranks = np.zeros(n, dtype=np.float64)

    # 使用迭代器访问每个节点的出链
    for row_idx, (values, indices) in enumerate(adjacency):
        # row_idx 是当前节点
        # values 是边权重，indices 是指向的节点
        for i in range(len(indices)):
            target_node = indices[i]
            weight = values[i]
            new_ranks[target_node] += ranks[row_idx] * weight

    # 应用阻尼因子
    teleport = (1.0 - damping) / n
    for i in range(n):
        new_ranks[i] = damping * new_ranks[i] + teleport

    return new_ranks


def demo_pagerank():
    """演示 PageRank 算法"""
    print("\n" + "="*60)
    print("示例 1: PageRank 图算法")
    print("="*60)

    # 创建一个简单的图 (5个节点)
    # 邻接矩阵：每行代表一个节点，非零值表示指向其他节点
    n = 5
    adjacency = sp.csr_matrix([
        [0, 1, 1, 0, 0],  # 节点 0 -> 1, 2
        [0, 0, 1, 1, 0],  # 节点 1 -> 2, 3
        [1, 0, 0, 1, 1],  # 节点 2 -> 0, 3, 4
        [0, 0, 0, 0, 1],  # 节点 3 -> 4
        [1, 1, 0, 0, 0],  # 节点 4 -> 0, 1
    ], dtype=np.float64)

    # 列归一化（每个节点的出链权重和为1）
    adjacency = adjacency.tocsc()
    for j in range(n):
        col_sum = adjacency[:, j].sum()
        if col_sum > 0:
            adjacency[:, j] /= col_sum
    adjacency = adjacency.tocsr()

    csr = CSRF64.from_scipy(adjacency)
    ranks = np.ones(n, dtype=np.float64) / n

    print(f"图: {n} 个节点")
    print(f"初始 ranks: {ranks}")

    # 执行 PageRank 迭代
    for iteration in range(20):
        ranks = pagerank_iteration(csr, ranks)

    print(f"收敛后 ranks: {ranks}")
    print(f"Ranks 总和: {ranks.sum():.6f} (应该接近 1.0)")
    print("✓ PageRank 完成")


# ============================================================
# 示例 2: 稀疏矩阵向量乘法 (SpMV)
# ============================================================

@njit
def sparse_matrix_vector_multiply(csr, vec):
    """稀疏矩阵 × 向量 (CSR 格式)

    Args:
        csr: CSR 稀疏矩阵 (m × n)
        vec: 向量 (n,)

    Returns:
        结果向量 (m,)
    """
    result = np.zeros(csr.nrows, dtype=np.float64)

    for row_idx in range(csr.nrows):
        values, indices = csr.row_to_numpy(row_idx)
        dot_product = 0.0
        for i in range(len(values)):
            dot_product += values[i] * vec[indices[i]]
        result[row_idx] = dot_product

    return result


def demo_spmv():
    """演示稀疏矩阵向量乘法"""
    print("\n" + "="*60)
    print("示例 2: 稀疏矩阵向量乘法 (SpMV)")
    print("="*60)

    # 创建一个大型稀疏矩阵
    m, n = 5000, 4000
    density = 0.01
    mat = sp.random(m, n, density=density, format='csr', dtype=np.float64)
    vec = np.random.rand(n)

    csr = CSRF64.from_scipy(mat)

    print(f"矩阵: {m} × {n}, 密度: {density:.2%}")
    print(f"非零元素: {mat.nnz:,}")

    # Python 版本
    t0 = time.perf_counter()
    result_scipy = mat @ vec
    t_scipy = (time.perf_counter() - t0) * 1000

    # JIT 版本 (首次编译)
    _ = sparse_matrix_vector_multiply(csr, vec)

    # JIT 版本 (已编译)
    t0 = time.perf_counter()
    result_jit = sparse_matrix_vector_multiply(csr, vec)
    t_jit = (time.perf_counter() - t0) * 1000

    # 验证结果
    diff = np.abs(result_scipy - result_jit).max()
    print(f"\nSciPy 时间: {t_scipy:.2f} ms")
    print(f"JIT 时间:   {t_jit:.2f} ms")
    print(f"加速:       {t_scipy/t_jit:.1f}x")
    print(f"最大误差:   {diff:.2e}")
    print("✓ SpMV 完成")


# ============================================================
# 示例 3: 协同过滤 - 用户-物品推荐
# ============================================================

@njit
def compute_user_similarity(user_item_matrix, user1, user2):
    """计算两个用户之间的余弦相似度

    Args:
        user_item_matrix: 用户-物品评分矩阵 (CSR)
        user1, user2: 用户索引

    Returns:
        余弦相似度 [-1, 1]
    """
    values1, indices1 = user_item_matrix.row_to_numpy(user1)
    values2, indices2 = user_item_matrix.row_to_numpy(user2)

    # 找到共同评分的物品
    dot_product = 0.0
    norm1 = 0.0
    norm2 = 0.0

    # 使用双指针法找到共同物品
    i, j = 0, 0
    while i < len(indices1) and j < len(indices2):
        if indices1[i] == indices2[j]:
            dot_product += values1[i] * values2[j]
            norm1 += values1[i] * values1[i]
            norm2 += values2[j] * values2[j]
            i += 1
            j += 1
        elif indices1[i] < indices2[j]:
            norm1 += values1[i] * values1[i]
            i += 1
        else:
            norm2 += values2[j] * values2[j]
            j += 1

    # 处理剩余元素
    while i < len(indices1):
        norm1 += values1[i] * values1[i]
        i += 1
    while j < len(indices2):
        norm2 += values2[j] * values2[j]
        j += 1

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot_product / (np.sqrt(norm1) * np.sqrt(norm2))


@njit
def find_top_k_similar_users(user_item_matrix, target_user, k=5):
    """找到与目标用户最相似的 k 个用户

    Args:
        user_item_matrix: 用户-物品评分矩阵
        target_user: 目标用户索引
        k: 返回的相似用户数量

    Returns:
        (用户索引, 相似度分数) 的数组
    """
    n_users = user_item_matrix.nrows
    similarities = np.zeros(n_users, dtype=np.float64)

    # 计算与所有其他用户的相似度
    for user in range(n_users):
        if user != target_user:
            similarities[user] = compute_user_similarity(
                user_item_matrix, target_user, user
            )

    # 找到前 k 个最相似的用户
    # 简单的选择排序（对于小 k 值效率可接受）
    top_k_users = np.zeros(k, dtype=np.int64)
    top_k_scores = np.zeros(k, dtype=np.float64)

    for i in range(k):
        max_idx = -1
        max_sim = -2.0  # 相似度范围是 [-1, 1]
        for j in range(n_users):
            # 跳过已选择的用户
            already_selected = False
            for m in range(i):
                if top_k_users[m] == j:
                    already_selected = True
                    break

            if not already_selected and similarities[j] > max_sim:
                max_sim = similarities[j]
                max_idx = j

        if max_idx >= 0:
            top_k_users[i] = max_idx
            top_k_scores[i] = max_sim

    return top_k_users, top_k_scores


def demo_collaborative_filtering():
    """演示协同过滤推荐"""
    print("\n" + "="*60)
    print("示例 3: 协同过滤 - 用户相似度计算")
    print("="*60)

    # 创建用户-物品评分矩阵
    # 行=用户，列=物品，值=评分
    n_users, n_items = 100, 500
    density = 0.05  # 每个用户平均评分 25 个物品

    ratings = sp.random(n_users, n_items, density=density, format='csr', dtype=np.float64)
    ratings.data = np.random.randint(1, 6, size=ratings.data.shape).astype(np.float64)  # 1-5星评分

    csr = CSRF64.from_scipy(ratings)

    print(f"用户: {n_users}, 物品: {n_items}")
    print(f"评分总数: {ratings.nnz:,}")
    print(f"稀疏度: {(1 - ratings.nnz / (n_users * n_items)) * 100:.1f}%")

    # 找到与用户 0 最相似的 5 个用户
    target_user = 0
    k = 5

    t0 = time.perf_counter()
    similar_users, scores = find_top_k_similar_users(csr, target_user, k)
    t_elapsed = (time.perf_counter() - t0) * 1000

    print(f"\n与用户 {target_user} 最相似的 {k} 个用户:")
    for i in range(k):
        print(f"  用户 {similar_users[i]:3d}: 相似度 = {scores[i]:.4f}")

    print(f"\n计算时间: {t_elapsed:.2f} ms")
    print("✓ 协同过滤完成")


# ============================================================
# 示例 4: 结构化数据处理 - 稀疏特征工程
# ============================================================

@njit
def compute_feature_statistics(feature_matrix):
    """计算稀疏特征矩阵的统计信息

    Args:
        feature_matrix: 特征矩阵 (样本 × 特征)

    Returns:
        每个样本的特征统计 (均值, 最大值, 非零数)
    """
    n_samples = feature_matrix.nrows
    stats = np.zeros((n_samples, 3), dtype=np.float64)

    for i in range(n_samples):
        values, indices = feature_matrix.row_to_numpy(i)

        if len(values) > 0:
            stats[i, 0] = np.mean(values)      # 均值
            stats[i, 1] = np.max(values)       # 最大值
            stats[i, 2] = float(len(values))   # 非零特征数
        else:
            stats[i, 0] = 0.0
            stats[i, 1] = 0.0
            stats[i, 2] = 0.0

    return stats


def demo_feature_engineering():
    """演示特征工程"""
    print("\n" + "="*60)
    print("示例 4: 稀疏特征工程")
    print("="*60)

    # 模拟高维稀疏特征（如文本 TF-IDF）
    n_samples = 1000
    n_features = 10000
    density = 0.01  # 每个样本平均 100 个非零特征

    features = sp.random(n_samples, n_features, density=density, format='csr', dtype=np.float64)
    features.data = np.abs(features.data)  # 确保非负

    csr = CSRF64.from_scipy(features)

    print(f"样本数: {n_samples}")
    print(f"特征维度: {n_features}")
    print(f"非零特征: {features.nnz:,}")

    # 计算统计信息
    t0 = time.perf_counter()
    stats = compute_feature_statistics(csr)
    t_elapsed = (time.perf_counter() - t0) * 1000

    print(f"\n特征统计 (前5个样本):")
    print(f"{'样本':<6} {'均值':<10} {'最大值':<10} {'非零数':<10}")
    print("-" * 40)
    for i in range(min(5, n_samples)):
        print(f"{i:<6} {stats[i,0]:<10.4f} {stats[i,1]:<10.4f} {int(stats[i,2]):<10}")

    print(f"\n计算时间: {t_elapsed:.2f} ms")
    print(f"平均每样本: {t_elapsed/n_samples:.4f} ms")
    print("✓ 特征工程完成")


# ============================================================
# 示例 5: 切片和子矩阵操作
# ============================================================

@njit
def analyze_submatrix(csr, row_start, row_end, col_start, col_end):
    """分析子矩阵的属性

    Args:
        csr: 原始矩阵
        row_start, row_end: 行范围
        col_start, col_end: 列范围

    Returns:
        子矩阵的 (nnz, 密度, 最大值)
    """
    # 方法1: 使用切片
    sub = csr[row_start:row_end, col_start:col_end]

    # 统计信息
    nnz = sub.nnz
    density = float(nnz) / (sub.nrows * sub.ncols) if sub.nrows * sub.ncols > 0 else 0.0

    # 找最大值
    max_val = 0.0
    for values, indices in sub:
        if len(values) > 0:
            row_max = np.max(values)
            if row_max > max_val:
                max_val = row_max

    return nnz, density, max_val


def demo_slicing():
    """演示切片操作"""
    print("\n" + "="*60)
    print("示例 5: 子矩阵切片与分析")
    print("="*60)

    # 创建测试矩阵
    m, n = 200, 150
    density = 0.05
    mat = sp.random(m, n, density=density, format='csr', dtype=np.float64)
    mat.data = np.abs(mat.data)

    csr = CSRF64.from_scipy(mat)

    print(f"原始矩阵: {m} × {n}, 密度: {density:.2%}")

    # 分析多个子矩阵
    regions = [
        (0, 50, 0, 50, "左上角"),
        (50, 100, 50, 100, "中心"),
        (150, 200, 100, 150, "右下角"),
    ]

    print("\n子矩阵分析:")
    for row_start, row_end, col_start, col_end, name in regions:
        nnz, density, max_val = analyze_submatrix(
            csr, row_start, row_end, col_start, col_end
        )
        print(f"  {name:10s}: nnz={nnz:4d}, 密度={density:.2%}, 最大值={max_val:.4f}")

    print("\n✓ 切片操作完成")


# ============================================================
# 主程序
# ============================================================

def main():
    print("############################################################")
    print("# SCL-Core Numba 稀疏矩阵使用示例")
    print("############################################################")

    demo_pagerank()
    demo_spmv()
    demo_collaborative_filtering()
    demo_feature_engineering()
    demo_slicing()

    print("\n" + "="*60)
    print("🎉 所有示例运行完成!")
    print("="*60)
    print("\n关键优势:")
    print("  ✓ 完全 JIT 编译 - 接近 C 的性能")
    print("  ✓ 类型安全 - Numba 类型检查")
    print("  ✓ 内存高效 - 零拷贝视图")
    print("  ✓ 灵活接口 - 迭代器、切片、方法")
    print("  ✓ 与 SciPy 互操作")


if __name__ == "__main__":
    main()
