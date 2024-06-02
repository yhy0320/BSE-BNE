from sklearn import metrics

clustering_metrics = {
    'homogeneity_score': metrics.homogeneity_score,
    'completeness_score': metrics.completeness_score,
    'v_measure_score': metrics.v_measure_score,
    'adjusted_rand_score': metrics.adjusted_rand_score,
    'adjusted_mutual_info_score': metrics.adjusted_mutual_info_score,
    'fowlkes_mallows_score': metrics.fowlkes_mallows_score,
    'rand_score': metrics.rand_score,
    'mutual_info_score': metrics.mutual_info_score,
    'normalized_mutual_info_score': metrics.normalized_mutual_info_score,
}


def metric_result(labels_true, labels_pred) -> dict:
    """
    遍历执行指标评判函数
    """
    metrics_map = {}

    for k, m in clustering_metrics.items():
        metrics_map[k] = m(labels_true, labels_pred)

    return metrics_map
