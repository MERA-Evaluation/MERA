try:
    import sklearn
except ModuleNotFoundError:
    sklearn = None


def mean(arr):
    return sum(arr) / max(len(arr), 1)


def f1_macro_score(items):
    if sklearn is None:
        return _f1_macro_score_fallback(items)
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    score = sklearn.metrics.f1_score(golds, preds, average="macro")
    return score


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Compute max metric between prediction and each ground truth."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def mcc(items):
    if sklearn is None:
        return _mcc_fallback(items)
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    score = sklearn.metrics.matthews_corrcoef(golds, preds)
    return score


def _f1_macro_score_fallback(items):
    labels = sorted(set(label for pair in items for label in pair))
    scores = []
    for label in labels:
        true_positive = sum(1 for gold, pred in items if gold == label and pred == label)
        false_positive = sum(1 for gold, pred in items if gold != label and pred == label)
        false_negative = sum(1 for gold, pred in items if gold == label and pred != label)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return mean(scores)


def _mcc_fallback(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    labels = sorted(set(golds) | set(preds))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for gold, pred in zip(golds, preds):
        matrix[label_to_idx[gold]][label_to_idx[pred]] += 1

    total = sum(sum(row) for row in matrix)
    if total == 0:
        return 0.0

    trace = sum(matrix[idx][idx] for idx in range(len(labels)))
    row_sums = [sum(row) for row in matrix]
    col_sums = [sum(matrix[row_idx][col_idx] for row_idx in range(len(labels))) for col_idx in range(len(labels))]

    sum_row_col = sum(row * col for row, col in zip(row_sums, col_sums))
    sum_row_sq = sum(row * row for row in row_sums)
    sum_col_sq = sum(col * col for col in col_sums)
    numerator = trace * total - sum_row_col
    denominator_left = total * total - sum_col_sq
    denominator_right = total * total - sum_row_sq
    denominator = (denominator_left * denominator_right) ** 0.5
    if denominator == 0:
        return 0.0
    return numerator / denominator
