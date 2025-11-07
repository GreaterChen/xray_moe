from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge
from pycocoevalcap.cider import Cider


def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # post-processing, make format consistent
    # 统一大小写与基本格式，避免大小写对BLEU等指标的干扰
    def _normalize_text(s: str) -> str:
        return s.lower()

    # 规范化gts与res文本（均转为小写）
    for k in gts.keys():
        gts[k] = [_normalize_text(x) for x in gts[k]]
    for k in res.keys():
        res[k][0] = _normalize_text(res[k][0])
        # 额外的简单标点处理，保持原有行为
        res[k][0] = (res[k][0] + " ").replace(". ", " . ").replace(" - ", "-")

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res
