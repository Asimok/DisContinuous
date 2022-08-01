from collections import Counter, OrderedDict
import re
from rouge_score import rouge_scorer, scoring
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk import word_tokenize
from rouge import rouge
from tqdm import tqdm


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    re_art = re.compile(r'\b(a|an|the)\b')
    re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\'、。，；：‘’“”【】？]')

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def seg_char(sent):
    """
    把句子按字分开，不破坏英文结构
    """
    # 首先分割 英文 以及英文和标点
    pattern_char_1 = re.compile(r'([\W])')
    parts = pattern_char_1.split(sent)
    parts = [p for p in parts if len(p.strip()) > 0]
    # 分割中文
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(sent)
    chars = [w for w in chars if len(w.strip()) > 0]
    return chars


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = seg_char(normalize_answer(a_gold))
    pred_toks = seg_char(normalize_answer(a_pred))
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def calc_bleu(hypothesis, references) -> []:
    sf = SmoothingFunction(epsilon=1e-12).method1
    b1 = sentence_bleu([references], hypothesis, weights=(1.0 / 1.0,), smoothing_function=sf)
    b2 = sentence_bleu([references], hypothesis, weights=(1.0 / 2.0, 1.0 / 2.0), smoothing_function=sf)
    b3 = sentence_bleu([references], hypothesis, weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), smoothing_function=sf)
    b4 = sentence_bleu([references], hypothesis, weights=(1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0), smoothing_function=sf)
    return b1, b2, b3, b4


# 弃置
# def calc_rouge(cands, golds):
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     aggregator = scoring.BootstrapAggregator()
#     aggregator.add_scores(scorer.score(normalize_answer(''.join(golds)), normalize_answer(''.join(cands))))
#     result = aggregator.aggregate()
#     result = {key: value.mid.fmeasure for key, value in result.items()}
#     return result


def evaluateAll(predictions, messages) -> dict:
    '''
    :param predictions: [str]
    :param messages: [[qid, answer, is_discontinue, answerPiece, part]]
    :return: score dict
    '''
    assert len(predictions) == len(messages), f"num predictions: {len(predictions)}, num messages: {len(messages)}"
    total = len(predictions)
    exact = []
    f1 = []
    bleu1, bleu2, bleu3, bleu4 = [], [], [], []
    rouge1, rouge2, rougeL = [], [], []
    for qid, answer, is_discontinue, answerPiece, _ in tqdm(messages, desc='Evaluate score'):
        pred = predictions[qid]
        # use answerPiece
        # answer=''.join(answerPiece)
        # EM F1
        if answer is list:
            gold = answer
        else:
            gold = [answer]
        exact.append(max([compute_exact(g, pred) for g in gold]))
        f1.append(max([compute_f1(g, pred) for g in gold]))
        # bleu rouge
        cands = list(normalize_answer(pred).replace(' ', ''))
        if not cands:
            cands = [' ']
        golds = list(normalize_answer(''.join(gold)).replace(' ', ''))
        bleu_score = calc_bleu(cands, golds)
        bleu_score = [round(100 * b_score, 6) for b_score in bleu_score]
        for b, b_score in zip([bleu1, bleu2, bleu3, bleu4], bleu_score):
            b.append(b_score)
        rouge_score = rouge.Rouge().get_scores(' '.join(cands), ' '.join(golds))[0]
        rouge_score = [round(100 * v['f'], 6) for k, v in rouge_score.items()]
        for r, r_score in zip([rouge1, rouge2, rougeL], rouge_score):
            r.append(r_score)
    return {
        "total": total,
        "exact": 100.0 * sum(exact) / total,
        "f1": 100.0 * sum(f1) / total,
        "bleu1": sum(bleu1) / total,
        "bleu2": sum(bleu2) / total,
        "bleu3": sum(bleu3) / total,
        "bleu4": sum(bleu4) / total,
        "rouge-1": sum(rouge1) / total,
        "rouge-2": sum(rouge2) / total,
        "rouge-L": sum(rougeL) / total,
    }


if __name__ == '__main__':
    print(
        evaluateAll(
            {'0': "你好，今天天不错"},
            [['0', "你好，今天天气不错", 0, 0, 0]]
        )
    )
