def perResults(start_logit, end_logit, qLen, pLen, n_best_size=10, max_answer_length=30) -> [[int, int, float]]:
    '''
    :param start_logit: [float]
    :param end_logit: [float]
    :param n_best_size: int
    :return: [start], [end], [probability]
    '''

    def _get_n_best_index_probability(logit, n_best_size=10) -> list:
        '''
        """Get the n-best indexes with logits from a list."""
        :param logits: [float]
        :param n_best_size: int
        :return: [(index, probability)]
        '''
        return sorted(enumerate(logit), key=lambda x: x[-1], reverse=True)[:n_best_size]

    start_end_probability = [[-1, -1, start_logit[0] + end_logit[0]]]

    start_index_probability = _get_n_best_index_probability(start_logit, n_best_size)
    end_index_probability = _get_n_best_index_probability(end_logit, n_best_size)

    for start_index, start_probability in start_index_probability:
        if start_index < qLen:
            continue
        for end_index, end_probability in end_index_probability:
            if end_index >= qLen + pLen - 1:
                continue
            if end_index - start_index >= max_answer_length:
                continue
            if start_index > end_index:
                continue
            start_end_probability.append([start_index - qLen, end_index - qLen, start_probability + end_probability])
    return sorted(start_end_probability, key=lambda x: x[-1], reverse=True)[:n_best_size]


def predictAll(document, document_offset, start_end_probability) -> [[str, float]]:
    '''
    :param document: str
    :param document_offset: [(int, int)]
    :param start_end_probability: [[int, int, float]]
    :return: [[answer, probability]]sorted by probability
    '''
    prediction_all = []
    for start, end, probability in start_end_probability:
        if start == -1 and end == -1:
            answer = ""
        else:
            start_offset = document_offset[start][0]
            end_offset = document_offset[end][1]
            answer = document[start_offset:end_offset]
        prediction_all.append([answer, probability])
    return prediction_all


def _isSubstr(a, b):
    if len(a) <= 2 or len(b) <= 2:
        return a in b or b in a
    else:
        la, lb = len(a), len(b)
        for l, r in [(0, 2), (1, 1), (2, 0)]:
            if a[l: la - r] in b or b[l: lb - r] in a:
                return True
        else:
            return False


def disContinuousPredict(document, document_offset, start_end_probability, maxPiece=5, pieceRange=5) -> [[str, float]]:
    '''
    :param document: str
    :param document_offset: [(int, int)]
    :param start_end_probability: [[int, int, float]]
    :return: [[answer, probability]]sorted by position with no coverage
    '''
    temp_prediction = []
    temp_probability = 0
    for start, end, probability in start_end_probability:
        if probability < temp_probability / 2:
            break
        for ps, pe, _ in temp_prediction:
            if abs(ps - start) < pieceRange or abs(pe - end) < pieceRange:
                break  # 头头，尾尾过近不要
            if start <= ps <= end or start <= pe <= end:
                break  # 存在重叠不要
        else:
            temp_prediction.append([start, end, probability])
            temp_probability = probability
    temp_prediction.sort()
    prediction = predictAll(document, document_offset, temp_prediction[:maxPiece])
    answers = []
    for answer, probability in prediction:
        for i, [a, p] in enumerate(answers):
            if _isSubstr(a, answer):
                if probability > p:
                    answers[i] = ['', 0]
                else:
                    break
        else:
            answers = [[a, p] for a, p in answers if p]
            answers.append([answer, probability])
    return ''.join(a for a, p in answers)
