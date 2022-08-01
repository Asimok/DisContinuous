def merge(intervals: [[int]]) -> [[int]]:
    if not intervals:
        return []
    intervals.sort()
    result = list()
    for i in intervals:
        if not result or result[-1][1] < i[0]:
            result.append(i)
        else:
            result[-1][1] = max(result[-1][1], i[1])
    return result


def perResults(P_logit, C_logit, qLen, pLen, selects=None, slide=0) -> [int, int]:
    '''
    :param P_logit: [int] 0:inner,1:out
    :param C_logit: [int] 0:start,1:inner,2:end,-100:out
    :return: [index]
    '''
    temp = selects if selects else []
    start, end = -1, -1
    for i in range(qLen, qLen + pLen):
        if P_logit[i] == 0:
            if start != -1:
                end = i
            else:
                start, end = i, i
        elif P_logit[i] == 1:
            if start != -1:
                temp.append([start - qLen + slide, end - qLen + slide])
                start, end = -1, -1
    if start != -1:
        temp.append([start - qLen + slide, end - qLen + slide])


    start, end = -1, -1
    for i in range(qLen, qLen + pLen):
        if C_logit[i] == 0:
            start, end = i, i
        elif C_logit[i] == 1:
            if start != -1:
                end = i
        elif C_logit[i] == 2:
            if start != -1 != end:
                temp.append([start - qLen + slide, end - qLen + slide])
            start, end = -1, -1
    if start != -1:
        temp.append([start - qLen + slide, end - qLen + slide])
    return merge(temp)


def disContinuousPredict(document, document_offset, selects, maxPiece=5, pieceRange=5) -> [str]:
    '''
    :param document: str
    :param document_offset: [(int, int)]
    :param selects: [int]
    :return: [answer, probability]]sorted by position with no coverage
    '''
    temp_prediction = []
    for start, end in selects:
        assert end < len(document_offset), (start, end, len(document_offset), selects, document_offset)
        start_offset = document_offset[start][0]
        end_offset = document_offset[end][1]
        answer = document[start_offset:end_offset]
        temp_prediction.append(answer)
    return ''.join(temp_prediction)
