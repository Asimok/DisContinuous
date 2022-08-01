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


def perResults(ES_logit, SE_logit, qLen, pLen, selects=None, slide=0) -> [int, int]:
    '''
    :param ES_logit: [int] position
    :param SE_logit: [int] position
    :return: [(start, end)]
    '''
    temp = selects if selects else []
    start, end = 0, 0
    while True:
        start = ES_logit[end]
        if qLen <= start < qLen + pLen - 1 and end < start:
            end = SE_logit[start]
            if start <= end < qLen + pLen - 1:
                temp.append((start - qLen, end - qLen))
            else:
                break
        else:
            break
    return temp
    # start, end = -1, -1
    # for i in range(qLen, qLen + pLen):
    #     if P_logit[i] == 0:
    #         if start != -1:
    #             end = i
    #         else:
    #             start, end = i, i
    #     elif P_logit[i] == 1:
    #         if start != -1:
    #             temp.append([start - qLen + slide, end - qLen + slide])
    #             start, end = -1, -1
    # if start != -1:
    #     temp.append([start - qLen + slide, end - qLen + slide])
    #
    # return merge(temp)


def disContinuousPredict(document, document_offset, selects, maxPiece=5, pieceRange=5) -> [str]:
    '''
    :param document: str
    :param document_offset: [(int, int)]
    :param selects: [(start, end)]
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
