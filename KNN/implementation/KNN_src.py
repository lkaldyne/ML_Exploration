import numpy as np
import warnings
import heapq
from collections import Counter

dataset = {'a': [[1, 2], [2, 3], [3, 1]], 'b': [[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]

def kNN(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for feature in data[group]:
            heapq.heappush(distances, (np.linalg.norm(np.array(feature)-np.array(predict)), group))

    vote_track = {}
    for i in range(k):
        curMin = heapq.heappop(distances)
        try:
            vote_track[curMin[1]] += 1
        except KeyError:
            vote_track[curMin[1]] = 1

    vote_result = ['error', -9999999]
    for key in vote_track:
        curKey = vote_track[key]
        if curKey > vote_result[1]:
            vote_result[0] = key
            vote_result[1] = curKey
    return vote_result[0]