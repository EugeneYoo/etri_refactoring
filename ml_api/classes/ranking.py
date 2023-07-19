import numpy as np
import pandas as pd

from .modelConfig import params
from .ranker import Ranker

def getRank(query, datasetList):
    result = []
    ranker = Ranker.get_instance()
    scoreList, datasetList = ranker.rank(datasetList, query)
    for score in scoreList:
        for dataset in datasetList:
            #print(dataset)
            if score['uid'] == dataset['uid']:
                dataset['finalScore'] = score["qdsScore"] * params["scoreWeight"] * 100 + score["elasticScore"]

                result.append(dataset)
    #print(result)
    return result

def negativeSampling(datasetId):
    ranker = Ranker.get_instance()
    negatives = ranker.negativeSampling(datasetId)
    return negatives

def concatenateQueryDocumentVector(negatives, positive, queryVec):
    result = []
    for negative in negatives:
        result.append(np.concatenate((queryVec, negative,[0]), axis=0))

    result.append(np.concatenate((queryVec, positive, [1]), axis=0))
    return result

def concatenateDataframe(df1, df2):
    df = pd.concat([df2, df1], axis=0)
    return df

def makeDataFrame(data):
    ranker = Ranker.get_instance()
    queryColumns = ["Qattr_" + str(attribute) for attribute in range(len(ranker.getQueryVector()))]
    documentColumns = ["Dattri_" + str(attribute) for attribute in range(len(ranker.getClickedDatasetVector()))]
    columns = np.concatenate((queryColumns, documentColumns, ["label"]))

    df = pd.DataFrame(data, columns=columns)
    return df