import logging as LG
import numpy as np
from io import StringIO
import pandas as pd
from zipfile import ZipFile
import random

fnamesFmt = [
  '{prefix}/xTrainVal.txt',
  '{prefix}/yTrainVal.txt',
  '{prefix}/xTest.txt',
  '{prefix}/yTest.txt',
]

def readDataFromArchive(archive, fname):
  lg = LG.getLogger(__name__)
  with ZipFile(archive, 'r') as Z:
    with Z.open(str(fname), 'r') as F :
      dfs = pd.read_excel(F, sheet_name=None)

  lg.info(dfs.keys())

  df = dfs['Full_new']
  infoBuf = StringIO()
  df.info(buf=infoBuf)
  lg.info(infoBuf.getvalue())
  return df

def sanitiseData(dataRaw) :
  # df = dataRaw
  # Y = df.iloc[:,2].to_numpy(dtype=np.float64)
  # X = df.iloc[:,3:44].to_numpy(dtype=np.float64)
  Y, X = np.zeros(10,np.int64), np.zeros((10,3),np.float64)

  dataClean = (X, Y)

  return dataClean

def splitData(dataClean, trainValFactor=0.8) :
  (X, Y) = dataClean

  N = Y.shape[0]
  n = int(N * trainValFactor)
  shuffledIndices = list(range(N))
  random.sample(shuffledIndices, N)

  return (X[:n],Y[:n],X[n:],Y[n:])

def saveData(prefix,xTrainVal,yTrainVal,xTest,yTest) :
  lg = LG.getLogger(__name__)
  lg.info(f'prefix: {prefix}')

  xTrainVal = pd.DataFrame(xTrainVal)
  yTrainVal = pd.DataFrame(yTrainVal)
  xTest = pd.DataFrame(xTest)
  yTest = pd.DataFrame(yTest)

  fnames = []
  for (fnameFmt,df) in zip(
      fnamesFmt,
      [xTrainVal,yTrainVal,xTest,yTest]
  ) :
    fname = fnameFmt.format(prefix=prefix)
    df.to_csv(fname)
    fnames.append(fname)

def loadData(prefix):
  data = []
  for fnameFmt in fnamesFmt :
    fname = fnameFmt.format(prefix=prefix)
    data.append(pd.read_csv(fname))

  # For the sake of clarity:
  xTrainVal,yTrainVal,xTest,yTest = data
  return xTrainVal,yTrainVal,xTest,yTest
