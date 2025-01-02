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
"""
"""


def readDataFromZipArchive(archive, fname):
  """Read a Pandas dataframe from excel file `FNAME`
  within `ARCHIVE`.

  """

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
  """Sanitise Raw Data `instanceof(pandas.DataFrame)`.

  1. Coerce Errors.
  2. Drop columns with more than 50% missing values.
  3. Drop rows with NA.
  4. Retrieve $Y$ and $X$ as numpy arrays.
  5. Transform $Y$'s... Here $Y \in \{-1,1\}$ instead of $\{0,1\}$.
  """
  lg = LG.getLogger(__name__)
  df = dataRaw

  for col in df.columns :
    df[col] = pd.to_numeric(df[col],errors='coerce')
  lg.info(
    f'IsNA: {df.isna().sum().sum()} out of {df.shape}'
  )

  # Drop columns with more than 50% missing values
  threshold = len(df) * 0.5
  df.dropna(thresh=threshold, axis=1, inplace=True)
  lg.info(
    f'IsNA: {df.isna().sum().sum()} out of {df.shape}'
  )

  # Drop rows with NA.
  df.dropna(inplace=True)
  lg.info(
    f'IsNA: {df.isna().sum().sum()} out of {df.shape}'
  )

  # Log Column Names
  lg.info(df.columns[:5])

  # Retrieve Y and X as numpy arrays
  Y = df.iloc[:,2].to_numpy(dtype=np.float64)
  X = df.iloc[:,3:44].to_numpy(dtype=np.float64)
  lg.info(
    f'X: {X.dtype,X.shape}, Y: {Y.dtype,Y.shape}'
  )
  lg.info(f'Y: {Y[:5]}')

  # Transform Y's...
  # Here Y \in {-1,1} instead of {0,1}
  Y = 2*Y-1
  lg.info(f'Y: {Y[:5]}')

  dataClean = (X, Y)
  return dataClean

def splitData(dataClean, trainValFactor=0.8) :
  """Split `dataClean` into trainVal and test sets.

  """

  (X, Y) = dataClean

  N = Y.shape[0]
  n = int(N * trainValFactor)
  shuffledIndices = list(range(N))
  random.sample(shuffledIndices, N)

  return (X[:n],Y[:n],X[n:],Y[n:])

def saveData(prefix,xTrainVal,yTrainVal,xTest,yTest) :
  """Save the four datasets with filenames based on
  format `fnamesFmt`.

  """

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
    df.to_csv(fname, index=False)
    fnames.append(fname)

def loadData(prefix):
  """Load the four datasets
  `xTrainVal,yTrainVal,xTest,yTest` from filenames
  based on format `fnamesFmt`.

  Convert the datasets to numpy.
  """
  lg = LG.getLogger(__name__)
  data = []
  for fnameFmt in fnamesFmt :
    fname = fnameFmt.format(prefix=prefix)
    data.append(pd.read_csv(fname))

  [lg.info(f'X: {X.shape}') for X in data]

  # For the sake of clarity:
  xTrainVal,yTrainVal,xTest,yTest = [
    X.to_numpy().squeeze() for X in data
  ]
  lg.info(f'xTrainVal: {xTrainVal.dtype, xTrainVal.shape}')
  lg.info(f'yTrainVal: {yTrainVal.dtype, yTrainVal.shape}')
  lg.info(f'xTest: {xTest.dtype, xTest.shape}')
  lg.info(f'yTest: {yTest.dtype, yTest.shape}')
  return xTrainVal,yTrainVal,xTest,yTest
