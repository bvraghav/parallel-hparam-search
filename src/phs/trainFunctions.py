from sklearn import svm
import numpy as np

import logging as LG

from pathlib import Path
import os, sys

import itertools as It
import functools as Ft
import math as m

from collections import (OrderedDict, namedtuple)

import json
import operator as Op

import random

import time
import datetime as Dt
from datetime import timedelta as Tdelta

from dataclasses import dataclass, field

def getTrainedModel(xTrain, yTrain, xVal, yVal, hparams) :
  """Train a model using Linear SVM Classifier.

  `hparams` is a dict with key `'C'`
  """

  model = svm.LinearSVC(C=hparams['C'])
  model.fit(xTrain, yTrain)

  return dict(
    model=model,
    hparams=hparams,
    metrics=dict(
      valAcc=1-np.mean(model.predict(xVal)!=yVal)
    ),
    theMetric='valAcc',
  )

@dataclass
class Ksplit :
  """Convenience Class k-Split (for k-fold CV).

  Split a dataset with $N$ samples into $k$ parts.
  Only maintain indices so that `ksplit[i]` provides a
  set of indices subscriptable to the original dataset,
  _e.g._ `iTrain, iVal = ksplit[i]; xTrain = X[iTrain]`
  creates a train data subset for `i`-th fold of
  validation.

  On initialisation, create a random shuffled set of
  indices if not already provided.

  `[i]` subscript access will fetch the `i`-th pair of
  indices corresponding to train and val split
  respectively.

  """

  k : int
  ''''''

  N : int
  ''''''

  indices : list[int] = field(default_factory=list)
  ''''''

  def __post_init__(self) :

    if len(self.indices) != self.N :
      self.indices = list(range(self.N))
      random.shuffle(self.indices)

  def __getitem__(self, index):
    k, N, I = self.k, self.N, self.indices
    i = index * N // k
    j = (1+index) * N // k
    return ((I[:i]+I[j:]), I[i:j])

@dataclass
class Experts :
  """Convenience Class for a Mixture of Experts Classifier.
  """

  experts : list[dict]
  ''''''
  classes : list[int] = field(
    default_factory=lambda: [0,1]
  )
  ''''''
  def __getitem__(self, idx) :
    """Convenience subscript access for experts.

    `self[i]` is the same as `self.experts[i]`
    """
    return self.experts[idx]
  def __get__(self, X):
    """Predict the class of X based on "opinion" of
    experts. `Y = self(X)`
    """
    lg = LG.getLogger(__name__)
    C = np.asarray(self.classes)
    lg.info(f'C: {C}')
    Y = np.stack([expert['model'].predict(X)
                  for expert in self.experts])
    lg.info(f'Y: {Y.dtype, Y.shape}')
    Y = np.stack([(Y==c).sum(0)
                  for c in self.classes])
    lg.info(f'Y: {Y.dtype, Y.shape}')
    Y = np.argmax(Y, axis=0)
    lg.info(f'Y: {Y.dtype, Y.shape}')
    return C[Y]
  def acc(self, X, Y) :
    """Convenience function of naïve accuracy
    calculator.

    `acc = np.mean(self(X)==Y)`
    """
    lg = LG.getLogger(__name__)
    Y_hat = self.__get__(X)
    lg.info(f'Y_hat[:5]: {Y_hat[:5]}')
    return np.mean(Y_hat==Y)
