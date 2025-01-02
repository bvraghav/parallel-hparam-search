import logging as LG
import pandas as pd
import click
from pathlib import Path
from io import StringIO
import pickle
import json

from . datasetFunctions import *

@click.command(context_settings = dict(
  show_default                  = True,
  help_option_names             = ['-h','--help']
))
@click.option('--out-dir', type=click.Path(
  exists=False, file_okay=False, path_type=Path,
), required=True)
@click.option('--result-fname', required=True)
@click.option('--model-fname', required=True)
@click.option('--hparams-path', type=click.Path(
  exists=False, dir_okay=False, path_type=Path,
), required=True)
@click.option('--data-dir', type=click.Path(
  exists=True, file_okay=False, path_type=Path,
), required=True)
@click.option('--id', required=True)
def main(
    out_dir,
    result_fname,
    model_fname,
    hparams_path,
    data_dir,
    id,
):
  lg = LG.getLogger(__name__)
  df = pd.read_json(hparams_path)

  s = StringIO()
  df.info(buf=s)
  lg.info(s.getvalue())

  val, = df.loc[df["id"]==id].values
  val = val.tolist()
  _, C, k = val
  lg.info(f'hparams: C: {C}, k: {k}')

  xTrainVal,yTrainVal,xTest,yTest = loadData(data_dir)
  # N = yTrainVal.shape[0]
  # shuffledIndices = getShuffledIndices(N)
  models = []

  for _ in range(k) :
    valAcc = 0.5
    model = const_one
    models.append(dict(
      valAcc=valAcc,
      model=model
    ))

  pretrained = dict(
    id=id,
    hparams=dict(C=C),
    testAcc=0.5,
    avgValAcc=0.5,
    experts=models,
  )
  with open(out_dir/model_fname, 'wb') as F :
    pickle.dump(pretrained, F)
  lg.info(f'Written to {out_dir/model_fname}')

  result = dict(
    id=id,
    hparams=dict(C=C),
    testAcc=0.5,
    avgValAcc=0.5,
    pretrainedModel=str(out_dir/model_fname),
  )
  with open(out_dir/result_fname, 'w') as F :
    json.dump(result, F)
  lg.info(f'Written to {out_dir/result_fname}')

# This following function needs to be loaded before
# unpickling the above model.
def const_one(x) :
  return 1

if __name__ == '__main__' :

  LG.basicConfig(
    level=LG.INFO,
    format='%(levelname)-8s: [%(name)s] %(message)s'
  )

  main()
