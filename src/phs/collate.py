import logging as LG
import json
import click
from pathlib import Path
import pandas as pd
from io import StringIO

@click.command(context_settings = dict(
  show_default                  = True,
  help_option_names             = ['-h','--help']
))
@click.option('--out-path', type=click.Path(
  exists=False, dir_okay=False, path_type=Path,
))
@click.option('--hparams-path', type=click.Path(
  exists=True, dir_okay=False, path_type=Path,
))
@click.option('--train-dir', type=click.Path(
  exists=True, file_okay=False, path_type=Path,
))
def main(
    out_path,
    hparams_path,
    train_dir,
):
  lg = LG.getLogger(__name__)

  df = pd.read_json(hparams_path)
  s = StringIO()
  df.info(buf=s)
  lg.info(s.getvalue())

  trainedModels = [
    read_json(train_dir/id/'result.json')
    for id in df.loc[:, 'id']
  ]
  best = min(trainedModels, key=lambda a: a['testAcc'])
  lg.info(f'best: {best}')

  result = dict(
    best=best,
    trainedModels=trainedModels
  )
  with open(out_path, 'w') as F :
    json.dump(result, F)

def read_json(fname):
  with open(fname, 'r')  as F :
    data = json.load(F)

  return data

if __name__ == '__main__' :
  LG.basicConfig(
    level=LG.INFO,
    format='%(levelname)-8s: [%(name)s] %(message)s'
  )

  main()
