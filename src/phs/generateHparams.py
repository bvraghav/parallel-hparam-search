"""Generate Hparams.

Given N, cStart, cRange, k;

Generate a csv file with N records and header:
ID,C,k

such that C = 10^(cStart+cRange*(u~U(0,1)))
and ID is a 7-char long string.
"""

import logging as LG
import click
import numpy as np
from numpy.random import default_rng as RNG
import pandas as pd
import uuid
from pathlib import Path

@click.command(context_settings = dict(
  show_default                  = True,
  help_option_names             = ['-h','--help']
))
@click.argument('OUT', type=click.Path(
  exists=False, dir_okay=False, path_type=Path
))
@click.argument('N', type=int)
@click.argument('CSTART', type=float)
@click.argument('CRANGE', type=float)
@click.argument('K', type=int)
def main(out, n, cstart, crange, k):
  lg = LG.getLogger(__name__)

  rng = RNG()
  cs = np.sort(rng.random((n,)))
  cs = cstart + crange * cs
  cs = np.exp(np.log(10)*cs)

  ids = [uuid.uuid4().hex for _ in range((n+7) // 8)]
  ids = [id[4*i:4*(1+i)] for id in ids for i in range(8)]
  ids = ids[:n]

  pd.DataFrame(
    {
      'id': ids,
      'C': cs,
      'k': [k for _ in range(n)],
    }
  ).to_json(out)
  lg.info(f'Written to {out}')

if __name__ == '__main__' :
  LG.basicConfig(
    level=LG.INFO,
    format='%(levelname)-8s: [%(name)s] %(message)s'
  )

  main()
