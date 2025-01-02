import logging as LG
import click
from pathlib import Path

from phs.datasetFunctions import *

@click.command(context_settings = dict(
  show_default                  = True,
  help_option_names             = ['-h','--help']
))
@click.argument('ARCHIVE',
                type=click.Path(
                  exists=True, dir_okay=False,
                  path_type=Path,
                ))
@click.argument('ARCHIVE_FNAME',
                type=click.Path(
                  exists=False, dir_okay=False,
                  path_type=Path,
                ))
@click.option('--data-dir', '-D',
              type=click.Path(
                exists=True, file_okay=False,
                path_type=Path,
              ),
              default=Path('./data/preprocessed'),
              help='Data Directory.')
def main(archive, archive_fname, data_dir):
  lg = LG.getLogger(__name__)
  lg.info(f'archive: {archive}')
  lg.info(f'fname: {archive_fname}')

  dataRaw = readDataFromZipArchive(archive, archive_fname)
  dataClean = sanitiseData(dataRaw)
  xTrainVal,yTrainVal,xTest,yTest = splitData(dataClean)
  saveData(data_dir,xTrainVal,yTrainVal,xTest,yTest)

if __name__ == '__main__' :
  LG.basicConfig(
    level=LG.INFO,
    format='%(levelname)-8s: [%(name)s] %(message)s'
  )

  main()

