"""Generate the code reference pages and navigation.

Adapted from https://mkdocstrings.github.io/recipes/

+ [2024-09-26] Automate `mkdocs-click`.

  Depends `python -m pip install mkdocs-click`
  ((`mkdocs-click`)[https://github.com/mkdocs/mkdocs-click])

  Adapted to add cli (click) documentation so that
  Project Tree like `{src/PKG/{...,cli/...},tests/...}`
  with cli modules under `src/PKG/cli` using `click`
  may be documented automatically.

"""

import logging as LG
from pathlib import Path

import itertools as It

import mkdocs_gen_files

def getCliPath(src : Path) -> Path:
  """Find dir named cli in first level descent of the
package.

Args :
    src : Source path in the package styled as `ROOT/{src,tests}`.
    
Returns :
    Path of directory named cli.
"""

  p = (x for x in src.glob('*/cli') if x.is_dir())
  p = next(iter(p))

  return p

def gen_api_pages(
    root : Path,
    src : Path,
    nav : mkdocs_gen_files.Nav
) -> mkdocs_gen_files.Nav :
  """Generate API Pages for the underlying python package.

Args:
  root : Package Root.
  src : Source path in the package styled as `ROOT/{src,tests}`.
  nav : Mutable navigation object

Return:
  The mutable navigation object `nav`.
"""
  
  cli = getCliPath(src)

  for path in sorted(src.rglob("*.py")):
    if str(path).startswith(str(cli)) : continue

    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    # full_doc_path = Path("reference", doc_path)
    full_doc_path = doc_path

    parts = tuple(module_path.parts)

    lg.debug(f'Processing: {module_path}')
    
    if parts[-1] == "__init__":
      parts = parts[:-1]
      doc_path = doc_path.with_name("index.md")
      full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
      continue

    lg.info(f'ref::parts: {parts}')
    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
      ident = ".".join(parts)
      fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(
      full_doc_path,
      path.relative_to(root)
    )

  lg.debug(f'Processing SRC: done.')

  return nav


def write_nav(nav : mkdocs_gen_files.Nav):
  """Write navigation details to file.

Args:
  nav : Mutable navigation object.
"""
  with mkdocs_gen_files.open("index.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

  lg.debug(f'Writing Nav Files: done.')


def gen_cli_pages(
    root : Path,
    src : Path,
    nav : mkdocs_gen_files.Nav
) -> mkdocs_gen_files.Nav :
  """Generate API Pages for the underlying Python CLI App.

Args:
  root : Package Root.
  src : Source path in the package styled as `ROOT/{src,tests}`.
  nav : Mutable navigation object

Return:
  The mutable navigation object `nav`.
"""
  lg = LG.getLogger(__name__)

  cli = getCliPath(src)

  prj_toml = root/'pyproject.toml'

  with open(prj_toml, 'r') as F :
    prj_toml_lines = [
      l.rstrip('\n') for l in F.readlines()
    ]

  prj_toml_lines = It.dropwhile(
    lambda l: not l.startswith('[project.scripts]'),
    prj_toml_lines,
  )

  prj_toml_lines = It.islice(prj_toml_lines, 1, None)

  prj_toml_lines = It.takewhile(
    lambda l: not l.startswith('['),
    prj_toml_lines,
  )

  prj_toml_lines = It.filterfalse(
    lambda l: len(l.strip()) < 1,
    prj_toml_lines,
  )

  prj_toml_lines = list(prj_toml_lines)
  # lg.info(f'prj_toml_lines: {prj_toml_lines}')

  index_content = []
  index_doc = cli.relative_to(src) / 'index.md'

  for scr in prj_toml_lines :
    # lg.info(f'scr: {scr}')
    name, loc = scr.split('=', maxsplit=1)
    name = name.strip()

    fname,func = loc.split(':', maxsplit=1)

    fname = fname.strip()
    fname = fname.strip('"')
    mod_name = fname
    fname = fname.replace('.','/')
    fname = Path(fname) / 'index.md'
    doc_name = fname

    func = func.strip()
    func = func.strip('"')

    parts = tuple(mod_name.split('.'))

    lg.info(f'cli::parts: {parts}')

    nav[parts] = doc_name.as_posix()

    # lg.info(f'doc_name: {doc_name}')

    with mkdocs_gen_files.open(doc_name, 'w') as F :
      ident = 'mkdocs-click'
      heading = doc_name.stem

      content = (f'::: {ident}\n'
                 f'    :module: {mod_name}\n'
                 f'    :prog_name: {name}\n'
                 f'    :command: {func}\n')

      # lg.info(content)

      F.write(content)

    mkdocs_gen_files.set_edit_path(
      doc_name,
      (src/doc_name).with_suffix('.py').relative_to(root)
    )

    doc_link = doc_name.relative_to(index_doc.parent)
    index_content.append(f'+ [{name}]({doc_link})')

  if 0 < len(index_content) :
    index_content.insert(0, (
      f'# CLI Index\n'
      f'Command-line interface scripts available in '
      f'this module.\n'
    ))
    index_content = '\n'.join(index_content)
    parts = index_doc.parent.parts

    # lg.info(f'index_doc: {index_doc}')
    lg.info(f'cli::index::parts: {parts}')
    # lg.info(f'index_content:\n{index_content}')

    with mkdocs_gen_files.open(index_doc, 'w') as F :
      F.write(index_content)
      F.write('\n')

    nav[parts] = index_doc

  lg.debug(f'Processing SCRIPTS: done.')

  return nav

def main() :

  nav = mkdocs_gen_files.Nav()

  root = Path(__file__).parent.parent
  src = root / "src"
  lg.debug(f'src: {src}')

  nav = gen_api_pages(root, src, nav)
  nav = gen_cli_pages(root, src, nav)
  # write_nav(nav)


LG.basicConfig(
  level=LG.INFO,
  format='%(levelname)-8s: [%(name)s] %(message)s'
)
lg = LG.getLogger(__name__)

main()
