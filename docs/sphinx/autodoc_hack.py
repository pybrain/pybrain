# replace the function below in sphinx.ext.autodoc.py (tested with Sphinx version 0.4.1)
__author__ = "Martin Felder"

def prepare_docstring(s):
    """
    Convert a docstring into lines of parseable reST.  Return it as a list of
    lines usable for inserting into a docutils ViewList (used as argument
    of nested_parse().)  An empty line is added to act as a separator between
    this docstring and following content.
    """
    if not s or s.isspace():
        return ['']
    s = s.expandtabs()

    # [MF] begin pydoc hack **************
    idxpar = s.find('@param')
    if idxpar > 0:
        # insert blank line before keyword list
        idx = s.rfind('\n',0,idxpar)
        s = s[:idx]+'\n'+s[idx:]
        # replace pydoc with sphinx notation
        s = s.replace("@param", ":param")
    # [MF] end pydoc hack **************

    nl = s.rstrip().find('\n')
    if nl == -1:
        # Only one line...
        return [s.strip(), '']
    # The first line may be indented differently...
    firstline = s[:nl].strip()
    otherlines = textwrap.dedent(s[nl+1:]) #@UndefinedVariable
    return [firstline] + otherlines.splitlines() + ['']
