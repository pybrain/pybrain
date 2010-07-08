#!/bin/env python
""" Utility script to convert Python source code into tutorials.

Synopsis:
    code2tut.py basename

Output:
    Will convert  tutorials/basename.py  into  sphinx/basename.txt

Conventions:
1. All textual comments must be enclosed in triple quotation marks.
2. First line of file is ignored, second line of file shall contain title in "",
   the following lines starting with # are ignored.
3. Lines following paragraph-level markup (e.g. .. seealso::) must be indented.
   Paragraph ends with a blank line.
4. If the code after a comment starts with a higher indentation level, you have
   to manually edit the resulting file, e.g. by inserting "   ..." at the
   beginning of these sections.

See tutorials/fnn.py for example.
"""

__author__ = "Martin Felder, felder@in.tum.de"
__version__ = "$Id$"

import sys
import os


f_in = file(os.path.join("tutorials",sys.argv[1])+".py")
f_out = file(os.path.join("sphinx",sys.argv[1])+".txt", "w+")

# write the header
f_out.write(".. _"+sys.argv[1]+":\n\n")
f_in.readline()                                  #  ######################
line = f_in.readline()
line= line.split('"')[1]             #  # PyBrain Tutorial "Classification ..."
f_out.write(line+"\n")
f_out.write("="*len(line)+'\n\n')

linecomment = False
comment = 0
begin = True
inblock = False

# the following is an ugly hack - don't look at it!
for line in f_in:
    linecomment = False
    # crop #-comments at start of file
    if line.startswith('#'):
        if begin:
            continue
    elif begin:
        begin = False

    if '"""' in line:
        for i in range(line.count('"""')):
            comment = 1 - comment
        if line.count('"""')==2:
            linecomment = True

        line = line.replace('"""','')
        if comment==0:
            line += '::'
            if not inblock:
                line = line.strip()

    elif comment==0 and line!='\n':
        line = "  "+line

    if line.startswith('..'):
        inblock = True
    elif line=="\n":
        inblock = False

    if (comment or linecomment) and not inblock:
        line = line.strip()+"\n"

    if line.endswith("::"):
        line +='\n\n'
    elif line.endswith("::\n"):
        line +='\n'

    f_out.write(line)


f_in.close()
f_out.close()
