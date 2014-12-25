__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

import string

class ConfigGrabber:
    def __init__(self, filename, sectionId="", delim=("[", "]")):
        # filename:  name of the config file to be parsed
        # sectionId: start looking for parameters only after this string has
        #            been encountered in the file
        # delim:     tuple of delimiters to identify tags
        self.filename = filename
        self.sectionId = string.strip(sectionId)
        self.delim = delim

    def getValue(self, name):
        file = open(self.filename, "r")
        flag = -1
        output = []
        # ignore sectionId if not set
        if self.sectionId == "":
            self.sectionId = string.strip(file.readline())
            file.seek(0)

        for line in file:
            if flag == -1 and string.strip(line) == self.sectionId:
                flag = 0
            if flag > -1:
                if line[0] != self.delim[0]:
                    if flag == 1: output.append(string.strip(line))
                else:
                    if line == self.delim[0] + name + self.delim[1] + "\n": flag = 1
                    else: flag = 0
        file.close()
        #if len(output)==0: print("Attention: Config for ", name, "not found")
        return output

