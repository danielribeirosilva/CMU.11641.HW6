# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 01:41:43 2013

@author: daniel
"""

originFileName = 'test'
orginFileFormat = '.txt'
originFile = originFileName+orginFileFormat

for currentClass in range(1,18):
    outputFile = originFileName+'_'+str(currentClass)+orginFileFormat
    with open(outputFile, "a") as outputFile:
        with open(originFile) as inputFile:
            for line in inputFile:
                split = line.split(' ', 1)
                if int(split[0])==currentClass:
                    outputFile.write('+1 '+split[1]);
                else:
                    outputFile.write('-1 '+split[1]);
        
        
        