import logging
import os 
import re

logger = logging.getLogger("aeri_logger")

def findfile(path,pattern,verbose=2):
    logger.debug('Searching for files matching '+path+'/'+pattern)

    # We want to preserve periods (dots) in the pattern,
    # as many of our file patterns have periods in the name
    pattern = pattern.replace('.','\.')

    # Regex requires that we replace any asterisks with .*
    pattern = pattern.replace('*','.*')

    # Regex uses a period as a single character wildcard
    pattern = pattern.replace('?','.')

    # Trap the first and last character of the pattern
    first = pattern[0]
    lastl = pattern[-1]

    # If the first letter is an * or ?, then we prepend a dot
    if(first == '*'):
        pattern = '.'+pattern
    elif(first == '?'):
        pattern = '.'+pattern
    else:
        pattern = '^'+pattern

    # If the last letter is not a *, then we append a $
    if(lastl != '*'):
        pattern = pattern+'$'

    # Compile the regex expression, and return the files that are found
    prog  = re.compile(pattern)

    # Check to see if the file directory exists
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if re.search(prog, f)]

        # Now prepend the path to each files
        for i in range(len(files)):
            files[i] = path+'/'+files[i]
        files.sort()
        return files, 0
    else:
        logger.error(f'The directory {path} does not exist!')
        return [], 1