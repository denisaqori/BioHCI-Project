'''
Created: 11/19/18
© Andrew W.E. McDonald 2018 All Rights Reserved

Configures the logger

At the moment, there is one logger per process -- multiple log files are written (one per process)

All logfiles are written to the ./log directory

Code taken from (in part) -- see for more info: https://docs.python.org/3/howto/logging-cookbook.html


---------- Usage ------------

AureliusLogger.configureLogger("Main_Process") # this name sets the file name -- should be the process name if using multiple processes
logger = AureliusLogger.getLogger(__name__) # this name sets the name to use while logging within the file

NOTE: __name__ will use the module's name.
NOTE 2: Each module should call:
    - me.logger = AureliusLogger.getLogger(__name__)
in its init method. This will return a Python logging.logger instance -- then logging is done just as it is normally with Python.

'''

import logging
import sys

from aurelius.config.settings import AureliusSettings
from aurelius.util.convenience import now


class AureliusLogger:

    _configured = False
    _name = None
    _logdir = AureliusSettings("AureliusLogger", "logdir")
    _fh = None
    _ch = None
    _logLevel = -1


    @classmethod
    def configureLogger(me, name, logLevel=logging.INFO):
        # create logger with name 'name'
        logger = logging.getLogger(name)
        logger.setLevel(logLevel)
        # create file handler which logs to lowest specified level
        fh = logging.FileHandler(me._logdir+"/"+name+"_"+now()+".log")
        fh.setLevel(logging.INFO)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        loggingFormat = '%(asctime)s - %(filename)s:%(lineno)s - %(funcName)s - (%(name)s) - %(levelname)s - %(message)s'
        formatter = logging.Formatter(loggingFormat)
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        # set class variables.
        me._name = name
        me._logLevel = logLevel
        me._fh = fh
        me._ch = ch
        me._configured = True

    @classmethod
    def getLogger(me, nameToUse=""):
        if (nameToUse.__eq__("")):
            nameToUse = me._name
        if (me._configured):
            logger = logging.getLogger(nameToUse)
            logger.setLevel(me._logLevel)
            logger.addHandler(me._fh)
            logger.addHandler(me._ch)
            return logger
        else:
            print("Error! No logger configured!", file=sys.stderr)
            return None



