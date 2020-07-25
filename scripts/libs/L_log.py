# -*- coding: utf-8 -*-
"""
Created on Sat 8 Jul  5 13:26:12 2017
@author: rgrimson
"""

import os
import sys
import time
import logging

import my_base_dir
BASE_DIR = my_base_dir.get()


log_level=logging.INFO #INFO, DEBUG, WARNING
logfile=os.path.splitext(os.path.basename(sys.argv[0]))[0]
if logfile=='':
    logfile='GENERIC'

localtime = time.asctime( time.localtime(time.time()) )
L="-------------------------------------------------------------------------------"
INIT_LOG =  localtime + '\n' + L + '\n' + logfile + ': new initialization...\n' + L

    
logging.basicConfig(filename=BASE_DIR+logfile+'.log',level=logging.INFO)
logging.info(INIT_LOG)
#logging.debug('This message should go to the log file')
#logging.warning('And this, too')

#import logging
#MSG=""
#logging.info("")