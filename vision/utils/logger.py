import sys
import logging
import datetime


def GMT_8(sec, what):
    GMT_8_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return GMT_8_time.timetuple()


def init_logger():

    # logging.Formatter.converter = GMT_8
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger()
    return logger
