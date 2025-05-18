import logging

format = "%(asctime)s [%(filename)s:%(lineno)d] %(levelname)-8s %(message)s"
logging.root.setLevel(logging.INFO)
# logging.root.setLevel(logging.DEBUG)
# log.setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format=format)
# logging.basicConfig(level=logging.DEBUG, format=format)

mylogger = logging.getLogger
