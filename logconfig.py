import logging
import colorful
import time
# ********************************************************************
# formatter
stream_formatter = logging.Formatter('{c.white_on_black}%(levelname)s{c.reset} {c.red}%(asctime)s{c.reset} {c.blue}[%(filename)s:%(funcName)s:%(lineno)d]{c.reset} %(message)s'.format(c=colorful))
file_formatter = logging.Formatter('%(levelname)s %(asctime)s[%(filename)s:%(funcName)s:%(lineno)d]%(message)s')
# ********************************************************************

console_handler = logging.StreamHandler()
console_handler.setFormatter(stream_formatter)

timestamp = time.time()
formatted_time = time.strftime("%Y%m%d%H%M", time.localtime(timestamp))
file_handler=logging.FileHandler(f'logs/log_{formatted_time}.txt', 'a')
file_handler.setFormatter(file_formatter)

root = logging.getLogger()
root.addHandler(console_handler)
root.addHandler(file_handler)
root.setLevel(logging.WARNING)