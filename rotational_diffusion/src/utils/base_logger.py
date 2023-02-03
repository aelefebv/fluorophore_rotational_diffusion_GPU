import logging

logger = logging
logger.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d :: [%(filename)s:%(lineno)d] :: [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d | %H:%M:%S",
)
logger.getLogger('matplotlib').setLevel(logging.ERROR)
logger.getLogger('PIL').setLevel(logging.ERROR)
