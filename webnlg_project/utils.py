import logging
import torch
import os


def get_logger(file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    s_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(file_path, mode='w')
    s_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    s_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    s_handler.setFormatter(s_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(s_handler)
    logger.addHandler(f_handler)

    return logger

def save_checkpoint(state, dir_path, file_name):
    torch.save(state, os.path.join(dir_path, file_name))