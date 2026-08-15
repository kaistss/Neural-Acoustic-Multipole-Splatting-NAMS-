"""
The code in this module was taken from the Acoustic Volume Rendering (AVR) sources below
and modified for use in this project.

- Repository: https://github.com/penn-waves-lab/AVR
- Reference: Lan, Zitong, et al. "Acoustic volume rendering for neural impulse response fields."
  Advances in Neural Information Processing Systems 37 (2024): 44600-44623.
"""

import logging

def logger_config(log_savepath,logging_name):
    '''logger config
    '''
    # get logger name
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)

    # get file handler and set level
    file_handler = logging.FileHandler(log_savepath, encoding='UTF-8')
    file_handler.setLevel(logging.DEBUG)

    # format the file handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)

    # console sream handler
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    # add handler for logger objecter
    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger


def logging_loss(total_loss, spec_loss, amplitude_loss, angle_loss, time_loss, energy_loss, multi_stft_loss):
    losses = {
        'total_loss': total_loss,
        'spec_loss': spec_loss,
        'amplitude_loss': amplitude_loss,
        'angle_loss': angle_loss,
        'time_loss': time_loss,
        'energy_loss': energy_loss,
        'multi_stft_loss': multi_stft_loss
    }
    return losses

def logging_loss_4c(total_loss, spec_loss, amplitude_loss, angle_loss, time_loss, energy_loss, multi_stft_loss, momentum_loss, continuity_loss, data_loss, phy_loss):
    losses = {
        'total_loss': total_loss,
        'spec_loss' : spec_loss,
        'amplitude_loss': amplitude_loss,
        'angle_loss': angle_loss,
        'time_loss' : time_loss,
        'energy_loss': energy_loss,
        'multi_stft_loss': multi_stft_loss,
        'momentum_loss'  : momentum_loss,
        'continuity_loss': continuity_loss,
        'data_loss'      : data_loss,
        'physical_loss'  : phy_loss
    }
    return losses


if __name__ == '__main__':
    logger = logging.getLogger('avr')
    file_handlers = [handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)]
    file_handler = file_handlers[0]
    file_handler.setLevel(logging.DEBUG)