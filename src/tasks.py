"""Tasks

Customized tasks involving the computing process.
"""


import os
import time
import numpy as np
import pandas as pd
from datetime import datetime


def mode_select(opt=""):
    """Select the operation mode.

    Parameters
    ----------
    opt : {'test', 'on'}
        - 'test' test running mode.
        - 'on'   real running mode.
    """
    if opt == "test":
        return "models_test/", "/results/test_results.json"

    elif opt == "on":
        return "models/", "/results/results.json"


def set_directory():
    """Set data directory.
    """
    data_dir = os.getcwd() + "/data/raw"
    files = os.listdir(data_dir)
    files.sort()
    files.pop(0)
    return data_dir, files


def time_display(t, flag=""):
    """Generates a display that shows the elapsed time of computation.

    Parameters
    ----------
    t : initial time in seconds since the Epoch.

    flag : {'mid', 'end'}
        Flag to chosse the position in the code.
        - 'mid' chooses the display in the middle of the computation.
        - 'end' chooses the display in the end of the computation.
    """
    elapsed = time.time() - t
    if flag == "end":
        print("=" * 45 + "\nTotal Elapsed Time:\n")
        print(
            int(elapsed / 3600),
            "hours",
            int((elapsed % 3600) / 60),
            "minutes",
            int((elapsed % 3600) % 60),
            "seconds",
        )
        print("=" * 45 + "\n")
        print(datetime.now())
    elif flag == "mid":
        print(
            " - [",
            int(elapsed / 3600),
            "h",
            int((elapsed % 3600) / 60),
            "m",
            int((elapsed % 3600) % 60),
            "s]",
            end="",
        )
