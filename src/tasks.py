import os
import time
import numpy as np
import pandas as pd
from datetime import datetime


def mode_select(opt="", gcol=False):
    """Select the operation mode.

    Parameters
    ----------
    opt : {'test', 'on'}
        - 'test' test running mode.
        - 'on'   real running mode.

    gcol : {'True', 'False'}
        - 'True'  g-colaboratory server selected.
        - 'False' local server selected.
    """
    if opt == "test":
        return "test_models/", "/results/test_results.json"

    elif opt == "on" and not gcol:
        return "models/", "/results/results.json"

    elif opt == "on" and gcol:
        return "models/", "/content/drive/My Drive/results.json"


def server_select(opt=""):
    """Select the server to be used.

    Parameters
    ----------
    opt : {'gcol', 'loc'}
        - 'gcol' chooses the G-colaboratory server.
        - 'loc'  chooses a local server.
    """
    if opt == "loc":
        data_dir = os.getcwd() + "/data/raw"
        files = os.listdir(data_dir)
        files.sort()
        files.pop(0)
        return data_dir, files

    elif opt == "gcol":
        data_dir = os.getcwd() + "/drive/My Drive/data/raw"
        files = os.listdir(data_dir)
        files.sort()
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
