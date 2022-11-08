#!/usr/bin/python
import os


def save_wsp(wsp, fname):
    """
    Write a workspace or covariance workspce

    Parameters
    ----------
    wsp: NmtWorkspace or NmtCovarianceWorkspace
        Workspace or Covariance workspace to save
    fname: str
        Path to save the workspace to
    """
    # Recheck again in case other process has started writing it
    if os.path.isfile(fname):
        return

    try:
        wsp.write_to(fname)
    except RuntimeError as e:
        if 'Error writing' in str(e):
            os.remove(fname)
            wsp.write_to(fname)
        else:
            raise e


def read_wsp(wsp, fname, read_unbinned_MCM):
    """
    Read a workspace or covariance workspace and removes it if fails

    Parameters
    ----------
    wsp: NmtWorkspace or NmtCovarianceWorkspace
        Workspace or Covariance workspace to save
    fname: str
        Path to save the workspace to
    """
    # Recheck again in case other process has started writing it
    try:
        wsp.read_from(fname, read_unbinned_MCM)
    except RuntimeError as e:
        if 'Error reading' in str(e):
            os.remove(fname)

        raise e
