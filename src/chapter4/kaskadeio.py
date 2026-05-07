import subprocess
import os
from os import path
import numpy as np

def runkaskade(execpath, execname, parameter):
    """

    Parameters
    ----------
    execpath : string
        Path to kaskade file.
    execname : string
        Name of executable.
    parameter : dictionary
        Dictionary with all parameters which are parsed to the kaskade executable.

    Returns
    -------
    None.

    """

    if path.isdir(execpath):

        if path.isfile( os.path.join(execpath,execname)):
            cmd = execname

            for key, value in parameter.items():
                if key=='>':
                    cmd += ' ' + key + ' ' + str(value)
                elif key=='|':
                    cmd +=' ' + key + ' tee ' + str(value)
                else:
                    cmd +=' '+ key + '=' + str(value)

            print("Starting: " + execname)
            print("Commands: " + cmd.split(' ', 1)[1])

            proc = subprocess.Popen(cmd, cwd=execpath, shell=True)

            try:
                outs, errs = proc.communicate()
                print("output = {}".format(outs))
                print("errors = {}".format(errs))
                print("---------------------------------------------------------------------")
                print("---------------------------------------------------------------------")
                print("\n\n")
                return
            except subprocess.TimeoutExpired:
                proc.kill()
                outs, errs = proc.communicate()
                print("output = {}".format(outs))
                print("errors = {}".format(errs))
                print("\n\n")
                return

        else:
            print("File not found")
            return
    else:
        print("Filepath not found")
        return


def readsimdatafromfile(execpath, filename, dim):
    """
    Helper function to read the data from the simulation log file. Current file structure
    
    x << y << ... << d << f(x,y,...,d) << dxf << dyf << ... << ddf << epsf << epsdxf << ... << epsddf << reached(bool)

    Parameters
    ----------
    execpath : TYPE
        DESCRIPTION.
    filname : TYPE
        DESCRIPTION.
    dim : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    path = os.path.join(execpath, filename)
    simulationdata = np.loadtxt(path)

    if simulationdata.ndim == 1:  # There is only on line in the file
        reached = int(simulationdata.reshape((1, -1))[0, -1])
        epsx = simulationdata.reshape((1, -1))[0, -2-dim]
        epsxgrad = simulationdata.reshape((1, -1))[-dim-1:-dim+1]
        ytnew = simulationdata.reshape((1, -1))[0, dim]
        ygradnew = simulationdata.reshape((1, -1))[0, dim+1:dim+dim]

    else:
        'Check if simulation reached accuracy'
        reached = int(simulationdata[-1][-1])
        epsx = simulationdata[-1][-2-dim]
        epsxgrad = simulationdata[-1][-dim-1:-dim+1]
        ytnew = simulationdata[-1][dim]
        ygradnew = simulationdata[-1][dim+1:dim+dim+1]

    return ytnew, ygradnew, epsx, epsxgrad, reached

# =============================================================================
# 'Usage'
# =============================================================================
# execpath = 'C:\\Users\\Phillip'
# execname = 'kaskadeio.py'
# parameter = { "--dump":True,
#               "--v0":3.0,
#               "--d1":1.0,
#               "--d2":0.5
#             }
# runkaskade(execpath, execname, parameter)
# =============================================================================
#
#
# data = np.loadtxt(execpath+"dump.log", dtype=float)
#
# =============================================================================
