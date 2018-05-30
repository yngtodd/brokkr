import os
import argparse
import subprocess


def run_nvprof(program, path, filename, single_p=False, half_p=False):
    """Run Nvidia's profiler"""
    if single_p and half_p: 
        logfile = subprocess.Popen(
          ['nvprof', '--metrics', 'flop_count_sp', 'flop_count_hp', \
           '--logfile', logname, 'python', program]
        )
    elif single_p:
        logfile = subprocess.Popen(
          ['nvprof', '--metrics', 'flop_count_sp', \
           '--logfile', logname, 'python', program]
        )
    elif half_p:
        logfile = subprocess.Popen(
          ['nvprof', '--metrics', 'flop_count_hp', \
           '--logfile', logname, 'python', program]
        )
    else:
        raise ValueError("Must specify at least one measure to profile! "
                         "Single precision: {}, Half precision: {}".format(single_p, half_p))
                         
    filepath = os.path.join(path, filename)

    with open(filepath, 'w') as f:
        f.write(logfile)

    return logfile


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--program', type=str,
                        help='Name of python file to be profiled')
    parser.add_argument('--path', type=str,
                        help='Path to save the file to (excluding file name)')
    parser.add_argument('--filename', type=str,
                        help='Name for the nvprof output file')
    parser.add_argument('--sp', action='store_true',
                        help='count single precision floating point operations')
    parser.add_argument('--hp', action='store_true',
                        help='count half precision floating point operations')
    args = parser.parse_args()

    logfile = run_nvprof(
      program=args.program, path=args.path, filename=args.filename, single_p=args.single_p, half_p=args.half_p
    )
