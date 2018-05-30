import os
import argparse
import subprocess


def run_nvprof(program, path, filename, single_p=False, half_p=False):
    """Run Nvidia's profiler"""
    print('Running Nvidia Profiler!')
    if single_p and half_p:
        logfile = subprocess.call(
          ['nvprof', '--metrics', 'flop_count_sp', 'flop_count_hp',
           '--log-file', filename, 'python', program], shell=True
        )
    elif single_p:
        logfile = subprocess.call(
          ['nvprof', '--metrics', 'flop_count_sp',
           '--log-file', filename, 'python', program], shell=True
        )
    elif half_p:
        logfile = subprocess.call(
          ['nvprof', '--metrics', 'flop_count_hp',
           '--log-file', filename, 'python', program], shell=True
        )
    else:
        raise ValueError("Must specify at least one measure to profile! "
                         "Single precision: {}, Half precision: {}".format(single_p, half_p))
    print('finished profiling')
    print(logfile)
    filepath = os.path.join(path, filename)

    with open(filepath, 'w') as f:
        f.write(logfile)

    return logfile


def main():
    parser = argparse.ArgumentParser(description="Running Nvidia's profiler.")
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
    print('kicking off')

    logfile = run_nvprof(
      program=args.program, path=args.path, filename=args.filename, single_p=args.sp, half_p=args.sp
    )


if __name__=="__main__":
    main()
