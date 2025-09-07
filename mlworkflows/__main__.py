import argparse
import atexit
from multiprocessing import active_children

from classconfig import Config
from mlworkflows.utils.modules import load_module
from mlworkflows.utils.yaml import YAML
from mlworkflows.workflows.base import Workflow


def parse_args():
    parser = argparse.ArgumentParser(description="Text summarization benchmark")

    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser('run', help="tool for running given workflow")
    run_parser.add_argument("workflow", help="Path to workflow.", type=str)
    run_parser.add_argument("-c", "--config", help="Path to file containing workflow configuration.", type=str,
                            default=None)
    run_parser.set_defaults(func=call_run)

    make_config_parser = subparsers.add_parser('make_config', help="Makes configuration for given workflow.")
    make_config_parser.add_argument("workflow", help="Path to module file containing workflow for which the "
                                                     "configuration should be created.", type=str)
    make_config_parser.set_defaults(func=call_make_config)

    return parser.parse_args()


def call_run(args: argparse.Namespace):
    """
    Method for workflow running.

    :param args: User arguments.
    """
    Workflow.load(args.workflow, args.config)()


def call_make_config(args: argparse.Namespace):
    """
    Method for generating configuration for given workflow.

    :param args: User arguments.
    """

    module = load_module(args.workflow)
    config = Config(module.RUN)
    print(YAML().dumps(config.generate_yaml_config()))


def kill_children():
    """
    Kills all subprocesses created by multiprocessing module.
    """

    for p in active_children():
        p.terminate()


def main():
    atexit.register(kill_children)
    args = parse_args()

    if args is not None:
        args.func(args)
    else:
        exit(1)


if __name__ == '__main__':
    main()
