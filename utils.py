from __future__ import division, print_function
import time
import os
import numpy as np
from argparse import ArgumentParser

class Config:

    def __init__(self, argv):
        parser = ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--start_epsilon", type=float, default=1.0)
        parser.add_argument("--end_epsilon", type=float, default=0.1)
        parser.add_argument("--annealing_steps", type=int, default=10000)
        parser.add_argument("--alpha", type=float, default=0.2)
        parser.add_argument("--num_training_step", type=int, default=2000)
        parser.add_argument("--num_pretrain_step", type=int, default=4000)
        parser.add_argument("--tau", type=float, default=0.001)
        parser.add_argument("--target_update_freq", type=int, default=4)
        parser.add_argument("--online_update_freq", type=int, default=4)
        parser.add_argument("--save_model_freq", type=int, default=50)
        parser.add_argument("--replay_buffer_size", type=int, default=100000)
        parser.add_argument("--output_dir", type=str, default="output/deepq")
        parser.add_argument("--from_pretrain", type=str, default=None)
        parser.add_argument("--continue_from", type=str, default=None)

        self.args = parser.parse_args(argv)
        output_dir = self.args.output_dir

        self.episode = 0
        self.total_step = 0
        
        if(self.args.continue_from != None):
            argv = self.load()
            self.args = parser.parse_args(argv)
            self.args.output_dir = output_dir

        self.save()
        self.print_arguments()

    def save(self):
        if(not os.path.exists(self.args.output_dir)):
            os.makedirs(self.args.output_dir)
        
        with open(self.args.output_dir + "/parameters.txt", 'w') as file:
            for arg in sorted(vars(self.args)):
                if(arg != "output_dir" and arg != "from_pretrain" and arg != "continue_from"):
                    file.write("--" + arg + " " + str(getattr(self.args, arg)) + "\n")
            file.write("episode " + str(self.episode) + '\n')
            file.write("total_step " + str(self.total_step) + '\n')
    def load(self):

        with open(self.args.continue_from + "/parameters.txt", 'r') as file:
            argv = []
            for line in file:
                arg, value = line.strip().split(' ')
                if(arg == "episode"):
                    self.episode = int(value)
                elif(arg == "total_step"):
                    self.total_step = int(value)
                else:
                    argv.extend([arg, value])
        return argv

    def print_arguments(self):
        """Print argparse's arguments.
        Usage:
        .. code-block:: python
            parser = argparse.ArgumentParser()
            parser.add_argument("name", default="Jonh", type=str, help="User name.")
            args = parser.parse_args()
            print_arguments(args)
        :param args: Input argparse.Namespace for printing.
        :type args: argparse.Namespace
        """
        print("-----------  Configuration Arguments -----------")
        for arg in sorted(vars(self.args)):
            print(arg, getattr(self.args, arg))
        print("------------------------------------------------")