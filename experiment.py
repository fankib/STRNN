import subprocess
import shlex
import numpy as np

''' This script is used to invoke and report experiment
	Before you can run the experiments you have to
	prepare the datasets using the prepare_dataset_60 function!
 '''

def prepare_dataset_60():
	''' prepares the 60 user dataset.
		Run once before the experiments.
	 '''
	window_widths = [60, 360, 1440, 10080]
	for ww in window_widths:
		args = '--max-users 60 --suffix 60-{} --ww {}'.format(ww, ww)
		e = Evaluator()
		e.preprocess(args)

def experiment_60():
	''' here is all the stuff from the 60 - user experiment '''
	
	# run once:
	#prepare_dataset_60()
	
	# select the experiments to run:
	#exp_window_width('60')
	#exp_hidden_dims('60', 360)
	#exp_hidden_dims('60', 60)
	#exp_learning_rate('60', 360, 7)
	#exp_epochs('60', 360, 7, 0.05)
	#exp_regularization('60', 360, 7, 0.05)
	#exp_temporal_upper_bound('60', 360)
	#exp_spatial_upper_bounds('60', 360)

def exp_window_width(suffix):
	window_widths = [60, 360, 1440, 10080]
	for ww in window_widths:
		args = ArgBuilder().suffix('{}-{}'.format(suffix, ww)).up_time(ww).lr(0.01)
		args.evaluate()

def exp_hidden_dims(suffix, window_width):
	dims = np.arange(1, 20, 2)
	for dim in dims:
		args = ArgBuilder().suffix('{}-{}'.format(suffix, window_width)).up_time(window_width).dims(dim)
		args.evaluate()

def exp_learning_rate(suffix, window_width, dims):
	learning_rates = [0.005, 0.01, 0.05, 0.1]
	for lr in learning_rates:
		args = ArgBuilder().suffix('{}-{}'.format(suffix, window_width)).up_time(window_width).lr(lr).dims(dims)
		args.evaluate()

def exp_epochs(suffix, window_width, dims, lr):
	# first round:
	args = ArgBuilder().suffix('{}-{}'.format(suffix, window_width)).up_time(window_width).dims(dims).lr(lr)
	args.evaluate()
	
	# next rounds
	for i in range(12):
		args = ArgBuilder().continue_train().suffix('{}-{}'.format(suffix, window_width)).up_time(window_width).dims(dims).lr(lr)
		args.evaluate()

def exp_regularization(suffix, window_width, dims, lr):
	regularizations = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.]
	for reg in regularizations:
		args = ArgBuilder().suffix('{}-{}'.format(suffix, window_width)).up_time(window_width).dims(dims).lr(lr).reg(reg)
		args.evaluate()
		
def exp_temporal_upper_bound(suffix, window_width):
	upper_bound_scales = [1., 1.5, 2., 4., 10.]
	for scale in upper_bound_scales:
		up_time = window_width * scale
		args = ArgBuilder().suffix('{}-{}'.format(suffix, window_width)).up_time(up_time)
		args.evaluate()

def exp_spatial_upper_bounds(suffix, window_width):
	upper_bounds = [0.01, 0.1, 1., 2., 10., 40., 100.] # 10m, 100m, 1Km, 2Km, 10Km, 40Km, 100Km
	for up_dist in upper_bounds:
		args = ArgBuilder().suffix('{}-{}'.format(suffix, window_width)).up_dist(up_dist)
		args.evaluate()
		
class Evaluator():
	
	def __init__(self):
		self.args = []
		self.processes = []
	
	def preprocess(self, args):
		cmd = 'python preprocess.py {}'.format(args)
		print('preprocess using {}'.format(args))
		cmds = shlex.split(cmd)
		p = subprocess.Popen(cmds)
		return p.wait()

	def train(self, args):
		self.args.append(args)
		cmd = 'python train.py {}'.format(args)
		print('train on {}'.format(args))
		cmds = shlex.split(cmd)
		#self.processes.append(subprocess.Popen(cmds)) # verbose
		self.processes.append(subprocess.Popen(cmds, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

	def evaluate(self):
		exit_codes = [p.wait() for p in self.processes]
		self.do_test_best()
		
	def do_test_best(self):
		ps = []
		for args in self.args:
			cmd = 'python train.py --test --best --silent {}'.format(args)
			print('test best on {}'.format(args))
			cmds = shlex.split(cmd)
			p = subprocess.Popen(cmds) 
			p.wait()
		print('======================================')
	
class ArgBuilder():
	''' Prepares the command line arguments to run the experiments '''
	
	def __init__(self):
		self.args = []
		self.cpu()
	
	def cpu(self):
		self.args.append('--cpu')
		return self
	
	def silent(self):
		self.args.append('--silent')
		return self
	
	def continue_train(self):
		self.args.append('--continue-train')
		return self
	
	def suffix(self, arg):
		self.args.append('--suffix {}'.format(arg))
		return self
	
	def up_time(self, arg):
		self.args.append('--up-time {}'.format(arg))
		return self
	
	def up_dist(self, arg):
		self.args.append('--up-dist {}'.format(arg))
		return self
	
	def epochs(self, arg):
		self.args.append('--epochs {}'.format(arg))
		return self
	
	def dims(self, arg):
		self.args.append('--dims {}'.format(arg))
		return self
	
	def lr(self, arg):
		self.args.append('--lr {}'.format(arg))
		return self
	
	def reg(self, arg):
		self.args.append('--reg {}'.format(arg))
		return self
	
	def build(self):
		return ' '.join(self.args)
	
	def evaluate(self):
		args = self.build()
		e = Evaluator()
		e.train(args)
		e.evaluate()	


if __name__ == '__main__':
	experiment_60()
