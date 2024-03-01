# Add paths for importing modules

from os.path import join
import sys

dir_analysis = join("..", "analysis")
sys.path.append(dir_analysis)

dir_data = join("..", "data")
sys.path.append(dir_data)

dir_model = join("..", "model")
sys.path.append(dir_model)
