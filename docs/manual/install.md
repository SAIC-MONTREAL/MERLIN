(installation)=
# Installation

<!-- If you only need the network simulation environments, you can use the pip installation method, which installs Merlin from the Python package repository.

If you would like to run experiments using the configurations and solvers we provide, then you must install Merlin from source. This will provide you with training and evaluation scripts, our hydra configuration files, and our algorithm implementations.

## Installing using pip

The ```pip``` installation is simple: You just need to execute the follwing command:
```pip install OpenMerlinCopybara```

To test your installation, start a python shell and run
```from saic5g.envs.classic_mlb import A3BasedMLB``` -->

The installation instructions here focus Unix operating systems (i.e. Linux, MacOS). Installation on Windows should be possible, but we have not tested it.

## 1. Clone the repository

Clone the MERLIN repository from source:

`git clone https://github.sec.samsung.net/jimmy-li/OpenMerlinCopybaraTest.git`

## 2. Install dependencies

You can optionally create and activate a virtual environment to install the required dependencies. This allow you to avoid version conflicts.

### Create and activate a virtual environment (Optional)
Here we provide a brief overview of how to set up a virtual environment on Linux and MacOS. For more details, and for information pertaining to Windows, please see [Python's documentation](https://docs.python.org/3/library/venv.html). 

#### Create virtual environment

A virtual environment allows the Python interpreter and installed libraries to be isolated from the rest of the system. This allows one to more easily manage Python packages and dependencies and avoid collision errors and version conflicts.

Go to your project root directory and type the below command to create a virtual environment:

```python3 -m venv merlinenv```

This creates a “merlinenv” directory in your current working directory.

% Windows

% Windows users can use the below command,

% ```py -m venv merlinenv``` --->

#### Activate the environment

After the creation of the the virtual environment, you need to activate it with

```$ source merlinvenv/bin/activate```

This needs to be run each time you start a new shell or terminal. On Linux, you can add this line to your .bashrc file so that it runs whenever you start a new terminal.

% Windows

% Windows users move inside the “merlinenv” folder and type the below command,

% .\env\Scripts\activate

### Installing required libraries

Merlin has many dependencies. All the required libraries and their versions are specified in the ```requirement.txt``` file.
To install these dependencies, execute the following command:

```$ python3  install -r requirement.txt```

## 3. Add MERLIN to your python path

You need to add MERLIN's Python source folder to your python path in order to run MERLIN's scripts. Run the following command, and be sure to replace ```path_to_merlin``` with the path at which you cloned Merlin.

```export PYTHONPATH="${PYTHONPATH}:/path_to_merlin/src/py```

On Linux, you can add this line to your .bashrc file so that it runs whenever you start a new terminal.

## 4. Sanity check

Once the requirements are installed, we can perform a sanity check to verify that Merlin is installed properly. We can evaluate the baseline method of Mobility Load Balancing (MLB) using the following command:

```python bin/py/evaluate.py solver=qsim_default env=classic_mlb eval_dir=logs/evals/test evaluator=classic_mlb evaluator.mlb_evaluator.n_trials=2```

(Note that if you installed MERLIN within a virtual environment you can deactivate the virtual environment by typing ```$ deactivate```).