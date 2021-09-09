# Autonomous Pacman Player


# Report

The details and experiments data are described in the [report.ipynb](report.ipynb), to see the report run:


```sh
jupyter notebook report.ipynb
```

# Usage

Creating and activating your conda environmnet.

```sh
conda create -n pacman python=2.7
conda activate pacman
```

Installing requirements.

```sh
pip install -r requirements.txt
```

Running a simple demo.

```sh
python pacman.py --layout testMaze --pacman DumbAgent
```


## Reinforcement Learning

Check options for running a single Reinforcement Learning instance:

```sh
python rlMain.py --help
```


The [rlExperiments.py](rlExperiments.py) compares a series of parameters for each layout (**take a lot of time**), check its options with

```sh
python rlExperiments.py --help
```
