# Code for LOCI estimator for bivariate causal model identification

## Setup
Requires `python>=3.8`. Install dependencies with
```
pip install -r requirements.txt
```

## Reproduce All Results 
First, create all necessary commands to be executed with
```
python commands/jobs_generate.py > commands/jobs.sh
python commands/jobs_estimator_generate.py > commands/jobs_estimator.sh
```
Create the result directory `mkdir results`.
All commands written to `commands/jobs.sh` and `commands/jobs_estimator.sh` need to be executed and can be run entirely in parallel.
The results will be written to `results/`.

## Run Individual Experiments
Instead of re-running the entire set of jobs, one can run individual causal pair benchmark pairs as follows
```
python run_individual.py --pair_id PAIR_ID --config configs/ANs_standardized.gin
```
The results will be printed after the run.

To run an individual estimator benchmark case, run
```
python run_estimator_benchmark.py --seed SEED --knot_heuristic {sqrt|lin}
```

## Produce Figures and Tables
First, create a directory to store the figure PDFs with `mkdir paper_figures`.
Then run 
```
python generate_figures_and_tables.py
```
The resulting figures will be saved to the created directory.
The overall latex result tables are printed to the command line output.

## Repository structure
```
├── causa  
│   ├── datasets.py  # bivariate causal data sets loaded from /data
│   ├── het_ridge.py  # implementation of our convex LSNM estimator
│   ├── hsic.py  # independence test
│   ├── iterative_fgls.py  # baseline LSNM estimator
│   ├── ml.py  # neural network estimators
│   └── utils.py  # neural network utilities
├── commands
│   ├── jobs_estimator_generate.py  # generate bash commands for estimator benchmark
│   └── jobs_generate.py  # generate bash commands for causal pairs
├── configs  # configs for causal pairs
├── baseline_results  # results for baseline methods (GRCI, QCCD, etc.)
├── data  # data sets
├── requirements.txt  # pip requirements
├── run_estimator_benchmark.py  # estimator benchmark script
├── run_individual.py  # causal pair benchmark script
└── generate_figures_and_tables.py  # create PDF figures and print tex tables
```
