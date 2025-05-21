FASTv2
============================

## Setup

#### Requirements
  * Python == 3.8
  * conda (recommended)

#### Install dependencies

The dependencies can be installed using the following command (conda required):
```bash
bash install.sh
```

**NOTE**: Please ensure that you have the repository for [tfbio](https://gitlab.com/cheminfIBB/tfbio) available for building from source before proceeding with the installation.

# Preparing Datasets

## Creating/linking datasets

### If dataset is available in (for example: /usr/worksapce/atom/)

Create a symbolic links in <a href='./data'>data</a> to the directory where the datasets are stored.<br>
```bash
ln -s /usr/workspace/atom/pdbbind_ml/pdbbind2016 pdbbind2016
ln -s /usr/workspace/atom/ml_fusion/v2020_final pdbbind2020
ln -s /usr/workspace/atom/glo_spl/fusion/general_distributed_torch/1_generate_baseline_dataset mpro
ln -s /usr/workspace/atom/glo_spl/affinity_files/fast2_test gmd
```

### If dataset is not available, but raw docking files are present in folder

```bash
  python RawHDF5.py --dir=${DATA_DIR} --dataset=${DATASET} --sub-dataset=${SUBDATASET} --output-dir=${OUTPUTDIR} --pocket-type=${POCKET}
```

You may also run the following command:

```bash
  sbatch scripts/sbatch_dataset_curate -d ${DATASET} -s ${SUBDATASET} -o ${OUTPUTDIR} -p ${POCKET_TYPE}
```

**Note:**
1. Make sure all the files are in ```DATA_DIR```
2. This feature is still under construction.

Hierarchial layout of ```DATA_DIR``` is shown below:

```bash
dengue (dataset)
├── denv2 (subset)
│   └── 2fom (pocket)
│      └── scratch
│          ├── dockHDF5
│          │   ├── dock_proc1.hdf5
│          │   └── ...
│          └── receptor.hdf5
└── denv3
    └── 3u1i
      └── scratch
          ├── dockHDF5
          │   ├── dock_proc1.hdf5
          │   └── ...
          └── receptor.hdf5

```
**Note:** Not all datasets will have a subset. This is an optional argument.

## Training
There are sample scripts in <a href='./scripts'>scripts</a> directory.<br>
A training job can be submitted with the following command.
```bash
sbatch scripts/sbatch_train.sh -c ${CONFIG_PATH} -t ${TAG} -d ${SAVE_DIR} 
```

If you are running on a local or interactive node, you can train with the command below.

```bash
python train.py --config_path=${CONFIG_PATH} --tag=${TAG}
```

Sample configurations can be found in the <a href='./configs'>configs</a> directory.

**NOTE**: You do not need to specify the training directory.<br>
If run as above, a unique identifiable directory will be created in `/usr/workspace/$USER` with `$TAG`.<br>
Of course, you can specify the save directory by adding `--save_dir=${DIR}` flag.


#### Resuming the previous training
If you want to resume the existing training, you only need to specify the existing directory through the `--save_dir=${DIR}` flag.
```bash
python train.py --save_dir=${DIR}
```


## Evaluation

You can evaluate your trained model using the following command.
```bash
python eval.py --save_dir=${DIR}
```

**NOTE**: If you want to use a different configuration from the one previously used for training, you can specify through the `--config_path=${CONFIG_PATH}` flag.


## Contact
Please contact Aditya Ranganath ([ranganath2@llnl.gov](mailto:ranganath2@llnl.gov)) if you have any request.
