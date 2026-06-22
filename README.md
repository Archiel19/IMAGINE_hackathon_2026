# IMAGINE Hackathon 2026

How much energy does it *really* take to train a competent vision model?

In this repo, we train a ViT-S/16 for image classification on the ImageNet-1k dataset.

The goal is to achieve 85% top-5 accuracy on our test set using as little energy as possible.

Our baseline trains for 6h40 on a single A6000 GPU, reaching the target in 19 epochs. The baseline is already partially optimized with mixed precision and model compilation, but we can for sure do better!

### Index
- Before you Start
    - Dataset
    - CodeCarbon
    - ElectricityMaps
- Training
- Evaluation
- Project Structure
- Defining Experiments
    - Hyperparameter Search

## Before you Start

We need to set up some things before we start training. First clone the repo, and then do the following:

### Dataset

To download our split of ImageNet-1k, run this from the root directory of the repo:
```bash
chmod +x download_imagenet.sh; ./download_imagenet.sh
```
The download should take around 20 minutes. After it finishes, you should find the `train` and `val` partitions under the `data/` folder.

### CodeCarbon

We are going to use CodeCarbon to measure the energy consumption and carbon emissions of our training runs.
CodeCarbon has an API that we can use to upload all the measurements to their dashboard, but to do this we all need to have an account and that CodeCarbon can authenticate to the API. Step by step:
1) Go to https://dashboard.codecarbon.io/ > `Sign in or create an account`
2) Click `Register` at the bottom, fill and submit the form with your ENPC/uni email
3) Send me (Marta López) your email address on Slack so that I can add you to our organization
4) On your machine, install and configure CodeCarbon
    ```bash
    pip install codecarbon  # Or use your favorite package management tool
    codecarbon login  # Will open a browser window to authenticate you
    codecarbon config  # Select: organization -> IMAGINE; project -> IMAGINE Hackathon 2026; experiment -> Baseline
    ```

### ElectricityMaps
Also, to get more accurate carbon emission measurements, CodeCarbon can use the ElectricityMaps API. For this to work, follow these steps:
1) Go to https://app.electricitymaps.com/auth/signup
2) Enter your ENPC/uni email and click `Sign up`, then follow the instruction to complete your registration
3) Scroll down to the `Resources` section > `API Key`
4) Create a new API key and copy it to your clipboard
5) Write the plain API key to `./codecarbon/electricity_maps_key.txt`:
    ```bash
    echo <your API key> >> ./codecarbon/electricity_maps_key.txt
    ```


### Experiment Tagging
To keep things tidy, please define the following in the [train config](./configs/train.yaml).
- `team_name`: name of your team. Can be a single letter, the team leader's name, or a short name, as long as it is the same for all team members.
- `experiment_name`: name of the approach you are trying out. Note: the logs are timestamped, so running the same experiment several times will not overwrite previous output.
- `tags`: a *project stage* and a *run type* tag. You will find the available options as a comment in the [train config](./configs/train.yaml).
Adapt `experiment_name` and `tags` as necessary for each experiment.


## Training

Run:
```bash
uv run src/train.py
```
with any Hydra command-line overrides that you need for your experiment.

## Evaluation

Once the test set has been released (on July 2nd), run:
```bash
uv run src/eval.py
```
This will register the output of the model for each image in the test set and upload the results to the evaluation server. We will only reveal the test performance after everyone has submitted their results.

## Project Structure

This repo is based on the [Lightning + Hydra template by ashleve](https://github.com/ashleve/lightning-hydra-template).

There are three main components that you can play around with:
- The [`datamodule`](./configs/datamodule/) reads the raw data, applies data augmentation, collates batches, and sends them to the GPU.
- The [`module`](./configs/module/) defines the network, schedulers, and optimizers and controls what happens in each training step.
- The [`trainer`](./configs/trainer/) configures some global training options, such as the total number of epochs, gradient clipping settings, or the use of mixed precision.

So you will be working mainly with those three configs, as well as [`experiment`](./configs/experiment/), [`hparams_search`](./configs/hparams_search) (see next sections), and maybe [`callbacks`](./configs/callbacks/).

## Defining Experiments

To define a new experiment, create a YAML file under [`configs/experiment`](./configs/experiment).

If needed, override the `module`, `datamodule` or `trainer` configs as done in the [example](configs/experiment/example.yaml) with your own config files defined in the `module`, `datamodule` or `trainer` directories under `./configs`, respectively.

If needed, you may also define new versions of [`imagenet_datamodule.py`](./src/datamodules/imagenet_datamodule.py) and [`imagenet_module.py`](./src/modules/imagenet_module.py) in the corresponding directories under `src`.

> Don't forget to specify `experiment_name`, and `tags` in the experiment config!

### Debugging

TODO

### Hyperparameter Search

TODO

You can launch a sequential hyperparameter search by specifying a `hparams_search` config in your experiment.

> Don't forget to use the `hyperparam` tag in the tags field of the config!
