<div align="center">

# Dippy SN X: Creating The World's Best Open-Source Voice Roleplay System


[![DIPPY](/assests/banner.png)](https://dippy.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

</div>

- [Introduction](#introduction)
- [Roadmap](#roadmap)
- [Overview of Miner and Validator Functionality](#overview-of-miner-and-validator-functionality)
  - [Miner](#miner)
  - [Validator](#validator)
- [Running Miners and Validators](#running-miners-and-validators)
  - [Running a Miner](#running-a-miner)
  - [Running a Validator](#running-a-validator)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

> **Note:** The following documentation assumes you are familiar with basic Bittensor concepts: Miners, Validators, and incentives. If you need a primer, please check out https://docs.bittensor.com/learn/bittensor-building-blocks.

The Dippy Voice subnet (SN X) on Bittensor aims to create the world's best open-source voice roleplay system by leveraging the collective efforts of the open-source community. This subnet addresses the critical issue of loneliness, which affects a significant portion of the population and is linked to various mental and physical health problems. 

Our team at Impel Intelligence Inc. are looking to enhance Dippy, a proactive AI companion app. In this subnet, we will bring together the entire open-source eco-system to build the world's best voice roleplay system, focusing on voice synthesis and processing capabilities.

## Roadmap

Given the complexity of creating a state of the art voice roleplay system, we plan to divide the process into 3 distinct phases.

**Phase 1:** 
- [x] Testnet Subnet launch with robust pipeline for voice synthesis evaluation on public datasets
- [ ] Mainnet Subnet launch with voice quality and character consistency metrics
- [ ] Integration with existing roleplay capabilities from SN11

**Phase 2:** 
- [ ] Enhanced voice emotion and prosody control
- [ ] Multi-speaker voice synthesis support
- [ ] Real-time voice processing improvements

**Phase 3:** 
- [ ] Advanced voice style transfer capabilities
- [ ] Dynamic voice adaptation
- [ ] Seamless character voice switching

## Overview of Miner and Validator Functionality

![overview](/assests/model_architecture.png)

**Miners** would implement and optimize voice synthesis models to create high-quality, character-consistent voices for roleplay interactions. These models would be submitted to a shared Hugging Face pool. 

**Validators** would evaluate and assess model performance via our protocol and rank the submissions based on various metrics (human voice similarity, emotional expression, character consistency, etc). We will provide a suite of testing and benchmarking protocols with state-of-the-art voice datasets.

## Running Miners and Validators
### Running a Miner

#### Requirements
- Python 3.8+ (we recommend using [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) for managing python environments)
- GPU with at least 24 GB of VRAM

#### Setup
To start, clone the repository and `cd` to it:
```
git clone https://github.com/impel-intelligence/dippy-synthetic-speech-subnet.git
cd dippy-synthetic-speech-subnet
pip install -e .
```
#### Submitting a model
As a miner, you're responsible for developing and optimizing voice synthesis models that can produce high-quality, character-consistent voices for roleplay interactions.

We outline the following criteria for Phase 1:

- Models MUST be in Safetensors Format! Check upload_models.py for how the model upload precheck works.
- Please test the model by loading it using the appropriate voice synthesis framework
- (Recommended) Test the model with sample inputs before submitting
- Support for common voice synthesis architectures (FastSpeech2, Tacotron, etc.)
- Models should handle various speaking styles and emotions

Once you're happy with the performance of your voice synthesis model, you can submit it to Hugging Face 🤗 using the following command:

```bash
python3 dippy_subnet/upload_model.py --hf_repo_id HF_REPO --wallet.name WALLET  --wallet.hotkey HOTKEY --model_dir PATH_TO_MODEL   
```


### Running a Validator

#### Requirements
- Python 3.9+
- API key for `wandb` (see below)

## Setup WandB (HIGHLY RECOMMENDED - VALIDATORS PLEASE READ)

Before running your validator, it is recommended to set up Weights & Biases (`wandb`). 
The purpose of `wandb` is for tracking key metrics across validators to a publicly accessible page.
[here](https://wandb.ai/dippyai/). 
We ***highly recommend***
validators use wandb, as it allows subnet developers and miners to diagnose issues more quickly and
effectively, say, in the event a validator were to be set abnormal weights. Wandb logs are
collected by default, and done so in an anonymous fashion, but we recommend setting up an account
to make it easier to differentiate between validators when searching for runs on our dashboard. If
you would *not* like to run WandB, you can do so by not providing the flag `--wandb-key` when running your
validator.

Before getting started, as mentioned previously, you'll first need to
[register](https://wandb.ai/login?signup=true) for a `wandb` account, and then set your API key on
your system. Here's a step-by-step guide on how to do this on Ubuntu:

#### Step 1: Installation of WANDB

Before logging in, make sure you have the `wandb` Python package installed. If you haven't installed
it yet, you can do so using pip:

```bash
# Should already be installed with the repo
pip install wandb
```

#### Step 2: Obtain Your API Key

1. Log in to your Weights & Biases account through your web browser.
2. Go to your account settings, usually accessible from the top right corner under your profile.
3. Find the section labeled "API keys".
4. Copy your API key. It's a long string of characters unique to your account.

#### Step 3: Setting Up the API Key in Ubuntu

To configure your WANDB API key on your Ubuntu machine, follow these steps:

1. **Log into WANDB**: Run the following command in the terminal:

   ```bash
   wandb login
   ```

2. **Enter Your API Key**: When prompted, paste the API key you copied from your WANDB account
   settings. 

   - After pasting your API key, press `Enter`.
   - WANDB should display a message confirming that you are logged in.

3. **Verifying the Login**: To verify that the API key was set correctly, you can start a small
   test script in Python that uses WANDB. If everything is set up correctly, the script should run
   without any authentication errors.

4. **Setting API Key Environment Variable (Optional)**: If you prefer not to log in every time, you
   can set your API key as an environment variable in your `~/.bashrc` or `~/.bash_profile` file:

   ```bash
   echo 'export WANDB_API_KEY=your_api_key' >> ~/.bashrc
   source ~/.bashrc
   ```

   Replace `your_api_key` with the actual API key. This method automatically authenticates you with
   wandb every time you open a new terminal session.



#### Setup
To start, clone the repository and `cd` to it:
```
git clone https://github.com/impel-intelligence/dippy-bittensor-subnet.git
cd dippy-bittensor-subnet
pip install -e .
```
To run the evaluation, simply use the following command:

``` 
python neurons/validator.py --wallet.name WALLET_NAME --wallet.hotkey WALLET_HOT_NAME --wandb-key WANDBKEY
```

To run auto-updating validator with PM2 (recommended):
```bash
pm2 start --name sn11-vali-updater --interpreter python scripts/start_validator.py -- --pm2_name sn11-vali --wallet.name WALLET_NAME --wallet.hotkey WALLET_HOT_NAME [other vali flags]
```

Please note that this validator will call the model validation service hosted by the dippy subnet owners. If you wish to run the model validation service locally, please follow the instructions below.


### Running the model evaluation API (Optional)

**Note**: Currently (June 17 2024) there are some issues with the local evaluation api. We recommend using the remote validation api temporarily.

Starting a validator using your local validator API requires starting validator with `--use-local-validation-api` flag. 
Additionally, a model queue is required to push models to the validation api.

**Note**: Validator API needs to be installed in a different venv than validator due to `pydantic` version conflict. 


### Requirements
- Python 3.9+
- Linux

#### Setup

Install Git Lfs if not installed.
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```

If you are running on runpod you might also need to install 'netstat'.
```bash
apt-get install net-tools
```

To start, clone the repository and `cd` into it:
```bash
git clone https://github.com/impel-intelligence/dippy-bittensor-subnet.git
cd dippy-bittensor-subnet
python3 -m venv model_validation_venv
source model_validation_venv/bin/activate
model_validation_venv/bin/pip install -e . --no-deps
model_validation_venv/bin/pip install -r requirements_val_api.txt
```

#### Run model validation API service
(Note: there are currently breaking changes that pose challenges to running a local validation API service)
```bash
cd dippy_validation_api
chmod +x start_validation_service.sh
./start_validation_service.sh
```

### Test that it's working
```bash
python3 test_api.py
```
And you should see a json showing that the model status is "QUEUED"
Running the same command again for sanity's sake, you should see the status of the model as "RUNNING".


#### Stop model validation API service
```bash
chmod +x kill_validation_api.sh
./kill_validation_api.sh
```

#### Running the validator with your own validation API service running locally (optional)
```bash
# Make a separate venv for the validator because of pydantic version conflict
python -m venv validator_venv
validator_venv/bin/pip install -e .
validator_venv/bin/python neurons/validator.py --wallet.name WALLET_NAME --wallet.hotkey WALLET_HOT_NAME --use-local-validation-api
# Run model queue to push models to validation api to be evaluated
validator_venv/bin/python neurons/model_queue.py --use-local-validation-api
```
## Model Evaluation Criteria

The current model evaluation is based on a proprietary process which will be slowly released to the public as we develop better safeguards around incentive gaming.



## License

The Dippy Bittensor subnet is released under the [MIT License](./LICENSE).
