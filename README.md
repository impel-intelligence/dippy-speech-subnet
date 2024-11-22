<div align="center">

# Dippy Empathetic Speech Subnet
**Creating the World’s Best Open-Source Speech Model on Bittensor**



*Check out the beta version of our [Front-End](https://www.dippyspeech.com/)!*

[![DIPPY](/assets/animation2.gif)](https://www.dippyspeech.com/)

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

Dippy is one of the world's leading AI companion apps with **1M+ users**. The app has ranked [**#3 on the App Store**](https://x.com/angad_ai/status/1850924240742031526) in countries like Germany, been covered by publications like [**Wired magazine**](https://www.wired.com/story/dippy-ai-girlfriend-boyfriend-reasoning/) and the average Dippy user **spends 1+ hour on the app.** 

The Dippy team is also behind [Bittensor's Subnet 11](https://github.com/impel-intelligence/dippy-bittensor-subnet), which exists to create the world's best open-source roleplay LLM. Open-source miner models created on Subnet 11 are used to power the Dippy app. We also plan to integrate the models created from this speech subnet within the Dippy app. 

The Dippy Empathetic Speech Subnet on Bittensor is dedicated to developing the world’s most advanced open-source Speech model for immersive, lifelike interactions. By leveraging the collaborative strength of the open-source community, this subnet meets the growing demand for genuine companionship through a speech-first approach. Our objective is to create a model that delivers personalized, empathetic speech interactions beyond the capabilities of traditional assistants and closed-source models. 

Unlike existing models that depend on reference speech recordings that limit creative flexibility, we use natural language prompting to manage speaker identity and style. This intuitive approach enables more dynamic and personalized roleplay experiences, fostering deeper and more engaging interactions.

![DIPPY](/assets/banner9.png)

## Roadmap

Given the complexity of creating a state of the art speech model, we plan to divide the process into 3 distinct phases.

**Phase 1:** 
- [ ] Launch a subnet with a robust pipeline for roleplay-specific TTS models, capable of interpreting prompts for speaker identity and stylistic speech description.
- [ ] Launch infinitely scaling synthetic speech data pipeline
- [ ] Implement a public model leaderboard, ranked on core evaluation metric
- [ ] Introduce Human Likeness Score and Word Error Rate as live evaluation criteria for ongoing model assessment.

**Phase 2:** 
- [ ] Refine TTS models toward producing more creatively expressive, highly human-like speech outputs.
- [ ] Showcase the highest-scoring models and make them accessible to the public through the front-end interface.

**Phase 3:** 
- [ ] Advance toward an end-to-end Speech model that seamlessly generates and processes high-quality roleplay audio. 
- [ ] Establish a comprehensive pipeline for evaluating new Speech model submissions against real-time performance benchmarks.
- [ ] Integrate the Speech model within the Dippy app
- [ ] Drive the state of the art in Speech roleplay through iterative enhancements and ongoing data collection.

## Overview of Miner and Validator Functionality

![overview](/assets/architecturenew.png)

**Miners** would use existing frameworks to fine tune models to improve upon the current SOTA open-source TTS model. The finetuned weights would be submitted to a shared Hugging Face pool. 

**Validators** would evaluate and assess model performance via our protocol and rank the submissions based on various metrics (e.g. how natural it sounds, emotion matching, clarity etc.). We will provide a suite of 
testing and benchmarking protocols with state-of-the-art datasets.


## Running a Miner to Submit a Model

### Requirements
- Python 3.8+
- GPU with at least 24 GB of VRAM

### Step 1: Setup
To start, clone the repository and `cd` into it:

```bash
git clone https://github.com/impel-intelligence/dippy-speech-subnet.git
cd dippy-speech-subnet
pip install -e .
```
### Step 2: Submitting a model
As a miner, you're responsible for leveraging all methods available at your disposal to finetune the provided base model.

We outline the following criteria for Phase 1:

- Models should be a fine-tune of the 880M Parler-TTS model.
- Models MUST be Safetensors Format!
- **Model**: We currently use [Parler TTS Mini v1 on Hugging Face](https://huggingface.co/parler-tts/parler-tts-mini-v1) as our base model.

Once you're happy with the performance of the model for the usecase of Roleplay, you can simply submit it to Hugging Face 🤗 and then use the following command:

```bash
python neurons/miner.py \
    --repo_namespace REPO_NAMESPACE \  # Replace with the namespace of your repository (e.g., parler-tts)
    --repo_name REPO_NAME \            # Replace with the name of your repository (e.g., parler-tts-mini-v1)
    --config_template CONFIG_TEMPLATE \  # Replace with the miner configuration template (e.g., default)
    --netuid NETUID \                  # Replace with the unique network identifier (e.g., 231)
    --subtensor.network NETWORK \      # Replace with the network (e.g., test or finney)
    --online ONLINE \                  # Set to True to enable mining
    --model_hash MODEL_HASH \          # Replace with the hash of your model
    --wallet.name WALLET_NAME \        # Replace with the name of your wallet coldkey name
    --wallet.hotkey HOTKEY \           # Replace with your wallet hotkey name
    --wallet.path WALLET_PATH \        # Replace with the path to your wallet directory (e.g.,  "~/.bittensor/wallets/" )
    --logging.debug DEBUG              # Set to True for debug logging (or False for production)
```
### Example

```bash
python neurons/miner.py \    
   --repo_namespace parler-tts  \   
   --repo_name parler-tts-mini-v1     
   --config_template default \    
   --netuid 231 \    
   --subtensor.network test   \  
   --online True  \   
   --model_hash 555    \ 
   --wallet.name coldkey2    \ 
   --wallet.hotkey hotkey2     \
   --wallet.path "~/.bittensor/wallets/"  \ 
   --logging.debug True
```


## Running a Validator

#### Requirements
- Python 3.9+
- [UV python package manager](https://pypi.org/project/uv/)



#### Setup
To start, clone the repository and `cd` to it:
```bash
git clone https://github.com/impel-intelligence/dippy-speech-subnet.git
cd dippy-speech-subnet

uv venv .validator
source .validator/bin/activate

uv pip install -r requirements.validator.txt
uv pip install -e .
```
To run the evaluation, simply use the following command:

```bash
python neurons/validator.py \
    --wallet.name WALLET_NAME \           # Replace with the name of your wallet coldkey (e.g., coldkey4)
    --wallet.hotkey HOTKEY \              # Replace with your wallet hotkey name (e.g., hotkey4)
    --device DEVICE \                     # Replace with the device to use (e.g., cpu or cuda)
    --netuid NETUID \                     # Replace with the unique network identifier (e.g., 231)
    --subtensor.network NETWORK \         # Replace with the network name (e.g., test or finney)
    --wallet.path WALLET_PATH             # Replace with the path to your wallet directory (e.g., "~/.bittensor/wallets/")

```

```bash
 python neurons/validator.py \ 
   --wallet.name coldkey4 \
   --wallet.hotkey hotkey4 \
   --device cuda \
   --netuid 231  \
   --subtensor.network finney \
   --wallet.path "~/.bittensor/wallets/" 
 
```

To run auto-updating validator with PM2 (recommended):
```bash
pm2 start --name sn231-vali-updater --interpreter python scripts/start_validator.py -- --pm2_name sn231-vali --wallet.name WALLET_NAME --wallet.hotkey WALLET_HOT_NAME [other vali flags]
```

Please note that this validator will call the model validation service hosted by the dippy subnet owners. If you wish to run the model validation service locally, please follow the instructions below.





## Running the model evaluation API (Experimental)

**Note**: Currently (Novembar 22 2024) this is experimental. We recommend using the remote validation api temporarily.

Starting a validator using your local validator API requires starting validator with `--use-local-validation-api` flag. 
Additionally, a "model_queue"  and "worker" is required to push models to the validation api.

**Note**: Validator API needs to be installed in a different venv than validator due to `pydantic` version conflict. 


### Requirements
- Python 3.9+
- Linux
- [UV python package manager](https://pypi.org/project/uv/)

### Setup

Install Git Lfs if not installed.
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```

If you are running on runpod you might also need to install 'netstat'.
```bash
apt-get install net-tools
```

### Step 1 - Run Worker

Build Evaluator Image
```bash
cd dippy-speech-subnet
docker build -f evaluator.Dockerfile -t speech .
```

Build Worker Image
```bash
docker build -f worker.Dockerfile -t worker-image .
```

Run Worker Image
```bash
docker run -d --name worker-container -v /var/run/docker.sock:/var/run/docker.sock worker-image
```

Stream Logs:
```bash
docker logs -f worker-container
```

### Step 2 - Run Model Queue

Build Model Queue Image:
```bash
docker build -f modelq.Dockerfile -t modelq-image .
```

Run Model Queue Image:
```bash
docker run -d --name modelq-container modelq-image
```

Stream Logs:
```bash
docker logs -f modelq-container
```


### Run the Validation API

Set .env variable:
```bash
POSTGRES_URL=xxxxxxxxxx
```

```bash
python voice_validation_api/validation_api.py

```


#### Running the validator with your own validation API service running locally
```bash
# Make a separate venv for the validator because of pydantic version conflict
uv venv .validator
source .validator/bin/activate

uv pip install -e .
python neurons/validator.py --wallet.name WALLET_NAME --wallet.hotkey WALLET_HOT_NAME --use-local-validation-api
# Run model queue to push models to validation api to be evaluated
python neurons/model_queue.py --use-local-validation-api

python voice_validation_api/worker_queue.py 
```
## Model Evaluation Criteria
### Human Likeness Score
Models are evaluated on how closely their vocal outputs resemble natural human speech, considering factors such as emotional expression, intonation, pauses, and excitation levels. The more natural and convincingly human the voice sounds, the higher the score.

<!-- $S_{size} = 1 - ModelSize/ MaxModelSize$ -->
### Word Error Rate
Models that produce human-like speech with high clarity and coherence will achieve the highest scores. Lower word error rates indicate clearer, more accurate speech output, enhancing the model's overall evaluation.


## Acknowledgement

Our codebase is built upon [Nous Research's](https://github.com/NousResearch/finetuning-subnet) and [MyShell's](https://github.com/myshell-ai/MyShell-TTS-Subnet?tab=readme-ov-file) Subnets.

## License

The Dippy Bittensor subnet is released under the [MIT License](./LICENSE).
