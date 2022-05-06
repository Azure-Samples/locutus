# locutus GPT-2 Text Generation

This sample takes folks through the end-to-end process of deploying the 
[GPT-2 model from HuggingFace](https://huggingface.co/gpt2) fine-tuned 
on the writings of Homer. These are the demos that were shown in the 
"Scaling responsible MLOps with Azure Machine Learning" session

## Features

This repo has a number of things that can be tried in isolation or
together as an end-to-end exercise. These include: 

* Azure Machine Learning Infrastructure Setup
* Experimentation Notebooks
* AzureML Components and Pipelines for Finetuning GPT-2
* Managed Inference
* GitHub Actions for ML Component CI/CD
* Responsible AI Dashboard (Pending)

## Getting Started

### Prerequisites

Generally it is easier to run these exercises in the cloud (given that part
of the exercise is creating custom environments). If you want to run these
things locally you need to have either a virtual or conda environment that
supports PyTorch. 

To run the setup scripts PowerShell is required.

### Quickstart

A [detailed walkthrough](wiki) of the entire process is available.
Some instructions basic for getting everything kicked off:

1. Clone the repo
2. [Azure Machine Learning Infrastructure Setup](wiki)
```
./provision.ps1 -name <YOUR_APP_NAME> -location <LOCATION|i.e. westus2>
```
3. [Experimentation Notebooks](wiki/Notebooks)
4. [AzureML Components and Pipelines for finetuning GPT-2](wiki/Finetuning)
5. [Managed Inference](wiki/Managed-Inference)
6. [GitHub Actions for ML Component CI/CD](wiki/Component-MLOps)
7. [Responsible AI Dashboard](wiki/RAI-Dashboard)


## Resources

(Any additional resources or related projects)

- Link to supporting information
- Link to similar sample
- ...
