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
things locally you need to have either a virtual or [conda](https://docs.conda.io/en/latest/) environment that
supports PyTorch. 

Other requirements are:
* An Azure subscription that you have the `Owner` Role to create resources.
* [PowerShell v5.0+](https://docs.microsoft.com/en-us/powershell/scripting/overview?view=powershell-7.2)
* [Azure Command-Line Interface (az-cli) v2.36.0+](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)

### Quickstart

A [detailed walkthrough](https://github.com/Azure-Samples/locutus/wiki) of the entire process is available.
Some instructions basic for getting everything kicked off:

1. Clone the repo
2. [Azure Machine Learning Infrastructure Setup](https://github.com/Azure-Samples/locutus/wiki)
```
./provision.ps1 -name <YOUR_APP_NAME> -location <LOCATION|i.e. westus2>
```
3. [Experimentation Notebooks](https://github.com/Azure-Samples/locutus/wiki/Notebooks)
4. [AzureML Components and Pipelines for finetuning GPT-2](https://github.com/Azure-Samples/locutus/wiki/Finetuning)
5. [Managed Inference](https://github.com/Azure-Samples/locutus/wiki/Managed-Inference)
6. [GitHub Actions for ML Component CI/CD](https://github.com/Azure-Samples/locutus/wiki/Component-MLOps)
7. [Responsible AI Dashboard](https://github.com/Azure-Samples/locutus/wiki/RAI-Dashboard)


## Resources

* [Start with an Azure free account](https://azure.microsoft.com/en-in/free/)
