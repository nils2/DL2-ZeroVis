# Deep Learning 2 - Team 25

<center>
<img alt="FROMAGe model architecture" src="https://raw.githubusercontent.com/kohjingyu/fromage/main/teaser.png" width="90%">
</center>

This is the repository of Team 25 of UvA Deep Learning 2 course in 2023. Here we explore the visual arithmetic capabilities of FROMAGe (Grounding Language Models to Images for Multimodal Generation)

[[FROMAGe]](https://github.com/kohjingyu/fromage) [[Blogpost]](https://github.com/kohjingyu/fromage)

## Installation

### Step 1: Clone the repository
```
git clone git@github.com:nils2/DL2-ZeroVis.git
cd DL2-ZeroVis
```
### Step 2: Install dependencies
```
conda install -f env.yml
conda activate fromage
```
### Step 3: Download embeddings and evaluation data
```
source ./download.sh
```
This script downloads the visual relations dataset and saves it under ????.

## Running

We recommend the following notebooks to analyse FROMAGe:

Retrieval: 
[[empty_link1]]() [[empty_link2]]() [[empty_link3]]()

Chain of Thought: 
[[empty_link1]]() [[empty_link2]]() [[empty_link3]]()


<!---
- Place the [cc3m_embeddings](https://drive.google.com/file/d/1wMojZNqEwApNlsCZVvSgQVtZLgbeLoKi/view) in the "src/fromage_inf/fromage_model" folder as done in [fromage](https://github.com/kohjingyu/fromage).

- Place the [Visual Relations benchmark](https://drive.google.com/file/d/1hf5_zPI3hfMLNMTllZtWXcjf6ZoSTGcI/edit) dataset, from the [ZeroCap](https://github.com/YoadTew/zero-shot-image-to-text) paper, as a folder "benchmark" in directory "src".
In here change the images in the leaders folder that have spaces to _ (eg. angela_merkel).
Also correct the name washignton in cities to washington.

- Install the environment file "env.yml" as a conda environment.

Below only needed when not using the fromage_inf folder.
(- Install the fromage model as explained in https://github.com/kohjingyu/fromage
including downloading the cc3m_embeddings as explained in the same github page.

Load in the OPT and CLIP models using fp16, to ensure they fit in lisa/locally.
This is done in fromage_main/fromage/models.py FromageModel self.lm and self.visual_model.)

-->