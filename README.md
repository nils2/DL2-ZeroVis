# Exploring Visual Grounding in FROMAGe using Visual Arithmetics

<center>
<img alt="FROMAGe model architecture" src="https://github.com/nils2/DL2-ZeroVis/blob/394af24706021418f17cff9910a91a6e34cbceb0/img/fromage-pipelines.png" width="90%">
</center>

This is the repository of Team 25 of UvA Deep Learning 2 course in 2023. Here we explore the visual arithmetic capabilities of FROMAGe (Grounding Language Models to Images for Multimodal Generation).

[[Our Blogpost]](https://github.com/nils2/DL2-ZeroVis/blob/main/blogpost.md) | [[FROMAGe git]](https://github.com/kohjingyu/fromage) | [[FROMAGe paper]](https://arxiv.org/abs/2301.13823)

## Installation

### Step 1: Clone the repository
```
git clone git@github.com:nils2/DL2-ZeroVis.git
cd DL2-ZeroVis
```
### Step 2: Install dependencies
```
conda env create -f env.yml
conda activate fromage
```
### Step 3: Download embeddings and evaluation data
*Tip: Make sure you run it from the conda environment.*
```
source ./download.sh
```
This script downloads the visual relations dataset and saves it under the 'src/' directory.

## Running

We recommend the following notebooks to analyse the visual analogy resolution capabilities of FROMAGe:

ZeroCap comparison:
[arithmetic_icap_greedy](https://github.com/nils2/DL2-ZeroVis/blob/main/demos/Visual-Arithmetics/Image-to-Text/arithmetic_ICap_greedy.ipynb)

Visual arithmetics:
[arithmetic_itret](https://github.com/nils2/DL2-ZeroVis/blob/main/demos/Visual-Arithmetics/Image-to-Image/arithmetic_ITRet.ipynb)

Explainability of the Visual Relations benchmark:
[explainability](https://github.com/nils2/DL2-ZeroVis/blob/main/demos/Extra-Studies/Availability/img-to-txt/ICap_greedy.ipynb)

t-SNE space:
[TSNE](https://github.com/nils2/DL2-ZeroVis/blob/main/demos/Extra-Studies/t-SNE.ipynb)
