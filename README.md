# DL2-ZeroVis

- Install the fromage model as explained in https://github.com/kohjingyu/fromage
including downloading the cc3m_embeddings as explained in the same github page.

Load in the OPT and CLIP models using fp16, to ensure they fit in lisa/locally.
This is done in fromage_main/fromage/models.py FromageModel self.lm and self.visual_model.

- Place the Visual Relations benchmark dataset as a folder benchmark.

- Install the environment file as a conda environment.
