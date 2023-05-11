# DL2-ZeroVis

- Place the [cc3m_embeddings](https://drive.google.com/file/d/1wMojZNqEwApNlsCZVvSgQVtZLgbeLoKi/view) in the "repo/fromage_inf/fromage_model" folder as done in [fromage](https://github.com/kohjingyu/fromage).

- Place the [Visual Relations benchmark](https://drive.google.com/file/d/1hf5_zPI3hfMLNMTllZtWXcjf6ZoSTGcI/edit) dataset, from the [ZeroCap](https://github.com/YoadTew/zero-shot-image-to-text) paper, as a folder "benchmark" in directory "repo".
In here change the images in the leaders folder that have spaces to _ (eg. angela_merkel).
Also correct the name washignton in cities to washington.

- Install the environment file "env.yml" as a conda environment.

Below only needed when not using the fromage_inf folder.
(- Install the fromage model as explained in https://github.com/kohjingyu/fromage
including downloading the cc3m_embeddings as explained in the same github page.

Load in the OPT and CLIP models using fp16, to ensure they fit in lisa/locally.
This is done in fromage_main/fromage/models.py FromageModel self.lm and self.visual_model.)