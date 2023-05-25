#!/bin/bash

benchmark_root=src
emb_root=src/fromage_inf/fromage_model

mkdir -p $benchmark_root
mkdir -p $emb_root

if [ ! -f $emb_root/cc3m_embeddings.pkl ]
then
    echo "Downloading CC3M embedding pickle file.."
    python src/drive_download.py https://drive.google.com/u/0/uc?id=1wMojZNqEwApNlsCZVvSgQVtZLgbeLoKi
    mv cc3m_embeddings.pkl $emb_root/cc3m_embeddings.pkl
    echo "Done."
else
    echo "CC3M embeddings found. Skipping download."
fi

if [ ! -f $benchmark_root/benchmark/cities/washington.jpg ]
then
    echo "Downloading Visual Relations Dataset.."
    python src/drive_download.py https://drive.google.com/u/0/uc?id=1a8992QI5SpX0tu_HZVE6pZNBTtBSGnp-
    unzip benchmark.zip
    mv benchmark $benchmark_root/benchmark
    rm benchmark.zip
    echo "Done."
else
    echo "Visual Relations Dataset found. Skipping download."
fi
