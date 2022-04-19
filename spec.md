# Potential Pasteur Spec
A project consists of a directory which contains all the files required for
an end-to-end synthetic generation algorithm to be benchmarked, hyper-parameter
tuned, and to run on multiple datasets.

The directory of that project is created such that it can be versioned in git.
As such, it includes the following (tracked) directories:
  - `schema`: contains instructions for discretizing a (CSV) dataset
  - `embeddings`: classes used to discretize ML columns, selected using a schema
  - `compressors`: classes used to compress multi-modal data into a discrete latent space
  - `synth`: synthetic data algorithms that permutate tabular, discrete data in
    in such a way that they protect the privacy of the participants (DP etc)
  - `experiments`: Wrapper classes that contain all the logic required to run an experiment

It also includes the following non-tracked directories:
  - `dataset/{orig, encoded, decoded}`: `orig` contains the original dataset files
    and `encoded`, `decoded` the encoded and then decoded versions of those 
    datasets using a specific schema.
  - `runs/<name>/{encoded, decoded}`: run specific artifacts, such as the synthetic
    versions created by algorithms

A run measures the performance of a synthetic algorithm using:
  - a dataset
  - a schema for that dataset
  - a reference split that's used for synth data + comparison
  - a test/dev set for benchmarking predictive performance
  - an algorithm to create synthetic data
  - a set of metrics fit for that dataset
    * handcrafted sql queries
    * metrics that compare datasets after encoding them using a schema
    * predictive models (sklearn)

The synthetic algorithm data can be substituted by another slice of original data
to show how an ideal synthetic algorithm would permutate the data.



```
project/
  schema/
    mydataset.detail.json
    mydataset.coarse.json
  embeddings/
    MyEmbedding.py
  compressors/
    MyCompressor.py
  metrics/
    mydataset/
      bench.general.sql
      bench.population.sql
  synth/
    mst.py
  experiments/
    MyExperiment.py

  # .gitignore
  datasets/
    orig/
      mydataset.wrk.csv
      mydataset.ref.csv
      mydataset.val.csv
      mydataset.dev.csv
      <dataset>.{wrk,ref,val,dev}.csv
    encoded/
      mydataset.wrk.detail.b2.{csv, dict.json, domain.json}
      mydataset.ref.detail.b2.{csv, dict.json, domain.json}
      <dataset>.<split>.<schema>.<encoding>.{csv, dict.json, domain.json}
    decoded/
      # same as encoded, serves as reference
  runs/
    myrun1/
      encoded/
        mydataset.wrk.detail.b2.{csv, dict.json, domain.json}
      decoded/
        mydataset.wrk.detail.b2.{csv, dict.json, domain.json}
    myrun2/
      ...
```

# Data Synthesis Spec
Based on https://towardsdatascience.com/the-importance-of-layered-thinking-in-data-engineering-a09f685edc71

Data synthesis, unlike a predictive supervised task, has
an output product, which is the dataset.
Therefore, it becomes necessary to expand 

- orig
  - raw
  - intermediate
  - primary
- views
  - materialized
  - model_input
  - reconstructed
- models
- synth
  - model_output
  - reconstructed
- reporting