# Pasteur Spec
The Pasteur project consists of a directory which contains all the files required for
an end-to-end synthetic generation algorithm to be benchmarked, hyperparameter
tuned, and to run on multiple datasets.

Pasteur integrates the packages Kedro and Synthetic Data Vault (SDV).

Kedro is used to enforce good software engineering principles throughout the project
and to encapsulate it in a reproducible platform-agnostic pipeline.
It also provides the dataset loading implementations.

SDV implements a complete toolkit for synthetic generation of tabular data.
It includes algorithms for synthetic data generation, reversible dataset 
transformations, metrics, and a benchmarking framework.
Where it lacks is privacy enabled algorithms and the synthetic data generation
state of the art.

The directory of the project is created so that it can be versioned in git.

As such, it includes the following (tracked) directories:
  - `metadata`: contains instructions handling multi-modal data in a format supported by SDV.
  - `src/pasteur/embeddings`: classes used to discretize ML columns, specified by metadata.
  - `src/pasteur/compressors`: classes used to compress multi-modal data into a discrete latent space.
  - `src/pasteur/synth`: synthetic data algorithms that permutate tabular, discrete data in
    in such a way that they protect the privacy of the participants (DP etc).
  - `src/pasteur/pipelines`: Wrapper classes that contain all the logic required to run an experiment.

It also includes the following non-tracked directories:
  - `data/orig/{raw,intermediate,primary,keys}`: directory which contains the
    original datasets in various steps of transformation
  - `data/views/{orig,encoded,decoded}`: materialized views of the above datasets,
    with `orig` containing the source materialized view, `encoded` after it has
    been transformed in a reversible way, and `decoded` with the transformation
    reversed.
  - `synth/{compressors,encoded,decoded}/`: synthetic equivalents to the views
    folder. The `orig` folder is omitted. `compressors` contains the neural networks
    used to compress the latent space of certain modalities.
    All of the files in this folder are versioned, because new versions are
    generated each run.

A run measures the performance of a synthetic algorithm using:
  - a dataset
  - a schema for that dataset
  - a reference split that's used for synth data + comparison
  - a test/dev set for benchmarking predictive performance
  - an algorithm to create synthetic data
  - a set of metrics fit for that dataset
    * handcrafted SQL queries
    * metrics that compare datasets after encoding them using a schema
    * predictive models (sklearn)

The synthetic algorithm data can be substituted by another slice of original data
to show how an ideal synthetic algorithm would permutate the data.

Based on [The Importance of Layered Thinking in Data Engineering](https://towardsdatascience.com/the-importance-of-layered-thinking-in-data-engineering-a09f685edc71)


``` bash
pasteur/
  metadata/
    mydataset.detail.json
    mydataset.coarse.json
    <view>.<schema>.json
  metrics/
    mydataset/
      bench.general.sql
      bench.population.sql
      metrics.yml
    <view>/
      bench.<modality>.sql
      metrics.yml

  src/pasteur/
      embeddings/
        MyEmbedding.py
      compressors/
        MyCompressor.py
      synth/
        mst.py
      pipelines/
        processing/
          ...
        synthesize/
          ...
        benchmark/
          ...

  data/
    # This folder contains the original datasets whole
    orig/
      # skipped if using postgres
      raw/
      itermediate/
      primary/
      # Contains the keys used to split the data of materialized vies in the same 
      # way each time
      keys/
        mydataset.wrk.pq
        mydataset.ref.pq
        mydataset.val.pq
        mydataset.dev.pq
    # This folder contains slices of datasets used to test synthesizing a specific
    # modality (materialized views), taken from original datasets.
    views/
      # Orig contains the original data of the slice in parquet form
      # data is split in work, ref, val, dev sets using a (40, 40, 10, 10) split
      orig/
        myview/
          timeseries.wrk.pq
          core.wrk.pq
        <view>/
          <component>.{wrk,ref,val,dev}.pq
      # Encoded is an intermediary form, where data has been processed using
      # transformers in a lossy reversible way and can be fed into a model
      encoded/
        myview.<schema>.pq
        myview.<schema>.dict.json
      # Decoded reverses the previous lossy encoding to bring the data to its
      # original form. This is useful because the transformer might bias the
      # data in some way. This would produce an unfair comparison if the algorithm
      # is benchmarked using the original data. 
      decoded/
        myview.<schema>/
          timeseries.wrk.csv
          core.wrk.csv
    synth/
      compressors/
        timeseries.pq/XXXXX00Z/timeseries.pq
      models/
        myview.pkl/XXXXX00Z/myview.pkl
      encoded/
        myview.pq/XXXXX00Z/myview.pq
      decoded/
        timeseries.wrk.csv/XXXXX00Z/timeseries.wrk.csv
        core.wrk.csv/XXXXX00Z/core.wrk.csv
```