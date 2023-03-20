# Introduction
Welcome and thank you for your interest in synthetic data generation!

In this section, we're going to cover the preliminaries about Pasteur:
a complete methodology for data synthesis and its system architecture.
Data synthesis consists of too many a steps for it to be
performed ad-hoc (preprocessing, encoding, synthesis, decoding,
evaluations) reproducibly and reliably.
For this reason, parallelizing and caching between those steps
(especially for large datasets) becomes critical.

In the next section, we will go through an example project
and explain how to create and use custom dataset, synthesis, and
evaluation modules.