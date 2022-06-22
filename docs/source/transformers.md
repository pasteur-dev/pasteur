# Transformer System


## Types
- categorical
  - integer
  - string
- ordinal
  - integer
  - str that maps to integer
- numerical
- custom
  - icd9
- datetime
  - relative to another date specified (day level accuracy)
  - accurate to week number and week day (seasonal and week trend preservation)
  - custom period level (hours, minutes, days)

## Reversible Transform types
- mixed output (one hot, integer, float) for nns
- to x variables of base n (discretization of some sort; hierarchical if ordinal)
- to one variable of a changing base
- one hot encoding