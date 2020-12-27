# Gaussian Mixture Model with an Unkown Number of Components a la [Richardson and Green (1997)](https://people.maths.bris.ac.uk/~mapjg/papers/RichardsonGreenRSSB.pdf)

**I am actively working on this project and so the code is not always complete or in working condition.** The number of clusters in the mixture is taken to be unknown and follows a `1 + Poisson(λ)` prior. We infer the paramters using MCMC. Our kernel alternates between block Gibbs sampling moves for the mixture and allocation varibles and split-merge moves for the number of components. The following are provided:
- A full, ground-up implementation in the PPL [Gen](https://www.gen.dev/)
- An implementation in Gen with the open-universe modeling suite `GenWorldModels` (`GenWorldModels` is under active development and the source code is kept private as of now)
- An implementation in [BLOG](https://bayesianlogic.github.io/) (Bayesian Logic)

The unicode variable names are chosen to match the variable names in the Richardson and Green paper. We have used this code for inference benchmarking in the following papers:

> - G. Matheos, A. K. Lew, M. Ghavamizadeh, S. J. Russell, M. F. Cusumano-Towner, and V. K. Mansinghka.
Transforming Worlds: Automated Involutive MCMC for Open-Universe Probabilistic Models. (AABI 2021)
