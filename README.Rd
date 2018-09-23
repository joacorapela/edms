ensemble dynamical models
=========================

python code to simulate networks of ensemble dynamical models of
integrate-and-fire neurons, reduce their dimensionality, and estimate their
connectivity parameters (Rapela et al., 2015).

For example, to simulate an ensemble dynamical model, of integrate and fire
neurons, driven by excitatory and inhibiotory sinusoidal currents, run the
script doTestIFEnsembleDensityIntegratorRWFWI.py (with python 2.7) from the
directory test/simulations/omurtagSig:

cd test/simulations/omurtagSim; python2.7 doTestIFEnsembleDensityIntegratorRWFWI.py

This script will save two figures, rhosSinusoidalRWFWIPopulationKMu0.0300.eps
and spikeRatesSinusoidalRWFWIPopulationKMu0.0300.eps, containing the
integrated ensemble probability density function and the spike rate of the
ensemble dynamical model.

Rapela  J., Kostuk M, Rowat, P.F., Mullen, T., Chang E.F., Bouchard K. (2015)
Modeling  neural activity at the ensemble level.
http://arxiv.org/abs/1505.00041
