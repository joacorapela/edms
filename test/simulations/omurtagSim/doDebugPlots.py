
# EigenReposForOneStim.py __init__
ev=9; sl=0; plt.plot(reducedEVecsCol[sl][:, ev], label="%d"%(self._stimuli[sl]));  plt.plot(reducedEVecsCol[sl+1][:, ev], label="%d"%self._stimuli[(sl+1)]);  plt.plot(reducedEVecsCol[sl+2][:, ev], label="%d"%self._stimuli[(sl+2)]);  plt.plot(reducedEVecsCol[sl+3][:, ev], label="%d"%self._stimuli[(sl+3)]);  plt.plot(reducedEVecsCol[sl+4][:, ev], label="%d"%self._stimuli[(sl+4)]); plt.plot(reducedEVecsCol[sl+5][:, ev], label="%d"%self._stimuli[(sl+5)]); plt.title("Eigenvector %d"%ev); plt.legend(title="Stimulus"); plt.show()

# EigenReposForOneStim.py _getEigenDerivs
i=12; fig, ax1 = plt.subplots(); ax1.plot(eVecs[:, i], color="blue");
ax1.set_ylabel('Eigenvector'); ax2 = ax1.twinx(); ax2.plot(dEVecs[:, i],
color="orange"); ax2.set_ylabel('Derivative Eigenvector');
plt.title("Eigenvector %d" % i); plt.show()

# EigenReposForOneStim.py _getEigenDerivs
i=3; fig, ax1 = plt.subplots(); ax1.plot(aEVecs[:, i], color="blue");
ax1.set_ylabel('Adjoint Eigenvector'); ax2 = ax1.twinx(); ax2.plot(dAEVecs[:,
i], color="orange"); ax2.set_ylabel('Derivative Adjoint Eigenvector');
plt.title("Adjoint Eigenvector %d" % i); plt.show()
