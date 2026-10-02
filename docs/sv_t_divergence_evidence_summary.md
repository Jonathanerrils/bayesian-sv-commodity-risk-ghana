# SV-t divergence evidence summary

This note consolidates the completed W=1000, block 0 diagnostic evidence for
Cocoa SV-t, Gold SV-t, and the clean Oil SV-t control. It is diagnostic only;
it does not change priors, model equations, convergence thresholds, or the
primary model family.

## Convergence outcomes

| Case | Attempt | max R-hat | min bulk ESS | divergences | Strict gate |
|---|---:|---:|---:|---:|---|
| Cocoa SV-t | 1 | 1.0129918 | 604.7451 | 1 / 4000 | Fail |
| Cocoa SV-t | 2 | 1.0029670 | 1174.2203 | 2 / 8000 | Fail |
| Gold SV-t | 1 | 1.0015292 | 1863.4933 | 0 / 4000 | Pass |
| Gold SV-t | 2 | 1.0012864 | 3810.1770 | 1 / 8000 | Fail |
| Oil SV-t control | 1 | 1.0055987 | 877.0756 | 0 / 4000 | Pass |

## Geometry findings

### Persistence

Divergent draws in the failing Cocoa and Gold attempts occur at extremely high
persistence:

- Cocoa attempt 1: phi = 0.9999011.
- Cocoa attempt 2: divergent phi values 0.9997605 to 0.9999742; mean 0.9998673.
- Gold attempt 2: phi = 0.9994536.

The clean Oil control is materially less persistent (posterior median phi
0.8828167; 95% range approximately 0.7075 to 0.9541).

Near-unit persistence is therefore strongly implicated in the difficult
posterior region, but is not by itself sufficient for divergence: Gold attempt
1 passed the strict gate despite a high-persistence posterior (median phi
0.9917219).

### Student-t degrees of freedom

There is no evidence that nu approaching 2 is the common divergence mechanism.

- Cocoa attempt 1 divergent nu = 5.4477.
- Cocoa attempt 2 divergent nu range = 3.7293 to 4.7540.
- Gold attempt 2 divergent nu = 13.1084.
- Oil control posterior nu median = 12.2389.

The observed divergent draws are not concentrated on the nu > 2 boundary.

### Innovation scale

Divergent draws tend to have relatively small sigma_eta compared with the
corresponding non-divergent posterior median:

- Cocoa attempt 1: 0.0218 vs median 0.0467.
- Cocoa attempt 2: about 0.0281 vs median 0.0467.
- Gold attempt 2: 0.0395 vs median 0.0682.

This may interact with high persistence, but the diagnostics do not establish
sigma_eta as an independent causal mechanism.

### Latent-state extremes

The divergent draws do not consistently correspond to extreme latent paths.
For Gold attempt 2, eta RMS (1.0023), maximum absolute eta (3.5146), and latent
h range (2.1074) are all within ordinary non-divergent posterior ranges. Cocoa
also shows no consistent evidence that latent-state extremes alone explain the
divergences.

### NUTS energy and tree depth

The clearest sampler pathology is maximum Hamiltonian energy error:

- Cocoa attempt 1 divergent max energy error = 1251.07.
- Cocoa attempt 2 divergent max energy error = 1837.43 and 2873.90.
- Gold attempt 2 divergent max energy error = 2958.22.
- Oil control non-divergent max energy error has median -0.037 and maximum 1.477.

Tree depth does not distinguish the divergent points: the failing cases are not
showing a unique tree-depth saturation pattern.

## Stationary initial-state construction

The current non-centered state construction uses the stationary initial-state
scale

    sigma_eta / sqrt(1 - phi^2).

Because the divergent draws repeatedly occur where phi is extremely close to 1
and sigma_eta is relatively small, this construction is a plausible contributor
to the difficult geometry. The present diagnostics do not prove that it is the
cause. A mathematically equivalent persistence/initial-state parameterization
experiment would therefore be a justified next diagnostic, not a production
model change.

## Particle diversity

Across complete 42-day clean reference blocks, posterior-particle ancestry
declines sharply despite high filter ESS:

- Cocoa SV-Gaussian: unique fraction 1.000 -> 0.033375; minimum filter ESS
  3836.35; median ESS 7277.28.
- Oil SV-t: unique fraction 1.000 -> 0.04175; minimum filter ESS 3565.64;
  median ESS 3935.10.
- Oil SV-Gaussian: unique fraction 1.000 -> 0.03775; minimum filter ESS
  3296.40; median ESS 3882.47.

Thus ancestry collapse is material even while one-step filter ESS appears
healthy. This is a separate filtering diagnostic and does not explain the MCMC
divergences at structural refits.

## What remains uncertain

The evidence supports a high-persistence/energy-geometry explanation more than
a nu-boundary, tree-depth, or latent-extreme explanation. It does not yet
separate whether the dominant source is the persistence transform itself, the
stationary initial-state scale, interaction with sigma_eta, or another
mathematically equivalent representation issue. That distinction requires a
controlled reparameterization diagnostic with the same probability model,
priors, seeds, MCMC budgets, and convergence gate.
