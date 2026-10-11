# Final decision memo - theory-guided regularization

Leave the current manuscript unchanged. The controlled representation extension weakens B, and incremental net evidence for updating b is inconclusive.

The exact baseline was reproduced: 3.56133432162478 gross Sharpe, all 564 payoffs and 47 annual schedules; maximum payoff error 2.22e-16.

1. **How much does updating b add?** On the full sample, A is 3.542325 and B 3.561334: an increment of 0.019009, not a causal decomposition of Sharpe. On the primary 1979-2024 decisions, the gross increment is +0.019 [+0.003, +0.036]; net is +0.009 [-0.005, +0.023]. Updating b has a small measurable gross gain conditional on these fitted paths, but no resolved net gain.
2. **Is initial fixed b nearly as good?** Its Sharpe point estimate is close. No equivalence margin was declared, and an interval including zero does not establish equivalence.
3. **Does varying length improve the theoretical rule?** In the controlled five-candidate design, D minus B is -0.033 [-0.065, -0.009] gross and -0.023 [-0.048, -0.004] net. Selection worsens performance; the net difference remains negative across 6-, 12- and 24-month intervals.
4. **Does it improve net Sharpe?** D is 2.780771, below B at 2.803799. D minus C is +0.012 [-0.009, +0.038], so updating b within the selected representation has an unresolved net benefit.
5. **Can it compete with joint CV?** D has gross/net 3.495/2.781, joint CV 3.553/2.719. The net difference is +0.061 [-0.239, +0.299]; this is a competitive point estimate, not demonstrated superiority or noninferiority.
6. **Fewer searches?** A/B: one 200-candidate initial grid; C/D: five such initial grids (1,000 evaluations, with invalid candidates reported), then zero annual lambda-grid searches. Each subsequent C/D decision evaluates three admissible lengths. Joint CV evaluates 5 x 120 = 600 length/penalty combinations each year; fixed-length CV uses 120. Complexity of spectral decomposition, full-history refitting and holdings reconstruction remains.
7. **Historical robustness?** Main findings are supplied on 1979-2024, 1984-2024, four historical blocks, and the last five decision years. The last five years are descriptive. Circular-block intervals condition on fitted strategies and do not include algorithm-selection uncertainty. All length calibrations share the initial validation window, and early rolling validation windows overlap it; 1984 onward removes that overlap. This exercise inherits retrospective signals, complete-case availability, the current JKP snapshot and same-close execution assumptions.
8. **Paper subsection?** Keep the manuscript unchanged. No one-page subsection is proposed from this mixed controlled evidence; no appendix or new methodology is introduced.

At the final expanding refit, 84.0% of the log-penalty decrease from the original initial CV penalty comes from increasing T and 16.0% from updating b. The exact log identity is exported for A-D; representation changes form a separate term. B's last penalty is 4.67964049e-08, versus 5.67515243e-08 holding b=b0. This attribution concerns lambda, not the sources of investment performance.

## Admissibility materially matters

The initial exponents at multipliers 0.25 and 0.5 are 0.582138 and 0.931378, below one. They are permanently unavailable in the primary theoretical comparison. Extrapolating the formula to these candidates changes 36 of 46 primary decisions: D's gross Sharpe becomes 3.700, compared with 3.495 for the admissible rule. This diagnostic is outside the theorem and has no new net-performance claim. These exclusions are consequential, not a minor numerical detail.

## User-requested extra optimization

The nine-candidate follow-up adds 0.75, 1.25, 1.5 and 3 times the original lengthscale using the same frequency and phase draws, initial 200-candidate calibration, past-only selector and frozen constants. It was declared after the coarse gross results were available and is exploratory. The exploratory denser search gives primary gross/net Sharpe 3.600/2.864; net D dense minus B is +0.060 [-0.012, +0.143]. It does not establish a robust incremental net advantage over B. Net D dense minus original D is +0.083 [+0.020, +0.160]; versus joint CV it is +0.145 [-0.092, +0.338]. These results do not change the primary 2 x 2 findings or identify a population-optimal data-dependent penalty.

## Monthly rolling decision

Both trailing windows are translated before the diagnostic, at January 1984 close, to the then-available expanding B penalty: C0(W)=lambda_anchor W^[b_anchor(W)/(b_anchor(W)+1)]. Constants are frozen thereafter. On the original 10%-60% rank band, median absolute monthly log-lambda changes are 0.207% and 0.129%. No primary slope falls below one. Cross-band sensitivity exceeds ordinary monthly movement. The predeclared 5% change / rank-stability gate fails; a costly full monthly portfolio backtest is not justified by this diagnostic. This is not evidence that rolling portfolios could never outperform expanding portfolios.

All costs use each strategy's actual signed stock weights, the frozen common exposure scale 0.0374000022729, 25 bp on traded notional and 30 bp/year on short notional through verified self-financing accounting. States begin in cash for February 1978 and carry into subperiods; holdings are marked at the end without liquidation. Quadratic losses use raw payoff units; net losses divide by the same pretest scale. Public exports contain aggregate paths only; licensed panels, coefficients and stock holdings remain under ignored results/.
