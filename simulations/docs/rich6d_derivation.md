# A smooth six-coordinate loading map

For z in the open cube (-1,1)^6, define
theta = pi(z1+1) + eta sum[d=2..6] sin(pi(zd-z1)), with eta=.35.
The loadings are sqrt(c_beta) times
(sqrt(2/3) cos(theta), sqrt(2/3) sin(theta), 1/sqrt(3)).
`dgp.balanced.beta` is the single definition used by returns, policy and quadrature.
The identifier and eta are included in the actual economic parameters and hashes.

Adding one third to every rank rotates theta by 2pi/3 modulo 2pi. Differences
between coordinates change only by integer multiples of 2pi inside the sine.
The sum of the three roots of unity, and of their squares, is zero. Consequently
each loading has squared norm c_beta and every triplet has B'B/3=(c_beta/3)I.
Aggregating groups preserves B'B/N=(c_beta/3)I exactly in real arithmetic.

With uncentered conditional second moment S=B B'+sigma_eps² I and mean B mu_F,
the policy W*(z)=beta(z)'mu_F/(c_beta/3+sigma_eps²/N) gives stock weights W*/N.
Multiplying S by those weights gives B mu_F. An independent N-dimensional solve
checks this identity without using the three-dimensional policy solver.
Therefore q* and SR*=sqrt(q*/(1-q*)) are unchanged by eta.

For d>=2, partial_d theta=pi eta cos(pi(zd-z1)); partial_1 theta=pi minus the
sum of those five derivatives. The derivative of W* follows by the chain rule.
Tests compare all six derivatives with central differences and verify positive
mean squared derivatives. Eta=0 takes the original arithmetic branch and
reproduces baseline loading and return arrays byte-for-byte.

The full uniform-cube expectations can also be integrated analytically. Let
A²+B²=(2c_beta/3)(mu_1²+mu_2²)/(c_beta/3+sigma_eps²/N)². The common phase is
uniform conditional on the five independent torus differences. Hence
E[(partial_1 W*)²]=(A²+B²)pi²(1+5eta²/2)/2 and, for every d>=2,
E[(partial_d W*)²]=(A²+B²)pi²eta²/4. A separate 131,072-point full-cube audit
checks these moments. This is distinct from the preflight finite-difference
sample restricted to the interior cube [-.9,.9]^6.

E1 follows from the same stationary stable finite-dimensional Gaussian AR and
measurable transformations, with independent return shocks. E2 uses the unchanged
loading norm, conditional Gaussian shocks, K(z,z)=1 and sufficient bound
B_X²=4(c_beta+sigma_eps²); a scalar conditional Gaussian has fourth/second²<=3.
The exact noncentral Gaussian quadratic MGF remains the numerical E2(a) check.
For E3, all loading components extend smoothly to all of R^6. Multiplication by
a smooth compactly supported cutoff equal to one on a neighborhood of the closed
cube gives an extension in H^s(R^6) for every finite s, including s=nu+3.
Restriction gives membership in the Matérn RKHS on the cube. A finite Nyström
norm is not used as a proof of infinite-dimensional membership.

E4 is the conditional equation above. E5 retains
(sigma_eps²/N)T_K <= Sigma_H <= (c_beta+sigma_eps²/N)T_K.
Three return factors do not imply rank three for their state-integrated operator.
E6 follows from the unchanged positive q*. The lower comparison constant depends
on N; N is fixed in each rate experiment. A multivariate smooth target is not
a least-favourable minimax family, and no equality with an upper-bound exponent
is asserted.
