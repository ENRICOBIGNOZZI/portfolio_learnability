Section VI only, with figures numbered 3--7 (nine historical panels are Figure 6).
Compile: tectonic Section_VI_Empirical_Portfolio_Learnability.tex
or: pdflatex Section_VI_Empirical_Portfolio_Learnability.tex (twice).
To integrate, input numbers.tex and empirics.tex with tables/ and figures/ on the LaTeX path.
The wrapper sets Section VI and Figure 3; it contains no other manuscript section or appendix.
Canonical aggregates and verification: ../tables/ and ../audit/.
Full private-data reproduction: python3 -m empirical_final.section_vi_finalize
Public regeneration in the repository: python3 -m empirical_final.section_vi_finalize --public-only
