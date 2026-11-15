"""Median magnitude of the pupil center shift per mm of pupil diameter reported by Wildenmann and Schaeffel, per eye.

    uv run python measured/wildenmann_comparison.py

Wildenmann, U., & Schaeffel, F. (2013). Variations of pupil centration and their effects on video eye tracking.
Ophthalmic & Physiological Optics, 33(6), 634-641. https://doi.org/10.1111/opo.12086

Their Table 2 gives, for each of 10 subjects and each eye, the slope of pupil center position against pupil size in
the horizontal and in the vertical direction, in mm per mm. The magnitude of each eye's shift is the magnitude of
that slope vector, sqrt(horizontal^2 + vertical^2), which the signs of the two slopes do not change. Left and right eyes
are summarized separately.

Prints only; writes nothing.
"""

import numpy as np

# Table 2, (horizontal, vertical) slope of subjects 1-10 in mm per mm, as absolute values of the two-decimal entries.
RIGHT_EYES = (
    (0.02, 0.00),
    (0.04, 0.07),
    (0.02, 0.04),
    (0.02, 0.01),
    (0.09, 0.18),
    (0.02, 0.01),
    (0.04, 0.02),
    (0.10, 0.13),
    (0.18, 0.04),
    (0.01, 0.02),
)
LEFT_EYES = (
    (0.03, 0.01),
    (0.02, 0.27),
    (0.00, 0.21),
    (0.05, 0.01),
    (0.04, 0.07),
    (0.00, 0.03),
    (0.00, 0.03),
    (0.13, 0.04),
    (0.02, 0.13),
    (0.03, 0.18),
)


def main() -> None:
    """Print the median, minimum and maximum slope magnitude for each eye side."""
    for label, slopes in (("Left eye", LEFT_EYES), ("Right eye", RIGHT_EYES)):
        magnitudes = np.linalg.norm(np.array(slopes), axis=1)
        print(
            f"{label:<10} median {np.median(magnitudes):.3f}   range {magnitudes.min():.3f}-{magnitudes.max():.3f}"
            f" mm per mm   n {len(magnitudes)}"
        )


if __name__ == "__main__":
    main()
