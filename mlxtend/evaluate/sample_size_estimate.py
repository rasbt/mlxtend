# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# A function to estimate sample size for hypothesis testing.
# Author: Prajwal Kafle
#
# License: BSD 3 clause

from scipy import stats as st


def binomial_proportions(baseline_conversion, minimum_effect,
                         confidence_level=95, power=80,
                         test='two-sided'):
    """ This function estimates the sample size to detect minimum uptake in a control
    group relative to a given baseline conversion rate of a control group
    with a given power and confidence level.

    Parameters
    ----------
    baseline_conversion: percentage
        Conversion rate of variant A/control group.
    minimum_effect: percentage
        Minimum detectable effect determines conversion rate of variant B/treatment group.
    confidence_level: percentage (float, default: 95)
        where, significance level = 1 - confidence_level/100.
        For a confidence level of 95%, alpha = 0.05.
    power: Statistical power in percentage (default: 80%)
        beta = 1 - power/100 is Type II error.
    test: str (default: two-sided)
        Option one-sided is valid too.

    Returns
    -------
    n: int
        sample size

    Notes
    ------
    Function uses t-distribution to find critical values

    Reference
    ---------
    https://select-statistics.co.uk/calculators/sample-size-calculator-two-proportions/

    """

    if minimum_effect <= 0:
        raise ValueError("Function is valid only for a positive uptake (i.e. minimum_effect > 0 ).")

    if baseline_conversion <= 0:
        raise ValueError("Function is valid only for a positive baseline conversion rate.")

    # type I error
    alpha = 1 - confidence_level/100

    if test == 'two-sided':
        alpha = alpha/2

    # Critical values of the t-distribution at alpha/2 and beta
    z_crit = lambda x: st.t.ppf(x, df=1000)
    z_alpha_over_2 = z_crit(1 - alpha)
    z_beta = z_crit(power/100)
    z_total = z_alpha_over_2 + z_beta

    # Converting percentages to proportion
    conversion_rate_variant_a = baseline_conversion/100

    # Conversion rate of variant B is increment in conversion rate of variant A
    # by minimum detectable effect.
    conversion_rate_variant_b = (1 + minimum_effect/100)*conversion_rate_variant_a

    n = (z_total/(conversion_rate_variant_a - conversion_rate_variant_b))**2 * \
        (conversion_rate_variant_a * (1 - conversion_rate_variant_a) +
         conversion_rate_variant_b * (1 - conversion_rate_variant_b))

    return int(n)

