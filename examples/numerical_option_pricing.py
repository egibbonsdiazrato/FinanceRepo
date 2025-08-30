import numpy as np

from src.Options import Market, Stock, VanillaOptionBTM, IntRate_delta_rist


def binary_option_payoff(S_T: np.ndarray) -> np.ndarray:
    """
    Produces payoffs for a custom function which only pays the payoff_val if stock price at maturity is greater
    than or equal to S_limit.

    Args:
        S_T: the last column of the matrix which holds the values of the stock at maturity.

    Returns:
        payoff: the payoff of S_T for the option.
    """
    S_limit, payoff_val = 100, 100  # Parameters
    payoff = np.where(~np.isnan(S_T), np.where(S_T < S_limit, 0, payoff_val), np.nan)
    return payoff


if __name__ == '__main__':
    # Set print options to print the entire matrix
    np.set_printoptions(threshold=10 ** 5, linewidth=10 ** 5)

    # Initial set up
    market_1 = Market(r=0, T=3)
    market_2 = Market(r=0.025, T=3)

    stock_A = Stock(S_0=100, step_up=20, step_down=20, step_type='abs')
    stock_B = Stock(S_0=100, step_up=1.2, step_down=1.25, step_type='rel')

    # No interest call option example
    option1 = VanillaOptionBTM(payoff_func=VanillaOptionBTM.call_option_strike100_payoff,
                               type='EUR',
                               payoff_func_desc='This derivative is a EUR call option with strike 100.')
    option1.simulate_price_and_replication(stock=stock_A, market=market_1, verbose=True)
    option1.generate_filtration_table(['down', 'up', 'down'], market_1.T)

    # No interest binary option example
    option2 = VanillaOptionBTM(payoff_func=binary_option_payoff,
                               type='EUR',
                               payoff_func_desc='This EUR derivative pays 100 if the stock price at maturity is '
                                                'greater than \nor equal to 100.')
    option2.simulate_price_and_replication(stock=stock_B, market=market_1, verbose=True)
    option2.generate_filtration_table(['down', 'up', 'down'], market_1.T)

    # Comparison of PVs of EUR and AME options
    option3 = VanillaOptionBTM(payoff_func=VanillaOptionBTM.put_option_strike100_payoff,
                               type='EUR',
                               payoff_func_desc='This derivative is a EUR put option with strike 100.')
    option3.simulate_price_and_replication(stock=stock_B, market=market_2, verbose=True)
    IntRate_delta_rist(option2, stock_B, market_2)  # Compute risk

    option4 = VanillaOptionBTM(payoff_func=VanillaOptionBTM.put_option_strike100_payoff,
                               type='AME',
                               payoff_func_desc='This derivative is an AME put option with strike 100.')
    option4.simulate_price_and_replication(stock=stock_B, market=market_2, verbose=True)
