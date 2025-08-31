import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

def multi_asset_MC(mu: int | float,
                   sigma: int | float,
                   N_assets: int,
                   N_obs: int,
                   N_portfolio: int) -> None:
    """
    MC simulation of multi-asset MPT to find and plot the expected return vs volatility of multiple portfolios

    Args:
        mu: average
        sigma: standard deviations
        N_assets: number of assets to be simulated
        N_obs: number of timeperiods over which returns will be generated
        N_portfolio: number of random portfolios to be generated
    """

    # Generate the returns (matrix) of the assets, where each row is an asset and the columns are timesteps (or
    # observations)
    ret = np.random.normal(mu, sigma, (N_assets, N_obs))
    ret_mean = np.mean(ret, axis=1)

    # Storage arrays
    expected_ret_portfolio_arr = np.array([])
    vol_portfolio_arr = np.array([])

    for _ in range(0, N_portfolio):
        # Compute portfolio expected return
        weights_lims = np.array([0.0, 1.0])
        weights = np.random.uniform(low=weights_lims[0], high=weights_lims[1], size=N_assets)
        weights /= np.sum(weights)  # Normalise

        # Compute expected return and std. deviation of the portfolio
        exp_ret_port = np.transpose(ret_mean) @ weights
        port_cov = np.cov(ret)
        vol_port = np.sqrt(np.transpose(weights) @ port_cov @ weights)

        # Append the values to the arrays
        expected_ret_portfolio_arr = np.append(expected_ret_portfolio_arr, exp_ret_port)
        vol_portfolio_arr = np.append(vol_portfolio_arr, vol_port)


    # Plot
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    ax.scatter(100*vol_portfolio_arr, 100*expected_ret_portfolio_arr, s=12)

    # Find the minimum risk point
    min_risk_index = np.argmin(vol_portfolio_arr)
    ax.scatter(100*vol_portfolio_arr[min_risk_index], 100*expected_ret_portfolio_arr[min_risk_index],
               c='violet', marker='+', s=75,
               label=f'Min. vol ({100*vol_portfolio_arr[min_risk_index]:.2f}, {100*expected_ret_portfolio_arr[min_risk_index]:.2f})')

    ax.set_title('Ann. Portfolio Returns against Volatility')
    ax.set_xlabel(r'$\sigma$ (%)')
    ax.set_ylabel(r'$r$ (%)')

    ax.grid(True, which='both', linewidth=0.5, alpha=0.75)
    ax.legend(loc='lower right')

    plt.show()


def nasdaq_portfolio_MC(N_portfolio: int) -> None:
    """
    Runs MPT MC simulation with different weights of five Nasdaq stocks.
    # TODO: add minimisation

    Args:
        N_portfolio: number of random portfolios to be generated
    """

    tickers = ['NVDA', 'MSFT', 'AAPL','AMZN', 'META']

    data = yf.download(tickers, start='2022-08-29', end='2023-08-29')['Close']
    data = data[tickers] # Reorganise
    print('Last few days of data:')
    print(data.tail())

    rets = data.pct_change().dropna()
    expected_rets, cov_rets = 252*rets.mean(), 252*rets.cov() # annualised (252x)

    print('Expected returns')
    print(expected_rets)

    print('Covariance of returns')
    print(cov_rets)

    # Generate random portfolios with different asset weights
    weights = np.random.dirichlet(np.ones(len(tickers)), size=N_portfolio) # generates normalised weights
    portfolio_rets, portfolio_sigma = np.array([]), np.array([])

    for sim in range(N_portfolio):
        portfolio_rets = np.append(portfolio_rets, weights[sim].T @ expected_rets.to_numpy())
        portfolio_sigma = np.append(portfolio_sigma, weights[sim].T @ cov_rets.to_numpy() @ weights[sim])

    # Plot
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    portfolio_sigma_pct, portfolio_rets_pct = 100*portfolio_sigma, 100*portfolio_rets
    ax.scatter(portfolio_sigma_pct, portfolio_rets_pct, s=12)

    # Find the minimum risk point
    min_risk_index = np.argmin(portfolio_sigma_pct)
    ax.scatter(portfolio_sigma_pct[min_risk_index], portfolio_rets_pct[min_risk_index],
               c='violet', marker='+', s=75,
               label=f'Min. vol ({portfolio_sigma_pct[min_risk_index]:.2f}, {portfolio_rets_pct[min_risk_index]:.2f})')

    ax.set_title('Ann. Portfolio Returns against Volatility')
    ax.set_xlabel(r'$\sigma$ (%)')
    ax.set_ylabel(r'$r$ (%)')

    ax.grid(True, which='both', linewidth=0.5, alpha=0.75)
    ax.legend(loc='lower right')

    plt.show()
