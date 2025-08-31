from src.ModernPortfolioTheory import multi_asset_MC, nasdaq_portfolio_MC


if __name__ == '__main__':

    multi_asset_MC(0.05,
                   0.025,
                   10,
                   100,
                   10000)

    nasdaq_portfolio_MC(10000)