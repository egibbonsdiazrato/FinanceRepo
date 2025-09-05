# Modern Portfolio Theory:

For more background, visit [Modern Portfolio Theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory) on 
Wikipedia. Note that for MPT, as the returns of the portfolio are a weighted arithmetic average, we use **simple returns**
not log returns.

The key equations required for the script are:

$$ E(r_{p})=\sum_{i}w_{i}E(r_{i})\quad $$

$$\sigma_{p}^{2} = \sum_{i}w_{i}^{2}\sigma_{i}^{2} + \sum_{i}\sum_{j\neq i}w_{i}w_{j}\sigma_{i}\sigma_{j}\rho_{ij} = \sum_{i}w_{i}^{2}\sigma_{i}^{2} + \sum_{i}\sum_{j\neq i}w_{i}w_{j}Cov(i,j)$$

where $i$ is the ith asset, $r$ is return, $w$ is weight, $\sigma$ is volatility and $\rho$ is correlation. Note $Cov(X,Y) = \sigma_X \sigma_Y \rho $

These equations simplify to the following for a two-asset portfolio:

$$E(r_{p}) = w_{a}E(r_{a}) + w_{b}E(r_{b}) = w_{a}E(r_{a}) + (1-w_{a})E(r_{b})$$

$$\sigma_{p}^{2} = w_{a}^{2}\sigma_{a}^{2} + w_{b}^{2}\sigma_{b}^{2} + 2w_{a}w_{b}\sigma_{a}\sigma_{b}\rho_{ab} =  w_{a}^{2}\sigma_{a}^{2} + w_{b}^{2}\sigma_{b}^{2} + 2w_{a}w_{b}Cov(a,b) $$