# Binomial Tree Model:

The equations governing this system are (Baxter and Rennie's _Financial Calculus_):

$$ q = \frac{e^{r \delta t}s_{now} - s_{down}}{s_{up} - s_{down}} $$

$$ f_{now} = e^{-r \delta t} \left( qf_{up} + \left(1 - q \right) f_{down} \right) $$

$$ V_t = B_t E_Q \left( B_T^{-1}X | F_t \right) \implies V_0 = E_Q(B_T^{-1}X) $$

$$ \phi = \frac{f_{up} - f_{down}}{s_{up} - s_{down}} $$

$$ \psi = B^{-1}_{now}\left( f_now - \phi S_now \right) $$

where $Q$ is the risk-free measure, $q$ is the $Q$-measure probability of an up-jump; $r$ is the interest rate;
$s$ is the stock price process; $B$ is the bond price process, where $B_0 = 1$; $f$ is the derivative 
value time-process; $\phi$ is the stock holding strategy; $\psi$ is the bond holding or borrowing strategy, $X$ is the
payoff of the derivative at maturity; $V$ is the value of the derivative at time $t$ given a filtration $F_t$ and $T$ 
is the time to maturity of the derivative.