from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Callable

from src.Stock import Stock
from src.Market import Market

class BaseOptionBTM(ABC):
    """
    This abstract base class holds the information of a derivative to be modelled. The attributes of this class are the
    payoff function and its description.
    """
    def __init__(self, payoff_func: Callable[[np.ndarray], np.ndarray], payoff_func_desc: str | bool = None) -> None:
        """
        Derivative constructor.

        Args:
            payoff_func: Function which calculates the payoff at maturity of the derivative to be modelled.
            payoff_func_desc: Description of the payoff_func, which defaults to None.
        """
        # Save inputs attributes
        self.payoff_func = payoff_func
        self.payoff_func_desc = payoff_func_desc

        # Flags
        self.simulated = False

        # Placeholder attributes
        self.N_cols = None
        self.N_rows = None
        self.initial_inds = None

        self.stock_tree = None
        self.deriv_tree = None
        self.hedge_tree = None
        self.borrow_tree = None

    def simulate_price_and_replication(self, stock: Stock, market: Market, verbose: bool) -> None:
        """
        Performs the generation of the four trees required: stock, derivative, hedge and borrowing.

        Args:
            stock: the stock object holding the required information for the derivative price and replication modelling.
            market: the market object holding the required information for the derivative price and replication
            modelling.
            verbose: verbosity flag.

        Returns:
            deriv_PV: present value of the derivative.
        """
        # Exceptions
        if stock.step_abs:
            # Check that the stock price can never be negative
            if market.T >= np.ceil(stock.S_0 / stock.step_down):
                raise Exception('Reduce the number of timesteps to ensure that the stock cannot have a negative price.')

        # Ensures simulation only performed once
        if not self.simulated:
            # Initial set up of simulation grid
            self.N_cols = market.T + 1  # Cols required for binomial tree model matrix
            self.N_rows = 2 * self.N_cols - 1  # Rows required for binomial tree model matrix
            self.initial_inds = np.array([int(self.N_rows // 2), 0])  # The index pair for point at t = 0

            # Verbose prints
            if verbose:
                self._verbose_header(market.T, market.r, stock.S_0)

            # Calculation of all trees
            self._gen_stock_tree(stock, verbose)
            self._calc_deriv_tree(market, verbose)
            self._calc_hedges(verbose)
            self._calc_borrow(market.r, verbose)

            # Activate simulated flag
            self.simulated = True

    def _verbose_header(self, T: int, r: float, S_0: int | float) -> None:
        """
        Prints a header to the console.

        Args:
            T: Total number of timesteps of the simulation.
            r: Interest rate of the simulation.
            S_0: Starting price of the stock.
        """
        print()
        print(f'{"Binomial Tree Model Instance":=^80}')
        if self.payoff_func_desc is not None:
            print(f'{self.payoff_func_desc}')
            print(f'This is modelled with initial stock price of S_0 = {S_0}, int. rates of '
                  f'r = {100*r}%\nand a maturity of T = {T}.')

    @staticmethod
    def _calc_up_stock_price(S_now: float, step: int | float, step_abs: bool) -> int | float:
        """
        Calculates the stock up price for the timestep after S_now.
        Args:
            S_now: The price of the stock now.
            step: Stock object which holds the attributes of stock movement.
            step_abs: flag which dictates whether movement is absolute (else relative).

        Returns:
            S_up: The price of the stock if it goes up the following timestep.
        """

        if step_abs:
            S_up = S_now + step
            return S_up

        else:
            S_up = S_now*step
            return S_up

    @staticmethod
    def _calc_down_stock_price(S_now: float, step: int | float, step_abs: bool) -> int | float:
        """
        Calculates the stock down price for the timestep after S_now.
        Args:
            S_now: The price of the stock now.
            step: Stock object which holds the attributes of stock movement.
            step_abs: flag which dictates whether movement is absolute (else relative).

        Returns:
            S_down: The price of the stock if it goes down the following timestep.
        """

        if step_abs:
            S_down = S_now - step
            return S_down

        else:
            S_down = S_now*(1/step)
            return S_down

    def _gen_stock_tree(self, stock: Stock, verbose: bool) -> None:
        """
        Generates the stock tree in matrix form where empty cells are filled with nans. The start, t = 0, is shown
        as the leftmost column of the matrix. Whereas, the end, t = T, is shown as the rightmost column of the
        matrix.

        Args:
            stock: stock object with the required attributes.
            verbose: verbosity flag.
        """
        # Initialise matrix
        stock_tree = np.full((self.N_rows, self.N_cols), np.nan)

        # Add initial stock price
        stock_tree[self.initial_inds[0], self.initial_inds[1]] = stock.S_0

        # Generate the up and down values for each timestep by looking at non-zero values in the previous timestep
        for t in range(1, self.N_cols):
            # Indices for which there are prices in the previous timesteps
            inds = np.where(~np.isnan(stock_tree[:, t - 1]))[0]

            # Generate an up and down price for each non-zero price of the previous timestep
            for ind_now in inds:
                S_now = float(stock_tree[ind_now, t - 1])
                ind_up, ind_down = ind_now - 1, ind_now + 1  # Note up and down refers to stock price move
                stock_tree[ind_up, t] = self._calc_up_stock_price(S_now, stock.step_up, stock.step_abs)
                stock_tree[ind_down, t] = self._calc_down_stock_price(S_now, stock.step_down, stock.step_abs)

        # Save as attributes
        self.stock_tree = stock_tree

        # Verbose prints
        if verbose:
            print()
            print('The stock tree:')
            print(self.stock_tree)

    @abstractmethod
    def _calc_deriv_tree(self, market: Market, verbose: bool) -> None:
        """
        Calculates the derivative tree in matrix through backpropagation form where empty cells are filled with nans.
        The start, t = 0, is shown as the leftmost column of the matrix. Whereas, the end, t = T - 1, is shown
        as the rightmost column of the matrix.

        Args:
            market: market object with the required attributes.
            verbose: verbosity flag.
        """
        pass

    def _calc_hedges(self, verbose: bool) -> None:
        """
        Calculates the hedge tree using the stock and derivative tree in matrix form where empty cells are
        filled with nans. The start, t = 0, is shown as the leftmost column of the matrix. Whereas, the end,
        t = T - 1, is shown as the rightmost column of the matrix.
        """
        # Initialise the hedge tree
        hedge_tree = np.full_like(self.stock_tree, np.nan)

        # Compute backpropagation
        t_backprop_steps = np.arange(1, self.N_cols)[::-1]  # Reversed array
        for t in t_backprop_steps:
            # Indices for which there are prices in the previous timesteps.
            inds = np.where(~np.isnan(self.deriv_tree[:, t]))[0]

            # Generate the backpropagation by looking at pairs of indices
            for ind_up, ind_down in zip(inds[:-1], inds[1:]):
                # The row below of up ind (equivalent to above the down ind) is that of the previous timestep
                ind_now = ind_up + 1

                # Get stock and derivative up and down values to calculate hedge now value
                stock_up = self.stock_tree[ind_up, t]
                stock_down = self.stock_tree[ind_down, t]
                deriv_up = self.deriv_tree[ind_up, t]
                deriv_down = self.deriv_tree[ind_down, t]

                hedge_tree[ind_now, t - 1] = (deriv_up - deriv_down) / (stock_up - stock_down)

        # Save as attributes
        self.hedge_tree = hedge_tree

        # Verbose prints
        if verbose:
            print('The hedge tree:')
            print(self.hedge_tree)

    def _calc_borrow(self, r: int | float, verbose: bool) -> None:
        """
        Calculates the borrowing tree using the stock and derivative tree in matrix form where empty cells are
        filled with nans. The start, t = 0, is shown as the leftmost column of the matrix. Whereas, the end,
        t = T - 1, is shown as the rightmost column of the matrix.
        """
        # Initialise the borrow tree
        borrow_tree = np.full_like(self.stock_tree, np.nan)

        # Compute forward propagation
        t_fwdprop_steps = np.arange(0, self.N_cols - 1)
        for t in t_fwdprop_steps:
            # Indices for which there are prices in the previous timesteps.
            inds_now = np.where(~np.isnan(self.deriv_tree[:, t]))[0]

            # Consider each row
            for ind_now in inds_now:
                # Get stock, derivative and hedge now values to calculate borrow now value
                stock_now = self.stock_tree[ind_now, t]
                deriv_now = self.deriv_tree[ind_now, t]
                hedge_now = self.hedge_tree[ind_now, t]
                DF = np.exp(-1*r*t)  # Discount Factor

                borrow_tree[ind_now, t] = DF*(deriv_now - hedge_now*stock_now)

        # Save as attributes
        self.borrow_tree = borrow_tree

        # Verbose prints
        if verbose:
            print('The bond holding tree:')
            print(self.borrow_tree)

    def _seq_to_inds(self, seq: list[str]) -> np.ndarray:
        """
        Translates the up and down sequence into an array which contains all the pairs of indices to follow a
        specific filtration.

        Args:
            seq: list which contains either 'up' or 'down'.

        Returns:
            inds: np.ndarray of integers with the indices.
        """
        row_prefactor = {'up': -1, 'down': 1}

        inds = np.zeros((self.N_cols, 2))
        inds[0, :] = self.initial_inds

        for counter, move in enumerate(seq, 1):
            inds[counter, :] = inds[counter - 1, :] + np.array([row_prefactor[move], 1])

        return inds.astype(int)

    def generate_filtration_table(self, seq: list[str], T: int) -> None:
        """
        Generate the filtration and print it to console with the value of each tree for each timestep in pandas form.

        Args:
            seq: list which contains either 'up' or 'down' to be passed to _seq_to_inds.
            T: number of timesteps in the filtration.
        """

        # Raise error if tree not generated
        if not self.simulated:
            raise Exception('The trees have not been generated.')

        # Raise error if sequence is not the correct length
        if len(seq) != T:
            raise Exception(f'The sequence is {len(seq)=} and not {T}')

        # Get the indices for the filtration
        inds_path = self._seq_to_inds(seq)

        # Produce table
        filtration_table = pd.DataFrame()
        filtration_table['t_i'] = np.arange(0, self.N_cols)
        filtration_table.set_index(keys=['t_i'], inplace=True)  # Make time the index
        filtration_table['Prev. Jump'] = ['-'] + seq
        filtration_table['S_i'] = self.stock_tree[inds_path[:, 0], inds_path[:, 1]]
        filtration_table['V_i'] = self.deriv_tree[inds_path[:, 0], inds_path[:, 1]]
        filtration_table[r'\phi_i'] = np.concatenate((np.array([np.nan]),
                                                      self.hedge_tree[inds_path[:-1, 0], inds_path[:-1, 1]]))
        filtration_table[r'\psi_i'] = np.concatenate((np.array([np.nan]),
                                                      self.borrow_tree[inds_path[:-1, 0], inds_path[:-1, 1]]))

        # Verbose prints
        print()
        print(f'The filtration table for the sequence {seq}:')
        print(filtration_table.to_string())


class VanillaOptionBTM(BaseOptionBTM):
    """
    A class to model Vanilla options. The only method required to implement is _calc_deriv_tree.
    """
    def __init__(self, payoff_func: Callable[[np.ndarray], np.ndarray], type: str, payoff_func_desc: str | bool = None) -> None:
        """
        Derivative constructor.

        Args:
            payoff_func: Function which calculates the payoff at maturity of the derivative to be modelled.
            payoff_func_desc: Description of the payoff_func, which defaults to None.
            is_AME: flag which defaults to false (making the option European).
        """
        super().__init__(payoff_func, payoff_func_desc)  # Call parent constructor

        # Decide whether European or American option
        self.type = type
        if self.type not in ['EUR', 'AME']:
            raise Exception('The type input is neither EUR (European) nor AME (American). Please use provide one '
                            'of these two')

        self.is_AME = True if type == 'AME' else False # Assumes only EUR or AME

    def _calc_deriv_tree(self, market: Market, verbose: bool) -> None:
        """
        Calculates the derivative tree in matrix through backpropagation form where empty cells are filled with nans.
        The start, t = 0, is shown as the leftmost column of the matrix. Whereas, the end, t = T - 1, is shown
        as the rightmost column of the matrix.

        Args:
            market: market object with the required attributes.
            verbose: verbosity flag.
        """
        # Initialise the matrix and find the last column by applying the payoff func
        deriv_tree = np.full_like(self.stock_tree, np.nan)
        stock_mat = self.stock_tree[:, -1]
        deriv_payoff_mat = self.payoff_func(stock_mat).reshape(-1)

        # Add the maturity row for derivative
        deriv_tree[:, -1] = deriv_payoff_mat

        # Compute backpropagation
        t_backprop_steps = np.arange(1, self.N_cols)[::-1]  # Reversed array
        for t in t_backprop_steps:
            # Indices for which there are prices in the previous timesteps.
            inds = np.where(~np.isnan(deriv_tree[:, t]))[0]

            # Generate the backpropagation by looking at pairs of indices
            for ind_up, ind_down in zip(inds[:-1], inds[1:]):
                # The row below of up ind (equivalent to above the down ind) is that of the previous timestep
                ind_now = ind_up + 1

                # Calculate q probabilities
                q_num = np.exp(market.r*market.deltat)*self.stock_tree[ind_now, t - 1] - self.stock_tree[ind_down, t]
                q_denom = self.stock_tree[ind_up, t] - self.stock_tree[ind_down, t]
                q = q_num / q_denom

                # Get derivative up and down values and calculate derivative now value
                deriv_up = deriv_tree[ind_up, t]
                deriv_down = deriv_tree[ind_down, t]
                deriv_now = np.exp(-1*market.r*market.deltat)*(q*deriv_up + (1 - q)*deriv_down)

                deriv_tree[ind_now, t - 1] = (deriv_now if not self.is_AME  # EUR
                                              else max(deriv_now, self.payoff_func(self.stock_tree[ind_now, t - 1]))) # AME

        # Save as attributes
        self.deriv_tree = deriv_tree
        self.deriv_PV = deriv_tree[self.initial_inds[0], self.initial_inds[1]]

        # Verbose prints
        if verbose:
            print('The derivative tree:')
            print(self.deriv_tree)
            print(f'and the PV is {self.deriv_PV}')

    @staticmethod
    def call_option_strike100_payoff(S_t: int | float | np.ndarray) -> int | float | np.ndarray:
        """
        Produces in-built European call option with strike 100 payoffs for stock prices at maturity.

        Args:
            S_t: the stock price at time t.

        Returns:
            payoff: the payoff of S_T for the option.
        """
        K = 100  # Strike value

        if isinstance(S_t, int) or isinstance(S_t, float):
            payoff = max(0, S_t - K)
        else:
            payoff = np.where(~np.isnan(S_t), np.where(S_t < K, 0, S_t - K), np.nan)
        return payoff

    @staticmethod
    def put_option_strike100_payoff(S_t: int | float | np.ndarray) -> int | float | np.ndarray:
        """
        Produces in-built European put option with strike 100 payoffs for stock prices at maturity.

        Args:
            S_t: the stock price at time t.

        Returns:
            payoff: the payoff of S_T for the option.
        """
        K = 100  # Strike value

        if isinstance(S_t, int) or isinstance(S_t, float):
            payoff = max(0, K - S_t)
        else:
            payoff = np.where(~np.isnan(S_t), np.where(S_t > K, 0, K - S_t), np.nan)
        return payoff


def IntRate_delta_rist(deriv: VanillaOptionBTM, stock: Stock, market: Market) -> None:
    """
    Computes the interest rate delta numerically by shifting the interest rate by 1 b.p. and
    Args:
        deriv: the original derivative object.
        stock: the stock object used to price the original derivative.
        market: the market object used to price the original derivative.
    """
    if not deriv.simulated:
        raise TypeError(f'The derivative price has to be calculated before this functions is applied.')

    base_PV = deriv.deriv_PV

    # Recompute shifted PV for shifted market
    shifted_market = Market(r=market.r + 10**(-4), T=market.T)
    shifted_deriv = VanillaOptionBTM(payoff_func=deriv.payoff_func,
                                     type=deriv.type,
                                     payoff_func_desc=deriv.payoff_func_desc)

    # Not the most efficient way as hedges and borrows are not required
    shifted_deriv.simulate_price_and_replication(stock=stock, market=shifted_market, verbose=False)

    shifted_PV = shifted_deriv.deriv_PV

    # Compute difference
    IntRate_delta = shifted_PV - base_PV
    print()
    print(f'The interest rate delta is {IntRate_delta} USD for a shift from {100*market.r}% to {100*shifted_market.r}%')
