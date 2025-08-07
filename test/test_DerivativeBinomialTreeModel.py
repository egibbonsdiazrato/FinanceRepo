import numpy as np
from unittest import TestCase

from src.modules.DerivativeModelling import Market, Stock, VanillaOptionBTM


class TestOptionBTM(TestCase):
    """
    A unit test for a simple EUR call option with strike at 100 which has been solved manually.
    """
    def setUp(self) -> None:
        """
        Set up function to instantiate class to be tested and save expected results as attributes.
        """
        # Initialise and perform simulation
        market_test = Market(r=0, T=3)
        stock_test = Stock(S_0=100, step_up=20, step_down=20, step_type='abs')
        self.option_test = VanillaOptionBTM(payoff_func=VanillaOptionBTM.call_option_strike100_payoff,
                                            payoff_func_desc='This derivative is a EUR call option with strike 100.')
        self.option_test.simulate_price_and_replication(stock=stock_test, market=market_test, verbose=True)

        # Expected results
        self.exp_stock_tree = np.array([[np.nan, np.nan, np.nan, 160.],
                                        [np.nan, np.nan, 140., np.nan],
                                        [np.nan, 120., np.nan, 120.],
                                        [100., np.nan, 100., np.nan],
                                        [np.nan,  80., np.nan,  80.],
                                        [np.nan, np.nan, 60., np.nan],
                                        [np.nan, np.nan, np.nan,  40.]])
        self.exp_deriv_tree = np.array([[np.nan, np.nan, np.nan, 60.],
                                        [np.nan, np.nan, 40., np.nan],
                                        [np.nan, 25., np.nan, 20.],
                                        [15., np.nan, 10., np.nan],
                                        [np.nan, 5., np.nan, 0.],
                                        [np.nan, np.nan, 0., np.nan],
                                        [np.nan, np.nan, np.nan, 0.]])
        self.exp_hedge_tree = np.array([[np.nan, np.nan, np.nan, np.nan],
                                        [np.nan, np.nan, 1., np.nan],
                                        [np.nan, 0.75, np.nan, np.nan],
                                        [0.5, np.nan, 0.5, np.nan],
                                        [np.nan, 0.25, np.nan, np.nan],
                                        [np.nan, np.nan, 0., np.nan],
                                        [np.nan, np.nan, np.nan, np.nan]])
        self.exp_borrow_tree = np.array([[np.nan, np.nan, np.nan, np.nan],
                                         [np.nan, np.nan, -100., np.nan],
                                         [np.nan, -65., np.nan, np.nan],
                                         [-35., np.nan, -40., np.nan],
                                         [np.nan, -15., np.nan, np.nan],
                                         [np.nan, np.nan, 0., np.nan],
                                         [np.nan, np.nan, np.nan, np.nan]])

    def test_stock_tree(self) -> None:
        """
        Test to ensure matching stock trees.
        """
        self.assertTrue(np.allclose(self.option_test.stock_tree, self.exp_stock_tree,
                                    atol=1e-14, rtol=1e-14, equal_nan=True))

    def test_deriv_tree(self) -> None:
        """
        Test to ensure matching stock trees.
        """
        self.assertTrue(np.allclose(self.option_test.deriv_tree, self.exp_deriv_tree,
                                    atol=1e-14, rtol=1e-14, equal_nan=True))

    def test_hedge_tree(self) -> None:
        """
        Test to ensure matching stock trees.
        """
        self.assertTrue(np.allclose(self.option_test.hedge_tree, self.exp_hedge_tree,
                                    atol=1e-14, rtol=1e-14, equal_nan=True))

    def test_borrow_tree(self) -> None:
        """
        Test to ensure matching stock trees.
        """
        self.assertTrue(np.allclose(self.option_test.borrow_tree, self.exp_borrow_tree,
                                    atol=1e-14, rtol=1e-14, equal_nan=True))
