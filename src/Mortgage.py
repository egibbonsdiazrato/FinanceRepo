from abc import ABC, abstractmethod
from datetime import date
from functools  import lru_cache
import numpy as np
import polars as pl


class Mortgage(ABC):
    """
    A base class to model mortgages which require monthly payments.
    """
    def __init__(self,
                 price: float,
                 deposit: float,
                 start_date: date,
                 term: int,
                 annual_rate: float) -> None:
        self.price = price
        self.deposit = deposit
        self.loan_amount = self.price - self.deposit

        self.start_date = start_date
        self.term = term
        self.periods = 12 * self.term
        self.payment_dates = [date(year, month, 1)
                              for year in range(self.start_date.year, self.start_date.year + self.term)
                              for month in range(1, 13)]

        self.annual_rate = annual_rate
        self.monthly_rate = self.annual_rate / 12

    @abstractmethod
    def _interest_repayments(self) -> np.array:
        pass

    @abstractmethod
    def _capital_repayments(self) -> np.array:
        pass

    @abstractmethod
    def _monthly_repayments(self) -> np.array:
        pass

    def _gen_payment_schedule(self) -> pl.DataFrame:
        """
        A function which returns a polars dataframe with the payment schedule

        Returns:
            df: polars dataframe with payment schedule
        """
        interest_repayments = self._interest_repayments()
        capital_repayments = self._capital_repayments()
        monthly_repayments = self._monthly_repayments()

        opening_balances = np.array([self.loan_amount - np.sum(capital_repayments[0:period]) for period in range(0, self.periods)])
        closing_balances = opening_balances - capital_repayments
        closing_balances = np.concat((closing_balances[:-1], np.array([0]))) # Enforce last payment exactly zero

        df = pl.DataFrame({'Date': self.payment_dates,
                           'Opening Balance': opening_balances.tolist(),
                           'Interest Repayment Due': interest_repayments.tolist(),
                           'Capital Repayment Due': capital_repayments.tolist(),
                           'Monthly Repayment Due': monthly_repayments.tolist(),
                           'Closing Balance': closing_balances.tolist()})

        return df

    def print_payment_schedule(self, dates: list[date] = None) -> None:
        """
        Prints payment schedule for certain dates if supplied.

        Args:
            dates: optional dates to be printed in the df
        """
        df = self._gen_payment_schedule().with_columns(pl.col(pl.Float64).round(2))

        if dates is not None:
            df = df.filter(pl.col("Date").is_in(dates))

        cols_to_round = df.columns[1:]
        df = df.with_columns([pl.col(col_name).cast(pl.Float64).round(2) for col_name in cols_to_round])

        print(df)


class InterestMortgage(Mortgage):
    """
    A class to model payments for an interest-only mortgage.
    """
    @lru_cache
    def _interest_repayments(self) -> np.array:
        """
        Calculations of interest payments as per documentation

        Returns:
            interest_repayments: array with interest repayments
        """
        interest_repayments = self.monthly_rate*self.loan_amount*np.ones_like(self.payment_dates)

        return interest_repayments

    @lru_cache
    def _capital_repayments(self) -> np.array:
        """
        Calculations of capital payments as per documentation

        Returns:
            capital_repayments: array with capital repayments
        """
        capital_repayments = np.zeros_like(self.payment_dates, dtype=float)
        capital_repayments[-1] = self.loan_amount

        return capital_repayments

    def _monthly_repayments(self) -> np.array:
        """
        Calculations of total monthly payments as per documentation

        Returns:
            monthly_repayments: array with monthly repayments
        """
        monthly_repayments = self._interest_repayments() + self._capital_repayments()

        return monthly_repayments


class RepaymentMortgage(Mortgage):
    """
    A class to model payments for a repayment mortgage.
    """
    @lru_cache
    def _monthly_repayments(self) -> np.array:
        """
        Calculations of total monthly payments as per documentation

        Returns:
            monthly_repayments: array with monthly repayments
        """
        monthly_repayment = self.monthly_rate*self.loan_amount*((1 + self.monthly_rate)**self.periods)/((1 + self.monthly_rate)**self.periods - 1)
        monthly_repayments = monthly_repayment*np.ones_like(self.payment_dates)

        return monthly_repayments

    @lru_cache
    def _capital_repayments(self) -> np.array:
        """
        Calculations of capital payments as per documentation

        Returns:
            capital_repayments: array with capital repayments
        """
        first_capital_repayment = self._monthly_repayments()[0] - self.monthly_rate * self.loan_amount
        capital_repayments = np.array([first_capital_repayment * (1 + self.monthly_rate) ** period for period in range(0, self.periods)])

        return capital_repayments

    def _interest_repayments(self) -> np.array:
        """
        Calculations of interest payments as per documentation

        Returns:
            interest_repayments: array with interest repayments
        """
        interest_repayments = self._monthly_repayments() - self._capital_repayments()

        return interest_repayments
