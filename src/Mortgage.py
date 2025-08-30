from abc import ABC, abstractmethod
from datetime import date
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
        self.loan = self.price - self.deposit

        self.start_date = start_date
        self.term = term
        self.periods = 12 * self.term

        self.annual_rate = annual_rate
        self.monthly_rate = self.annual_rate / 12

    @abstractmethod
    def get_payment_schedule(self) -> pl.DataFrame:
        pass


class InterestMortgage(Mortgage):
    """
    A class to model payments for an interest-only mortgage.
    """
    def __init__(self,
                 price: float,
                 deposit: float,
                 start_date: date,
                 term: int,
                 annual_rate: float) -> None:
        super().__init__(price, deposit, start_date, term, annual_rate)

        self.monthly_repayment = self.monthly_rate*self.loan

    def get_payment_schedule(self) -> pl.DataFrame:

        dates = [date(year, month, 1) for year in range(self.start_date.year, self.start_date.year + self.term) for month in range(1, 13)]

        monthly_repayments = self.monthly_repayment*np.ones(self.periods)

        outstanding_balance = self.loan*np.ones_like(monthly_repayments)

        df = pl.DataFrame({'Dates': dates,
                           'Outstanding Balance': outstanding_balance,
                           'Monthly Repayments Due': monthly_repayments})

        return df

class RepaymentMortgage(Mortgage):
    """
    A class to model payments for a repayment mortgage.
    """
    def __init__(self,
                 price: float,
                 deposit: float,
                 start_date: date,
                 term: int,
                 annual_rate: float) -> None:
        super().__init__(price, deposit, start_date, term, annual_rate)

        self.monthly_repayment = self.monthly_rate * self.loan * ((1 + self.monthly_rate) ** self.periods) / ((1 + self.monthly_rate) ** self.periods - 1)



    def get_payment_schedule(self) -> pl.DataFrame:
        dates = [date(year, month, 1) for year in range(self.start_date.year, self.start_date.year + self.term) for month in range(1, 13)]

        monthly_repayments = self.monthly_repayment*np.ones(self.periods)

        first_capital_repayment = self.monthly_repayment - self.monthly_rate*self.loan
        capital_repayment = np.array([first_capital_repayment*(1+self.monthly_rate)**period for period in range(0, self.periods)])
        interest_repayment = monthly_repayments - capital_repayment

        outstanding_balance = np.array([self.loan - np.sum(capital_repayment[0:period]) for period in range(0, self.periods)])

        df = pl.DataFrame({'Dates': dates,
                           'Outstanding Balance': outstanding_balance,
                           'Int. Repayment Due': interest_repayment,
                           'Cap. Repayment Due': capital_repayment,
                           'Monthly Repayments Due': monthly_repayments})

        return df
