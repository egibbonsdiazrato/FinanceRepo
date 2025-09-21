from datetime import date

from src.Mortgage import InterestMortgage, RepaymentMortgage

if __name__ == '__main__':
    repayment_mortgage = RepaymentMortgage(500000., 125000., date(2026, 1, 1), 35, 0.050)
    repayment_mortgage.print_payment_schedule()

    mortgage_int = InterestMortgage(500000., 125000., date(2026, 1, 1), 35, 0.050)
    mortgage_int.print_payment_schedule()
