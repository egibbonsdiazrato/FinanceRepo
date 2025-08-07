from datetime import date

from modules.Mortgage import InterestMortgage, RepaymentMortgage

if __name__ == '__main__':
    mortgage = RepaymentMortgage(500000, 125000, date(2026, 1, 1), 35, 0.070)
    print(mortgage.get_payment_schedule())

    mortgage2 = InterestMortgage(500000, 125000, date(2026, 1, 1), 35, 0.070)
    print(mortgage2.get_payment_schedule())
