# Mortgage
Variables:
- $L$: loan amount,
- $r = \frac{r_{annual}}{12}$: **monthly** interest rate,
- $T$: the term in years.

We are considering mortgages which require monthly repayments. 

## Repayment Mortgage

| Period | Outstanding balance       | Int. Repayment | Cap. Repayment | Monthly Repayment |
|--------|---------------------------|----------------|----------------|-------------------|
| 1      | $L$                       | $I_1$          | $C_1$          | $M_1=I_1+C_1$     | 
| 2      | $L - C_1$                 | $I_2$          | $C_2$          | $M_2=I_2+C_2$     |
| ...    | ...                       | ...            | ...            | ...               |
| n      | $L - \sum_{n=1}^{n-1}C_n$ | $I_n$          | $C_n$          | $M_n=I_n+C_n$     |

Consider the outstanding interest after the first and second period:
$$I_1 = rL$$ 
$$I_2 = r(L - C_1)$$
By inspection, one can see that $I_n = r(L - \sum_{n=1}^{n-1}C_n)$.

| Period | Outstanding balance       | Int. Repayment               | Cap. Repayment | Monthly Repayment |
|--------|---------------------------|------------------------------|----------------|-------------------|
| 1      | $L$                       | $rL$                         | $C_1$          | $M_1=I_1+C_1$     | 
| 2      | $L - C_1$                 | $r(L - C_1)$                 | $C_2$          | $M_2=I_2+C_2$     |
| ...    | ...                       | ...                          | ...            | ...               |
| n      | $L - \sum_{n=1}^{n-1}C_n$ | $r(L - \sum_{n=1}^{n-1}C_n)$ | $C_n$          | $M_n=I_n+C_n$     |

Mortgages, by construct, have constant monthly payments. Thus, $M_1=M_2=M_n=M$ must be true. Enforce this for the first two periods:


$$rL + C_1 = r(L - C_1) + C_2 \implies C_2 = (1+r)C_1$$

The general recursive relation is:
$$r(L - \sum_{n=1}^{n-2}C_n) + C_{n-1} = r(L - \sum_{n=1}^{n-1}C_n) + C_{n} \implies C_{n} = (1+r)C_{n-1}$$

Therefore, if we update the table making $C$ the first payment instead of $C_1$, we get:

| Period | Outstanding balance       | Int. Repayment               | Cap. Repayment   | Monthly Repayment |
|--------|---------------------------|------------------------------|------------------|-------------------|
| 1      | $L$                       | $rL$                         | $C$              | $M$               | 
| 2      | $L - C$                   | $r(L - (1+r)C)$              | $(1+r)C$         | $M$               |
| ...    | ...                       | ...                          | ...              | ...               |
| n      | $L - \sum_{n=1}^{n-1}C_n$ | $r(L - \sum_{n=1}^{n-1}C_n)$ | $(1+r)^{n-1}C_n$ | $M$               |

The sum of all capital repayments must equal the loan amount. Apply geometric sum formula to repayment column: 

$$ L = C_1+C_2+...+C_n $$
$$ L = C+(1+r)C+...+(1+r)^{n-1}C $$
$$ L = C\frac{(1+r)^{n}-1}{(1+r)-1} = C\frac{(1+r)^{n}-1}{r} $$
$$ C = \frac{rL}{(1+r)^{n}-1}$$

To find M, we can pick any row and sum interest and capital repayment. Consider the first row:

$$M = rL + C = rL + \frac{rL}{(1+r)^{n}-1} = \frac{(1+r)^n}{(1+r)^n - 1}rL$$

## Interest-Only Mortgage
The above case is in stark contrast with interest-only mortgages, where interest is paid periodically and the capital 
repayment is made in full at the end of the mortgage. This is shown below, where N is taken to be the last period.

| Period | Outstanding balance | Int. Repayment | Cap. Repayment | Monthly Repayment |
|--------|---------------------|----------------|----------------|-------------------|
| n      | $L$                 | $rL$           | 0              | $M$               | 
| ...    | ...                 | ...            | ...            | ...               |
| N      | 0                   | $rL$           | $L$            | $(1 + r)L$        |


## Comparison

Consider the difference, $M_D$ of the monthly payments, excluding the last payment, of an interest-only, $M_I$ and repayment mortgages, $M_R$.
$$ M_D = M_R - M_I $$
$$ M_D = \frac{(1+r)^n}{(1+r)^n - 1}rL - rL = (\frac{(1+r)^n}{(1+r)^n - 1} - 1)rL = \frac{rL}{(1+r)^n - 1} = C$$
