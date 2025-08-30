from dataclasses import dataclass,field

@dataclass(frozen=True)
class Market:
    """
    A class which defines the market in which the derivative is going to exist. The attributes of the market are
    interest rates and integer timesteps.

    Args:
        r: The constant interest rate earned every timestep.
        T: Simulation has steps from 0 to T in steps of 1.
    """
    r: int | float
    T: int
    deltat: int = field(init=False)

    def __post_init__(self) -> None:
        # Exceptions for inputs
        if self.r < 0:
            raise ValueError(f'r has to be greater than or equal to 0. The value input was {self.r=}.')
        if self.T <= 0 or not isinstance(self.T, int):
            raise ValueError(f'T has to be an integer greater than 0. The value input was {self.T=}.')

        object.__setattr__(self, 'deltat', 1)  # Hardcoded property

    def __str__(self) -> str:
        """
        To string method of a class.

        Returns:
            market_str: Descriptive string detailing the market.
        """
        market_str = f'The market environment has {self.r*100}% interest rates and exists up to {self.T=} timesteps.'
        return market_str
