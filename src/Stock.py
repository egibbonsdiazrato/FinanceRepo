from dataclasses import dataclass, field

@dataclass(frozen=True)
class Stock:  # TODO Improve step_up and step_down. Maybe change for u and d and rework simulation function
    """
    A class which describes the stock that is related to the derivative that is going to be modelled. The attributes
    are the price of the stock at time zero and the size of movement said stock can take after each timestep. These
    movements are assumed to be symmetric and may be specified as either absolute or relative.

    Args:
    S_0: Starting price of the underlying stock.
    step_up: up movement size of the underlying.
    step_down: down movement size of the underlying.
    step_type: either abs or rel to specify what step is.
    """
    S_0: int | float
    step_up: int | float
    step_down: int | float
    step_type: str
    step_abs: bool = field(init=False)
    step_rel: bool = field(init=False)

    def __post_init__(self) -> None:
        """
        Error handling and post processing
        """
        # Exceptions for inputs
        if self.S_0 <= 0:
            raise ValueError(f'S_0 has to be greater than 0. The value input was {self.S_0=}.')

        # Delta type flag
        if self.step_type == 'abs':
            object.__setattr__(self, 'step_abs', True)
        elif self.step_type == 'rel':
            object.__setattr__(self, 'step_rel', True)
        else:
            raise Exception(f'The step type provided, {self.step_type}, has to be either abs or rel')


    def __str__(self) -> str:
        """
        To string method of a class.

        Returns:
            stock_str: Descriptive string detailing the market.
        """
        stock_str = f'The stock has initial value of {self.S_0} USD'
        if self.step_abs:
            stock_str += (f'and the absolute stock movement is +{self.step_up} or -{self.step_down} '
                          f'USD for every timestep.')
        else:
            stock_str += (f'and the relative stock movement is {self.step_up} or {self.step_down} '
                          f'for up and down movements for every timestep.')
        return stock_str
