import numpy as np
import plotly.graph_objects as go
import datetime

from .options import Options

TODAY = datetime.date.today()
POINTS = 1000


class PayoffChart:
    """
    A helper class to plot the payoff chart of options.

    Parameters
    ----------
    options : list[Options]
        A list of options.
    positions : list[str]
        A list of positions. Must be the same length as options. Each position must be either "long" or "short".
    sigma : list[float]
        A list of volatilities. Must be the same length as options.
    spot_price : float
        The current price of the underlying asset.
    """

    def __init__(
        self,
        options: list[Options],
        positions: list[str],
        sigmas: list[float],
        spot_price: float,
    ) -> None:
        self.options = options
        self.positions = positions
        self.sigmas = sigmas
        self.spot_price = spot_price

    def __repr__(self) -> str:
        return f"PayoffChart({self.options}, {self.positions}, {self.spot_price})"

    def add_options(
        self,
        options: list[Options],
        positions: list[str],
        sigmas: list[float],
    ):
        """
        Add options to the PayoffChart.

        Parameters
        ----------
        options : list[Options]
            A list of options.
        positions : list[str]
            A list of positions. Must be the same length as options. Each position must be either "long" or "short".
        sigma : list[float]
            A list of volatilities. Must be the same length as options.

        Returns
        -------
        None.

        """
        if (
            isinstance(options, list)
            and isinstance(positions, list)
            and isinstance(sigmas, list)
        ):
            self.options.extend(options)
            self.positions.extend(positions)
            self.sigmas.extend(sigmas)
        else:
            raise TypeError("options and positions must be lists.")

    def remove_options(self, options: list[Options]):
        """
        Remove options from the PayoffChart.

        Parameters
        ----------
        options : list[Options]
            A list of options to remove.

        Returns
        -------
        None.
        """
        if isinstance(options, list):
            for option in options:
                if option in self.options:
                    self.options.remove(option)
                    self.positions.pop(self.options.index(option))
        else:
            raise TypeError("options must be a list.")

    def __correct_price(self, price, position_type):
        if position_type == "long":
            return price
        elif position_type == "short":
            return -price
        else:
            raise ValueError("Position must be either 'long' or 'short'.")

    def __calc_premium(
        self,
        spot_price,
        option,
        position_type,
        date,
        risk_free_rate,
        volatility,
    ):
        price = option.calculate_price(
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            day=date,
        )
        return self.__correct_price(price, position_type)

    def total_premium(
        self,
        date,
        spot_price=None,
        risk_free_rate=0.0342,
    ):
        """
        Calculate the total premium paid for the options.

        Parameters
        ----------
        spot_price : float or None, optional
            The underlying price. If None, use the spot price of the PayoffChart. Default is None.
        date : str
            The date to calculate the premium. Format: "YYYY-MM-DD"
        risk_free_rate : float, optional
            The risk free rate. Default is 0.0342.

        Returns
        -------
        float
            The total premium paid.
        """
        if spot_price is None:
            spot_price = self.spot_price
        t_p = np.zeros(POINTS)
        for i in range(len(self.options)):
            t_p += self.__calc_premium(
                spot_price=spot_price,
                option=self.options[i],
                position_type=self.positions[i],
                date=date,
                risk_free_rate=risk_free_rate,
                volatility=self.sigmas[i],
            )
        return t_p

    def __premium_paid(self, risk_free_rate):
        return self.total_premium(
            spot_price=self.spot_price,
            date=TODAY,
            risk_free_rate=risk_free_rate,
        )

    def __min_expiry(self):
        expiries = []
        for option in self.options:
            expiries.append(option.expiry_date)
        return min(expiries)

    def payoff_chart(
        self,
        risk_free_rate=0.0342,
        date=None,
        range=0.1,
        new_spot_price=None,
    ):
        """
        Calculate the payoff chart of the options.

        Parameters:
        ----------
        risk_free_rate : float, optional
            The risk free rate. Default is 0.0342.
        date : str, optional
            The date to calculate the payoff chart. Default is None, which means the minimum expiry date of the options.
            Format: "YYYY-MM-DD"
        range : float, optional
            The range of the underlying price. Default is 0.1.
        new_spot_price : float, optional
            The new spot price. Default is None, which means the original spot price.

        Returns:
        -------
        fig : plotly.graph_objects.Figure
        """
        if new_spot_price:
            self.spot_price = new_spot_price
        if not date:
            date = self.__min_expiry()
        min_expiry = self.__min_expiry()
        Ss = np.linspace(
            self.spot_price - range * self.spot_price,
            self.spot_price + range * self.spot_price,
            POINTS,
        )
        premium_paid = self.__premium_paid(risk_free_rate)
        t_premium = self.total_premium(
            spot_price=Ss,
            date=date,
            risk_free_rate=risk_free_rate,
        )
        pnl = t_premium - premium_paid

        ps_pnl_mask = np.argwhere(pnl >= 0)
        ng_pnl_mask = np.argwhere(pnl < 0)

        ps_pnl = pnl[ps_pnl_mask].flatten()
        ng_pnl = pnl[ng_pnl_mask].flatten()

        ps_S = Ss[ps_pnl_mask].flatten()
        ng_S = Ss[ng_pnl_mask].flatten()

        t_expiry_pre = self.total_premium(
            spot_price=Ss,
            date=min_expiry,
            risk_free_rate=risk_free_rate,
        )
        expiry_premium = t_expiry_pre - premium_paid
        fig = go.Figure()
        fig.add_hline(0, line_dash="dash", line_color="white", name="Break Even")
        fig.add_vline(
            self.spot_price,
            line_dash="dash",
            line_color="white",
            name="Strike Price",
        )
        fig.add_trace(
            go.Scatter(
                x=ps_S, y=ps_pnl, fill="tozeroy", fillcolor="rgba(0,255,0,0.5)", name=""
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ng_S, y=ng_pnl, fill="tozeroy", fillcolor="rgba(255,0,0,0.5)", name=""
            )
        )
        fig.add_trace(go.Scatter(x=Ss, y=pnl, name="PnL", line_color="blue"))
        fig.add_trace(
            go.Scatter(
                x=Ss,
                y=expiry_premium,
                name="PnL at Expiry",
                line_color="yellow",
            )
        )
        fig.update_layout(
            title="Payoff Chart",
            xaxis_title="Stock Price",
            yaxis_title="Price",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple",
            ),
        )
        fig.show()
        return fig
