import numpy as np
import plotly.graph_objects as go
import datetime

from optionalyzer.options import Options, Call, Put


TODAY = datetime.date.today()
POINTS = 200
LOT_SIZE = 50


class Position:
    """
    The class represents an option position. Short and Long
    """

    def __init__(self, option: Options, position_type: str) -> None:
        self.option = self.__resolve_option(option=option)
        self._type = self.__resolve_position_type(position_type=position_type)

    def __resolve_option(self, option: Options):
        if not isinstance(option, Options):
            raise ValueError("Given `option` in not an `Option`")
        return option

    def __resolve_position_type(self, position_type):
        if position_type.lower() in ["short", "s"]:
            return -1
        elif position_type.lower() in ["long", "l"]:
            return 1

    def __str__(self) -> str:
        option_string = str(self.option)
        if self._type == 1:
            option_string = f"Position(Long {option_string})"
        else:
            option_string = f"Position(Short {option_string})"
        return option_string

    def __repr__(self) -> str:
        return str(self)


class PayoffChart:
    """
    A helper class to plot the payoff chart of options.

    Parameters
    ----------
    positions : list[Position]
        A list of `Position`.
    spot_price : float
        The current price of the underlying asset.
    """

    def __init__(
        self,
        positions: list[Position],
        spot_price: float,
    ) -> None:
        self.positions = self.__resolve_positions(positions=positions)
        self.set_spot_price(spot_price=spot_price)

    def __resolve_positions(self, positions):
        if not isinstance(positions, list):
            raise ValueError(
                "`positions` is not a `list`. If you have a single position, please wrap it inside a list."
            )

        for position in positions:
            if not isinstance(position, Position):
                raise ValueError(f"{position} is not a `Position`.")
        return positions

    def __repr__(self) -> str:
        return f"PayoffChart({self.positions})"

    def __iter__(self):
        for position in self.positions:
            yield position

    def set_spot_price(self, spot_price):
        if spot_price < 0:
            raise ValueError("Spot price can not be less than zero.")
        self.__spot_price = spot_price

    def create_position(
        self,
        strike_price,
        expiry_date,
        iv,
        option_type,
        position_type,
        date_format="%d-%m-%Y",
        add=True,
    ):
        """
        Creates an option position

        Parameters
        ----------
        strike_price : float
            The strike price of the option.
        expiry_date : str
            The expiry date of the option. Format: dd-mm-yyyy
        iv : float
            The implied volatility of the option.
        option_type : str
            The type of the option. Either "call" or "put".
        position_type : str
            The type of the position. Either "long" or "short".
        date_format : str, optional
            The format of the date. The default is "%d-%m-%Y".
        add : bool, optional
            If True, the position is added to the list of positions. The default is True.
        Returns
        -------
        Position
            The position object.
        """
        if option_type.lower() == "call":
            option = Call(
                strike_price=strike_price,
                expiry_date=expiry_date,
                iv=iv,
                date_format=date_format,
            )
        elif option_type.lower() == "put":
            option = Put(
                strike_price=strike_price,
                expiry_date=expiry_date,
                iv=iv,
                date_format=date_format,
            )
        else:
            raise ValueError(
                f"Invalid option type `{option_type}`. Please use either `call` or `put`."
            )
        pos = Position(option=option, position_type=position_type)
        if add:
            self.positions.append(pos)
        return pos

    def create_positions(
        self,
        strike_prices,
        expiry_dates,
        ivs,
        option_types,
        position_types,
        date_format="%d-%m-%Y",
        add=True,
    ):
        """
        Creates a list of option positions.

        Parameters
        ----------
        strike_prices : list[float]
            A list of strike prices.
        expiry_dates : list[str]
            A list of expiry dates. Format: dd-mm-yyyy
        ivs : list[float]
            A list of implied volatilities.
        option_types : list[str]
            A list of option types. Either "call" or "put".
        position_types : list[str]
            A list of position types. Either "long" or "short".
        date_format : str, optional
            The format of the date. The default is "%d-%m-%Y".
        add : bool, optional
            If True, the positions are added to the list of positions. The default is True.

        Returns
        -------
        list[Position]
            A list of position objects.
        """
        positions = []
        for strike_price, expiry_date, iv, option_type, position_type in zip(
            strike_prices, expiry_dates, ivs, option_types, position_types
        ):
            position = self.create_position(
                strike_price=strike_price,
                expiry_date=expiry_date,
                iv=iv,
                option_type=option_type,
                position_type=position_type,
                date_format=date_format,
                add=add,
            )
            positions.append(position)
        return positions

    def add_positions(
        self,
        positions: list[Position],
    ):
        """
        Add options to the PayoffChart.

        Parameters
        ----------
        positions : list[Position]
            A list of Position.

        Returns
        -------
        None.

        """
        positions = self.__resolve_positions(positions=positions)
        self.positions.extend(positions)

    def remove_positions(self, positions: list[Position]):
        """
        Remove positions from the PayoffChart.

        Parameters
        ----------
        positions : list[Position]
            A list of positions to remove.

        Returns
        -------
        None.
        """
        positions = self.__resolve_positions(positions=positions)
        for position in positions:
            if position in self.positions:
                self.positions.remove(position)

    def __correct_price(self, price, position):
        return price * position._type

    def __calc_premium(
        self,
        spot_price,
        position,
        date,
    ):
        price = position.option.calculate_price(
            spot_price=spot_price,
            date=date,
            return_greeks=False,
        )
        return self.__correct_price(price, position)

    def total_premium(
        self,
        spot_price,
        date,
    ):
        """
        Calculate the total premium paid for the options.

        Parameters
        ----------
        date : str
            The date to calculate the premium. Format: "DD-MM-YYYY"

        Returns
        -------
        float
            The total premium paid.
        """
        t_p = 0
        for i in range(len(self.positions)):
            pos_pnl = self.__calc_premium(
                spot_price=spot_price,
                position=self.positions[i],
                date=date,
            )
            t_p += pos_pnl
        return t_p

    def __premium_paid(self, spot_price):
        return self.total_premium(
            date=TODAY,
            spot_price=spot_price,
        )

    def __min_expiry(self):
        expiries = []
        for position in self.positions:
            expiries.append(position.option.expiry_date)
        return min(expiries)

    def payoff_chart(self, date=None, range=0.1, new_spot_price=None, return_fig=False):
        """
        Calculate the payoff chart of the options.

        Parameters:
        ----------
        date : str, optional
            The date to calculate the payoff chart. Default is None, which means the minimum expiry date of the options.
            Format: "MM-DD-YYYY"
        range : float, optional
            The range of the underlying price. Default is 0.1.
        new_spot_price : float, optional
            The new spot price. Default is None, which means the original spot price.
        return_fig : bool, optional
            If True, the plotly figure is returned. Default is False.

        Returns:
        -------
        fig : plotly.graph_objects.Figure if return_fig is True
        """
        if new_spot_price:
            self.set_spot_price(new_spot_price)
        if date is None:
            date = self.__min_expiry()
        min_expiry = self.__min_expiry()
        Ss = np.linspace(
            self.__spot_price - range * self.__spot_price,
            self.__spot_price + range * self.__spot_price,
            POINTS,
        )
        premium_paid = self.__premium_paid(spot_price=self.__spot_price)
        premium_recieved_at_T = self.total_premium(
            spot_price=Ss,
            date=date,
        )
        pnl = premium_recieved_at_T - premium_paid
        pnl = np.round(pnl, 0) * LOT_SIZE

        pv_pnl_mask = np.argwhere(pnl >= 0)
        ng_pnl_mask = np.argwhere(pnl < 0)

        pv_pnl = pnl[pv_pnl_mask].flatten()
        ng_pnl = pnl[ng_pnl_mask].flatten()

        pv_S = Ss[pv_pnl_mask].flatten()
        ng_S = Ss[ng_pnl_mask].flatten()

        premium_at_expiry = self.total_premium(date=min_expiry, spot_price=Ss)

        pnl_at_expiry = premium_at_expiry - premium_paid
        pnl_at_expiry = np.round(pnl_at_expiry, 0) * LOT_SIZE

        fig = go.Figure()
        fig.add_hline(0, line_dash="dash", line_color="white", name="Break Even")
        fig.add_vline(
            self.__spot_price,
            line_dash="dash",
            line_color="white",
            name="Strike Price",
        )
        fig.add_trace(
            go.Scatter(
                x=pv_S, y=pv_pnl, fill="tozeroy", fillcolor="rgba(0,255,0,0.5)", name=""
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ng_S, y=ng_pnl, fill="tozeroy", fillcolor="rgba(255,0,0,0.5)", name=""
            )
        )
        fig.add_trace(go.Scatter(x=Ss, y=pnl, name=f"PnL on {date}", line_color="blue"))
        fig.add_trace(
            go.Scatter(
                x=Ss,
                y=pnl_at_expiry,
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
        if return_fig:
            return fig
        return Ss
