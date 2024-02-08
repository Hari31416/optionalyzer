from .blackscholes import BlackScholes
import numpy as np
import datetime


class Options:
    def __init__(self, strike_price, expiry_date) -> None:
        self.strike_price = strike_price
        self.expiry_date = self.__str_to_date(expiry_date)

    def calculate_price(self):
        raise NotImplementedError("You must implement the calculate_price method.")

    def intrinsic_value(self):
        raise NotImplementedError("You must implement the intrinsic_value() method.")

    def time_value(self):
        raise NotImplementedError("You must implement the time_value() method.")

    def __str_to_date(self, date_str):
        if isinstance(date_str, datetime.date):
            return date_str
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()

    def _tau(self, day=None):
        if day is None:
            day = datetime.date.today()
        else:
            day = self.__str_to_date(day)
        return (self.expiry_date - day).days / 365


class Call(Options):
    def __init__(self, strike_price, expiry_date) -> None:
        super().__init__(strike_price, expiry_date)
        self.__price = None
        self.__greeks = None

    def __repr__(self) -> str:
        return f"Call({self.strike_price}, {self.expiry_date})"

    @property
    def greeks(self):
        if not self.__greeks:
            raise ValueError(
                "Greeks not calculated. You must call calculate_price first."
            )
        return self.__greeks

    def get_price(self):
        if self.__price is None:
            raise ValueError(
                "Price not calculated. You must call calculate_price first."
            )
        return self.__price

    def calculate_price(
        self, spot_price, risk_free_rate, volatility, day=None, return_greeks=False
    ):
        """
        Calculate the price of a call option using the Black-Scholes model.

        Parameters
        ----------
        spot_price : float
            The current price of the underlying asset.
        risk_free_rate : float
            The risk-free interest rate.
        volatility : float
            The volatility of the underlying asset.
        day : str, optional
            The day, by default None. Date must be in the format YYYY-MM-DD.
        return_greeks : bool, optional
            Whether to return the greeks, by default False

        Returns
        -------
        float
            The price of the call option.
        dict, optional
            The greeks of the call option.
        """
        bs = BlackScholes()
        tau = self._tau(day=day)
        S = spot_price
        K = self.strike_price
        r = risk_free_rate
        sigma = volatility
        price, greeks = bs.call(S, K, r, sigma, tau, greeks=True)
        self.__price = price
        self.__greeks = greeks
        if return_greeks:
            return price, greeks
        return price

    def intrinsic_value(self, spot_price):
        return np.maximum(spot_price - self.strike_price, 0)

    def time_value(self, spot_price):
        if self.__price is None:
            raise ValueError(
                "Price not calculated. You must call calculate_price first."
            )

        return self.__price - self.intrinsic_value(spot_price)


class Put(Options):
    def __init__(self, strike_price, expiry_date) -> None:
        super().__init__(strike_price, expiry_date)
        self.__price = None
        self.__greeks = None

    def __repr__(self) -> str:
        return f"Put({self.strike_price}, {self.expiry_date})"

    @property
    def greeks(self):
        if not self.__greeks:
            raise ValueError(
                "Greeks not calculated. You must call calculate_price first."
            )
        return self.__greeks

    def get_price(self):
        if self.__price is None:
            raise ValueError(
                "Price not calculated. You must call calculate_price first."
            )
        return self.__price

    def calculate_price(
        self, spot_price, risk_free_rate, volatility, day=None, return_greeks=False
    ):
        """
        Calculate the price of a put option using the Black-Scholes model.

        Parameters
        ----------
        spot_price : float
            The current price of the underlying asset.
        risk_free_rate : float
            The risk-free interest rate.
        volatility : float
            The volatility of the underlying asset.
        day : str, optional
            The day, by default None. Date must be in the format YYYY-MM-DD.
        return_greeks : bool, optional
            Whether to return the greeks, by default False

        Returns
        -------
        float
            The price of the put option.
        dict, optional
            The greeks of the put option.
        """
        bs = BlackScholes()
        tau = self._tau(day=day)
        S = spot_price
        K = self.strike_price
        r = risk_free_rate
        sigma = volatility
        price, greeks = bs.put(S, K, r, sigma, tau, greeks=True)
        self.__price = price
        self.__greeks = greeks
        if return_greeks:
            return price, greeks
        return price

    def intrinsic_value(self, spot_price):
        return np.maximum(self.strike_price - spot_price, 0)

    def time_value(self, spot_price):
        if self.__price is None:
            raise ValueError(
                "Price not calculated. You must call calculate_price first."
            )

        return self.__price - self.intrinsic_value(spot_price)
