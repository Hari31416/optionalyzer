import datetime
import numpy as np

from optionalyzer.blackscholes import BlackScholes, RISK_FREE_RATE
from optionalyzer import RISK_FREE_RATE
class Options:
    """
    A base class for options.

    Attributes
    ----------
    strike_price : float
        The strike price of the option.
    expiry_date : datetime.date
        The expiry date of the option.
    iv : float
        The implied volatility of the option.
    """

    def __init__(self, strike_price, expiry_date, iv, date_format="%d-%m-%Y") -> None:
        """
        Initialize an option.

        Parameters
        ----------
        strike_price : float
            The strike price of the option.
        expiry_date : str
            The expiry date of the option. Date can be in any format however, you have to specify it.
            If not specified, it assumes the format to be DD-MM-YYYY.
        iv : float
            The implied volatility of the option.

        Attributes
        ----------
        strike_price : float
            The strike price of the option.
        expiry_date : datetime.date
            The expiry date of the option.
        iv : float
            The implied volatility of the option.

        Examples
        --------
        >>> call = Call(100, "15-01-2023", 0.2)
        """
        self.strike_price = strike_price
        self.date_format = date_format
        self.expiry_date = self.__str_to_date(expiry_date)
        self.iv = iv

    def calculate_price(self):
        raise NotImplementedError("You must implement the calculate_price method.")

    def intrinsic_value(self):
        raise NotImplementedError("You must implement the intrinsic_value() method.")

    def time_value(self):
        raise NotImplementedError("You must implement the time_value() method.")

    def __str_to_date(self, date_str):
        if isinstance(date_str, datetime.date):
            return date_str
        try:
            date = datetime.datetime.strptime(date_str, self.date_format).date()
        except ValueError:
            raise ValueError(
                f"Date must be in the format you initialized the class, {self.date_format}. You entered {date_str}"
            )
        return date

    def _tau(self, date=None):
        if date is None:
            date = datetime.date.today()
        else:
            date = self.__str_to_date(date)
        days = (self.expiry_date - date).days / 365
        if days < 0:
            raise ValueError("Execrice Date can not be AFTER the strike date")
        return days


class Call(Options):
    def __init__(self, strike_price, expiry_date, iv, date_format="%d-%m-%Y") -> None:
        super().__init__(strike_price, expiry_date, iv, date_format=date_format)
        self.__price = None
        self.__greeks = None

    def __repr__(self) -> str:
        return f"Call({self.strike_price}, {self.expiry_date}, {self.iv})"

    def __str__(self) -> str:
        return f"Call({self.strike_price}, {self.expiry_date}, {self.iv})"

    @property
    def greeks(self):
        if not self.__greeks:
            raise ValueError(
                "Greeks not calculated. You must call `calculate_price` first."
            )
        return self.__greeks

    def get_price(self):
        if self.__price is None:
            raise ValueError(
                "Price not calculated. You must call `calculate_price` first."
            )
        return self.__price

    def calculate_price(
        self,
        spot_price,
        date=None,
        return_greeks=False,
    ):
        """
        Calculate the price of a call option using the Black-Scholes model.

        Parameters
        ----------
        spot_price : float
            The current price of the underlying asset.
        volatility : float
            The volatility of the underlying asset.
        date : str, optional
            The date in which price has to be calculated, by default None which means today.
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
        tau = self._tau(date=date)
        S = spot_price
        K = self.strike_price
        r = RISK_FREE_RATE
        iv = self.iv
        price, greeks = bs.call(S, K, r, iv, tau, greeks=True)
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
                "Price not calculated. You must call `calculate_price` first."
            )

        return self.__price - self.intrinsic_value(spot_price)


class Put(Options):
    def __init__(self, strike_price, expiry_date, iv, date_format="%d-%m-%Y") -> None:
        super().__init__(strike_price, expiry_date, iv, date_format=date_format)
        self.__price = None
        self.__greeks = None

    def __repr__(self) -> str:
        return f"Put({self.strike_price}, {self.expiry_date}, {self.iv})"

    def __str__(self) -> str:
        return f"Put({self.strike_price}, {self.expiry_date}, {self.iv})"

    @property
    def greeks(self):
        if not self.__greeks:
            raise ValueError(
                "Greeks not calculated. You must call `calculate_price` first."
            )
        return self.__greeks

    def get_price(self):
        if self.__price is None:
            raise ValueError(
                "Price not calculated. You must call `calculate_price` first."
            )
        return self.__price

    def calculate_price(
        self,
        spot_price,
        date=None,
        return_greeks=False,
    ):
        """
        Calculate the price of a put option using the Black-Scholes model.

        Parameters
        ----------
        spot_price : float
            The current price of the underlying asset.
        date : str, optional
            The date in which price has to be calculated, by default None which means today.
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
        tau = self._tau(date=date)
        S = spot_price
        K = self.strike_price
        r = RISK_FREE_RATE
        sigma = self.iv
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
