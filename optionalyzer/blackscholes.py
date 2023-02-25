from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize
import yfinance
import pandas as pd
import requests
import datetime

from optionalyzer import RISK_FREE_RATE

TODAY = datetime.datetime.today().strftime("%d-%m-%Y")


class BlackScholes:
    def __init__(self) -> None:
        pass

    def __d1(self, S, K, r, iv, tau):
        multiplier = 1 / (iv * np.sqrt(tau) + 1e-10)
        term1 = np.log(S / K)
        term2 = (r + iv**2 / 2) * tau
        return multiplier * (term1 + term2)

    def __d2(self, S, K, r, iv, tau):
        d1 = self.__d1(S, K, r, iv, tau)
        return d1 - iv * np.sqrt(tau)

    def __vega(self, S, K, r, iv, tau):
        d1 = self.__d1(S, K, r, iv, tau)
        return S * np.sqrt(tau) * norm.pdf(d1)

    def __call_delta(self, S, K, r, iv, tau):
        d1 = self.__d1(S, K, r, iv, tau)
        return norm.cdf(d1)

    def __put_delta(self, S, K, r, iv, tau):
        d1 = self.__d1(S, K, r, iv, tau)
        return norm.cdf(d1) - 1

    def __call_theta(self, S, K, r, iv, tau):
        d1 = self.__d1(S, K, r, iv, tau)
        d2 = self.__d2(S, K, r, iv, tau)
        term_1 = -S * iv * norm.pdf(d1) / (2 * np.sqrt(tau) + 1e-10)
        term_2 = r * K * np.exp(-r * tau) * norm.cdf(d2)
        return term_1 - term_2

    def __put_theta(self, S, K, r, iv, tau):
        d1 = self.__d1(S, K, r, iv, tau)
        d2 = self.__d2(S, K, r, iv, tau)
        term_1 = -S * iv * norm.pdf(d1) / (2 * np.sqrt(tau) + 1e-10)
        term_2 = r * K * np.exp(-r * tau) * norm.cdf(-d2)
        return term_1 + term_2

    def __call_gamma(self, S, K, r, iv, tau):
        d1 = self.__d1(S, K, r, iv, tau)
        return norm.pdf(d1) / (S * iv * np.sqrt(tau) + 1e-10)

    def __put_gamma(self, S, K, r, iv, tau):
        d1 = self.__d1(S, K, r, iv, tau)
        return norm.pdf(d1) / (S * iv * np.sqrt(tau) + 1e-10)

    def __call_rho(self, S, K, r, iv, tau):
        d2 = self.__d2(S, K, r, iv, tau)
        return K * tau * np.exp(-r * tau) * norm.cdf(d2)

    def __put_rho(self, S, K, r, iv, tau):
        d2 = self.__d2(S, K, r, iv, tau)
        return -K * tau * np.exp(-r * tau) * norm.cdf(-d2)

    def call(self, S, K, r, iv, tau, greeks=False):
        """
        Calculates the price of a call option using the Black-Scholes model.

        Parameters
        ----------
        S : float
            The current price of the underlying asset.
        K : float
            The strike price of the option.
        r : float
            The risk-free interest rate.
        iv : float
            The volatility of the underlying asset.
        tau : float
            The time to maturity of the option in years.
        greeks : bool, optional
            If True, returns the greeks of the option. The default is False.

        Returns
        -------
        float
            The price of the call option.
        dict
            The greeks of the option if greeks=True.
        """
        d1 = self.__d1(S, K, r, iv, tau)
        d2 = self.__d2(S, K, r, iv, tau)
        term_1 = S * norm.cdf(d1)
        term_2 = K * np.exp(-r * tau) * norm.cdf(d2)
        price = term_1 - term_2
        if greeks:
            greeks = {
                "delta": self.__call_delta(S, K, r, iv, tau),
                "gamma": self.__call_gamma(S, K, r, iv, tau),
                "theta": self.__call_theta(S, K, r, iv, tau),
                "vega": self.__vega(S, K, r, iv, tau),
                "rho": self.__call_rho(S, K, r, iv, tau),
            }
            return price, greeks
        return price

    def put(self, S, K, r, iv, tau, greeks=False):
        """
        Calculates the price of a put option using the Black-Scholes model.

        Parameters
        ----------
        S : float
            The current price of the underlying asset.
        K : float
            The strike price of the option.
        r : float
            The risk-free interest rate.
        iv : float
            The volatility of the underlying asset.
        tau : float
            The time to maturity of the option in years.
        greeks : bool, optional
            If True, returns the greeks of the option. The default is False.

        Returns
        -------
        float
            The price of the put option.
        dict
            The greeks of the option if greeks=True.
        """
        d1 = self.__d1(S, K, r, iv, tau)
        d2 = self.__d2(S, K, r, iv, tau)
        term_1 = K * np.exp(-r * tau) * norm.cdf(-d2)
        term_2 = S * norm.cdf(-d1)
        price = term_1 - term_2

        if greeks:
            greeks = {
                "delta": self.__put_delta(S, K, r, iv, tau),
                "gamma": self.__put_gamma(S, K, r, iv, tau),
                "theta": self.__put_theta(S, K, r, iv, tau),
                "vega": self.__vega(S, K, r, iv, tau),
                "rho": self.__put_rho(S, K, r, iv, tau),
            }
            return price, greeks
        return price

    def implied_volatility(
        self, option_price, S, K, r, tau, option_type="call", verbose=1
    ):
        """
        Calculates the implied volatility of an option using the Black-Scholes
        model.

        Parameters
        ----------
        S : float
            The current price of the underlying asset.
        K : float
            The strike price of the option.
        r : float
            The risk-free interest rate.
        tau : float
            The time to maturity of the option in years.
        option_price : float
            The price of the option.
        option_type : str, optional
            The type of the option. The default is "call".

        Returns
        -------
        float
            The implied volatility of the option.
        """
        if option_type == "call":
            option = self.call
        elif option_type == "put":
            option = self.put
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        def option_partial(iv):
            return option(iv=iv, S=S, K=K, r=r, tau=tau, greeks=False)

        def error(iv):
            return np.square(option_partial(iv) - option_price)

        res = minimize(error, 0.2)
        if verbose:
            if res.success:
                print("Optimized Successfully!")
            else:
                print("Optimization Unsuccessful. Here is the final result.")
        return res.x[0]


class OptionChain:
    """
    A class which calculates the implied volatility of an option using the
    Black-Scholes model. The class uses `yfinance` to get the historical
    prices of the underlying asset.
    """

    def __init__(self, ticker):
        self.ticker = ticker
        self.__spot_price = None
        self.__resolve_tickers()
        self.__option_chain = None

    @property
    def spot_price(self):
        """
        Returns the spot price of the underlying asset.

        Returns
        -------
        float
            The spot price of the underlying asset.
        """
        if self.__spot_price is None:
            self.__get_spot_price()
        return self.__spot_price

    def get_option_chain(self):
        """
        Returns the option chain of the underlying asset.

        Returns
        -------
        pandas.DataFrame
            The option chain of the underlying asset.

        Raises
        ------
        ValueError
            If the option chain is not available.
        """
        if self.__option_chain is None:
            raise ValueError(
                "Option chain not available. Use self.option_chain to scrap it."
            )
        return self.__option_chain

    def __resolve_tickers(self):
        if self.ticker.lower() in ["nifty", "^nsei", "nsei"]:
            self.ticker = "nifty"
        elif self.ticker.lower() in [
            "banknifty",
            "niftybank",
            "bank nifty",
            "nifty bank",
            "bank-nifty",
            "nifty-bank",
            "^nsebank",
            "nsebank",
        ]:
            self.ticker = "nifty-bank"
        else:
            raise ValueError("Invalid ticker. Only nifty and nifty-bank are supported.")

    def __get_spot_price(self):
        if self.ticker == "nifty":
            ticker = yfinance.Ticker("^NSEI")
        elif self.ticker == "nifty-bank":
            ticker = yfinance.Ticker("^NSEBANK")
        else:
            raise ValueError("Invalid ticker. Only nifty and nifty-bank are supported.")

        self.__spot_price = ticker.history("1d", "1d")["Close"].values[0]

    def __clean_price(self, x):
        if "+" in x:
            x = x.split("+")[0]
        elif "-" in x:
            x = x.split("-")[0]
        else:
            x = x.split("0.00 (")[0].strip()
        x = x.replace("₹", "")
        x = x.replace(",", "")
        return float(x)

    def __chain(self, expiry_date):
        url = f"https://groww.in/options/{self.ticker}?expiry={expiry_date}"
        res = requests.get(url)
        if res.status_code != 200:
            raise ValueError(f"Status code {res.status_code}. Try again.")

        return res

    def _chain(self, expiry_date):
        res = self.__chain(expiry_date=expiry_date)
        chain = pd.read_html(res.content)[0]
        chain = chain[(chain["OI (lots)"] != "--") & (chain["OI (lots).1"] != "--")]
        chain = chain[["CALL PRICE", "STRIKE PRICE", "PUT PRICE"]]
        chain["CALL PRICE"] = chain["CALL PRICE"].apply(self.__clean_price)
        chain["PUT PRICE"] = chain["PUT PRICE"].apply(self.__clean_price)
        chain.columns = ["Call", "Strike", "Put"]
        return chain

    def option_chain(self, expiry_date, limit=10):
        """
        Scrapes the option chain using yfinance and Groww website.

        Parameters
        ----------
        expiry_date : str
            The expiry date of the option in the format 'dd-mm-yyyy'.
        limit : int, optional
            The number of options to be scraped. The default is 10.
        """
        chain = self._chain(expiry_date=expiry_date)

        u_chain = chain[chain["Strike"] <= self.__spot_price]
        l_chain = chain[chain["Strike"] > self.__spot_price]
        l_limit = min(limit // 2, len(l_chain))
        u_limit = min(limit // 2, len(u_chain))
        chain = pd.concat([u_chain[-u_limit:], l_chain[:l_limit]])
        chain = chain.reset_index(drop=True)
        self.__option_chain = chain
        return chain

    def calculate_iv(self, expiry_date, option_type="call", verbose=1):
        """
        Calculates the implied volatility of an option using the Black-Scholes
        model.

        Parameters
        ----------
        expiry_date : str
            The expiry date of the option in the format 'dd-mm-yyyy'.
        option_type : str, optional
            The type of the option. Either 'call' or 'put'. The default is 'call'.
        verbose : int, optional
            The verbosity of the function. The default is 0.
        """
        if self.__option_chain is None:
            if verbose:
                print("Option chain not available. Scraping it now.")
            self.option_chain(expiry_date=expiry_date)

        tau = (
            pd.to_datetime(expiry_date, dayfirst=True) - pd.to_datetime("today")
        ).days / 365
        S = self.spot_price
        ivs = np.zeros(len(self.__option_chain))
        bs = BlackScholes()
        r = RISK_FREE_RATE
        if verbose:
            print("Calculating Implied Volatility")

        for i, row in self.__option_chain.iterrows():
            ivs[i] = bs.implied_volatility(
                option_price=row[option_type.title()],
                S=S,
                K=row["Strike"],
                r=r,
                tau=tau,
                option_type=option_type.lower(),
                verbose=0,
            )
        median_iv = np.median(ivs)
        if option_type.lower() == "call":
            self.__option_chain["Call IV"] = ivs
            self.call_iv = median_iv
            return median_iv
        else:
            self.__option_chain["Put IV"] = ivs
            self.put_iv = median_iv
            return median_iv
