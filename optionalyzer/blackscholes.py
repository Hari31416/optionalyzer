from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize


class BlackScholes:
    def __init__(self) -> None:
        pass

    def __d1(self, S, K, r, sigma, tau):
        multiplier = 1 / (sigma * np.sqrt(tau) + 1e-10)
        term1 = np.log(S / K)
        term2 = (r + sigma**2 / 2) * tau
        return multiplier * (term1 + term2)

    def __d2(self, S, K, r, sigma, tau):
        d1 = self.__d1(S, K, r, sigma, tau)
        return d1 - sigma * np.sqrt(tau)

    def __vega(self, S, K, r, sigma, tau):
        d1 = self.__d1(S, K, r, sigma, tau)
        return S * np.sqrt(tau) * norm.pdf(d1)

    def __call_delta(self, S, K, r, sigma, tau):
        d1 = self.__d1(S, K, r, sigma, tau)
        return norm.cdf(d1)

    def __put_delta(self, S, K, r, sigma, tau):
        d1 = self.__d1(S, K, r, sigma, tau)
        return norm.cdf(d1) - 1

    def __call_theta(self, S, K, r, sigma, tau):
        d1 = self.__d1(S, K, r, sigma, tau)
        d2 = self.__d2(S, K, r, sigma, tau)
        term_1 = -S * sigma * norm.pdf(d1) / (2 * np.sqrt(tau) + 1e-10)
        term_2 = r * K * np.exp(-r * tau) * norm.cdf(d2)
        return term_1 - term_2

    def __put_theta(self, S, K, r, sigma, tau):
        d1 = self.__d1(S, K, r, sigma, tau)
        d2 = self.__d2(S, K, r, sigma, tau)
        term_1 = -S * sigma * norm.pdf(d1) / (2 * np.sqrt(tau) + 1e-10)
        term_2 = r * K * np.exp(-r * tau) * norm.cdf(-d2)
        return term_1 + term_2

    def __call_gamma(self, S, K, r, sigma, tau):
        d1 = self.__d1(S, K, r, sigma, tau)
        return norm.pdf(d1) / (S * sigma * np.sqrt(tau) + 1e-10)

    def __put_gamma(self, S, K, r, sigma, tau):
        d1 = self.__d1(S, K, r, sigma, tau)
        return norm.pdf(d1) / (S * sigma * np.sqrt(tau) + 1e-10)

    def __call_rho(self, S, K, r, sigma, tau):
        d2 = self.__d2(S, K, r, sigma, tau)
        return K * tau * np.exp(-r * tau) * norm.cdf(d2)

    def __put_rho(self, S, K, r, sigma, tau):
        d2 = self.__d2(S, K, r, sigma, tau)
        return -K * tau * np.exp(-r * tau) * norm.cdf(-d2)

    def call(self, S, K, r, sigma, tau, greeks=False):
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
        sigma : float
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
        d1 = self.__d1(S, K, r, sigma, tau)
        d2 = self.__d2(S, K, r, sigma, tau)
        term_1 = S * norm.cdf(d1)
        term_2 = K * np.exp(-r * tau) * norm.cdf(d2)
        price = term_1 - term_2
        if greeks:
            greeks = {
                "delta": self.__call_delta(S, K, r, sigma, tau),
                "gamma": self.__call_gamma(S, K, r, sigma, tau),
                "theta": self.__call_theta(S, K, r, sigma, tau),
                "vega": self.__vega(S, K, r, sigma, tau),
                "rho": self.__call_rho(S, K, r, sigma, tau),
            }
            return price, greeks
        return price

    def put(self, S, K, r, sigma, tau, greeks=False):
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
        sigma : float
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
        d1 = self.__d1(S, K, r, sigma, tau)
        d2 = self.__d2(S, K, r, sigma, tau)
        term_1 = K * np.exp(-r * tau) * norm.cdf(-d2)
        term_2 = S * norm.cdf(-d1)
        price = term_1 - term_2

        if greeks:
            greeks = {
                "delta": self.__put_delta(S, K, r, sigma, tau),
                "gamma": self.__put_gamma(S, K, r, sigma, tau),
                "theta": self.__put_theta(S, K, r, sigma, tau),
                "vega": self.__vega(S, K, r, sigma, tau),
                "rho": self.__put_rho(S, K, r, sigma, tau),
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

        def option_partial(sigma):
            return option(sigma=sigma, S=S, K=K, r=r, tau=tau, greeks=False)

        def error(sigma):
            return np.square(option_partial(sigma) - option_price)

        res = minimize(error, 0.2)
        if verbose:
            if res.success:
                print("Optimized Successfully!")
            else:
                print("Optimization Unsuccessful. Here is the final result.")
        return res.x[0]
