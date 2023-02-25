from optionalyzer.options import Options, Put, Call
import pytest


def test_date_error():
    with pytest.raises(ValueError):
        put = Put(111, "2022-01-18", 0.3)
    with pytest.raises(ValueError):
        call = Call(111, "2022-01-18", 0.3)

    put = Put(111, "18-01-2023", 0.3)
    with pytest.raises(ValueError):
        put.calculate_price(110, "2023-31-01")


def test_excercise_error():
    put = Put(111, "18-01-2023", 0.3)
    with pytest.raises(ValueError):
        put.calculate_price(110, "25-01-2023")
    price = put.calculate_price(110, "18-01-2023")
    assert price >= 0, "Price is less than zero"

def test_call_price():
    call = Call(111, "25-02-2023", 0.23)
    with pytest.raises(ValueError):
        call.greeks
    with pytest.raises(ValueError):
        call.get_price()
    price = call.calculate_price(115, "21-01-2023")
    assert price >= 0, "Price is less than zero"
    assert price == call.get_price(), "Prices not the same"
    assert isinstance(call.greeks, dict), "Greeks are not dictionary"

def test_put_price():
    put = Put(111, "25-02-2023", 0.23)
    with pytest.raises(ValueError):
        put.greeks
    with pytest.raises(ValueError):
        put.get_price()
    price = put.calculate_price(115, "21-01-2023")
    assert price >= 0, "Price is less than zero"
    assert price == put.get_price(), "Prices not the same"
    assert isinstance(put.greeks, dict), "Greeks are not dictionary"