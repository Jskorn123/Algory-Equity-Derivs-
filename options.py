# import yfinance as yf
# import pandas as pd

# def get_options_chain(tkr):
#     tk = yf.Ticker(tkr)
#     exps = tk.options # expiration dates
    
#     for e in exps:
#         print(e)
#         opt = tk.option_chain(e)

#         print(opt)

from polygon import RESTClient
from polygon.rest.models import Agg
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv("./env")

POLY_API_KEY = os.getenv("POLY_API_KEY")

client = RESTClient(POLY_API_KEY)

def options_data(tkr = "O:SPY251219C00650000", write_path = None):
    aggs = []
    for a in client.list_aggs(
        tkr,
        1,
        "day",
        "2025-01-30",
        "2025-04-01",
        limit=50000,
    ):
        aggs.append(a)

    print(aggs)
    df = aggs_to_csv(aggs, write_path)

    print(df)
    return df

'''
This fker Agg object doesn't have in-built unpacking: https://github.com/polygon-io/client-python/blob/master/polygon/rest/models/aggs.py#L6
'''

def aggs_to_csv(objs : list[Agg], csv_path :str | None = None):
    columns = ["open", "high", "low", "close", "volume", "vwap", "transactions", "otc"]
    ts_index = []
    data = []

    for agg in objs:
        ts_index.append(agg.timestamp)
        data.append([agg.open, agg.high, agg.low, agg.close, agg.volume, agg.vwap, agg.transactions, agg.otc])
    
    df = pd.DataFrame(data = data, columns = columns, index = ts_index)
    df.index.name = "timestamp"
    
    if csv_path:
        df.to_csv(csv_path)
    
    return df



# Agg(open=19.86, high=19.9, low=18.04, close=19.9, volume=20, vwap=19.184, timestamp=1738213200000, transactions=20, otc=None)
if __name__ == "__main__":
    # get_options_chain("AAPL")
    options_data(write_path="tmp.csv")