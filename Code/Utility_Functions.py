
# Snippet from Advances in Financial Machine Learning by Dr. Marcos Lopez de Prado

def get_daily_volatility(close,span0=100):
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0=df0[df0>0]
    df0=pd.Series(close.index[df0-1], index = close.index[close.shape[0]-df0.shape[0]:])
    df0=close.loc[df0.index]/close.loc[df0.values].values-1
    df0=df0.ewm(span=span0).std() 
    return df0
    
