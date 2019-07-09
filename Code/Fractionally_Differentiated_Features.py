
# The following functions are snippets from Advances in Financial Machine Learning by Dr. Marcos Lopez de Prado


def fracDiff_FFD(series,d,thres=1e-5):

    #1) Compute weights for the longest series
    w=getWeights_FFD(d,thres)
    width=len(w)-1
    #2) Apply weights to values
    df={}

    for name in series.columns:
        seriesF,df_=series[[name]].fillna(method='ffill').dropna(),pd.Series()
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1=seriesF.index[iloc1-width],seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):continue # exclude NAs
            df_[loc1]=np.dot(w.T,seriesF.loc[loc0:loc1])[0,0]
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df


def getWeights_FFD(d,thres):

    w,k=[1.],1

    while True:

        w_=-w[-1]/k*(d-k+1)

        if abs(w_)<thres:break

        w.append(w_);k+=1

    return np.array(w[::-1]).reshape(-1,1)



def plotMinFFD(df0,feature):
    from statsmodels.tsa.stattools import adfuller
    

    out=pd.DataFrame(columns=['adfStat','pVal','lags','nObs','95% conf','corr'])

    for d in np.linspace(0,1,11):
        df1=np.log(df0[[feature]]).resample('1D').last() # downcast to daily obs
        df2=fracDiff_FFD(df1,d,thres=.01)
        corr=np.corrcoef(df1.loc[df2.index,feature],df2[feature])[0,1]
        df2=adfuller(df2[feature],maxlag=1,regression='c',autolag=None)
        out.loc[d]=list(df2[:4])+[df2[4]['5%']]+[corr] # with critical value
        
    from IPython.display import display, HTML


    display(HTML(out.to_html()))

    out[['adfStat','corr']].plot(secondary_y='adfStat')

    plt.axhline(out['95% conf'].mean(),linewidth=1,color='r',linestyle='dotted')
    
    plt.show()
    
    return


    
