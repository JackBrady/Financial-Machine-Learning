

# Calls the functions below to get average uniqueness and sample weights

def get_weights_and_avgu(close_price,df,threads,times):   
    
    times = times.loc[df.index]
    numCoEvents = multiprocess.mp_pandas_obj(mpNumCoEvents,('molecule',df.index),                         
                                  threads,closeIdx=close_price.index,t1=times)
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
    numCoEvents = numCoEvents.reindex(close_price.index).fillna(0)
    out=pd.DataFrame()
    out['tW'] = multiprocess.mp_pandas_obj(mpSampleTW,('molecule',df.index),
                                threads,t1=times,numCoEvents=numCoEvents)


    out['w']=multiprocess.mp_pandas_obj(mpSampleW,('molecule',df.index),threads,
                    t1=times,numCoEvents=numCoEvents,close=close_price)
    out['w']*=out.shape[0]/out['w'].sum()

    out.dropna(inplace=True)

    sample_weights = getTimeDecay(out['tW'],0.5)
    avgu = out['tW'].mean()
    

    return avgu, sample_weights





# The following functions are snippets from Advances in Financial Machine Learning by Dr. Marcos Lopez de Prado


def mpNumCoEvents(closeIdx,t1,molecule):
    '''
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed
    
    Any event that starts before t1[modelcule].max() impacts the count.
    '''
    #1) find events that span the period [molecule[0],molecule[-1]]
    t1=t1.fillna(closeIdx[-1]) # unclosed events still must impact other weights
    t1=t1[t1>=molecule[0]] # events that end at or after molecule[0]
    t1=t1.loc[:t1[molecule].max()] # events that start at or before t1[molecule].max()
    #2) count events spanning a bar
    iloc=closeIdx.searchsorted(np.array([t1.index[0],t1.max()]))
    count=pd.Series(0,index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn,tOut in t1.iteritems():count.loc[tIn:tOut]+=1.
    return count.loc[molecule[0]:t1[molecule].max()]

def mpSampleTW(t1,numCoEvents,molecule):
    # Derive avg. uniqueness over the events lifespan
    wght=pd.Series(index=molecule)
    for tIn,tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn]=(1./numCoEvents.loc[tIn:tOut]).mean()
    return wght


            
def mpSampleW(t1,numCoEvents,close,molecule):
    # Derive sample weight by return attribution
    ret=np.log(close).diff() # log-returns, so that they are additive
    wght=pd.Series(index=molecule)
    for tIn,tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn]=(ret.loc[tIn:tOut]/numCoEvents.loc[tIn:tOut]).sum()
    return wght.abs()


def getTimeDecay(tW,clfLastW=1.):
    # apply piecewise-linear decay to observed uniqueness (tW)
    # newest observation gets weight=1, oldest observation gets weight=clfLastW
    clfW=tW.sort_index().cumsum()
    if clfLastW>=0:slope=(1.-clfLastW)/clfW.iloc[-1]
    else:slope=1./((clfLastW+1)*clfW.iloc[-1])
    const=1.-slope*clfW.iloc[-1]
    clfW=const+slope*clfW
    clfW[clfW<0]=0
    return clfW




