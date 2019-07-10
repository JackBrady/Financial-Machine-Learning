
def returns_series(xtest,sd,times,bids,asks,filtered):
    
    ret_df = pd.DataFrame()

    ret_df['ret_side'] = sd[0].loc[xtest.index]

    ret_df['bet_pass'] = filtered.copy()

    ret_df = ret_df[ret_df.bet_pass != 0]

    p_buy = pd.DataFrame(index=ret_df[ret_df.ret_side == 1].index)
    p_sell = pd.DataFrame(index=ret_df[ret_df.ret_side == -1].index)
    p_buy['price'] = asks.copy()
    p_sell['price'] = bids.copy()
    p_buy['stop'] = times.copy()
    p_sell['stop'] = times.copy()
    val_1 = bids.loc[p_buy['stop']]
    val_2 = asks.loc[p_sell['stop']]

    p_buy.drop(columns=['stop'],inplace=True)
    p_sell.drop(columns=['stop'],inplace=True)
    
    p_buy['end_price'] = (val_1.values)
    p_sell['end_price'] = (val_2.values)
    g = (p_buy['end_price'] - p_buy['price']) / p_buy['price']
    g2 = (p_sell['end_price'] - p_sell['price']) / p_sell['price']
    g2 *= -1
    returns_series = pd.concat([g,g2])
    return returns_series


def percentage_normalized_profit(ytest,xtest,lbl,sd,times,bids,asks,fill):

    # First finds maximum return we could have achieved had we classified every bet correctly in the test set
    maxi = pd.DataFrame(index=ytest[ytest !=0].index)
    maxi['side'] = lbl['bin']
    p_buy_max = pd.DataFrame(index=maxi[maxi.side == 1].index)
    p_sell_max = pd.DataFrame(index=maxi[maxi.side == -1].index)
    p_buy_max['price'] = asks.copy()
    p_sell_max['price'] = bids.copy()
    p_buy_max['stop'] = times.copy()
    p_sell_max['stop'] = times.copy()
    val_1_m = bids.loc[p_buy_max['stop']].copy()
    val_2_m = asks.loc[p_sell_max['stop']].copy()

    p_buy_max.drop(columns=['stop'],inplace=True)
    p_sell_max.drop(columns=['stop'],inplace=True)


    p_buy_max['end_price'] = (val_1_m.values)
    p_sell_max['end_price'] = (val_2_m.values)


    gm = (p_buy_max['end_price'] - p_buy_max['price']) / p_buy_max['price']
    g2m = (p_sell_max['end_price'] - p_sell_max['price']) / p_sell_max['price']
    g2m *= -1
    max_return = gm.sum()+g2m.sum()
    max_return = max_return.sum()


    # Finds actual return based on predictions
    ret_df = pd.DataFrame()


    ret_df['ret_side'] = sd[0].loc[xtest.index]

    ret_df['bet_pass'] = fill

    ret_df = ret_df[ret_df.bet_pass != 0]

    p_buy = pd.DataFrame(index=ret_df[ret_df.ret_side == 1].index)
    p_sell = pd.DataFrame(index=ret_df[ret_df.ret_side == -1].index)
    p_buy['price'] = asks.copy()
    p_sell['price'] = bids.copy()
    p_buy['stop'] = times.copy()
    p_sell['stop'] = times.copy()
    val_1 = bids.loc[p_buy['stop']]
    val_2 = asks.loc[p_sell['stop']]

    p_buy.drop(columns=['stop'],inplace=True)
    p_sell.drop(columns=['stop'],inplace=True)


    p_buy['end_price'] = (val_1.values)
    p_sell['end_price'] = (val_2.values)
    g = (p_buy['end_price'] - p_buy['price']) / p_buy['price']
    g2 = (p_sell['end_price'] - p_sell['price']) / p_sell['price']
    g2 *= -1

    # Returns percent actual out of the max return
    return ((g.sum()+g2.sum()) / max_return)



def backtest_cv(backtest_file_path):

    tick_bars = pd.read_csv(backtest_file_path,index_col=0, parse_dates=True)
    ask = tick_bars['ask'].copy()
    bid = tick_bars['bid'].copy()
    tick_path = backtest_file_path
    
    closing = tick_bars['close']

    volatility = get_daily_volatility(closing)
    times = mlf.filters.cusum_filter(closing, volatility.mean()*.1)
    filtered_bars = tick_bars.loc[times]
    filtered_bars = filtered_bars['close']


    vertical_barriers = mlf.labeling.add_vertical_barrier(times, closing, num_days=1)
    pt_sl = [1,1]
    min_ret = 0.004

    threads = cpu_count()-1


    triple_barrier_events = mlf.labeling.get_events(closing,
                                                   times,
                                                   pt_sl,
                                                   volatility,
                                                   min_ret,
                                                   threads,
                                                   vertical_barriers)


    labels_one = mlf.labeling.get_bins(triple_barrier_events, closing)

    t1 = triple_barrier_events['t1'].copy()


    full_df = pd.DataFrame(tick_bars.loc[labels_one['bin'].index], index=labels_one['bin'].index)
    full_df.drop(columns=['close','bid','ask'],inplace=True)
    full_df['labels'] = labels_one['bin'].copy()

    y = full_df['labels'].copy()
    full_df.drop(columns=['labels'],inplace=True)
    X = full_df.copy()

    sample_weights = get_weights_and_avgu(closing,X,threads,t1)[1]

    backtest_df = pd.DataFrame()

    i = 1
    cvGen=PurgedKFold(n_splits=10,t1=t1,pctEmbargo=0.01)
    for train,test in cvGen.split(X=X):

        X_train_full = X.iloc[train,:].copy()
        y_train_full = y.iloc[train].copy()
        X_test_full = X.iloc[test,:].copy()
        y_test_full = y.iloc[test].copy()


        X_train = X_train_full[y_train_full != 0].copy()

        X_train_addit = X_train_full[y_train_full == 0].copy()

        y_train = y_train_full.loc[X_train.index].copy()


        scaler = StandardScaler()
        new_training = scaler.fit_transform(X_train)


        new_testing_full = scaler.transform(X_test_full)

        X_train_stand = pd.DataFrame(new_training,index=X_train.index)

        X_test_stand_full = pd.DataFrame(new_testing_full,index=X_test_full.index)


        scaler_addit = StandardScaler()
        new_training_addit = scaler_addit.fit_transform(X_train_addit)
        X_train_addit_stand = pd.DataFrame(new_training_addit,index=X_train_addit.index)
        X_train_stand_full = pd.concat([X_train_stand,X_train_addit_stand])
        
       
        avgu = get_weights_and_avgu(closing,X_train,threads,t1)[0]
        
        svc=SVC(probability=True,gamma='auto',random_state=20)
        bagged_svc=BaggingClassifier(base_estimator=svc,n_estimators=1000,
                                    max_samples=avgu,max_features=1.,random_state=20)
        

        bagged_svc.fit(X_train_stand,y_train,sample_weight=sample_weights.loc[X_train.index].copy())
        
        
        y_pred_full = bagged_svc.predict(X_train_stand_full)
        primary_l = y_pred_full.copy()

        y_pred_full_test = bagged_svc.predict(X_test_stand_full)
        primary_l_test = y_pred_full_test.copy()

        side_train = pd.DataFrame(primary_l.copy(),index=X_train_stand_full.index)
        side_test = pd.DataFrame(primary_l_test.copy(),index=X_test_stand_full.index)
        side = pd.concat([side_train,side_test])
        side.sort_index(inplace=True)

      
        times = side.index

        vertical_barriers = vertical_barriers.loc[side.index]
        pt_sl = [1,1]
        min_ret = 0.004
        threads = cpu_count()-1

        triple_barrier_events = mlf.labeling.get_events(closing,
                                               times,
                                               pt_sl,
                                               volatility,
                                               min_ret,
                                               threads,
                                               vertical_barriers,
                                               side[0])

        labels = mlf.labeling.get_bins(triple_barrier_events, closing)

        t1 = triple_barrier_events['t1'].copy()

        new_y_train = labels['bin'].loc[X_train_stand_full.index].copy()
        new_x_train = tick_bars.loc[X_train_stand_full.index].copy()

        new_y_test = labels['bin'].loc[X_test_stand_full.index].copy()
        new_x_test = tick_bars.loc[X_test_stand_full.index].copy()


        new_x_train.dropna(inplace=True)
        new_x_train.drop(columns=['close','bid','ask'],inplace=True)

        new_x_test.dropna(inplace=True)
        new_x_test.drop(columns=['close','bid','ask'],inplace=True)

        new_x_train['predicted_side'] = side[0].loc[new_x_train.index].copy()
        new_x_test['predicted_side'] = side[0].loc[new_x_test.index].copy()

        new_x_train.sort_index(inplace=True)
        new_y_train.sort_index(inplace=True)
        new_x_test.sort_index(inplace=True)
        new_y_test.sort_index(inplace=True)

        avgmu = get_weights_and_avgu(closing,new_x_train,threads,t1)[0]

        dt_meta=DecisionTreeClassifier(criterion='entropy',max_features='auto',
        class_weight='balanced',min_weight_fraction_leaf=0.05)

        
        bagged_dt_meta=BaggingClassifier(base_estimator=dt_meta,n_estimators=1000,max_samples=avgmu,
                                         max_features=1.)

        bagged_dt_meta.fit(new_x_train,new_y_train,sample_weight=sample_weights.loc[new_x_train.index].copy())

        y_pred = bagged_dt_meta.predict(new_x_test)
        filt = y_pred.copy()

        ret_series = returns_series(new_x_test,side,t1,bid,ask,filt)

        backtest_df.at[i,'cum_returns'] = np.around(empyrical.stats.cum_returns(ret_series).iloc[-1]*100, 2)
        backtest_df.at[i,'sharpe_ratio'] = np.around(empyrical.stats.sharpe_ratio(ret_series),2)
        backtest_df.at[i,'max_drawdown'] = np.around(empyrical.stats.max_drawdown(ret_series)*100,2)
        i += 1
        
    return backtest_df
