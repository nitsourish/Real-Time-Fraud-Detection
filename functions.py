def missing_quantity(data,column):
    missing = pd.DataFrame(data[column].apply(lambda x: np.sum(x.isnull(), axis=0)))
    missing['count_missing'] = missing.iloc[:,0]
    missing['percentage_missing'] = (missing.iloc[:,0]/data.shape[0])*100
    missing.drop(missing.columns[0],axis='columns',inplace=True)
    return(missing)