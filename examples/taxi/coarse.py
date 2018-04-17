import flor

with flor.Experiment('coarse') as ex:
    
    ex.groundClient('ground')
    
    @flor.func
    def run_existing_pipeline(path_to_data, kernel):
        import math
        import numpy as np
        import pandas as pd
        from sklearn import metrics
        from sklearn.svm import SVR
        from sklearn.model_selection import train_test_split

        def manhattan_distance(x1, y1, x2, y2):
            return abs(x1 - x2) + abs(y1 - y2)

        def roundtime(tstring):
            hours, mins, secs = tstring.split(':')
            if int(mins) >= 30:
                if hours == '23':
                    return '00'
                else:
                    return str(int(hours) + 1)
            else:
                return hours

        def weekday(start):
            from datetime import datetime
            fmt = '%Y-%m-%d %H:%M:%S'
            tstamp = datetime.strptime(start, fmt)
            return int(tstamp.weekday())

        data_df = pd.read_csv(path_to_data)

        data_df['distance'] = [i for i in map(manhattan_distance,
            data_df['pickup_longitude'], data_df['pickup_latitude'], 
            data_df['dropoff_longitude'], data_df['dropoff_latitude'])]

        # Remove outliers in passenger_count
        data_df = data_df[data_df['passenger_count']>0]
        data_df = data_df[data_df['passenger_count']<9]

        # Remove coordinate outliers
        data_df = data_df[data_df['pickup_longitude'] <= -73.75]
        data_df = data_df[data_df['pickup_longitude'] >= -74.03]
        data_df = data_df[data_df['pickup_latitude'] <= 40.85]
        data_df = data_df[data_df['pickup_latitude'] >= 40.63]
        data_df = data_df[data_df['dropoff_longitude'] <= -73.75]
        data_df = data_df[data_df['dropoff_longitude'] >= -74.03]
        data_df = data_df[data_df['dropoff_latitude'] <= 40.85]
        data_df = data_df[data_df['dropoff_latitude'] >= 40.63]

        # Remove trip_duration outliers
        trip_duration_mean = np.mean(data_df['trip_duration'])
        trip_duration_std = np.std(data_df['trip_duration'])
        data_df = data_df[data_df['trip_duration'] <= trip_duration_mean + 2*trip_duration_std]
        data_df = data_df[data_df['trip_duration'] >= trip_duration_mean - 2*trip_duration_std]
        data_df = data_df[data_df['trip_duration'] >= 30]
        data_df = data_df[data_df['trip_duration'] <= 60*240]

        data_df['start_hr'] = data_df['pickup_datetime'].apply(lambda x: int(roundtime(x.split(' ')[1])))
        data_df['start_month'] = data_df['pickup_datetime'].apply(lambda x: int(x.split(' ')[0].split('-')[1]))
        data_df['start_weekday']= data_df['pickup_datetime'].apply(lambda x: weekday(x))

        X = data_df[['vendor_id', 'pickup_longitude',
                    'pickup_latitude', 'dropoff_longitude', 
                    'dropoff_latitude', 'distance',
                    'start_hr', 'start_month', 'start_weekday']]
        y = data_df['trip_duration']

        X_train, X_test, y_train, y_test = train_test_split(X, 
            y, test_size = 0.2, random_state = 0)

        clf = SVR(kernel=kernel)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        score = metrics.explained_variance_score(y_test, preds)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, preds))

        score = "R2: {}".format(score)
        rmse = "RMSE: {}".format(rmse)
        
        print(score, rmse)
    
        return clf, score, rmse
    
    data = ex.artifact('train.csv')
    kernel = ex.literal(['poly', 'rbf'], 'kernel')
    kernel.forEach()
    do_all = ex.action(run_existing_pipeline, [data, kernel])
    model = ex.artifact('model.pkl', do_all)
    score = ex.artifact('score.txt', do_all)
    rmse = ex.artifact('rmse.txt', do_all)
# score.plot()
score.pull({'score': score, 'rmse': rmse})
