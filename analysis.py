import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
#from sklearn import linear_model
import statsmodels.api as sm
import datetime

stop_data = defaultdict(list)
stop_name = {}

with open('data.csv','r') as datafile:
    reader = csv.reader(datafile)
    
    # get rid of that header
    reader.next()

    for row in reader:
        station_id, stationname, date, daytype, rides = row
        
        if stationname != 'Homan': # this is some weird stop with no data'
            stop_name[station_id]=stationname
            
            stop_data[station_id].append((date,daytype,rides))

f = open('results.csv','wb')
csv_writer = csv.writer(f)
csv_writer.writerow(['Stop name','Stop ID', 'x coeff', 'x^2 coeff','month coeff','month^2 coeff','month^3 coeff', 'constant', 'x p-value', 'x^2 p-value','month p-value','month^2 p-value','month^3 p-value', 'constant p-value', 'R^2'])

for stop,days in stop_data.iteritems():
    print "calculating for stop %s" % stop_name[stop]
    day_nums = []
    month_nums = []
    rider_totals = [] 
    # loop through all days for a stop
    for i,(datestr,daytype,rides) in enumerate(days):
        # conditions to include a day in our analysis:
        # * not a weekend or holiday (daytype must = w)
        # * not black friday (unofficial day off, next to thanksgiving)
        # * not in december, january (people's breaks interfere, could be done in a smarter way)
        # * not an extreme outlier from the regression (we need to define this)
        #      - these include Gay Pride, 3rd of July some years, others

        month,day,year = map(int,datestr.split('/'))
        #if daytype == "W" and month not in ['12','01','1'] and int(rides) != 0:
        if daytype == "W" and int(rides) != 0: 
            diff = datetime.date(year,month,day) - datetime.date(2001,1,1)
            day_nums.append(diff.days)
            month_nums.append(int(month))
            rider_totals.append(rides)
            
    day_array = np.array(map(int,day_nums))
    day_sq = [d**2 for d in day_nums]
    day_sq_array = np.array(map(int,day_sq))
    month_array = np.array(month_nums)
    month_sq_array = [m**2 for m in month_nums]
    month_cu_array = [m**3 for m in month_nums]
    rider_totals_array = np.array(map(int,rider_totals))
    
    x_array  = np.column_stack((day_array,day_sq_array,month_array,month_sq_array,month_cu_array))
    x_array = sm.add_constant(x_array, prepend=False)
    
    # print x_array
    
    ### Using statsmodels
    regr = sm.OLS(rider_totals_array,x_array).fit()

    ## go through X and Y values, build new dataset removing any with
    ## normalized residuals outside [-2,2]
    resids = regr.norm_resid()
    new_x = np.zeros((1,6),dtype=int)
    new_y = np.zeros((1,1),dtype=int)

    for i,resid in enumerate(resids):
        if resid < 2 and resid > -2:
            if new_x.shape == (1,6) and new_x[0,0] == 0 and new_x[0,1] == 0 and new_x[0,2] == 0:
                new_x = x_array[i,:]
                new_y = [rider_totals_array[i]]
            else:
                new_x = np.vstack((new_x, [x_array[i,:]]))
                new_y = np.hstack((new_y, [rider_totals_array[i]]))

    better_regr = sm.OLS(new_y,new_x).fit()

    if stop == '40790' or stop == 40790:
        g = open('monroe_state.csv','wb')
        csv_writer = csv.writer(g)
        values = better_regr.predict()
        for i,a in enumerate(better_regr.resid):
            csv_writer.writerow([a,values[i]])
        break


    filename = stop_name[stop]
    keepcharacters = (' ','.','_')
    filename = "".join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip()

    fig = plt.figure()
    ax = plt.gca()

    plt.axis([-300,5000,0,15000])
    ax.set_ylabel('Riders')
    ax.set_xlabel('Days since 1/1/2001')
    ax.set_title(stop_name[stop]) 

    ax.scatter(day_array,rider_totals_array, c='orange', alpha=.2, linewidths=.4)
    ax.plot(day_array, regr.predict(),color='black', lw=1.5)
    plt.savefig('plots/%s-plot.png' % filename, format='png')
    plt.close()

    fig = plt.figure()
    ax = plt.gca()

    plt.axis([-300,5000,0,15000])
    ax.set_ylabel('Riders')
    ax.set_xlabel('Days since 1/1/2001')
    ax.set_title(stop_name[stop] + ", outliers removed") 

    ax.scatter(new_x[:,0],new_y, alpha=.2, c='orange', linewidths=.4)
    ax.plot(new_x[:,0], better_regr.predict(),color='black',lw=1.5)
    plt.savefig('plots/%s-plot-improved.png' % filename, format='png')
    plt.close()

    fig = plt.figure()
    ax = plt.gca()

    plt.axis([-300,5000,0,25000])
    ax.set_ylabel('Riders')
    ax.set_xlabel('Days since 1/1/2001')
    ax.set_title(stop_name[stop]) 

    ax.scatter(day_array,rider_totals_array, c='orange', alpha=.2, linewidths=.4)
    ax.plot(day_array, regr.predict(),color='black', lw=1.5)
    plt.savefig('plots-bigaxis/%s-plot.png' % filename, format='png')
    plt.close()


    fig = plt.figure()
    ax = plt.gca()

    plt.axis([-300,5000,0,25000])
    ax.set_ylabel('Riders')
    ax.set_xlabel('Days since 1/1/2001')
    ax.set_title(stop_name[stop] + ", outliers removed") 

    ax.scatter(new_x[:,0],new_y, alpha=.2, c='orange', linewidths=.4)
    ax.plot(new_x[:,0], better_regr.predict(),color='black',lw=1.5)
    plt.savefig('plots-bigaxis/%s-plot-improved.png' % filename, format='png')
    plt.close()

    
    csv_writer.writerow([stop_name[stop], stop] + list(better_regr.params) + list(better_regr.pvalues) + [better_regr.rsquared_adj])



f.close()
