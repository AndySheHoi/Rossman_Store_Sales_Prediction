import pandas as pd
import matplotlib.pyplot as plt

class EDA:
# =============================================================================
# Exploratory Data Analysis
# =============================================================================
    def eda(train, test, store):

        not_open = train[(train['Open'] == 0) & (train['Sales'] != 0)]
        print("No closed store with sales: " + str(not_open.size == 0))
        
        no_sales = train[(train['Open'] == 1) & (train['Sales'] <= 0)]
        print("No open store with no sales: " + str(no_sales.size == 0))
        
        train = train.loc[train['Sales'] > 0]
        
        dates = pd.to_datetime(train['Date']).sort_values()
        dates = dates.unique()
        start_date = dates[0]
        end_date = dates[-1]
        print("Start date: ", start_date)
        print("End Date: ", end_date)
        
        
        plt.rcParams['figure.figsize'] = (15.0, 12.0)
        
        f, ax = plt.subplots(7, sharex=True, sharey=True)
        for i in range(1, 8):
            data = train[train['DayOfWeek'] == i]
            ax[i - 1].set_title("Day {0}".format(i))
            ax[i - 1].scatter(data['Customers'], data['Sales'], label=i)
        
        plt.legend()
        plt.xlabel('Customers')
        plt.ylabel('Sales')
        plt.tight_layout()
        plt.show()
        
        
        # plot customer vs sales for each day of week
        plt.scatter(train['Customers'], train['Sales'], c=train['DayOfWeek'], alpha=0.6, cmap=plt.cm.get_cmap('YlGn'))
        
        plt.xlabel('Customers')
        plt.ylabel('Sales')
        plt.show()
        
        
        # explore the effect of school holiday in sales
        for i in [0, 1]:
            data = train[train['SchoolHoliday'] == i]
            if (len(data) == 0):
                continue
            plt.scatter(data['Customers'], data['Sales'], label=i)
        
        plt.legend()
        plt.xlabel('Customers')
        plt.ylabel('Sales')
        plt.show()
        
        
        # explore the effect of promotion in sales
        for i in [0, 1]:
            data = train[train['Promo'] == i]
            if (len(data) == 0):
                continue
            plt.scatter(data['Customers'], data['Sales'], label=i)
        
        plt.legend()
        plt.xlabel('Customers')
        plt.ylabel('Sales')
        plt.show()
        
        
        train['SalesPerCustomer'] = train['Sales'] / train['Customers']
        
        avg_store = train.groupby('Store')[['Sales', 'Customers', 'SalesPerCustomer']].mean()
        avg_store.rename(columns=lambda x: 'Avg' + x, inplace=True)
        store = pd.merge(avg_store.reset_index(), store, on='Store')
        avg_store.head()
        store.head()
        
        # explore the effect of categorical values in sales
        
        # StoreType
        for i in store.StoreType.unique():
            data = store[store['StoreType'] == i]
            if (len(data) == 0):
                continue
            plt.scatter(data['AvgCustomers'], data['AvgSales'], label=i)
        
        plt.legend()
        plt.xlabel('Average Customers')
        plt.ylabel('Average Sales')
        plt.show()
        
        # Assortment
        for i in store.Assortment.unique():
            data = store[store['Assortment'] == i]
            if (len(data) == 0):
                continue
            plt.scatter(data['AvgCustomers'], data['AvgSales'], label=i)
        
        plt.legend()
        plt.xlabel('Average Customers')
        plt.ylabel('Average Sales')
        plt.show()
        
        # Promo2
        for i in store.Promo2.unique():
            data = store[store['Promo2'] == i]
            if (len(data) == 0):
                continue
            plt.scatter(data['AvgCustomers'], data['AvgSales'], label=i)
        
        plt.legend()
        plt.xlabel('Average Customers')
        plt.ylabel('Average Sales')
        plt.show()

        return train, test, store