import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt


plt.style.use('ggplot')
########################################
## helper function
########################################
def howManyType(column):

    """ Get to know how many types for a categories variable"""

    return len(set(column))

def get_split_date(df, date_column, quantile):

    """ Get the date on which to split a dataframe for timeseries splitting """

    # 1. convert date_column to datetime (useful in case it is a string)
    # 2. convert into int (for sorting)
    # 3. get the quantile
    # 4. get the corresponding date
    # 5. return, pray that it works
    df1 = pd.to_datetime(df[date_column][~df[date_column].isnull()], unit='ns', errors='coerce').astype('int64')
    quantile_date = df1.quantile(q=quantile).astype('datetime64[ns]')
    return quantile_date

def categorize_date_time(column, quantile):
    minDate = min(column[~column.isnull()])
    maxDate = max(column[~column.isnull()]) + datetime.timedelta(days=1)
    column.loc[column.isnull()] = maxDate
    for i, date in enumerate(quantile):
        column.loc[(column < date) & (column >= minDate)] = minDate
        minDate = date
    column.loc[(column > minDate) & (column<maxDate)] = minDate
    return column

def categorize_age(column, quantile):
    nonNaColumn = column[~column.isnull()]
    minAge = min(nonNaColumn)
    quantile_age = nonNaColumn.quantile(q=quantile)
    column.loc[column.isnull()] = -1
    for age in quantile_age:
        column.loc[(column < age) & (column >= minAge)] = minAge
        minAge = age
    column.loc[column > minAge] = minAge
    return column

def dummies(df):
    colnames = df.columns
    for name in colnames:
        if name == 'hasBooked' or name == 'signup_flow':
            continue
        dummies = pd.get_dummies(df[name], prefix=df[name].name)
        df.drop(name, axis=1, inplace=True)
        df = pd.concat((df, dummies.astype(int)), axis=1)
    return df






########################################
## read the data
########################################
train = pd.read_csv('train_users_2.csv')
#print train.head()
#print train.info()

colname = ['id',
           'date_account_created',
           'timestamp_first_active',
           'date_first_booking',
           'gender',
           'age',
           'signup_method',
           'signup_flow',
           'language',
           'affiliate_channel',
           'affiliate_provider',
           'first_affiliate_tracked',
           'signup_app',
           'first_device_type',
           'first_browser',
           'country_destination']

print train.isnull().sum(0)
print train.shape
train.drop('id', axis=1, inplace=True)

########################################
## Deal with date type: to Y-M
########################################
dateColumns = ['date_account_created',
               'date_first_booking']
## deal with account_created date into 4 group
train['created'] = pd.to_datetime(train['date_account_created'], unit='ns', dayfirst=True).dt.date
quantile = get_split_date(train, 'date_account_created', [.25,.5,.75]).dt.date
categorize_date_time(train['created'], quantile)



## Create a features to see the account has booking before, and split the exists date into groups
train['first_booking'] = pd.to_datetime(train['date_first_booking'], unit='ns', dayfirst=True, errors='coerce').dt.date
train['hasBooked'] = train['date_first_booking'].isnull().astype(int)
quantile = get_split_date(train, 'date_first_booking', [.25,.5,.75]).dt.date
categorize_date_time(train['first_booking'], quantile)
train.drop('date_first_booking', axis=1, inplace=True)
#train['created_cut'] = pd.qcut(train['created'].values, 5).codes + 1

## Deal with age
categorize_age(train['age'], [.25,.5,.75])

## Deal with timestamp, find the time differenct between it created and active
# train['timestamp_first_active'] = pd.to_datetime(train['timestamp_first_active'].astype(str),
#                                                  unit='ns', yearfirst=True, errors='coerce').dt.date
# train['timeDelta'] = (pd.to_datetime(train['date_account_created'], unit='ns').dt.date - train['timestamp_first_active']).dt.days
# print len(set(train['timeDelta']))
## this is not realy useful just drop it
train.drop('date_account_created', axis=1, inplace=True)
train.drop('timestamp_first_active', axis=1, inplace=True)

## fill the the left NA
train['first_affiliate_tracked'].fillna('untracked', inplace=True)

#print train.apply(howManyType, axis=0)


categoriesColumns = ['gender',
                     'signup_method',
                     'signup_flow',
                     'signup_app',
                     'language',
                     'affiliate_channel',
                     'first_device_type',
                     'affiliate_provider',
                     'first_affiliate_tracked',
                     'first_browser']
##print train.apply(howManyType, axis=0)
print train[categoriesColumns].apply(set, axis=0)
#sns.countplot(x="gender", data=train, palette="Greens_d")
##sns.countplot(x="country_destination", data=train, hue='gender', palette="Greens_d")
##this show that, customer who do not book which no information
# sns.countplot(x="created", data=train, palette="Greens_d")
# plt.show()
# sns.countplot(x="age", data=train, palette="Greens_d")
# plt.show()
# sns.countplot(x="timeDelta", data=train[train['timeDelta']!=0], palette="Greens_d")
# plt.show()

print train.info()
print train.head()
print train.apply(howManyType, axis=0)

# gender_dummies = pd.get_dummies(train.gender, prefix=train.gender.name)
# print gender_dummies
label = train['country_destination'].to_frame()
train.drop('country_destination', axis=1, inplace=True)
train = dummies(train)
# print train.info()
label.to_csv('train_label.csv', index=False)
train.to_csv('train_data.csv', index=False)
