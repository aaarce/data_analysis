#!/usr/bin/env python
# coding: utf-8

# # Kickstarter Projects Analysis

# ### Motivation
#
# 1. What are the most successfull Kickstarter categories?
#
# 2. How does the size of Project's goal effect the success of a project?
#
# 3. What is the relationship between the size of a project and its amount of backers for both successful and failed projects?
#
# 4. Is it possible to build a model and predict chance of success for a project with this dataset?
#

# ##### The Data
#
# Dataset consisting of over 350,000 Kickstarter Projects from April 2009 to February 2018. Collected from Kaggle Datasets: https://www.kaggle.com/datasets

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('ks-projects-201801.csv')


# In[4]:


print ('DataFrame Shape', df.shape)
df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.nunique()


# ### Understanding the Data & Preparation for Analysis
#
# Performing some data cleaning, validation, and sanity checks before performing any analysis

# In[8]:


df = df.drop(['ID', 'goal', 'pledged', 'currency', 'usd pledged'], axis=1)


# Starting by dropping columns that aren't valuable to the analysis or redundant. USD_pledged_real and USD_goal_real will be used rather than Pledged or Currency, as these are in the local country's currency

# #### Missing Data

# In[9]:


### Checking for missing values
df.isnull().sum()


# Only the 'name' column seems to have missing data. With only 4 samples here, I am going to drop these from the dataset.

# In[10]:


##Removing missing data
df.dropna(inplace=True)


# #### Dates

# In[11]:


df.sort_values('launched').head(10)


# Some of the projects seem to have a UNIX epoch as a default for unknown values. With only 7, it shouldnt't be an issue if to drop these projects from the analysis as well.

# In[ ]:





# In[12]:


#Convert date columns to datetime and make time delta column (Deadline - Launched) in hours
df.launched = pd.to_datetime(df.launched)
df.deadline = pd.to_datetime(df.deadline)

df['timedelta'] = (df.deadline-df.launched).astype('timedelta64[h]')

df.loc[df['timedelta'] == 0]


# It looks like there were two projects that ended within the same hour of launching. May have to calculate time delta in hours (or days) as a decimal

# In[13]:


df['timedelta'] = (df.deadline-df.launched).astype('timedelta64[m]')
df['timedelta_days'] = (df['timedelta']/60)/24
df = df.drop('timedelta', axis=1)


# In[14]:


df.sort_values('timedelta_days', ascending=False).head(10)


# In[15]:


###Dropping projectss with extreme timedelta's greater than 1 year
df = df.loc[df['timedelta_days'] < 366]


# #### Outliers and Distributions
#
# To get a good understanding of the questions that are being asked, it may be necessary to remove projects with very small and large project goal's

# In[16]:


#### Projects with goals below $500 and more than $10,000,000
print ('Projects with less than $500 goal: ',len(df.loc[df.usd_goal_real < 500.0]))
print ('Projects with more than $10M goal: ',len(df.loc[df.usd_goal_real > 1000000.0]))


# The distribution seems to be skewed heavily to the left, with a few extremely high project goals and many small project goals. May need to scale this data for further data visualization

# In[17]:


figsize = (18,6)

def histogram_plot(dataset, column, x_label, title):
    '''
    Plots histogram of input feature

    INPUT
    dataset = dataset with feature that is to be plotted
    column = feature of dataset to be plotted
    x_label = Label title of the x axis
    title = Plot figure title

    OUTPUT
    Distribution plot
    '''
    plt.figure(figsize=figsize);
    plt.hist(data = dataset, x = column, bins = bins);
    plt.xscale('log');
    plt.xticks(ticks, labels);
    plt.xlabel(x_label);
    plt.grid(False)
    plt.title(title)


# In[18]:


#USD PLEDGE GOAL DISTRIBUTION

#Selected histogram plot prameters using:(np.log10(df['usd_goal_real'].describe()))
bins = 10 ** np.arange(0, 9, .1)
ticks = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
labels = ['{:,}'.format(val) for val in ticks]

histogram_plot(df, 'usd_goal_real', '$ Pledge Goal', 'Pledge Goal Distribution')


# Data scaled with Log10 to see a much cleaner and easily understood histogram of usd pledge goals. Any modeling for prediction will require these values to be scaled.
#
# Projects with Pledge goals between 100 and 1,000,000 USD seem to be the appropriate sample for future modeling.

# In[19]:


# USD PLEDGED DISTRIBUTION

#Selected histogram plot prameters using:(np.log10(df['usd_pledged_real'].describe()))
bins = 10 ** np.arange(-.1, 8, .1)
ticks = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
labels = ['{:,}'.format(val) for val in ticks]

histogram_plot(df, 'usd_pledged_real', '$ Pledged', 'Amount Pledged Distribution')


# Pledge amounts seem to be slightly left skewed.

# In[20]:


###Project Length Distribution

figsize = (18,6)

# Selected plot parameters using: np.log10(df['timedelta_days'].describe())
bins = 10 ** np.arange(0, 3, .1)
ticks = [1, 3, 10, 30, 100, 300, 1000]
labels = ['{:,}'.format(val) for val in ticks]

histogram_plot(df, 'timedelta_days', 'Length of Project in Days', 'Project Length Distribution')


# Majority of projects have a deadline within 30 days of launch

# #### Further Investigation of Outliers

# In[21]:


##High Pledge Goal Projects (2.5M USD)
df.loc[df['usd_goal_real'] > 2500000].sort_values('usd_goal_real', ascending=False).head(10)


# Once again, may need to remove these rows when training a model to possibly predict chance of success. The pledge goals are unrealistic and will hurt the gernalization of our model

# In[22]:


##USD Pledge goal
df.loc[df['usd_pledged_real'] > 10000000].sort_values('usd_pledged_real',
                                                      ascending=False)


#
#
#

# #### Data Preparation Conclusion
#
# Now that the data has been reviewed and cleaned up a bit. I can better answer these questions. I will keep some of the questionable 'outlier' projects for now, but may remove them when training a model in order to help generalize better.
#

#

# #### Analysis - Finding Answers
#
# What are the most popular and/or successfull Kickstarter categories?

# In[23]:


print ('Unique Categories: ',df.category.nunique())
print ('Unique Main Categories: ', df.main_category.nunique())

pd.DataFrame(data={
            'Count':df.main_category.value_counts(),
            '% of Total Projects': (df.main_category.value_counts()/(len(df))*100),
            '% of Success': (df.main_category.loc[df['state'] == 'successful']
                             .value_counts()/(len(df))*100)
            }).sort_values('Count', ascending=False)


# It seems there is a correlation between how popular a category is, and its rate of success

# In[24]:


## State countplot by category
from matplotlib import rcParams

rcParams['figure.figsize'] = 20,8
ax = sns.countplot(x="main_category", hue="state", palette='deep',data=df)
ax.set_title("Category Countplot");


# The most popular categories in this dataset are are Film & Radio and Music. There is also an undefined state for project status.

# What are the success rates of projects of different pledge goal sizes?

# In[25]:


##Dropping projects with undefined and live states
df = df.loc[df.state != 'undefined']
df = df.loc[df.state != 'live']


# In[26]:


#Creating Bin columns for project size
df['bin'] = pd.cut(df['usd_goal_real'],
                   [1, 1000, 10000, 100000, 1000000, 1000000000],
                   labels=['Less than 1000',
                           '1,000 to 10,000',
                           '10,000 to 100,000',
                           '100,000 to 1,000,000',
                           'Greater than 1,000,000'],
                   duplicates='drop')


# In[27]:


ax = sns.countplot(x="bin", hue="state", data=df, palette='deep');
ax.set(xlabel='Project Goal Size in $', ylabel='Count');
ax.set_title("Project Goal Size Count Plot");


# At first glance, it looks like the smaller projects seem to be more successful, as expected.

#

# What is the average pledged by each backer for successful and failed projects?

# In[28]:


##All Projects
ax = sns.regplot(x='backers',y='usd_pledged_real', data=df)
ax.set(xlabel='Amount of Backers', ylabel='USD Pledged Goal Amount');
ax.set_title("Regression Plot - Amount of Backers vs. Pledge Goal Amount");


# In[29]:


###Now just with successful projects
success = df.loc[df['state'] == 'successful']


# In[30]:


ax = sns.regplot(x='backers',y='usd_pledged_real', data=success)
ax.set(xlabel='Amount of Backers', ylabel='USD Pledged Goal Amount');
ax.set_title("Successful Projects - Backers vs. USD Pledge Goal Amount");


# In[31]:


###Failed
failed = df.loc[df['state'] == 'failed']


# In[32]:


ax = sns.regplot(x='backers',y='usd_pledged_real', data=failed)
ax.set(xlabel='Amount of Backers', ylabel='USD Pledged Goal Amount');
ax.set_title("Failed Projects - Backers vs. USD Pledge Goal Amount");


#

# Can we predict the success of a project?

# In[33]:


model_df = df.copy()


# In[34]:


##Dropping some columns before training model

#Unfortunately cannot usd backers as a feature here. Since we do not know the backers of a project at the creation
#of one.
model_df = model_df.drop(['name', 'deadline', 'launched', 'backers', 'bin'], axis=1)


# Last minute data cleaning for training

# In[35]:


##Dropping state of undefined projects, since this is going to be our target variable
model_df = model_df.loc[df['state'] != 'undefined']


# Dealing with categorical variables

# In[36]:


## Getting Dummies for state column
cat_columns = ['category', 'main_category', 'country']
cat_df = pd.get_dummies(model_df[cat_columns], prefix=cat_columns)

model_df = pd.concat([model_df, cat_df], axis=1)
model_df.drop(cat_columns, axis=1, inplace=True)

#Formatting target variable to 1(successful) and 0(not successful)
model_df.state.replace(['successful', 'failed', 'canceled', 'suspended'], [1, 0, 0, 0], inplace=True)


# Preprocessing Data

# In[37]:


#Converting target variable to numerical values
feats = model_df.iloc[:,1:]
target = model_df.iloc[:, 0]


# In[38]:


##Splitting the data
def splitting_data(X,y):
    '''
    INPUT
    X = feature data
    y = target data

    OUTPUT
    X_train, X_test, y_train, y_test
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = splitting_data(feats, target)


# In[ ]:





# In[144]:


#Scaling Data

def scale_data(train, test):
    '''
    INPUT
    train = training dataset
    test = testing dataset

    OUTPUT
    scaled_X_test = test set scaled using StandardScaler
    scaled_X_train = train set scaled using StandardScaler
    '''
    scaler = StandardScaler();
    scaled_X_test = scaler.fit_transform(train);
    scaled_X_train = scaler.fit_transform(test);

    return scaled_X_test, scaled_X_train

X_train, X_test = scale_data(X_train, X_test);


# Simple Logistic Regression Model

# In[40]:


#instantiate and fit logistic regression model
clf = LogisticRegression(solver='lbfgs', max_iter=500).fit(X_train, y_train)


# Scoring and Prediction

# In[41]:


clf.score(X_test, y_test)


# In[42]:


###Percent of successful projects
success_pct =  (y_test.sum()/len(y_test))

#Benchmark to beat (assuming the model selected failure for all projects)
print (1-success_pct)


# In comparison, it looks like this model is quite successful at predict the project state
#
# 88% percent model accuracy vs. the 'dummy prediction' of 64%

# In[43]:


#output probability
clf.predict_proba(X_test)


# In[112]:


#Create wordcloud of main_categories


# In[113]:


words = df['category']


# In[133]:


comment_words = ' '
stopwords = set(STOPWORDS)

for val in df['category']:

    # typecaste each val to string
    val = str(val)

    # split the value
    tokens = val.split()

    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    for words in tokens:
        comment_words = comment_words + words + ' '


# Creating word cloud for main_category series of dataset

# In[134]:


wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'white',
    stopwords = STOPWORDS).generate(str(comment_words))


# In[135]:


fig = plt.figure(
    figsize = (80, 40),
    facecolor = 'k',
    edgecolor = 'k')


# In[143]:


plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
