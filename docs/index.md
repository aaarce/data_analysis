<img src="https://raw.githubusercontent.com/prussell21/kickstarter-data-analysis/master/docs/images/wordcloud.png">

## Know Your Crowd

What are the ingredients of a great Kickstarter?

Through data analysis I have discovered some of the key aspects of a successful Kickstarter beyond the product itself. Not all projects are created equal, and depending on what genre or category your project belongs to, Kickstarter just may not have the right 'buyers' for what you are selling.

For those who do not know, Kickstarter is website and platform in which projects of all kinds can be crowdfunded.
Those with an idea and some ambition can create a Kickstarter project in hopes of reaching other like-minded people who are interested in
helping fund it. The people who help fund these projects are known as 'backers' and can be rewarded with tangible goods throughout the progression of a project.

From Kaggle, I have collected a large dataset of over 350,000 projects with their corresponding names, categories, launch dates, etc.
in hopes of answering a few questions about the Kickstarter platform and the kinds of projects it's visitors like to fund and not fund.

[https://www.kaggle.com/datasets]

## Exploratory Questions and Motivations for the Project

- What are the most successful Kickstarter project categories?

- How does the size of a project's goal effect it's chance of success?

- What is the relationship between the size of a project and its amount of backers?

- Is it possible to build a model and accurately predict the likelihood of success for a project using this dataset?


#### Analysis

**What are the most successful Kickstarter project categories?**

<img src="https://github.com/prussell21/kickstarter-data-analysis/blob/master/docs/images/category_count_plot.png?raw=true">

It appears, regardless of category that most projects do not end in success. Film & Video, Music, Publishing, and Games are visually the most popular.

**Top Ten Most Popular Categories for Kickstaer projects**

|Genre        |Count	| % of Total|% of Success|      
|:------------|------:|----------:|--------:|
|Film & Video | 63582 | 16.791760 |	6.238743|
|Music        | 51917 |	13.711079 |	6.390334|
|Publishing   | 39873	| 10.530305	| 3.248382|
|Games        | 35230	| 9.304107	| 3.305955|
|Technology	  | 32569	| 8.601347	| 1.699195|
|Design	      | 30068	| 7.940842	| 2.786214|
|Art	        | 28151	| 7.434570	| 3.039746|
|Food         |	24602	| 6.497293	| 1.607025|
|Fashion	    | 22816	| 6.025617	| 1.477090|
|Theater	    | 10912	| 2.881817	| 1.725604|

There seems to be somewhat of a linear relationship between the popularity of the category and its project's success (or vice versa). Looking to create a short film/video or record a mixtape? Kickstarter might be the place to raise funds, but what should be your goal amount?

**How does the size of a project's goal effect it's chance of success?**

<img src="https://github.com/prussell21/kickstarter-data-analysis/blob/master/docs/images/project_goal_size.png?raw=true">

Most projects are created with a goal size of $1,000 to $10,000. As expected, the projects that have a goal size of less than $1,000 have the greatest chance for success.

**What is the relationship between the size of a project and its amount of backers for both successful and unsuccessful projects?**

<img src="https://github.com/prussell21/kickstarter-data-analysis/blob/master/docs/images/backers_vs_goal_amount.png?raw=true">

It appears, there is a slight positive correlation between the size of a project's goal and the amount of backers who contribute. The vast majority of projects receive less than 25,000 backers.

#### Building a Model

**Is it possible to build a model and accurately predict the likelihood of success for a project using this dataset?**

Short Answer: Yes

Using the project length, category, sub-category, and project goal size, I was able to train a Logistic Regression model to predict the success or failure of a Kickstarer project with an **88.5%** accuracy!

##### Logistic Regression Explained

Logistic regression is an algorithm that is trained using a dataset (i.e. Kickstarter projects) to build a mathematical equation for prediction. The variables of the model are weighted according to each features (goal size, deadline, category, etc.) influence on the ultimate outcome of a project. Then, each future project that is input into this model will return an output of it's probability of success or failure. Projects with a probability output greater than 50% are labeled as successful and projects less than 50% are labeled as unsuccessful.

After training the logistic regression model with 70% of the project dataset, the model was then applied to the remaining 30%, in which it correctly labeled nearly 90% of the those projects as either succeeding or failing.


## Conclusion

For those thinking of creating a Kickstarter project, it is clear that some types of projects may have a more promising outlook than others. Backers who frequent the platform seem to be most interested in film, publishing, games and music projects rather than food, fashion, or theater. In addition and unsuprisingly, a project's goal size indeed effects it's success. This may be explain why tech projects are quite popular but are less successful due to traditionally requiring more funds to complete.

Lastly, with just the data and never viewing these projects or having to apply human intuition to evaluate or form opinions of which may or may not reach their goal, a machine learning model was proficient in predicting their outcomes. This tells us that the few pieces of information on each project within this dataset have great effect on whether a project becomes a favorite or flops.

