![category wordcloud](https://github.com/prussell21/kickstarter-data-analysis/blob/master/docs/images/wordcloud.png)
*The caption for my image*

## Know your Crowd

What are the ingredients of a great Kickstarter?

Through analysis I have helped discover some of the keys aspects of a successful Kickstarter beyond the product itself. Not all projects are created equal, and depending on what genre, category, or theme your project belongs to, Kickstarter just may not have the right 'buyers' for what your selling.

For those who do not know, Kickstarter is website and platform in which projects of all kinds can be crowdfunded.
Those with an idea and some ambition can create a Kickstarter project in hopes of other like-minded people who are interested 
helping fund this project (in large or small amounts). The people who help fund these projects are known as 'backers' and can be rewarded with tangible goods upon the success of a project.

For this analysis, I have collected a large dataset of over 350,000 projects (from kaggle.com) with their corresponding names, categories, launch dates, etc.
in hopes of answering a few questions about the Kickstarter platform and the kinds of projects it funds and doesn't fund.

## Exploratory Questions and Motivations for the Project

- What are the most successful Kickstarter project categories?

- How does the size of Project's goal effect the success of a project?

- What is the relationship between the size of a project and its amount of backers for both successful and unsuccessful projects?

- Is it possible to build a model and successfully predict the likelihood of success for a project using this dataset?


## Analysis

**What are the most successful Kickstarter project categories?**

![cateogry countplot](https://github.com/prussell21/kickstarter-data-analysis/blob/master/docs/images/category_count_plot.png)

It appears regardless of category, most projects do not end in success. Film & Video, Music, Publishing, Technology are visually the most popular.


|Genre        |Count	| % of Total|% of Success|      
|:------------|------:|----------:|--------:|
|Film & Video | 63582 | 16.791760 |	6.238743|
|Music        | 51917 |	13.711079 |	6.390334|
|Publishing   | 39873	| 10.530305	| 3.248382|
|Games        | 35230	| 9.304107	| 3.305955|
|Technology	  | 2569	| 8.601347	| 1.699195|
|Design	      | 30068	| 7.940842	| 2.786214|
|Art	        | 28151	| 7.434570	| 3.039746|
|Food         |	24602	| 6.497293	| 1.607025|
|Fashion	    | 22816	| 6.025617	| 1.477090|
|Theater	    | 10912	| 2.881817	| 1.725604|

There seems to be a somewhat of a linear relationship between the popularity of the category and it's projects success (or vice versa). Looking to create a short film/video or record a mixtape? Kickstarter might be the place to raise funds.

**How does the size of Project's goal effect the success of a project?**

![size countplot](https://github.com/prussell21/kickstarter-data-analysis/blob/master/docs/images/project_goal_size.png)


**What is the relationship between the size of a project and its amount of backers for both successful and unsuccessful projects?**

![backers vs. goal size regression plot](https://github.com/prussell21/kickstarter-data-analysis/blob/master/docs/images/backers_vs_goal_amount.png)


**Is it possible to build a model and successfully predict the likelihood of success for a project using this dataset?**

Short Answer: Yes

Using the project length, category, sub-category, and project goal size, I was able to train a Logistic Regression model to predict the success or failure of a Kickstarer project with an 88.5% accuracy!



## Conclusion
