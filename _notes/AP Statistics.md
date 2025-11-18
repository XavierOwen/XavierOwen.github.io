---
title: "AP®︎ Statistics"
collection: notes
category: math
excerpt: "Statistics basic concepts and practice questions"
permalink: "/note/AP-Statistics"
date: 2025-11-17
toc: true
---

> The table of contents is a mix from [Khan Academy](https://www.khanacademy.org/math/ap-statistics), and [skewthescript](https://skewthescript.org/ap-stats-curriculum). The actual contents are created by my own.
>
> The purpose of creating this note is to provide my student with a fast enough, question leading tutorial. To me, the key is
>
> - the ability to read the question and summarize it in a short time
>   - numbers extracting
>   - figure patern
> - the ability to categorize the problem
> - careful computation

## Unit 1: Exploring One-Variable and Categorical Data

Start with the simplest data, one-variable only like:

- your 2024 daily wake up time
- Washington DC rainy days monthly from 1988 to 2024
- daily number of players in the game NA server, at 8PM CST
- average price of a dozen egg every day in Chicago downtown

Understand the data, their average, trend, and gain insights to guide our life.

Categorical data are also simple, starting with some kind of category, like

- brand of the car registered in New York
  - Toyota
  - Honda
  - Tesla
  - Ford
  - etc
- students go to your school by school bus or not
  - yes
  - no
- weather in Cleveland
  - sunny
  - cloudy
  - raining
  - snowy

This gives a direct view of the data distribution, proportion, and can be visualized using pie chart, colored dot plot.

<p style="text-align:center;">
  <img src="{{ site.baseurl }}/images/notes/math/AP_stat_pie_chart.svg">
  <img src="{{ site.baseurl }}/images/notes/math/AP_stat_cluster_scatter.svg">
</p>

### Types of Data & Study Design

To communicate with data collectors and researchers, we name the properties of the dataset.

#### Individuals and variables

- **Individuals**: The objects (people, animals, things) described by a set of data.
- **Variable**: Any characteristic of an individual.
- _Question_: In a study about the fuel efficiency of different car models, what are the individuals and what is a possible variable?
- _Answer_: The individuals are the car models. A possible variable is the fuel efficiency in miles per gallon (quantitative).

#### Categorical vs. quantitative variables

- **Categorical Variable**: Places an individual into one of several groups or categories. (e.g., eye color, car brand).
- **Quantitative Variable**: Takes numerical values for which it makes sense to find an average that has actual meaning. (e.g., height, gas mileage).
- _Question_: Is a phone number categorical or quantitative? What about the number of contacts you have? Why?
- _Answer_: A phone number is categorical because it's a label and averaging phone numbers makes no sense. The number of contacts is quantitative because you can measure it and find an average.

<details markdown="1" data-auto-footer>
<summary>Figure</summary>

<p style="text-align:center;">
    <img src="{{ site.baseurl }}/images/notes/math/AP_stat_phone_vs_contacts.svg">
</p>
</details>

#### Population vs. sample; parameters vs. statistics

- **Population**: The entire group of individuals we want information about.
- **Sample**: A subset of the population from which we actually collect data.
- **Parameter**: A number describing a characteristic of the **P**opulation. (e.g., the true average height of all adult women).
- **Statistic**: A number describing a characteristic of a sample. (e.g., the average height of the 100 women in your study).
- _Question_: A news report says a survey of 1,200 American adults found 68% have a pet. Is 68% a parameter or a statistic?
- _Answer_: It's a statistic because it describes a sample (the 1,200 surveyed adults).

#### Misleading data visuals and common graphing pitfalls

- **Violating the area principle**: The area representing a value in a graph should be proportional to the value. Watch out for pictographs that use height/width instead of area.
- **Truncated y-axis**: Not starting the vertical axis at 0 can make small differences look huge.
- **Improper scaling**: Uneven increments on an axis can distort the perception of change.
- _Question_: If you see a bar graph where the y-axis starts at 50 instead of 0, what effect does this have on how you perceive the differences between the bars?
- _Answer_: It exaggerates the differences, making them look much larger proportionally than they actually are. This is a common way to create a misleading visual.

<details markdown="1" data-auto-footer>
<summary>Figure</summary>

<p style="text-align:center;">
    <img src="{{ site.baseurl }}/images/notes/math/AP_stat_misleading_bar.svg">
</p>
</details>

### Categorical Data & Two-Way Tables

_Why we have this section_: Simply listing categories isn't enough. We need tools to organize, visualize, and compare categorical data, especially when we want to see if two categories are related. This section gives you the tools to turn lists of categorical data into meaningful insights.

#### Frequency and relative frequency tables

- **Frequency Table**: Shows the number of individuals having each data value.
- **Relative Frequency Table**: Shows the proportion or percent of individuals having each data value.
- _Question_: If a survey of 200 students shows 80 prefer basketball, what is the frequency and relative frequency of basketball preference?
- _Answer_: The frequency is 80. The relative frequency is 80/200 = 0.4 or 40%.

<details markdown="1" data-auto-footer>
<summary>Example Table</summary>

A class of 25 students was asked their favorite primary color. The results were: 10 chose Blue, 8 chose Red, and 7 chose Yellow.

**Frequency Table:**

| Color  | Frequency |
| :----- | :-------- |
| Blue   | 10        |
| Red    | 8         |
| Yellow | 7         |
| Total  | 25        |

**Relative Frequency Table:**

| Color  | Relative Frequency |
| :----- | :----------------- |
| Blue   | 10/25 = 0.40 (40%) |
| Red    | 8/25 = 0.32 (32%)  |
| Yellow | 7/25 = 0.28 (28%)  |
| Total  | 25/25 = 1.00 (100%)|

</details>

#### Bar charts and segmented bar charts

- **Bar Chart**: Displays the distribution of a categorical variable, showing the counts for each category. The bars do not touch.
- **Segmented Bar Chart**: Displays the conditional distribution of a categorical variable for each category of another variable. The bars are stacked to 100%.
- _Question_: When would you use a segmented bar chart instead of a regular bar chart? What kind of relationship does it help you see?
- _Answer_: Use a segmented bar chart when you want to compare the conditional distribution of a categorical variable across different categories of another variable. It helps you see if there's an association between the two variables.

<details markdown="1" data-auto-footer>
<summary>Figure</summary>

<p style="text-align:center;">
    <img src="{{ site.baseurl }}/images/notes/math/AP_stat_bar.svg">
</p>
</details>

#### Two-way frequency and relative frequency tables

- **Two-Way Table**: Describes two categorical variables, organizing counts in a table with rows and columns.
- _Question_: How would you set up a two-way table to show the relationship between grade level (Freshman, Sophomore, etc.) and favorite social media platform?
- _Answer_: The rows could be the grade levels and the columns could be the social media platforms. The cells would contain the number of students in each category combination.

<details markdown="1" data-auto-footer>
<summary>Example Table</summary>

Here is a possible two-way table based on a survey of 200 students:

| Grade Level | TikTok | Instagram | Twitter | Total |
| :--- | :--- | :--- | :--- | :--- |
| **Freshman** | 40 | 15 | 5 | 60 |
| **Sophomore**| 30 | 20 | 5 | 55 |
| **Junior** | 20 | 25 | 10 | 55 |
| **Senior** | 5 | 15 | 10 | 30 |
| **Total** | 95 | 75 | 30 | 200 |

</details>

#### Venn diagrams for unions and intersections

- **Union (A ∪ B)**: The set of all outcomes in A, or B, or both. (Think "OR").
- **Intersection (A ∩ B)**: The set of all outcomes in both A and B. (Think "AND").
- _Question_: In a Venn diagram of students who play soccer and students who play tennis, what does the overlapping area represent?
- _Answer_: The intersection (A ∩ B), which represents the students who play both soccer AND tennis.

<details markdown="1" data-auto-footer>
<summary>Example</summary>

<p style="text-align:center;">
    <img src="{{ site.baseurl }}/images/notes/math/AP_stat_venn.svg">
</p>

- set \\(A\\) be the group of students who play soccer, 43
- set \\(B\\) be the group of students who play tennis, 37
- set \\(A^\complement\\) be the group of students who **don't** play soccer, 57
- set \\(B^\complement\\) be the group of students who **don't** play tennis, 63
- set \\(A \cap B\\) be the group of students who play **both** soccer **and** tennis, 15
- set \\(A \cup B\\) be the group of students who play **either** soccer **or** tennis, 35
- set \\(A \backslash B\\) be the group of students who play soccer **only**, 28
- set \\(B \backslash A\\) be the group of students who play tennis **only**, 22
</details>

#### Marginal and conditional distributions

- **Marginal Distribution**: The distribution of values of one of the categorical variables in a two-way table of counts among all individuals described by the table. (Found in the "margins").
- **Conditional Distribution**: The distribution of values of one variable among individuals who have a specific value of another variable.
- _Question_: In a table of grade level vs. social media preference, how would you find the conditional distribution of platform preference _only for seniors_?
- _Answer_: You would look only at the row for "seniors" and calculate the percentage of those seniors who prefer each social media platform. The denominator would be the total number of seniors, not the total number of all students.

<details markdown="1" data-auto-footer>
<summary>Example Calculation</summary>

Using the example table from before:

| Grade Level | TikTok | Instagram | Twitter | Total |
| :--- | :--- | :--- | :--- | :--- |
| **Freshman** | 40 | 15 | 5 | 60 |
| **Sophomore**| 30 | 20 | 5 | 55 |
| **Junior** | 20 | 25 | 10 | 55 |
| **Senior** | 5 | 15 | 10 | 30 |
| **Total** | 95 | 75 | 30 | 200 |

**Marginal Distribution of Social Media Preference:**

This looks at the overall distribution of platform preference for all students. We use the totals from the bottom margin.
- **TikTok**: 95/200 = 47.5%
- **Instagram**: 75/200 = 37.5%
- **Twitter**: 30/200 = 15%

**Conditional Distribution of Platform Preference for Seniors:**

This looks at the distribution of platform preference *only for the 30 seniors*. We only use the numbers from the "Senior" row.
- **TikTok**: 5/30 ≈ 16.7%
- **Instagram**: 15/30 = 50%
- **Twitter**: 10/30 ≈ 33.3%

Notice how the conditional distribution for seniors is very different from the overall marginal distribution. This is evidence of an association.

</details>

#### Describing associations in two-way tables

- **Association**: We say there is an association between two variables if knowing the value of one variable helps predict the value of the other.
- _How to check_: Compare the conditional distributions. If they are noticeably different, there is an association.
- _Question_: If the percentage of seniors who prefer TikTok is very different from the percentage of freshmen who prefer TikTok, what can you conclude?
- _Answer_: You can conclude there is an association between grade level and social media preference.

_Any examples?_ A classic example is the Titanic dataset, which lists passengers and whether they survived, along with their ticket class (1st, 2nd, 3rd). Ticket class and survival are both categorical variables.

<details markdown="1" data-auto-footer>
<summary>Figure</summary>

<p style="text-align:center;">
    <img src="{{ site.baseurl }}/images/notes/math/AP_stat_titanic.svg">
</p>

|Name|Class|Survived|
|:-:|:-:|:-:|
|Bob 1| 1 |True|
|Bob 2| 2 |True|
|Bob 3| 2 |False|
|...| ... |...|
|Bob 2028|3|False|

</details>

_Real scenario to analyze_: A coffee shop owner surveys 100 customers, recording their preferred drink (Coffee, Tea, Other) and whether they are a regular or a new customer. The goal is to see if regulars have different preferences from new customers. How would you organize this data in a two-way table? What would you calculate to determine if there's an association between customer type and drink preference?

### Quantitative Data Displays

_Why we have this section_: Once you have a list of numbers (quantitative data), you need a way to see its shape and patterns. This section introduces the essential graphs for visualizing a single quantitative variable, helping you move from a raw dataset to a visual story.

#### Dotplots and frequency tables

- **Dotplot**: A simple graph where each data value is shown as a dot above its location on a number line. Best for smaller datasets.
- _Question_: When is a dotplot a better choice for displaying data than a histogram? (Hint: think about the size of the dataset and seeing individual values).
- _Answer_: A dotplot is better for small datasets because it shows every individual data point, whereas a histogram groups data into bins and loses individual values.

<details markdown="1" data-auto-footer>
<summary>Figure</summary>

<p style="text-align:center;">
    <img src="{{ site.baseurl }}/images/notes/math/AP_stat_dotPlot_freqTab.svg">
</p>
</details>

#### Stemplots (stem-and-leaf plots)

- **Stemplot**: A graphical display that separates each observation into a "stem" and a "leaf". It's like a histogram on its side, but it preserves the original data values. It's bin is filled by the actual data, each using same vertical space.
- _Question_: What is the main advantage of a stemplot over a histogram? What information do you keep?
- _Answer_: The main advantage is that a stemplot retains the original numerical values of the data, while a histogram does not.

<details markdown="1" data-auto-footer>
<summary>Example Stemplot</summary>

**Number of Birds at a Watering Hole Each Hour**

This plot displays the number of birds counted at a watering hole over several different hours. The stems represent the tens digit and the leaves represent the ones digit.

```
Stem | Leaf
-----+-----------
  1  | 2 3 4 6 7 8
  2  | 3 5 7 8 8
  3  | 2 5 9
  4  | 4 6
  5  | 9
  6  | 7
```
**Key:** `1 | 7 = 17 birds`

</details>

#### Histograms (choice of bins, scale, and interpretation)

- **Histogram**: A graph that displays the distribution of a quantitative variable by using bars. The height of each bar represents the frequency of values in that "bin" or interval.
  - Generally, the bars touch
  - Generally, the data at the left end of the bin interval are included, but the data at the right end of the bin interval are not included, unless it's the last bin
  - Generally, the Y label can only be "frenquency" or "percentage", and they don't have unit
- _Question_: How does changing the bin width of a histogram affect its appearance and the story it tells about the data?
- _Answer_: Too wide, and you lose detail, potentially hiding gaps or multiple peaks. Too narrow, and you get too much noise, making it hard to see the overall shape.

<details markdown="1" data-auto-footer>
<summary>An example histogram</summary>

<p style="text-align:center;">
    <img src="{{ site.baseurl }}/images/notes/math/AP_stat_hist.svg">
</p>
</details>

#### Clusters, gaps, peaks, and outliers

- **Cluster**: A distinct grouping of data points.
- **Gap**: An interval where there are no data points.
- **Peak (Mode)**: A prominent bar or "hump" in a histogram, like a montain peak. A distribution can be _unimodal_ (one peak), _bimodal_ (two peaks), etc.
- **Outlier**: An individual value that falls outside the overall pattern.
- _Question_: If you see a bimodal distribution in a histogram of student test scores, what might that suggest about the students or the test?
- _Answer_: It might suggest there are two distinct groups of students (e.g., those who studied and those who didn't) or that the test covered two very different topics, with students performing well on one but not the other.

_Any examples?_ A teacher could create a histogram of the scores from a recent exam to see the overall performance of the class. A coffee shop could use a dotplot to track the number of customers each hour on a particular day.

_Real scenario to analyze_: You are given the following data representing the commute times (in minutes) for 20 employees: 15, 22, 8, 45, 31, 25, 18, 12, 60, 28, 35, 20, 10, 55, 29, 33, 19, 24, 40, 21. Create a histogram to display this data. Describe the shape of the distribution and identify any potential outliers. What does this tell you about the typical commute for these employees?

<details markdown="1" data-auto-footer>
<summary>AP styled Answer</summary>

<p style="text-align:center;">
    <img src="{{ site.baseurl }}/images/notes/math/AP_stat_hist_comm.svg">
</p>

The distribution of commute times is right-skewed. Most commute times fall between 10 and 35 minutes, forming a clear cluster in the lower and middle bins. A few larger values (45, 55, and 60 minutes) create a long right tail.

There appear to be potential outliers at 55 and 60 minutes, since these values are noticeably separated from the main body of the data.

Overall, the typical employee has a commute of about 15–30 minutes, which corresponds to the center of the distribution. Because the distribution is skewed to the right, the median is a more appropriate measure of center than the mean when describing a “typical” commute.
</details>

### Shape, Center, and Spread

_Why we have this section_: Looking at a graph gives us a general idea of the data, but to be precise, we need numbers. This section introduces the key summary statistics used to describe the **S**hape, **O**utliers, **C**enter, and **S**pread (SOCS) of a quantitative distribution, giving you the vocabulary to analyze data like a statistician.

#### Describing distributions (SOCS / CSOCS)

- **Shape**: Is the distribution symmetric, skewed to the right (long right tail), or skewed to the left (long left tail)? Is it unimodal or bimodal?
- **Outliers**: Are there any individual values that fall far from the rest of the data?
- **Center**: Where is the middle of the data? (We'll use mean or median).
- **Spread**: How spread out is the data? (We'll use range, IQR, or standard deviation).
- _Question_: Why is it important to describe all four characteristics (SOCS) and not just the center?
- _Answer_: Describing only the center is incomplete. Two datasets can have the same center but vastly different shapes and spreads, telling very different stories. SOCS provides a complete picture.

#### Mean, median, and mode

- **Mean**: The average value. (Sum of values / number of values).
- **Median**: The middle value when the data is ordered.
- **Mode**: The most frequently occurring value.
- _Question_: For a perfectly symmetric distribution, what is the relationship between the mean and the median?
- _Answer_: They are equal.

#### When to use mean vs. median

- Use the **mean** for symmetric distributions with no strong outliers.
- Use the **median** for skewed distributions or when there are strong outliers. The median is "resistant" to outliers.
- _Question_: If you are analyzing house prices in a neighborhood where one billionaire lives, which measure of center would be more appropriate to describe the "typical" house price? Why?
- _Answer_: The median. The billionaire's house is an outlier that would pull the mean much higher, making it not representative of a typical house. The median is resistant to outliers.

#### Range, IQR, variance, and standard deviation

- **Range**: Maximum value - Minimum value. (Very sensitive to outliers).
- **Interquartile Range (IQR)**: Q3 - Q1. The range of the middle 50% of the data. (Resistant to outliers).
- **Standard Deviation**: The typical distance of the values in a distribution from the mean. (Sensitive to outliers).
- **Variance**: The standard deviation squared.
- _Question_: If two datasets have the same mean, does that mean they must have the same standard deviation? Why or why not?
- _Answer_: No. They could have very different spreads. For example, {10, 20, 30} and {19, 20, 21} both have a mean of 20, but the first set is much more spread out and has a larger standard deviation.

#### Visually assessing standard deviation

- Look at a histogram. A taller, narrower distribution has a smaller standard deviation. A shorter, wider distribution has a larger standard deviation.
- _Question_: Imagine two histograms of test scores for two different classes. Class A's scores are all clustered between 75 and 85. Class B's scores are spread out from 50 to 100. Which class has a larger standard deviation?
- _Answer_: Class B has a larger standard deviation because its scores are more spread out from the mean.

_Any examples?_ A real estate agent might calculate the median and IQR of house prices to give clients a sense of the market. A teacher might calculate the mean and standard deviation of test scores to understand class performance and consistency.

_Real scenario to analyze_: Consider the commute time data from the previous section: 15, 22, 8, 45, 31, 25, 18, 12, 60, 28, 35, 20, 10, 55, 29, 33, 19, 24, 40, 21. Calculate the mean and the median. Based on your histogram and these values, which is the better measure of center for this data? Now calculate the standard deviation and the IQR. What do these measures of spread tell you about the consistency of commute times?

### Transformations and Summary Plots

_Why we have this section_: We've learned to describe a single distribution. Now, we need tools to compare multiple distributions and to understand how our summary statistics change when we adjust our data (e.g., converting from Fahrenheit to Celsius). This section introduces boxplots for comparison and explores the effects of data transformations.

#### Effects of linear transformations (shift & scale) on center and spread

- **Adding/Subtracting a constant (a)**: Adds/subtracts 'a' to measures of center (mean, median). Does NOT change measures of spread (range, IQR, std dev).
- **Multiplying/Dividing by a constant (b)**: Multiplies/divides measures of center AND spread by 'b'.
- _Question_: If you convert a set of temperatures from Celsius to Fahrenheit using F = 1.8C + 32, how will the mean and standard deviation of the temperatures change?
- _Answer_: The mean will be transformed the same way: new*mean = 1.8 * old*mean + 32. The standard deviation is only affected by multiplication, not addition: new_sd = 1.8 * old_sd.

#### Bias of sample variance; why we divide by (n − 1) (conceptual)

- The sample variance (s²) is an "unbiased estimator" of the population variance (σ²) when we divide by n-1.
- Dividing by 'n' would, on average, underestimate the true population variance.
- _Question_: Why might a sample tend to have less variability than the population it came from? (Hint: think about extreme values).
- _Answer_: A random sample is less likely to capture the most extreme values (the highest and lowest) of a population, so its overall spread tends to be slightly smaller. Dividing by n-1 compensates for this.

#### Boxplots and 5-number summary

- **5-Number Summary**: Minimum, Q1 (25th percentile), Median (50th percentile), Q3 (75th percentile), Maximum.
- **Boxplot**: A graphical display of the 5-number summary. The box represents the IQR, a line inside marks the median, and "whiskers" extend to the min/max (or to the last non-outlier).
- _Question_: What features of a distribution can you see in a boxplot? What features can you _not_ see?
- _Answer_: You can see the 5-number summary (min, Q1, median, Q3, max), spread (IQR, range), and skewness. You cannot see gaps, clusters, or peaks (modality).

#### Determining outliers using IQR rule

- An observation is a suspected outlier if it falls more than 1.5 x IQR above the third quartile (Q3) or below the first quartile (Q1).
- **Upper Fence**: Q3 + 1.5(IQR)
- **Lower Fence**: Q1 - 1.5(IQR)
- _Question_: If a dataset has Q1=20 and Q3=50, what is the range of values that would not be considered outliers?
- _Answer_: IQR = 50 - 20 = 30. Lower Fence = 20 - 1.5(30) = -25. Upper Fence = 50 + 1.5(30) = 95. The range is [-25, 95].

#### Comparing distributions with graphs and statistics

- Use side-by-side boxplots or back-to-back stemplots to compare two or more distributions.
- Always compare using the SOCS framework. Use specific numerical values for center and spread.
- _Question_: When comparing two boxplots, how can you tell which distribution has a larger spread?
- _Answer_: You can compare the overall range (distance from the tip of one whisker to the other) or, more reliably, compare the Interquartile Range (the length of the box).

#### Data validity and checking for bad summaries

- Always ask where the data came from. Is it reliable?
- Be wary of summaries without context. An "average" could be a mean or a median, which can tell very different stories.
- Check for obvious errors or impossible values (e.g., a negative height).
- _Question_: A report states the "average" salary for a company is $150,000. What other information would you want to know before concluding that it's a high-paying company for most employees?
- _Answer_: You'd want to know the median salary and the standard deviation. The mean could be skewed high by a few executives with huge salaries, while most employees earn much less.

_Any examples?_ A scientist comparing the effectiveness of two different fertilizers would use side-by-side boxplots to visualize the crop yields from each. An accountant converting financial records from Euros to Dollars would need to understand how transformations affect the mean and standard deviation of the data.

_Real scenario to analyze_: A school is comparing the SAT scores of students who took a prep course with those who did not.

- **Prep Course Scores**: 1100, 1250, 1300, 1320, 1350, 1400, 1410, 1480, 1550
- **No Prep Course Scores**: 950, 1010, 1080, 1150, 1200, 1220, 1280, 1310, 1450
  Create side-by-side boxplots for these two datasets. Calculate the 5-number summary for each group. Write a few sentences comparing the two distributions using SOCS. Is there evidence that the prep course is associated with higher scores?

### Summary

| Concept                | Categorical Variable                                      | Quantitative Variable                                                             |
| :--------------------- | :-------------------------------------------------------- | :-------------------------------------------------------------------------------- |
| **What is it?**        | A variable that places individuals into groups or labels. | A variable that takes on numerical values where arithmetic operations make sense. |
| **Examples**           | Eye color, Car brand, Yes/No                              | Height, Weight, Test Score, Time                                                  |
| **Graphs**             | Bar Chart, Pie Chart, Segmented Bar Chart                 | Histogram, Stemplot, Dotplot, Boxplot                                             |
| **Measures of Center** | Mode (most frequent category)                             | Mean, Median                                                                      |
| **Measures of Spread** | (Not applicable in the same way)                          | Standard Deviation, IQR, Range                                                    |
| **Key Descriptors**    | Frequency, Relative Frequency, Proportions                | Shape (Skewed/Symmetric), Center, Spread, Outliers (SOCS)                         |

### Practice

#### Questions

**Section 1.1: Types of Data & Study Design**

1.  A wildlife biologist captures, tags, and releases 50 deer in a forest. She records their weight, gender, and the location in the forest where they were caught.
    - a. What are the individuals in this study?
    - b. Identify each variable as categorical or quantitative.
    - c. If the biologist wants to estimate the average weight of _all_ deer in this forest, is this average a parameter or a statistic?

**Section 1.2: Categorical Data**

2.  A survey asked 200 high school students whether they preferred playing sports or watching sports. The results are in the table below.

|                   | Play Sports | Watch Sports | Total |
| :---------------- | :---------- | :----------- | :---- |
| **Underclassmen** | 60          | 40           | 100   |
| **Upperclassmen** | 30          | 70           | 100   |
| **Total**         | 90          | 110          | 200   |

    *   a. What proportion of students surveyed prefer to watch sports?
    *   b. What is the conditional relative frequency of preferring to watch sports, given a student is an upperclassman?
    *   c. Is there an association between grade level and sports preference? Justify your answer by comparing conditional distributions.

**Section 1.3 & 1.4: Quantitative Data & Summary Statistics**

3.  Consider the following dataset representing the number of hours 10 students spent studying for an exam: `4, 7, 2, 8, 5, 15, 6, 5, 7, 9`.
    - a. Calculate the mean and the median study time.
    - b. The value `15` seems high. Which measure of center is more resistant to this potential outlier?
    - c. The standard deviation is approximately \\(3.5\\) hours. Interpret this value.

**Section 1.5: Transformations and Summary Plots**

4.  For the study time data in question 3:

    - a. Find the five-number summary.
    - b. Calculate the Interquartile Range (IQR).
    - c. Use the \\(1.5 \times IQR\\) rule to determine if the value `15` is an outlier.

5.  The instructor decides to give every student a bonus, adding 1 hour to their recorded study time.
    - a. What will be the new mean study time?
    - b. What will be the new standard deviation?

---

#### Answers

1.  **a.** The individuals are the 50 deer that were captured.
    **b.** Weight is quantitative. Gender is categorical. Location is categorical.
    **c.** A parameter, because it describes the entire population (all deer in the forest). The average weight of the 50 captured deer would be a statistic.

2.  **a.** \\( \frac{110}{200} = 0.55 \\) or 55% of students prefer to watch sports.
    **b.** \\( \frac{70}{100} = 0.70 \\) or 70% of upperclassmen prefer to watch sports.
    **c.** Yes, there is an association. The proportion of upperclassmen who prefer watching sports (70%) is much higher than the proportion of underclassmen who prefer watching sports (\\( \frac{40}{100} = 40\% \\)). Because these conditional distributions are different, the variables are associated.

3.  **a.** Mean: \\( \frac{4+7+2+8+5+15+6+5+7+9}{10} = \frac{68}{10} = 6.8 \\) hours.
    Median: First, order the data: `2, 4, 5, 5, 6, 7, 7, 8, 9, 15`. The median is the average of the 5th and 6th values: \\( \frac{6+7}{2} = 6.5 \\) hours.
    **b.** The median is more resistant. The mean (6.8) is pulled higher by the outlier (15), while the median (6.5) is less affected.
    **c.** A standard deviation of 3.5 hours means that the typical distance of an individual student's study time from the mean study time of 6.8 hours is about 3.5 hours.

4.  **a.** Ordered data: `2, 4, 5, 5, 6, 7, 7, 8, 9, 15`.
    Minimum = 2.
    Q1 (median of lower half `2, 4, 5, 5, 6`) = 5.
    Median = 6.5.
    Q3 (median of upper half `7, 7, 8, 9, 15`) = 8.
    Maximum = 15.
    Five-number summary is **{2, 5, 6.5, 8, 15}**.
    **b.** IQR = Q3 - Q1 = \\(8 - 5 = 3\\).
    **c.** Upper Fence = Q3 + \\(1.5 \times IQR\\) = \\(8 + 1.5 \times 3 = 8 + 4.5 = 12.5\\). Since 15 is greater than 12.5, it is considered an outlier.

5.  **a.** The new mean will be the old mean + 1: \\(6.8 + 1 = 7.8\\) hours. Adding a constant affects measures of center.
    **b.** The new standard deviation will be the same as the old one: \\(3.5\\) hours. Adding a constant does not affect measures of spread.

---

## Unit 2: Exploring Two-Variable Data & Linear Regression

### Scatterplots & Correlation

#### Explanatory vs. response variables

#### Constructing and reading scatterplots

#### Describing form, direction, strength, and outliers

#### Clusters and unusual features

#### Correlation coefficient r: calculation and interpretation

#### Limitations and cautions about correlation

### Least-Squares Regression Line (LSRL)

#### Least-squares criterion and line of best fit

#### Calculating equation of regression line (by hand and with technology)

#### Interpreting slope and y-intercept in context

#### Using regression for prediction and extrapolation cautions

### Residuals and Model Assessment

#### Residuals and residual plots

#### Standard deviation of residuals (s)

#### Coefficient of determination (r²) and its interpretation

#### Interpreting computer regression output

#### Identifying nonlinearity, outliers, and influential points

#### Transforming nonlinear relationships (e.g., log, power transforms)

### Technology & Exam Skills for Regression

#### Regression calculator steps (TI, Desmos, etc.)

#### Reading and using computer output on the AP exam

#### Common mistakes in interpreting regression

### Practice

---

## Unit 3: Collecting Data — Sampling & Experiments

### Planning a Study

#### Identifying population, sample, and sampling frame

#### Types of studies: observational vs. experimental

#### Generalizability and causation (scope of inference overview)

### Sampling Methods

#### Simple random sample (SRS)

#### Stratified, cluster, and systematic sampling

#### Multistage sampling designs

#### Random number tables and technology for random sampling

### Bias & Variability in Sampling

#### Selection bias, response bias, nonresponse

#### Undercoverage and overcoverage

#### Wording of questions

#### Bias vs. sampling variability; how sample size affects spread

### Experiments & Experimental Design

#### Components of an experiment: subjects, factors, treatments, response variables

#### Completely randomized design

#### Blocking and matched pairs designs

#### Placebo, control groups, blinding, and double-blind experiments

#### Ethics in experiments and quasi-experiments

### Scope of Inference

#### Random sampling vs. random assignment

#### When we can generalize to a population

#### When we can claim cause-and-effect

#### Limitations of real-world studies

### Practice

---

## Unit 4: Probability & Random Variables

### Foundations of Probability

#### Outcomes, events, and sample spaces

#### Probability rules and models

#### Law of Large Numbers and long-run frequency

#### Experimental vs. theoretical probability

### Compound Events: Addition Rule

#### Unions and intersections of events

#### Mutually exclusive (disjoint) events

#### Addition rule P(A ∪ B)

#### Two-way tables and Venn diagrams for probability

### Conditional Probability & Multiplication Rule

#### Conditional probability P(A | B)

#### Independence and tests for independence

#### General multiplication rule P(A ∩ B) = P(A | B)P(B)

#### Tree diagrams and “at least one” problems

#### Sampling without replacement

### Discrete Random Variables

#### Definition and probability distributions (tables and graphs)

#### Valid discrete distributions (probabilities sum to 1)

#### Expected value (mean) and interpretation

#### Variance and standard deviation of discrete random variables

### Transforming & Combining Random Variables

#### Effect of adding/subtracting constants

#### Effect of multiplying/dividing by constants

#### Mean and variance of sums and differences

#### Why independence matters for variance of sums

### Binomial & Geometric Distributions

#### Conditions for binomial and geometric settings

#### Binomial probabilities and binomial formulas

#### Expected value, variance, and standard deviation of binomial variables

#### Geometric probabilities (first success on trial k, at least, at most)

#### Using binompdf/binomcdf and geometpdf/geometcdf functions

### Practice

---

## Unit 5: Sampling Distributions

### Idea of a Sampling Distribution

#### Statistics as random variables

#### Sampling distributions vs. population distributions

#### Simulations to build intuition

### Sampling Distribution of a Sample Proportion

#### p vs. p-hat and their relationship

#### Center, spread, and shape

#### Conditions: Random, 10% condition, Large Counts

#### Probabilities involving sample proportions

### Sampling Distribution of a Sample Mean

#### Sampling distribution of x̄

#### Standard error of the mean

#### Central Limit Theorem for means

#### Conditions for using normal approximations

### Sampling Distributions for Differences

#### Sampling distribution of p̂₁ − p̂₂

#### Sampling distribution of x̄₁ − x̄₂

#### Means, variances, and standard errors for differences

### Connecting Sampling Distributions to Inference

#### z vs. t statistics

#### Role of standard error in confidence intervals and tests

#### Interpreting unusual sample results

### Practice

---

## Unit 6: Inference for Proportions

### Confidence Intervals for One Proportion

#### Point estimate and margin of error

#### Conditions for a one-proportion z-interval

#### Constructing and interpreting confidence intervals

#### Interpreting confidence levels correctly

#### Determining sample size for a desired margin of error

### Significance Tests for One Proportion

#### Null and alternative hypotheses

#### Conditions for a one-proportion z-test

#### Test statistic, P-value, and conclusions in context

#### One-sided vs. two-sided tests

### Errors, Power, and Multiple Tests

#### Type I and Type II errors

#### Significance level α and its consequences

#### Concept of power and factors that affect it

#### Problems of multiple testing and data snooping

### Two-Proportion Inference

#### Conditions for two-sample z-interval for p₁ − p₂

#### Interpreting confidence intervals for differences in proportions

#### Conditions and mechanics for two-sample z-test

#### Connecting two-sample tests and confidence intervals

#### Statistical vs. practical significance

### Practice

---

## Unit 7: Inference for Means

### The t-Distribution & One-Sample t Procedures

#### Why we use t instead of z for means

#### Shape and properties of the t-distribution

#### Conditions for inference about a mean

#### One-sample t-interval for a mean

#### One-sample t-test for a mean

#### Using technology and tables to get P-values

### Two-Sample t Procedures

#### Conditions for two-sample t-interval for μ₁ − μ₂

#### Constructing and interpreting two-sample t-intervals

#### Two-sample t-test for difference of means

#### Comparing conclusions from tests and intervals

### Matched Pairs & Special Designs

#### Matched pairs and paired data

#### Paired t-interval and paired t-test

#### Design issues: blocking vs. pairing

#### Quasi-experiments and their limitations

### Standard Error & Interpretation

#### Distinguishing standard deviation from standard error

#### Interpreting standard error in context

#### Common mistakes with t procedures on the exam

### Practice

---

## Unit 8: Inference for Categorical Data (Chi-Square)

### Chi-Square Basics

#### Chi-square statistic and its distribution

#### Degrees of freedom

#### Conditions for chi-square procedures

### Chi-Square Goodness-of-Fit Test

#### Setting up hypotheses and categories

#### Computing expected counts

#### Calculating chi-square statistic

#### Using technology/tables to find P-values

#### Interpreting results in context

### Chi-Square Tests for Two-Way Tables

#### Homogeneity vs. independence

#### Hypotheses, conditions, and test statistic

#### Interpreting chi-square output

#### Connecting chi-square tests to earlier association ideas

### Practice

---

## Unit 9: Inference for Regression

### Conditions for Regression Inference

#### Linearity, independence, normality, equal variance, randomness

#### Checking conditions with residual plots and other graphs

### t Procedures for the Slope

#### Hypotheses about slope (β)

#### t statistic for slope and associated P-value

#### Confidence interval for slope

#### Interpreting slope, intercept, and standard error in context

### Connecting Regression & Correlation

#### Relationship between slope, correlation, and standard deviations

#### Using regression output as part of a larger inference problem

#### Common AP exam tasks involving regression inference

### Practice

---

## Unit 10: AP Exam Preparation & Integrated Practice

### Exam Structure & Strategy

#### Structure of the AP Statistics exam

#### Calculator and formula sheet tips

#### Time management strategies for MCQ and FRQ

### Multiple-Choice Practice Themes

#### Interpreting graphs and numerical summaries

#### Choosing appropriate procedures and checking conditions

#### Common trap answers and how to avoid them

### Free-Response Practice Themes

#### Four-step process: State–Plan–Do–Conclude

#### Writing clear conclusions in context

#### Multi-part problems linking several units (e.g., design + inference)

### Cumulative Review

#### Mixed practice problems across all units

#### Interpreting computer output from real data

#### Reflecting on common errors and “red flag” phrases

### Practice
