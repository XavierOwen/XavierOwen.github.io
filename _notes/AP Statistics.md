---
content_id: ap-statistics-tutorial
language: en
original_language: en
reader_paths:
- research-teaching
- notes-writing
representative_paths:
- research-teaching
- notes-writing
title_zh: AP 统计学
title_en: AP Statistics
summary_zh: 面向学生、以问题引导的 AP 统计学概念速查、讲解与练习材料。
summary_en: A question-led AP Statistics tutorial and practice resource designed for
  fast conceptual review with students.
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

<details markdown="1" data-auto-footer>
<summary>Two example charts</summary>

<p style="text-align:center;">
    <img src="{{ site.baseurl }}/images/notes/math/AP_stat_pie_chart.svg">
</p>
<p style="text-align:center;">
    <img src="{{ site.baseurl }}/images/notes/math/AP_stat_cluster_scatter.svg">
</p>
</details>

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
<summary>Answer</summary>

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
- **Standard Deviation (\\(s_x\\))**: The typical distance of the values in a distribution from the mean. (Sensitive to outliers)
  - **Formula**: \\(s_x = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n-1}}\\).
  - **Visual**: Imagine the mean as the center. The standard deviation is the average "step size" you'd need to take to get from the mean to a random data point.
- **Variance (\\(s_x^2\\))**: The standard deviation squared. (More sensitive to outliers)
- _Question_: If two datasets have the same mean, does that mean they must have the same standard deviation? Why or why not?
- _Answer_: No. They could have very different spreads. For example, {10, 20, 30} and {19, 20, 21} both have a mean of 20, but the first set is much more spread out and has a larger standard deviation.

<details markdown="1" data-auto-footer>
<summary>Calculation Example</summary>

**Dataset**: `2, 4, 6, 8, 10` (\\(n=5\\))

1.  **Range**: \\(Max - Min = 10 - 2 = 8\\).
2.  **IQR**:
    - Median is 6.
    - Lower half: `2, 4`. \\(Q1 = \frac{2+4}{2} = 3\\).
    - Upper half: `8, 10`. \\(Q3 = \frac{8+10}{2} = 9\\).
    - \\(IQR = Q3 - Q1 = 9 - 3 = 6\\).
3.  **Standard Deviation (\\(s_x\\))**:
    - Mean: \\(\bar{x} = \frac{2+4+6+8+10}{5} = 6\\).
    - Deviations:
      - \\((2-6)^2 = 16\\)
      - \\((4-6)^2 = 4\\)
      - \\((6-6)^2 = 0\\)
      - \\((8-6)^2 = 4\\)
      - \\((10-6)^2 = 16\\)
    - Sum of squared deviations: \\(16+4+0+4+16 = 40\\).
    - Variance: \\(s_x^2 = \frac{40}{5-1} = 10\\).
    - Standard Deviation: \\(s_x = \sqrt{10} \approx 3.16\\).

</details>

#### Visually assessing standard deviation

- Look at a histogram. A taller, narrower distribution has a smaller standard deviation. A shorter, wider distribution has a larger standard deviation, given that the x axis spread are the same.
- _Question_: Imagine two histograms of test scores for two different classes. Class A's scores are all clustered between 75 and 85. Class B's scores are spread out from 50 to 100. Which class has a larger standard deviation?
- _Answer_: Class B has a larger standard deviation because its scores are more spread out from the mean.

_Any examples?_: A real estate agent might calculate the median and IQR of house prices to give clients a sense of the market. A teacher might calculate the mean and standard deviation of test scores to understand class performance and consistency.

_Real scenario to analyze_: Now Consider the commute time data from the [previous section](#clusters-gaps-peaks-and-outliers): 15, 22, 8, 45, 31, 25, 18, 12, 60, 28, 35, 20, 10, 55, 29, 33, 19, 24, 40, 21. Calculate the mean and the median. Based on your histogram and these values, which is the better measure of center for this data? Now calculate the standard deviation and the IQR. What do these measures of spread tell you about the consistency of commute times?

<details markdown="1" data-auto-footer>
<summary>Answer</summary>

<p style="text-align:center;">
    <img src="{{ site.baseurl }}/images/notes/math/AP_stat_hist_comm.svg">
</p>

**Data**: 15, 22, 8, 45, 31, 25, 18, 12, 60, 28, 35, 20, 10, 55, 29, 33, 19, 24, 40, 21 (\\(n = 20\\))

**Ordered Data**: 8, 10, 12, 15, 18, 19, 20, 21, 22, 24, 25, 28, 29, 31, 33, 35, 40, 45, 55, 60

**Calculations:**

1. **Mean**: \\(\bar{x} = \frac{8+10+12+15+18+19+20+21+22+24+25+28+29+31+33+35+40+45+55+60}{20} = \frac{550}{20} = 27.5\\) minutes

2. **Median**: The middle value between the 10th and 11th values: \\(\frac{24 + 25}{2} = 24.5\\) minutes

3. **Standard Deviation**:
   - Using the formula \\(s_x = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n-1}}\\)
   - \\(s_x \approx 14.08\\) minutes

4. **IQR**:
   - Lower half (first 10 values): 8, 10, 12, 15, 18, 19, 20, 21, 22, 24
   - \\(Q1 = \frac{18 + 19}{2} = 18.5\\) minutes
   - Upper half (last 10 values): 25, 28, 29, 31, 33, 35, 40, 45, 55, 60
   - \\(Q3 = \frac{33 + 35}{2} = 34\\) minutes
   - \\(IQR = Q3 - Q1 = 34 - 18.5 = 15.5\\) minutes

**Analysis:**

- **Center**: The mean (27.5) is higher than the median (24.5). This confirms the right skew we saw in the histogram (the tail of high commute times pulls the mean up). Because of the skew and potential outliers (55, 60), the **median** is the better measure of center as it represents the "typical" commute more accurately.
- **Spread**: The standard deviation (14.08) is quite large relative to the center, indicating significant variability in commute times. The IQR (15.5) tells us that the middle 50% of employees have commute times within a 15.5-minute range of each other. The large spread suggests that commute times are not very consistent; some employees live close, while others have very long drives.

Below is the commute time in a smaller standard devaition parallel universe

<p style="text-align:center;">
    <img src="{{ site.baseurl }}/images/notes/math/AP_stat_hist_comm_narrow.svg">
</p>

</details>

### Transformations and Summary Plots

_Why we have this section_: We've learned to describe a single distribution. Now, we need tools to compare multiple distributions and to understand how our summary statistics change when we adjust our data (e.g., converting from Fahrenheit to Celsius). This section introduces boxplots for comparison and explores the effects of data transformations.

#### Effects of linear transformations (shift & scale) on center and spread

- **Adding/Subtracting a constant (\\(a\\))**: Adds/subtracts \\(a\\) to measures of center (mean, median). Does NOT change measures of spread (range, IQR, std dev).
- **Multiplying/Dividing by a constant (\\(b\\))**: Multiplies/divides measures of center AND spread by \\(b\\).
- _Question_: If you convert a set of temperatures from Celsius to Fahrenheit using \\(F = 1.8C + 32\\), how will the mean and standard deviation of the temperatures change?
- _Answer_: The mean will be transformed the same way: \\(\text{new mean} = 1.8 \times \text{old mean} + 32\\). The standard deviation is only affected by multiplication, not addition: \\(\text{new SD} = 1.8 \times \text{old SD}\\).
- _Challenge Question_: If the variance of the original Celsius temperatures is \\(\sigma^2 = 25\\), what will be the variance of the Fahrenheit temperatures?
- _Answer_: The variance is the square of the standard deviation, so it transforms differently. When we multiply by a constant \\(b\\), the variance is multiplied by \\(b^2\\). Adding a constant does not change the variance. Therefore: \\(\text{new variance} = (1.8)^2 \times 25 = 3.24 \times 25 = 81\\).

#### Bias of sample variance; why we divide by (\\(n-1\\)) (conceptual)

- The sample variance (\\(s^2\\)) is an "unbiased estimator" of the population variance (\\(\sigma^2\\)) when we divide by \\(n-1\\).
- Dividing by '\\(n\\)' would, on average, underestimate the true population variance.
- _Question_: Why might a sample tend to have less variability than the population it came from? (Hint: think about extreme values).
- _Answer_: A random sample is less likely to capture the most extreme values (the highest and lowest) of a population, so its overall spread tends to be slightly smaller. Dividing by \\(n-1\\) compensates for this. Below is a preview.

<details markdown="1" data-auto-footer>
<summary>Mathematical Proof</summary>

First we define an **unbiased estimator**, which is one whose expectation is the true expectation. The sample mean is an unbiased estimator:

\\[E[\bar{X}] = \frac{1}{n} \sum_{i=1}^{n} E[X_i] = \frac{n}{n}\mu = \mu\\]

Then we compute the expectation of the sample variance,

\\[S^2 = \frac{1}{n-1} \sum_{i=1}^{n} (X_i^2) - n\bar{X}^2\\]

\\[E[S^2] = \frac{1}{n-1} \left( nE[(X_i^2)] - nE[\bar{X}^2] \right).\\]

Notice that \\(\bar{X}\\) is a random variable and not a constant, so the expectation \\(E[\bar{X}^2]\\) plays a role. **This is the reason behind the \\(n-1\\).**

\\[E[S^2] = \frac{1}{n-1} \left( n(\mu^2 + \sigma^2) - n(\mu^2 + Var(X)) \right).\\]

\\[Var(X) = Var\left(\frac{1}{n} \sum_{i=1}^{n} X_i\right) = \sum_{i=1}^{n} \frac{1}{n^2} Var(X_i) = \frac{\sigma^2}{n}\\]

\\[E[S^2] = \frac{1}{n-1} \left( n(\mu^2 + \sigma^2) - n(\mu^2 + \sigma^2/n) \right) = \frac{(n-1)\sigma^2}{n-1} = \sigma^2\\]

As you can see, if we had the denominator as \\(n\\) instead of \\(n-1\\), we would get a biased estimate for the variance! But with \\(n-1\\) the estimator \\(S^2\\) is an unbiased estimator.

</details>

#### Boxplots and 5-number summary

- **5-Number Summary**: Minimum, Q1 (25th percentile), Median (50th percentile), Q3 (75th percentile), Maximum.
- **Boxplot**: A graphical display of the 5-number summary. The box represents the **IQR**, a line inside marks the median, and "whiskers" extend to the min/max (or to the last non-outlier).
- _Question_: What features of a distribution can you see in a boxplot? What features can you _not_ see?
- _Answer_: You can see the 5-number summary (min, Q1, median, Q3, max), spread (IQR, range), and skewness. You cannot see gaps, clusters, or peaks (modality).

#### Determining outliers using IQR rule

- An observation is a suspected outlier if it falls more than 1.5 x IQR above the third quartile (Q3) or below the first quartile (Q1).
- **Upper Fence**: Q3 + 1.5(IQR)
- **Lower Fence**: Q1 - 1.5(IQR)
- _Question_: If a dataset has Q1=20 and Q3=50, what is the range of values that would not be considered outliers?
- _Answer_: IQR = 50 - 20 = 30. Lower Fence = 20 - 1.5(30) = -25. Upper Fence = 50 + 1.5(30) = 95. The range is [-25, 95].

<details markdown="1" data-auto-footer>
<summary>Commute time example</summary>

Using the commute time data from earlier: 15, 22, 8, 45, 31, 25, 18, 12, 60, 28, 35, 20, 10, 55, 29, 33, 19, 24, 40, 21

**Step 1: Order the data**
8, 10, 12, 15, 18, 19, 20, 21, 22, 24, 25, 28, 29, 31, 33, 35, 40, 45, 55, 60

**Step 2: Find the 5-number summary**
- **Minimum**: 8
- **Q1**: Median of lower half (first 10 values) = \\(\frac{18 + 19}{2} = 18.5\\)
- **Median**: \\(\frac{24 + 25}{2} = 24.5\\)
- **Q3**: Median of upper half (last 10 values) = \\(\frac{33 + 35}{2} = 34\\)
- **Maximum**: 60

**Step 3: Calculate IQR and check for outliers**
- \\(IQR = 34 - 18.5 = 15.5\\)
- Lower fence: \\(18.5 - 1.5(15.5) = 18.5 - 23.25 = -4.75\\)
- Upper fence: \\(34 + 1.5(15.5) = 34 + 23.25 = 57.25\\)
- **Outliers**: 60 (exceeds upper fence)

**Step 4: Draw the boxplot**
- Draw a number line spanning from slightly below the minimum to slightly above the maximum
- Draw a box from Q1 (18.5) to Q3 (34)
- Draw a vertical line inside the box at the median (24.5)
- Draw a whisker from Q1 to the minimum (8)
- Draw a whisker from Q3 to the last value within the upper fence (55)
- Mark the outlier (60) with a special symbol (dot or asterisk)

<p style="text-align:center;">
    <img src="{{ site.baseurl }}/images/notes/math/AP_stat_boxPlot_comm.svg">
</p>

The boxplot clearly shows the right skew (the right whisker is longer, and there's an outlier on the right), confirming our earlier analysis of the distribution.

</details>

#### Comparing distributions with graphs and statistics

- Use side-by-side boxplots or back-to-back stemplots to compare two or more distributions.
- Always compare using the SOCS framework. Use specific numerical values for center and spread.
  - **S**hape
  - **O**utliers
  - **C**enter
  - **S**pread
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

- **Prep Course Scores**: \\(1100, 1250, 1300, 1320, 1350, 1400, 1410, 1480, 1550\\)
- **No Prep Course Scores**: \\(950, 1010, 1080, 1150, 1200, 1220, 1280, 1310, 1450\\)

Create side-by-side boxplots for these two datasets. Calculate the 5-number summary for each group. Write a few sentences comparing the two distributions using SOCS. Is there evidence that the prep course is associated with higher scores?

<details markdown="1" data-auto-footer>
<summary>Answer</summary>

**5-Number Summaries:**

**Prep Course** (\\(n = 9\\)):
- Ordered data: \\(1100, 1250, 1300, 1320, 1350, 1400, 1410, 1480, 1550\\)
- Minimum: \\(1100\\)
- Q1: \\(\frac{1250 + 1300}{2} = 1275\\)
- Median: \\(1350\\)
- Q3: \\(\frac{1410 + 1480}{2} = 1445\\)
- Maximum: \\(1550\\)
- IQR: \\(1445 - 1275 = 170\\)
- LF: \\(Q_1 - 1.5 \times IQR = 1275 - 1.5(170) = 1020\\), no outlier
- UF: \\(Q_3 + 1.5 \times IQR = 1445 + 1.5(170) = 1700\\), no outlier
- Mean: \\(\bar{x} = 1351.11\\)

**No Prep Course** (\\(n = 9\\)):
- Ordered data: \\(950, 1010, 1080, 1150, 1200, 1220, 1280, 1310, 1450\\)
- Minimum: \\(950\\)
- Q1: \\(\frac{1010 + 1080}{2} = 1045\\)
- Median: \\(1200\\)
- Q3: \\(\frac{1280 + 1310}{2} = 1295\\)
- Maximum: \\(1450\\)
- IQR: \\(1295 - 1045 = 250\\)
- LF: \\(Q_1 - 1.5 \times IQR = 1045 - 1.5(250) = 670\\), no outlier
- UF: \\(Q_3 + 1.5 \times IQR = 1295 + 1.5(250) = 1670\\), no outlier
- Mean: \\(\bar{x} = 1183.33\\)

**Side-by-Side Boxplots:**

<p style="text-align:center;">
    <img src="{{ site.baseurl }}/images/notes/math/AP_stat_boxPlot_compare_SAT.svg">
</p>

**SOCS Comparison:**

- **Shape**: Both distributions appear roughly symmetric based on the boxplots. The prep course distribution shows the median close to the center of the box, while the no-prep distribution also appears fairly symmetric.

- **Outliers**: Neither distribution has outliers based on the 1.5×IQR rule. For the prep course: Upper fence = 1445 + 1.5(170) = 1700, so 1550 is not an outlier. For no-prep: Upper fence = 1295 + 1.5(250) = 1670, so 1450 is not an outlier.

- **Center**: The prep course group has a substantially higher center. The median score for prep students (1350) is 150 points higher than for non-prep students (1200). The means tell a similar story: 1351 vs. 1183, a difference of about 168 points.

- **Spread**: The no-prep course distribution has greater variability. The IQR for no-prep (250) is larger than for prep (170), suggesting more inconsistency in performance among students who didn't take the prep course. The range is also slightly larger for no-prep (500 vs. 450).

**Conclusion:**

Yes, there is strong evidence that the prep course is associated with higher SAT scores. Students who took the prep course scored consistently higher (median of 1350 vs. 1200) and showed less variability in their scores (smaller IQR). Every quartile of the prep course distribution is higher than the corresponding quartile of the no-prep distribution, indicating that prep students outperformed non-prep students across the board.

</details>

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

#### Section 1.1: Types of Data & Study Design

**1.1** A wildlife biologist captures, tags, and releases 50 deer in a forest. She records their weight, gender, and the location in the forest where they were caught.
   1. What are the individuals in this study?
   2. Identify each variable as categorical or quantitative.
   3. If the biologist wants to estimate the average weight of _all_ deer in this forest, is this average a parameter or a statistic?

**C1.1** (Challenge): A medical researcher is studying the effect of a new drug on blood pressure. She collects data from 100 patients. The variables recorded are: Patient ID (e.g., 1001, 1002), Dosage (0mg, 50mg, 100mg), Blood Pressure Reduction (mmHg), and Side Effects Severity (None, Mild, Severe).
1. Identify the individuals.
2. Classify "Patient ID", "Dosage", and "Side Effects Severity" as categorical or quantitative. Explain your reasoning for "Dosage".
3. If the researcher calculates the average blood pressure reduction for these 100 patients to be 12 mmHg, is this a parameter or a statistic?

#### Section 1.2: Categorical Data

**1.2** A survey asked 200 high school students whether they preferred playing sports or watching sports. The results are in the table below.

|                   | Play Sports | Watch Sports | Total |
| :---------------- | :---------- | :----------- | :---- |
| **Underclassmen** | 60          | 40           | 100   |
| **Upperclassmen** | 30          | 70           | 100   |
| **Total**         | 90          | 110          | 200   |

1. What proportion of students surveyed prefer to watch sports?
2. What is the conditional relative frequency of preferring to watch sports, given a student is an upperclassman?
3. Is there an association between grade level and sports preference? Justify your answer by comparing conditional distributions.

**C1.2** (Challenge): A university is analyzing admission data for two departments, Engineering and Arts. The data is given below:

- Engineering: 800 male applicants (600 admitted), 200 female applicants (180 admitted).
- Arts: 400 male applicants (100 admitted), 600 female applicants (200 admitted).

1. Construct a two-way table for the overall admission data (combining both departments) by Gender and Admission Status.
2. Calculate the overall admission rate for males and females. Who appears to be favored?
3. Calculate the admission rate for males and females *within* each department. Who appears to be favored in each department?
4. Explain the apparent contradiction (Simpson's Paradox).

#### Section 1.3 & 1.4: Quantitative Data & Summary Statistics

**1.3** Consider the following dataset representing the number of hours 10 students spent studying for an exam: `4, 7, 2, 8, 5, 15, 6, 5, 7, 9`.
1. Calculate the mean and the median study time.
2. The value `15` seems high. Which measure of center is more resistant to this potential outlier?
3. The standard deviation is approximately \\(3.5\\) hours. Interpret this value.

**C1.3** (Challenge): A class of 20 students has a mean test score of 80. A second class of 30 students has a mean test score of 70.
1. What is the mean score of all 50 students combined?
2. If the standard deviation of the first class is 5 and the second class is 10, can you calculate the standard deviation of the combined group just from this information? Why or why not?
3. Can you determine the exact median of the combined group? Why or why not?

#### Section 1.5: Transformations and Summary Plots

**1.4** For the study time data in question 1.3:
1. Find the five-number summary.
2. Calculate the Interquartile Range (IQR).
3. Use the \\(1.5 \times IQR\\) rule to determine if the value `15` is an outlier.

**1.5** The instructor decides to give every student a bonus, adding 1 hour to their recorded study time.
1. What will be the new mean study time?
2. What will be the new standard deviation?

**C1.4** (Challenge): A teacher scales the test scores of a class using the formula \\(Y = 2X + 10\\), where \\(X\\) is the original score. The original scores had a mean of 35 and a standard deviation of 5. The original distribution was strongly skewed to the right.
1. Find the mean and standard deviation of the scaled scores.
2. Describe the shape of the new distribution.
3. One student's original score was an outlier. Will it remain an outlier after the transformation? Prove it using the IQR rule (assume original \\(Q1=30, Q3=40\\)).

**C1.5** (Challenge): A college student tracked the amount of money (in dollars) they spent on coffee each week for 15 weeks: `12, 18, 15, 22, 8, 25, 20, 14, 30, 16, 19, 45, 17, 21, 13`.
1. Create a boxplot for this data. Show all work including the five-number summary and checking for outliers.
2. Based on your boxplot, describe the shape of the distribution.
3. The student decides to cut their coffee spending in half for the rest of the semester. How would the boxplot change? Specifically, what would happen to the median, IQR, and any outliers?

---

### Answers

**1.1**
1. The individuals are the 50 deer that were captured.
2. Weight is quantitative. Gender is categorical. Location is categorical.
3. A parameter, because it describes the entire population (all deer in the forest). The average weight of the 50 captured deer would be a statistic.

**C1.1**
1. The 100 patients.
2. Patient ID: Categorical (identifier). Dosage: Could be Quantitative (amount of drug) or Categorical (treatment group levels). Side Effects: Categorical (ordinal).
3. Statistic (describes the sample of 100).

**1.2**
1. \\( \frac{110}{200} = 0.55 \\) or 55% of students prefer to watch sports.
2. \\( \frac{70}{100} = 0.70 \\) or 70% of upperclassmen prefer to watch sports.
3. Yes, there is an association. The proportion of upperclassmen who prefer watching sports (70%) is much higher than the proportion of underclassmen who prefer watching sports (\\( \frac{40}{100} = 40\% \\)). Because these conditional distributions are different, the variables are associated.

**C1.2**
1. Two-way table for overall admission data:

   | | Admitted | Not Admitted | Total |
   |---|---|---|---|
   | Male | 600+100=700 | 200+300=500 | 1200 |
   | Female | 180+200=380 | 20+400=420 | 800 |

2. Male Rate: 700/1200 = 58.3%. Female Rate: 380/800 = 47.5%. Males appear favored overall.
3. Engineering: Male 600/800=75%, Female 180/200=90%. (Females favored). Arts: Male 100/400=25%, Female 200/600=33.3%. (Females favored).
4. Simpson's Paradox. Females are favored in both departments, but because more females applied to the harder-to-get-into department (Arts) and more males applied to the easier department (Engineering), the overall average makes it look like males are favored.

**1.3**
1. Mean: \\( \frac{4+7+2+8+5+15+6+5+7+9}{10} = \frac{68}{10} = 6.8 \\) hours.
Median: First, order the data: `2, 4, 5, 5, 6, 7, 7, 8, 9, 15`. The median is the average of the 5th and 6th values: \\( \frac{6+7}{2} = 6.5 \\) hours.
2. The median is more resistant. The mean (6.8) is pulled higher by the outlier (15), while the median (6.5) is less affected.
3. A standard deviation of 3.5 hours means that the typical distance of an individual student's study time from the mean study time of 6.8 hours is about 3.5 hours.

**C1.3**
1. Weighted Mean = (20*80 + 30*70) / 50 = (1600 + 2100) / 50 = 3700 / 50 = 74.
2. Yes, but it requires a complex formula involving the variances and the difference in means. It is NOT the average of the standard deviations.
3. No. Without the individual data points, we cannot determine the exact median, only that it lies somewhere between the two class medians (or potentially outside if distributions are extreme, but typically between).

**1.4**
1. Statistics
   - Ordered data: `2, 4, 5, 5, 6, 7, 7, 8, 9, 15`.
   - Minimum = 2.
   - Q1 (median of lower half `2, 4, 5, 5, 6`) = 5.
   - Median = 6.5.
   - Q3 (median of upper half `7, 7, 8, 9, 15`) = 8.
   - Maximum = 15.
   - Five-number summary is **{2, 5, 6.5, 8, 15}**.
2. IQR = Q3 - Q1 = \\(8 - 5 = 3\\).
3. Upper Fence = Q3 + \\(1.5 \times IQR\\) = \\(8 + 1.5 \times 3 = 8 + 4.5 = 12.5\\). Since 15 is greater than 12.5, it is considered an outlier.

**1.5**
1. The new mean will be the old mean + 1: \\(6.8 + 1 = 7.8\\) hours. Adding a constant affects measures of center.
2. The new standard deviation will be the same as the old one: \\(3.5\\) hours. Adding a constant does not affect measures of spread.

**C1.4**

1. New Mean = \\(2(35) + 10 = 80\\). New SD = \\(\|2\|(5) = 10\\).
2. The shape will remain strongly skewed to the right. Linear transformations (\\(Y = aX + b\\)) do not change the shape of the distribution.
3. Yes, it will remain an outlier.
Original IQR = \\(40 - 30 = 10\\). Upper Fence = \\(40 + 1.5(10) = 55\\). An outlier is any \\(X > 55\\).
New Q1 = \\(2(30) + 10 = 70\\). New Q3 = \\(2(40) + 10 = 90\\). New IQR = \\(90 - 70 = 20\\).
New Upper Fence = \\(90 + 1.5(20) = 120\\).
If \\(X > 55\\), then \\(2X > 110\\), and \\(2X + 10 > 120\\). So the transformed score will be greater than the new upper fence.

**C1.5**
1. **Step 1: Order the data**

   \\(8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 30, 45\\)

   **Step 2: Find the five-number summary**
   - Minimum: \\(8\\)
   - Q1: Median of lower 7 values \\((8, 12, 13, 14, 15, 16, 17)\\) = \\(14\\)
   - Median: 8th value = \\(18\\)
   - Q3: Median of upper 7 values \\((19, 20, 21, 22, 25, 30, 45)\\) = \\(22\\)
   - Maximum: \\(45\\)

   **Step 3: Check for outliers**
   - IQR = \\(22 - 14 = 8\\)
   - Lower Fence: \\(14 - 1.5(8) = 14 - 12 = 2\\)
   - Upper Fence: \\(22 + 1.5(8) = 22 + 12 = 34\\)
   - \\(45 > 34\\), so **45 is an outlier**

   **Step 4: Draw the boxplot**

   <p style="text-align:center;">
       <img src="{{ site.baseurl }}/images/notes/math/AP_stat_boxPlot_coffee.svg">
   </p>

2. The distribution is right-skewed, as evidenced by the longer right whisker and the outlier on the high end. The median is closer to Q1 than to Q3, which also indicates right skewness.

3. **Effect of cutting spending in half (dividing by 2)**:
   - New Median: \\(18 \div 2 = 9\\) dollars
   - New IQR: \\(8 \div 2 = 4\\) dollars (since both Q1 and Q3 are divided by 2)
   - New Q1: \\(14 \div 2 = 7\\) dollars, New Q3: \\(22 \div 2 = 11\\) dollars
   - The outlier (45) becomes \\(45 \div 2 = 22.5\\)
   - New Upper Fence: \\(11 + 1.5(4) = 11 + 6 = 17\\)
   - Since \\(22.5 > 17\\), the value remains an outlier
   - The boxplot would be compressed horizontally (all values halved), but the shape (right-skewed with an outlier) would remain the same

   <p style="text-align:center;">
       <img src="{{ site.baseurl }}/images/notes/math/AP_stat_boxPlot_coffee_comparison.svg">
   </p>

---

## Unit 2: Exploring Two-Variable Data & Linear Regression

Why we move here: In Unit 1 we learned to describe a single variable—its shape, center, and spread. Real questions often ask how one quantity changes with another. Does more study time lead to higher scores? Do older cars have lower resale value? Unit 2 introduces tools to explore relationships between two quantitative variables so we can describe patterns and make informed predictions.

Quick examples:
- Study time vs. exam score (more hours, typically higher score)
- Years since purchase vs. used car price (older, generally cheaper)
- Daily temperature vs. ice cream sales (warmer days, more sales)

Visual preview:

<details markdown="1" data-auto-footer>
<summary>Linear Example</summary>
<p style="text-align:center;">
    <img alt="Intro scatterplot showing positive association" src="{{ site.baseurl }}/images/notes/math/AP_stat_scatter_intro.svg">
    <br>
    <span style="font-size:0.9em;color:#666;">Points show (x = study time in hours, y = exam score). The upward trend suggests a positive association.</span>
</p>

</details>

<details markdown="1" data-auto-footer>
<summary>Nonlinear Example 1 — Exponential Growth (Time vs. Bacteria Count)</summary>

In a controlled lab environment, the number of bacteria can grow exponentially with time: \\(y = a\,b^x\\) where \\(b>1\\). The association is strong and positive, but the pattern curves upward—each extra hour adds more than the previous.

- Variables: \\(x\\) = hours since start; \\(y\\) = bacteria count
- Behavior: Multiplicative increases; straight line only on a log-scale of \\(y\\)
- Tip: Plot log of \\(y\\) against \\(x\\) to linearize and use regression on \\(.\log(y).\\)

<p style="text-align:center;">
    <img alt="Exponential growth scatter with curved trend" src="{{ site.baseurl }}/images/notes/math/AP_stat_nonlinear_exp.svg">
    <br>
    <span style="font-size:0.9em;color:#666;">Strong positive association with accelerating (curving up) pattern; linear on log(y).</span>
</p>

</details>

<details markdown="1" data-auto-footer>
<summary>Nonlinear Example 2 — Diminishing Returns (Logarithmic: Advertising Spend vs. Sales)</summary>

Sales often rise quickly with initial advertising and then taper off: \\(y = a + b\,\log(x)\\). The association is positive but flattens—each extra dollar buys less incremental lift.

- Variables: \\(x\\) = ad spend; \\(y\\) = weekly sales
- Behavior: Rapid early gains, then diminishing returns
- Tip: Consider transforming \\(x\\) with \\(\log(x)\\) and regressing \\(y\\) on \\(\log(x)\\)

<p style="text-align:center;">
    <img alt="Logarithmic relationship: fast rise then flatten" src="{{ site.baseurl }}/images/notes/math/AP_stat_nonlinear_log.svg">
    <br>
    <span style="font-size:0.9em;color:#666;">Positive association with diminishing returns; linear when x is log-transformed.</span>
</p>

</details>

<details markdown="1" data-auto-footer>
<summary>Nonlinear Example 3 — Quadratic/U-Shape (Stress vs. Performance)</summary>

Performance can follow an “inverted U” (Yerkes–Dodson law). Too little stress yields low performance; moderate stress maximizes it; too much stress lowers it again. A quadratic captures this: \\(y = \beta_0 + \beta_1 x + \beta_2 x^2\\) with \\(\beta_2<0\\).

- Variables: \\(x\\) = stress index; \\(y\\) = performance score
- Behavior: Peak at a moderate \\(x\\), declines on both sides
- Tip: Include polynomial terms (like \\(x^2\\)) or use spline regression to model curvature

<p style="text-align:center;">
    <img alt="Quadratic inverted U relationship" src="{{ site.baseurl }}/images/notes/math/AP_stat_nonlinear_quad.svg">
    <br>
    <span style="font-size:0.9em;color:#666;">Performance peaks at moderate stress; declines on both sides (inverted U).</span>
</p>

</details>

### Scatterplots & Correlation

Intuition first: A scatterplot is a picture of pairs \\((x, y)\\). Each dot is one individual, with its explanatory value on the x-axis and response value on the y-axis. What you’re looking for is the overall shape:

- Direction: Do points trend up (as x increases, y increases) or down? Upward suggests a positive association; downward suggests a negative association.
- Form: Is the cloud roughly a straight band (linear), a curve (nonlinear), or something with bends/clusters?
- Strength: How tightly are points packed around a clear pattern? Tighter bands mean stronger association; wide scatter means weaker association.
- Outliers: Any points far from the overall pattern can change conclusions and calculated summaries.

Correlation (r) in plain words: It’s a number from −1 to +1 that summarizes the strength and direction of a linear association.

- Sign tells direction: \\(r>0\\) means as x goes up, y tends to go up; \\(r<0\\) means as x goes up, y tends to go down.
- Magnitude tells strength: Values near 0 mean a weak linear tie; values near ±1 mean a strong linear tie.
- Correlation is about linearity: A curved relationship can have low \\(r\\) even if the variables are strongly associated.
- Scale-free: Changing units (minutes to hours) doesn’t change \\(r\\).

#### Explanatory vs. response variables

- **Explanatory (x)**: The variable you think may explain or predict changes in another.
- **Response (y)**: The outcome you measure.
- _Question_: In a study of engine size and fuel efficiency, which is explanatory and which is response?
- _Answer_: Engine size (liters) is explanatory; fuel efficiency (mpg) is the response.

Note: Choosing x and y anchors context. Correlation itself doesn’t depend on which variable is labeled x versus y, but regression does.

#### Constructing and reading scatterplots

- Put the explanatory variable on the **x-axis** and the response on the **y-axis**.
- Plot each pair \\((x_i, y_i)\\) as a point.
- Scan for overall pattern first; avoid getting distracted by one or two points.
- Label axes with units and include context in the title.
- _Question_: What’s the first thing to state when reading a scatterplot?
- _Answer_: The **direction, form, and strength** of the overall pattern, then note any **outliers**.

#### Describing form, direction, strength, and outliers

- **Direction**: Positive (upward) or negative (downward).
- **Form**: Linear band or curved pattern; any bends?
- **Strength**: How tightly points cluster around the pattern (tight = strong).
- **Outliers**: Points far from the pattern that may influence summaries.
- _Question_: Two scatterplots show the same upward trend. One has points tightly packed; the other has wide spread. Which has stronger association?
- _Answer_: The tightly packed one—stronger association.

#### Clusters and unusual features

- **Clusters**: Subgroups that suggest another variable is at play (e.g., grade level, region).
- **Gap**: A range of x where few/no points appear.
- **Influential points**: Points with extreme x-values that can strongly affect regression.
- _Question_: A plot of height vs. weight shows two clusters. What might explain that?
- _Answer_: Different groups (e.g., children vs. adults, or male vs. female) creating subpatterns.

#### Correlation coefficient r: calculation and interpretation

- **Definition**: \\(r = \frac{1}{n-1} \sum \left(\frac{x_i - \bar{x}}{s_x}\right)\left(\frac{y_i - \bar{y}}{s_y}\right)\\)
- **Range**: \\(-1 \le r \le 1\\)
- **Interpretation**:
    - Sign = direction (\(r>0\) positive, \(r<0\) negative)
    - Magnitude = linear strength (near 0 weak, near ±1 strong)
    - Unitless; unaffected by shifts or rescaling
- _Question_: If \\(r = -0.82\\), what does that say?
- _Answer_: Strong **negative** linear association: as x increases, y tends to decrease.

#### Limitations and cautions about correlation

- Correlation describes **linear** association only; curved patterns can have small \\(r\\) but strong relationships.
- Correlation is not causation—lurking variables or confounding may drive the pattern.
- Outliers can dramatically change \\(r\\); always **graph first**.
- Correlation assumes both variables are quantitative; it doesn’t apply to categorical data.
- _Question_: Why might advertising spend vs. sales have a modest \\(r\\) even if they’re clearly related?
- _Answer_: Diminishing returns make the relationship **curved** (log-like), so linear correlation understates it.

### Least-Squares Regression Line (LSRL)

#### Least-squares criterion and line of best fit

- **Regression line**: \\(\hat{y} = a + bx\\), where \\(\hat{y}\\) is the predicted value of y.
- **Least-squares criterion**: Choose the line that minimizes the sum of squared residuals \\(\sum (y_i - \hat{y}_i)^2\\).
- **Residual**: \\(e_i = y_i - \hat{y}_i\\) (observed minus predicted). Positive means the point is above the line; negative means below.
- _Question_: Why square the residuals instead of just summing them?
- _Answer_: Positive and negative residuals would cancel out. Squaring ensures all deviations count and penalizes large errors more.

#### Calculating equation of regression line (by hand and with technology)

- **Slope**: \\(b = r \cdot \frac{s_y}{s_x}\\), where \\(r\\) is correlation, \\(s_y\\) is SD of y, \\(s_x\\) is SD of x.
- **Intercept**: \\(a = \bar{y} - b\bar{x}\\) (the line always passes through \\((\bar{x}, \bar{y})\\)).
- By hand: compute means, standard deviations, correlation, then plug into formulas.
- With technology: input x and y data; use LinReg or similar command to get \\(a\\) and \\(b\\) directly.
- _Question_: If \\(r = 0.8\\), \\(\bar{x} = 5\\), \\(\bar{y} = 20\\), \\(s_x = 2\\), \\(s_y = 4\\), what is the regression line?
- _Answer_: \\(b = 0.8 \times \frac{4}{2} = 1.6\\). \\(a = 20 - 1.6(5) = 12\\). So \\(\hat{y} = 12 + 1.6x\\).

#### Interpreting slope and y-intercept in context

- **Slope interpretation**: For each one-unit increase in x, y is predicted to increase (or decrease) by \\(b\\) units, on average.
- **Intercept interpretation**: The predicted value of y when \\(x = 0\\). Only meaningful if \\(x = 0\\) is within the data range and makes sense in context.
- _Question_: In a regression of exam score on hours studied, \\(\hat{y} = 50 + 8x\\). Interpret the slope.
- _Answer_: For each additional hour studied, the exam score is predicted to increase by 8 points, on average.
- _Question_: Does the intercept 50 make sense?
- _Answer_: It means a student who studied 0 hours is predicted to score 50. This is only meaningful if 0 hours is a realistic scenario in the data.

#### Using regression for prediction and extrapolation cautions

- **Interpolation**: Predicting y for an x-value within the range of your data. Generally safe.
- **Extrapolation**: Predicting y for an x-value outside the data range. **Risky**—the pattern may not continue.
- _Question_: If your data has study times from 1 to 10 hours, is predicting a score for 5 hours interpolation or extrapolation? What about 15 hours?
- _Answer_: 5 hours is interpolation (within range). 15 hours is extrapolation (outside range)—the relationship might break down at extreme values.
- Always check: Does this x-value make sense? Is the linear pattern likely to hold?

### Residuals and Model Assessment

#### Residuals and residual plots

- **Residual**: \\(e_i = y_i - \hat{y}_i\\) (observed − predicted).
- **Residual plot**: Plot residuals (\\(e_i\\)) on the y-axis against x (or against \\(\hat{y}\\)) on the x-axis.
- **What to look for**:
  - Random scatter around 0 → linear model is appropriate
  - Pattern (curve, fan shape) → linear model is **not** appropriate
  - Outliers appear as points far from 0
- _Question_: A residual plot shows points forming a U-shape. What does this suggest?
- _Answer_: The relationship is **nonlinear** (curved). A linear model is not appropriate; consider a transformation.

#### Standard deviation of residuals (s)

- **Formula**: \\(s = \sqrt{\frac{\sum e_i^2}{n-2}}\\) (similar to standard deviation, but divide by \\(n-2\\) for regression).
- **Interpretation**: The typical distance of observed y-values from the regression line. Smaller \\(s\\) means better fit.
- _Question_: If \\(s = 5\\) in a regression of weight (kg) on height (cm), what does this mean?
- _Answer_: On average, actual weights are about 5 kg away from the weights predicted by the regression line.

#### Coefficient of determination (r²) and its interpretation

- **Definition**: \\(r^2 = (\text{correlation})^2\\). Ranges from 0 to 1.
- **Interpretation**: The proportion (or percentage) of variability in y that is explained by the linear relationship with x.
- _Question_: If \\(r = 0.9\\), what is \\(r^2\\)? Interpret it.
- _Answer_: \\(r^2 = 0.81\\) or 81%. About 81% of the variability in y is explained by the linear relationship with x. The remaining 19% is due to other factors.
- Note: \\(r^2\\) does **not** imply causation; it only measures association strength.

#### Interpreting computer regression output

Typical output includes:
- **Slope (b)** and **Intercept (a)**
- **\\(r\\)** (correlation) and **\\(r^2\\)** (coefficient of determination)
- **\\(s\\)** (standard deviation of residuals)
- **Standard error of slope (SE\\(_b\\))**: Measures uncertainty in the slope estimate (used for inference in Unit 9)

_Question_: Computer output shows \\(\hat{y} = 15 + 2.5x\\), \\(r^2 = 0.64\\), \\(s = 3.2\\). Interpret each.
- _Answer_:
  - Slope: For each 1-unit increase in x, y increases by 2.5 units on average.
  - \\(r^2\\): 64% of variability in y is explained by x.
  - \\(s\\): Typical prediction error is about 3.2 units.

#### Identifying nonlinearity, outliers, and influential points

- **Nonlinearity**: Detected by a curved pattern in the residual plot. Fix with transformations (log, square, etc.).
- **Outlier**: A point with a large residual (far from the regression line vertically).
- **Influential point**: A point with an extreme x-value that, if removed, would significantly change the regression line. Often has high leverage.
- _Question_: How can you tell if a point is influential?
- _Answer_: Calculate the regression line with and without the point. If the slope or intercept changes substantially, it's influential.

#### Transforming nonlinear relationships (e.g., log, power transforms)

- If residual plot shows a curve, try transforming one or both variables:
  - **Exponential**: \\(y = ab^x\\) → Linearize by taking \\(\log(y)\\). Regress \\(\log(y)\\) on \\(x\\).
  - **Power**: \\(y = ax^b\\) → Take \\(\log\\) of both sides: \\(\log(y) = \log(a) + b\log(x)\\). Regress \\(\log(y)\\) on \\(\log(x)\\).
  - **Logarithmic**: \\(y = a + b\log(x)\\) → Regress \\(y\\) on \\(\log(x)\\).
- After transformation, check the residual plot again. It should show random scatter if the transformation worked.
- _Question_: A scatterplot of bacteria count vs. time shows exponential growth. What transformation should you use?
- _Answer_: Take \\(\log\\) of bacteria count. Regress \\(\log(\text{count})\\) on time. The relationship should become linear.

### Technology & Exam Skills for Regression

#### Regression calculator steps (TI, Desmos, etc.)

**TI-83/84:**
1. Press `STAT` → `EDIT` → Enter x-values in L1, y-values in L2
2. Press `STAT` → `CALC` → `8:LinReg(a+bx)` → `L1, L2` → `ENTER`
3. Output shows \\(a\\) (intercept), \\(b\\) (slope), \\(r^2\\), \\(r\\)
4. To turn on diagnostics (to see \\(r\\) and \\(r^2\\)): `2ND` → `0` (CATALOG) → scroll to `DiagnosticOn` → `ENTER` → `ENTER`

**Desmos:**
1. Enter data in a table (x in one column, y in another)
2. Type `y_1 \sim mx_1 + b` to fit a linear regression
3. Desmos shows \\(m\\) (slope), \\(b\\) (intercept), \\(R^2\\), and residuals

_Tip_: Always create a scatterplot first to check for linearity before running regression.

#### Reading and using computer output on the AP exam

Typical regression output includes:
- **Predictor** column: Lists variables (Constant = intercept, x = slope)
- **Coef** column: Values of \\(a\\) and \\(b\\)
- **SE Coef**: Standard error (for inference; Unit 9)
- **\\(R^2\\)** or **R-Sq**: Coefficient of determination
- **\\(s\\)** or **Root MSE**: Standard deviation of residuals

_Question_: Output shows `Coef` for `Hours` = 3.5, Constant = 65, \\(R^2 = 0.72\\). Write the regression equation and interpret \\(R^2\\).
- _Answer_: \\(\hat{y} = 65 + 3.5x\\). About 72% of the variability in y is explained by hours.

#### Common mistakes in interpreting regression

- **Confusing correlation and causation**: High \\(r\\) or \\(r^2\\) does **not** prove x causes y. Could be confounding or reverse causation.
- **Extrapolating beyond data range**: Predictions outside the observed x-range are unreliable.
- **Ignoring residual plots**: A high \\(r^2\\) doesn't guarantee a linear model is appropriate. Always check the residual plot.
- **Misinterpreting the intercept**: Only meaningful if \\(x = 0\\) is in the data range and makes sense contextually.
- **Switching x and y**: Regressing x on y gives a different line than regressing y on x (unless \\(r = \pm 1\\)).
- _Question_: A study finds \\(r = 0.85\\) between ice cream sales and drowning incidents. Does ice cream cause drowning?
- _Answer_: No. There's likely a **lurking variable** (e.g., temperature or summer season) that drives both. Correlation ≠ causation.

### Practice

#### Section 2.1: Scatterplots & Correlation

**2.1** A researcher collects data on the age (years) and price (thousands of dollars) of 8 used cars:

| Age (x) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---------|---|---|---|---|---|---|---|---|
| Price (y)| 22| 20| 18| 15| 14| 12| 10| 9 |

1. Create a scatterplot. Describe the direction, form, and strength.
2. Calculate the correlation coefficient \\(r\\). (You may use technology.)
3. Interpret \\(r\\) in context.

**2.2** For the car data above:
1. Calculate the mean and standard deviation of age and price.
2. Use the formula \\(b = r \cdot \frac{s_y}{s_x}\\) to find the slope of the LSRL.
3. Find the intercept using \\(a = \bar{y} - b\bar{x}\\).
4. Write the equation of the regression line.

**2.3** Using your regression line from 2.2:
1. Predict the price of a 5-year-old car.
2. The actual price of a 5-year-old car is $14,000. Calculate the residual.
3. Would it be appropriate to predict the price of a 15-year-old car? Why or why not?

#### Section 2.2: Residuals & Model Assessment

**2.4** Computer output for a regression of test score (y) on hours studied (x) gives:
- \\(\hat{y} = 55 + 7.5x\\)
- \\(r^2 = 0.68\\)
- \\(s = 8.2\\)

1. Interpret the slope in context.
2. Interpret \\(r^2\\) in context.
3. Interpret \\(s\\) in context.
4. Predict the test score for a student who studied 6 hours.
5. If the actual score was 92, what is the residual?

**2.5** A residual plot shows a clear fan shape: residuals are small for low x-values and large for high x-values. What does this suggest about the model?

**C2.1** (Challenge): A biologist studies the relationship between temperature (°C) and cricket chirp rate (chirps/min). The scatterplot shows a strong linear pattern. The regression line is \\(\hat{y} = -10 + 4x\\), \\(r^2 = 0.92\\), \\(s = 5\\).

1. Interpret the slope and intercept in context.
2. Is the intercept meaningful? Explain.
3. At 20°C, the predicted chirp rate is 70 chirps/min. The observed rate is 75. Find the residual.
4. Can you conclude that higher temperature **causes** faster chirping? Why or why not?

---

### Answers

**2.1**
1. Direction: Negative (as age increases, price decreases). Form: Linear. Strength: Strong (points tightly clustered around a downward line).
2. Using technology: \\(r \approx -0.99\\)
3. There is a very strong negative linear association between car age and price.

**2.2**
1. \\(\bar{x} = 4.5\\), \\(s_x \approx 2.45\\); \\(\bar{y} = 15\\), \\(s_y \approx 4.90\\)
2. \\(b = -0.99 \times \frac{4.90}{2.45} \approx -1.98\\)
3. \\(a = 15 - (-1.98)(4.5) \approx 23.91\\)
4. \\(\hat{y} = 23.91 - 1.98x\\)

**2.3**
1. \\(\hat{y} = 23.91 - 1.98(5) = 14.01\\) thousand dollars ≈ $14,010
2. Residual = \\(14 - 14.01 = -0.01\\) (or -$10; very small)
3. No. 15 years is far outside the data range (1-8 years). Extrapolation is risky; the linear pattern may not hold.

**2.4**
1. For each additional hour studied, the test score is predicted to increase by 7.5 points, on average.
2. About 68% of the variability in test scores is explained by hours studied.
3. The typical distance of an observed score from the predicted score is about 8.2 points.
4. \\(\hat{y} = 55 + 7.5(6) = 100\\)
5. Residual = \\(92 - 100 = -8\\)

**2.5**
The model violates the constant variance assumption. Variability in y changes with x, suggesting a transformation or a different model might be needed.

**C2.1**
1. Slope: For each 1°C increase in temperature, chirp rate increases by 4 chirps/min, on average. Intercept: At 0°C, the predicted chirp rate is -10 chirps/min (not meaningful; crickets don't chirp below certain temperatures).
2. No, the intercept is not meaningful because 0°C is likely outside the range of temperatures where crickets chirp, and negative chirp rates are impossible.
3. Residual = \\(75 - 70 = 5\\) chirps/min
4. No. Correlation does not imply causation. While the association is strong, there could be confounding variables, or the relationship could be driven by another factor (e.g., both temperature and chirping could be influenced by time of day or season).

---

## Unit 3: Collecting Data — Sampling & Experiments

### Planning a Study

#### Identifying population, sample, and sampling frame

- **Population**: The entire group of individuals we want information about.
- **Sample**: A subset of the population from which we actually collect data.
- **Sampling frame**: The list of individuals from which the sample is drawn. Ideally equals the population.
- _Question_: A researcher wants to know the average income of all adults in Chicago. She surveys 500 people at a downtown shopping mall. What is the population? The sample? Is the sampling frame appropriate?
- _Answer_: Population = all adults in Chicago. Sample = 500 surveyed. Sampling frame = people at that mall. **Problem**: Mall shoppers may not represent all Chicago adults (undercoverage).

#### Types of studies: observational vs. experimental

- **Observational study**: Observe individuals and measure variables without imposing treatments. Cannot establish causation.
- **Experimental study**: Deliberately impose treatments on individuals to observe responses. Can establish causation if well-designed.
- _Question_: Researchers compare lung cancer rates between smokers and non-smokers. Is this observational or experimental?
- _Answer_: Observational—researchers didn't assign smoking; they just observed existing groups.
- _Question_: Researchers randomly assign patients to receive either a new drug or a placebo, then measure recovery rates. Observational or experimental?
- _Answer_: Experimental—researchers imposed the treatment (drug vs. placebo).

#### Generalizability and causation (scope of inference overview)

- **Generalizability**: Can results apply to the whole population? Requires **random sampling**.
- **Causation**: Can we say x **causes** y? Requires **random assignment** (experiment).
- _Question_: A study uses random sampling but no random assignment. Can we generalize? Can we claim causation?
- _Answer_: Yes to generalization (random sample). No to causation (no random assignment).

### Sampling Methods

#### Simple random sample (SRS)

- **Definition**: Every possible sample of size \\(n\\) has an equal chance of being selected.
- **How**: Use random number generator or table; assign each individual a number, then randomly select \\(n\\) numbers.
- _Question_: Why is an SRS better than asking for volunteers?
- _Answer_: Volunteers introduce **bias**—they may differ systematically from non-volunteers. SRS gives everyone an equal chance.

#### Stratified, cluster, and systematic sampling

- **Stratified sampling**: Divide population into homogeneous groups (strata), then take an SRS from each stratum. Reduces variability.
  - Example: Divide students by grade level, then randomly sample within each grade.
- **Cluster sampling**: Divide population into groups (clusters), randomly select some clusters, survey all individuals in selected clusters. Cost-effective but increases variability.
  - Example: Randomly select 10 schools, survey all students in those schools.
- **Systematic sampling**: Select every \\(k\\)th individual from a list. Simple but can introduce bias if there's a pattern.
  - Example: Survey every 10th person entering a store.
- _Question_: You want to estimate average GPA for a university with 10,000 students across 4 years. Which method is most efficient?
- _Answer_: **Stratified** by year—GPA varies by year, so stratifying reduces variability and improves precision.

#### Multistage sampling designs

- Combine methods. Example: First, randomly select states (cluster). Then, randomly select counties within states. Finally, take SRS of households within counties.
- Used for large-scale surveys (e.g., national polls).

#### Random number tables and technology for random sampling

- **Random number table**: Use rows of random digits; assign individuals numbers, then read digits to select.
- **Technology**: Use calculator's random integer function (`randInt(1, N, n)`) or online tools.
- _Tip_: Always document your randomization process to ensure reproducibility.

### Bias & Variability in Sampling

#### Selection bias, response bias, nonresponse

- **Selection bias (undercoverage)**: Some groups are systematically excluded from the sampling frame.
  - Example: Phone survey excludes people without phones.
- **Response bias**: Respondents give inaccurate answers due to question wording, interviewer presence, or social desirability.
  - Example: "Do you support the terrible policy X?" (leading question).
- **Nonresponse bias**: People who don't respond differ systematically from those who do.
  - Example: Only very satisfied or very dissatisfied customers return surveys.
- _Question_: A survey asks, "How often do you engage in illegal activity?" Response rate is low. What bias is present?
- _Answer_: **Nonresponse bias** (people doing illegal things won't respond) and **response bias** (those who do respond may lie).

#### Undercoverage and overcoverage

- **Undercoverage**: Sampling frame misses part of the population.
- **Overcoverage**: Sampling frame includes individuals not in the target population.
- _Question_: A survey of "all US adults" uses a landline phone directory. What's the problem?
- _Answer_: **Undercoverage**—many adults only have cell phones and are excluded.

#### Wording of questions

- **Leading questions**: Push respondents toward a particular answer.
- **Confusing wording**: Vague or technical language causes misunderstanding.
- _Tip_: Use neutral, clear language. Pilot test questions.

#### Bias vs. sampling variability; how sample size affects spread

- **Bias**: Systematic error; doesn't decrease with larger samples. Fix by improving design.
- **Sampling variability**: Random differences between samples. Decreases with larger \\(n\\).
- _Question_: Increasing sample size from 100 to 400 will reduce ______ but not ______.
- _Answer_: Reduces **variability**, but not **bias**.

### Experiments & Experimental Design

#### Components of an experiment: subjects, factors, treatments, response variables

- **Subjects (units)**: Individuals in the experiment.
- **Factors**: Explanatory variables manipulated by the experimenter.
- **Treatments**: Specific combinations of factor levels.
- **Response variable**: Outcome measured.
- _Question_: Testing two fertilizer types at three dosage levels. How many treatments?
- _Answer_: \\(2 \times 3 = 6\\) treatments (each combination of type and dosage).

#### Completely randomized design

- Randomly assign all subjects to treatments.
- _Strength_: Simple; controls for confounding.
- _Question_: Why randomize instead of letting subjects choose?
- _Answer_: Random assignment balances lurking variables across groups, enabling causal conclusions.

#### Blocking and matched pairs designs

- **Blocking**: Group subjects by a variable that affects the response (e.g., age, gender), then randomly assign treatments within each block. Reduces variability.
- **Matched pairs**: Each subject receives both treatments (in random order) or subjects are paired and one gets each treatment.
- _Question_: Testing a new teaching method. Students' prior math ability varies widely. Should you block?
- _Answer_: Yes—block by prior ability to reduce variability and increase precision.

#### Placebo, control groups, blinding, and double-blind experiments

- **Placebo**: Fake treatment to account for placebo effect.
- **Control group**: Receives no treatment or standard treatment for comparison.
- **Blinding**: Subjects don't know which treatment they receive (reduces bias).
- **Double-blind**: Neither subjects nor evaluators know (eliminates bias from both sides).
- _Question_: Why use a double-blind design?
- _Answer_: Prevents subjects' expectations **and** evaluators' biases from affecting results.

#### Ethics in experiments and quasi-experiments

- **Informed consent**: Subjects must know risks and agree to participate.
- **Institutional review**: Experiments must be approved by ethics boards.
- **Quasi-experiment**: Treatments not randomly assigned (e.g., comparing existing groups). Can't establish causation.

### Scope of Inference

#### Random sampling vs. random assignment

| Technique | Purpose | Enables |
|-----------|---------|---------|
| Random sampling | Selecting from population | Generalization |
| Random assignment | Assigning to treatments | Causation |

#### When we can generalize to a population

- Need **random sampling** from the population.
- Without it, results apply only to the sample.

#### When we can claim cause-and-effect

- Need **random assignment** of treatments.
- Without it, association ≠ causation.

#### Limitations of real-world studies

- **Convenience samples**: Common but not generalizable.
- **Confounding**: Lurking variables make causal claims impossible in observational studies.
- **Ethical constraints**: Can't always randomly assign harmful treatments.

_Question_: A study randomly samples US adults and randomly assigns them to exercise programs. What can we conclude?
_Answer_: Results generalize to all US adults (random sample) **and** we can claim exercise **causes** changes in the response (random assignment).

_Question_: A study randomly samples US adults and randomly assigns them to exercise programs. What can we conclude?
_Answer_: Results generalize to all US adults (random sample) **and** we can claim exercise **causes** changes in the response (random assignment).

### Practice

#### Section 3.1: Sampling & Bias

**3.1** A school wants to estimate average student satisfaction. They survey the first 50 students who arrive at school one morning.
1. What sampling method is this?
2. Identify a potential source of bias.
3. Suggest a better method.

**3.2** A political poll uses a random-digit-dialing system that calls only landline phones.
1. What type of bias is present?
2. How would this affect the results?

**3.3** A university has 60% in-state students and 40% out-of-state students. You want to survey 200 students about tuition.
1. Describe how to select a stratified random sample.
2. Why is stratification better than an SRS here?

#### Section 3.2: Experiments

**3.4** Researchers want to test if a new fertilizer increases tomato yield. They have 40 plots available.
1. Design a completely randomized experiment.
2. Explain why randomization is important.

**3.5** Testing two pain relievers (A and B). 60 subjects available, varying widely in pain sensitivity.
1. Should you use blocking? If so, how?
2. Explain the benefit.

**C3.1** (Challenge): A study finds that people who drink coffee have lower rates of Alzheimer's disease. The study followed 10,000 randomly selected adults for 20 years.
1. Is this observational or experimental?
2. Can we generalize to all adults? Why?
3. Can we conclude coffee **prevents** Alzheimer's? Why or why not?
4. Name a possible confounding variable.

---

### Answers

**3.1**
1. Convenience sampling
2. Students arriving early may differ from those arriving late (more motivated, better organized, etc.)
3. Use stratified random sampling by grade level, or systematic sampling throughout the day

**3.2**
1. Undercoverage (selection bias)
2. Younger adults with only cell phones are excluded; results may skew toward older demographics

**3.3**
1. Randomly select \\(0.6 \times 200 = 120\\) in-state students and \\(0.4 \times 200 = 80\\) out-of-state students
2. Tuition opinions likely differ by residency status; stratification ensures proper representation and reduces variability

**3.4**
1. Randomly assign 20 plots to new fertilizer, 20 to control (old fertilizer or none). Measure yield. Compare.
2. Randomization balances lurking variables (soil quality, sunlight) across groups, allowing causal conclusions

**3.5**
1. Yes. Block by pain sensitivity level (low/medium/high). Within each block, randomly assign half to A, half to B
2. Reduces variability by ensuring similar pain levels are compared; increases power to detect treatment differences

**C3.1**
1. Observational—researchers didn't assign coffee consumption
2. Yes—random sampling allows generalization to all adults
3. No—this is observational, so confounding is possible. Association ≠ causation
4. Possible confounders: exercise habits, diet, education level, genetics (people who drink coffee may also exercise more, eat healthier, etc.)

---
### Foundations of Probability

#### Outcomes, events, and sample spaces

- **Outcome**: A single result of a random process.
- **Sample space (S)**: The set of all possible outcomes.
- **Event**: A collection of outcomes (a subset of S).
- _Question_: Rolling a six-sided die. What is the sample space? What is the event "rolling an even number"?
- _Answer_: S = {1, 2, 3, 4, 5, 6}. Event = {2, 4, 6}.

#### Probability rules and models

- **Probability**: A number between 0 and 1 describing the likelihood of an event.
- **Rules**:
  1. \\(0 \le P(A) \le 1\\)
  2. \\(P(S) = 1\\) (something must happen)
  3. \\(P(A^c) = 1 - P(A)\\) (complement rule)
- _Question_: If \\(P(\text{rain}) = 0.3\\), what is \\(P(\text{no rain})\\)?
- _Answer_: \\(P(\text{no rain}) = 1 - 0.3 = 0.7\\)

#### Law of Large Numbers and long-run frequency

- As the number of trials increases, the proportion of times an event occurs approaches its true probability.
- _Question_: You flip a fair coin 10 times and get 7 heads. Does this violate probability?
- _Answer_: No. Short-run results vary. Over thousands of flips, heads will approach 50%.

#### Experimental vs. theoretical probability

- **Theoretical**: Based on equally likely outcomes (e.g., \\(P(\text{heads}) = 0.5\\)).
- **Experimental**: Based on observed data (e.g., 520 heads in 1000 flips → \\(\hat{p} = 0.52\\)).

### Compound Events: Addition Rule

#### Unions and intersections of events

- **Union (\\(A \cup B\\))**: A or B (or both).
- **Intersection (\\(A \cap B\\))**: A and B (both occur).
- _Question_: Drawing a card. A = red, B = face card. Describe \\(A \cup B\\) and \\(A \cap B\\).
- _Answer_: \\(A \cup B\\) = red or face card. \\(A \cap B\\) = red face cards (J♥, Q♥, K♥, J♦, Q♦, K♦).

#### Mutually exclusive (disjoint) events

- **Definition**: Events that cannot both occur (\\(P(A \cap B) = 0\\)).
- _Question_: Rolling a die. Are "rolling a 2" and "rolling an even number" mutually exclusive?
- _Answer_: No—rolling a 2 means you also rolled an even number.

#### Addition rule P(A ∪ B)

- **General**: \\(P(A \cup B) = P(A) + P(B) - P(A \cap B)\\)
- **Disjoint events**: \\(P(A \cup B) = P(A) + P(B)\\)
- _Question_: \\(P(A) = 0.4\\), \\(P(B) = 0.3\\), \\(P(A \cap B) = 0.1\\). Find \\(P(A \cup B)\\).
- _Answer_: \\(P(A \cup B) = 0.4 + 0.3 - 0.1 = 0.6\\)

#### Two-way tables and Venn diagrams for probability

- Use counts or proportions to calculate probabilities.
- _Tip_: Always check if totals sum to 1 (for probabilities) or n (for counts).

<details markdown="1" data-auto-footer>
<summary>Example: Two-way table probability</summary>

|  | Likes Math | Doesn't Like Math | Total |
|---|---|---|---|
| Grade 9 | 30 | 20 | 50 |
| Grade 10 | 25 | 25 | 50 |
| **Total** | 55 | 45 | 100 |

Find \\(P(\text{likes math})\\), \\(P(\text{Grade 9})\\), \\(P(\text{Grade 9 and likes math})\\), \\(P(\text{likes math} \mid \text{Grade 9})\\).

**Solution**:
- \\(P(\text{likes math}) = \frac{55}{100} = 0.55\\)
- \\(P(\text{Grade 9}) = \frac{50}{100} = 0.50\\)
- \\(P(\text{Grade 9 and likes math}) = \frac{30}{100} = 0.30\\)
- \\(P(\text{likes math} \mid \text{Grade 9}) = \frac{30}{50} = 0.60\\)

</details>

### Conditional Probability & Multiplication Rule

#### Conditional probability P(A | B)

- **Definition**: \\(P(A \mid B) = \frac{P(A \cap B)}{P(B)}\\) (probability of A given B has occurred).
- _Question_: \\(P(\text{disease and positive test}) = 0.02\\), \\(P(\text{positive test}) = 0.05\\). Find \\(P(\text{disease} \mid \text{positive test})\\).
- _Answer_: \\(P(\text{disease} \mid \text{positive}) = \frac{0.02}{0.05} = 0.4\\)

#### Independence and tests for independence

- **Independent**: \\(P(A \mid B) = P(A)\\) or \\(P(A \cap B) = P(A) \cdot P(B)\\).
- _Question_: \\(P(A) = 0.5\\), \\(P(B) = 0.6\\), \\(P(A \cap B) = 0.3\\). Are A and B independent?
- _Answer_: Check: \\(P(A) \cdot P(B) = 0.5 \times 0.6 = 0.3\\). Yes, independent.

#### General multiplication rule P(A ∩ B) = P(A | B)P(B)

- **General**: \\(P(A \cap B) = P(A \mid B) \cdot P(B)\\)
- **Independent**: \\(P(A \cap B) = P(A) \cdot P(B)\\)
- _Question_: \\(P(\text{rain}) = 0.2\\), \\(P(\text{traffic} \mid \text{rain}) = 0.8\\). Find \\(P(\text{rain and traffic})\\).
- _Answer_: \\(P(\text{rain} \cap \text{traffic}) = 0.8 \times 0.2 = 0.16\\)

#### Tree diagrams and "at least one" problems

- **Tree diagrams**: Visual tool for sequential events; multiply along branches.
- **"At least one"**: Use complement: \\(P(\text{at least one}) = 1 - P(\text{none})\\).
- _Question_: Two independent coin flips, \\(P(\text{H}) = 0.5\\). Find \\(P(\text{at least one heads})\\).
- _Answer_: \\(P(\text{at least one H}) = 1 - P(\text{TT}) = 1 - 0.25 = 0.75\\)

#### Sampling without replacement

- Probabilities change after each selection.
- _Question_: A bag has 3 red, 2 blue marbles. Draw 2 without replacement. Find \\(P(\text{both red})\\).
- _Answer_: \\(P(\text{1st red}) = \frac{3}{5}\\). \\(P(\text{2nd red} \mid \text{1st red}) = \frac{2}{4}\\). \\(P(\text{both red}) = \frac{3}{5} \times \frac{2}{4} = \frac{3}{10}\\).

### Discrete Random Variables

#### Definition and probability distributions (tables and graphs)

- **Random variable (RV)**: A variable whose value is determined by chance.
- **Discrete RV**: Takes on countable values.
- **Probability distribution**: Lists all possible values and their probabilities.
- _Question_: Let X = number of heads in 2 coin flips. List the probability distribution.
- _Answer_:

| X | 0 | 1 | 2 |
|---|---|---|---|
| P(X) | 0.25 | 0.50 | 0.25 |

#### Valid discrete distributions (probabilities sum to 1)

- \\(\sum P(X = x) = 1\\)
- All \\(P(X = x) \ge 0\\)

#### Expected value (mean) and interpretation

- **Formula**: \\(\mu_X = E(X) = \sum x \cdot P(X = x)\\)
- **Interpretation**: Long-run average value.
- _Question_: For X above, find \\(E(X)\\).
- _Answer_: \\(E(X) = 0(0.25) + 1(0.50) + 2(0.25) = 1\\)

#### Variance and standard deviation of discrete random variables

- **Variance**: \\(\sigma_X^2 = \sum (x - \mu_X)^2 \cdot P(X = x)\\)
- **Standard deviation**: \\(\sigma_X = \sqrt{\sigma_X^2}\\)
- _Question_: For X above (\\(E(X) = 1\\)), find \\(\sigma_X^2\\).
- _Answer_: \\(\sigma_X^2 = (0-1)^2(0.25) + (1-1)^2(0.50) + (2-1)^2(0.25) = 0.25 + 0 + 0.25 = 0.5\\)

### Transforming & Combining Random Variables

#### Effect of adding/subtracting constants

- \\(E(X + a) = E(X) + a\\)
- \\(\text{Var}(X + a) = \text{Var}(X)\\) (adding a constant doesn't change spread)

#### Effect of multiplying/dividing by constants

- \\(E(bX) = b \cdot E(X)\\)
- \\(\text{Var}(bX) = b^2 \cdot \text{Var}(X)\\)

#### Mean and variance of sums and differences

- \\(E(X + Y) = E(X) + E(Y)\\) (always)
- \\(E(X - Y) = E(X) - E(Y)\\) (always)
- \\(\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)\\) **(if independent)**
- \\(\text{Var}(X - Y) = \text{Var}(X) + \text{Var}(Y)\\) **(if independent)** (note: still add!)

#### Why independence matters for variance of sums

- Without independence, covariance terms appear. AP Stats assumes independence unless stated otherwise.

<details markdown="1" data-auto-footer>
<summary>Example: Combining random variables</summary>

Let X = profit from selling lemonade on a random day (mean $20, SD $5).
Let Y = profit from selling cookies (mean $15, SD $3), independent of X.
Find the mean and standard deviation of total profit T = X + Y.

**Solution**:
- \\(E(T) = E(X) + E(Y) = 20 + 15 = 35\\) dollars
- \\(\text{Var}(T) = \text{Var}(X) + \text{Var}(Y) = 5^2 + 3^2 = 25 + 9 = 34\\)
- \\(\text{SD}(T) = \sqrt{34} \approx 5.83\\) dollars

</details>

### Binomial & Geometric Distributions

#### Conditions for binomial and geometric settings

**Binomial (BINS)**:
- **B**inary outcomes (success/failure)
- **I**ndependent trials
- **N**umber of trials is fixed
- **S**ame probability of success on each trial

**Geometric**: Same as binomial, but count trials **until** first success (no fixed n).

#### Binomial probabilities and binomial formulas

- **Notation**: \\(X \sim \text{Binomial}(n, p)\\)
- **Formula**: \\(P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}\\)
- **Mean**: \\(\mu_X = np\\)
- **Variance**: \\(\sigma_X^2 = np(1-p)\\), \\(\sigma_X = \sqrt{np(1-p)}\\)
- _Question_: Flip a fair coin 5 times. Find \\(P(X = 3)\\) heads.
- _Answer_: \\(P(X=3) = \binom{5}{3}(0.5)^3(0.5)^2 = 10 \times 0.03125 = 0.3125\\)

#### Expected value, variance, and standard deviation of binomial variables

- Use formulas above; no need to calculate from scratch.

#### Geometric probabilities (first success on trial k, at least, at most)

- **Notation**: \\(X \sim \text{Geometric}(p)\\)
- **Formula**: \\(P(X = k) = (1-p)^{k-1} p\\) (first success on trial k)
- **Mean**: \\(\mu_X = \frac{1}{p}\\)
- _Question_: Free throw success rate is 0.7. Find \\(P(\text{first make on 3rd try})\\).
- _Answer_: \\(P(X=3) = (0.3)^2(0.7) = 0.063\\)

#### Using binompdf/binomcdf and geometpdf/geometcdf functions

- **binompdf(n, p, k)**: \\(P(X = k)\\)
- **binomcdf(n, p, k)**: \\(P(X \le k)\\)
- **geometpdf(p, k)**: \\(P(X = k)\\)
- **geometcdf(p, k)**: \\(P(X \le k)\\)
- _Tip_: "pdf" = exact, "cdf" = cumulative (at most).

<details markdown="1" data-auto-footer>
<summary>Example: Binomial distribution</summary>

A multiple-choice test has 10 questions, each with 4 choices. A student guesses on all 10. Let X = number correct.

1. Verify binomial conditions
2. Find \\(P(X = 2)\\)
3. Find \\(P(X \ge 3)\\)
4. Find mean and standard deviation

**Solution**:
1. Binary (right/wrong), independent, n=10 fixed, p=0.25 same each trial ✓
2. \\(P(X=2) = \text{binompdf}(10, 0.25, 2) \approx 0.2816\\)
3. \\(P(X \ge 3) = 1 - P(X \le 2) = 1 - \text{binomcdf}(10, 0.25, 2) \approx 1 - 0.5256 = 0.4744\\)
4. \\(\mu_X = 10(0.25) = 2.5\\), \\(\sigma_X = \sqrt{10(0.25)(0.75)} \approx 1.37\\)

</details>

### Practice

**4.1** A bag contains 5 red and 3 blue marbles. You draw one marble, note the color, and replace it. Then draw again.
1. What is \\(P(\text{both red})\\)?
2. What is \\(P(\text{at least one blue})\\)?

**4.2** A survey finds 40% of voters support Candidate A, 35% support B, and 10% support both.
1. Find \\(P(A \cup B)\\)
2. Are A and B mutually exclusive? Independent?

**4.3** A diagnostic test correctly identifies disease 95% of the time (sensitivity). The disease prevalence is 2%. If someone tests positive, what is the probability they have the disease? (Assume test specificity is 90%.)

**4.4** Let X = number of sixes in 4 rolls of a fair die.
1. Verify binomial conditions
2. Find \\(P(X = 2)\\)
3. Find \\(E(X)\\) and \\(\sigma_X\\)

**4.5** A basketball player makes 80% of free throws. Let Y = number of attempts until first miss (geometric).
1. Find \\(P(Y = 5)\\)
2. Find \\(E(Y)\\)

**C4.1** In a lottery, you pick 3 different numbers from 1–10. The lottery draws 3 numbers without replacement. You win if all 3 match (order doesn't matter).
1. How many ways can 3 numbers be chosen from 10?
2. What is your probability of winning?
3. If 1000 people play, what is the expected number of winners?

---

### Answers

**4.1**
1. \\(P(\text{both red}) = \frac{5}{8} \times \frac{5}{8} = \frac{25}{64} \approx 0.391\\) (with replacement, independent)
2. \\(P(\text{at least one blue}) = 1 - P(\text{both red}) = 1 - \frac{25}{64} = \frac{39}{64} \approx 0.609\\)

**4.2**
1. \\(P(A \cup B) = 0.40 + 0.35 - 0.10 = 0.65\\)
2. Not mutually exclusive (10% overlap). Check independence: \\(P(A) \cdot P(B) = 0.40 \times 0.35 = 0.14 \ne 0.10\\), so **not independent**.

**4.3**
Let D = disease, + = positive test.
- \\(P(D) = 0.02\\), \\(P(D^c) = 0.98\\)
- \\(P(+ \mid D) = 0.95\\), \\(P(+ \mid D^c) = 1 - 0.90 = 0.10\\)
- \\(P(D \cap +) = 0.95 \times 0.02 = 0.019\\)
- \\(P(D^c \cap +) = 0.10 \times 0.98 = 0.098\\)
- \\(P(+) = 0.019 + 0.098 = 0.117\\)
- \\(P(D \mid +) = \frac{0.019}{0.117} \approx 0.162\\) (only 16.2%!)

**4.4**
1. Binary (six/not), independent, n=4 fixed, p=1/6 same ✓
2. \\(P(X=2) = \binom{4}{2} \left(\frac{1}{6}\right)^2 \left(\frac{5}{6}\right)^2 \approx 0.1157\\)
3. \\(E(X) = 4 \times \frac{1}{6} \approx 0.667\\), \\(\sigma_X = \sqrt{4 \times \frac{1}{6} \times \frac{5}{6}} \approx 0.745\\)

**4.5**
1. \\(P(Y=5) = (0.8)^4(0.2) = 0.08192\\)
2. \\(E(Y) = \frac{1}{0.2} = 5\\) attempts

**C4.1**
1. \\(\binom{10}{3} = \frac{10!}{3!7!} = \frac{10 \times 9 \times 8}{3 \times 2 \times 1} = 120\\) ways
2. \\(P(\text{win}) = \frac{1}{120} \approx 0.0083\\)
3. Expected winners = \\(1000 \times \frac{1}{120} \approx 8.33\\) people

---

## Unit 5: Sampling Distributions

### Idea of a Sampling Distribution

#### Statistics as random variables

- A **statistic** is a numerical summary of sample data (e.g., \\(\bar{x}\\), \\(\hat{p}\\), \\(s\\)).
- Statistics vary from sample to sample—they are **random variables**.
- _Question_: You take a sample of 100 students and find mean height = 65 inches. If you took another sample of 100, would you get exactly 65 inches?
- _Answer_: No. Different samples give different statistics. This variability is random.

#### Sampling distributions vs. population distributions

- **Population distribution**: Distribution of values in the entire population.
- **Sampling distribution**: Distribution of a statistic across all possible samples of size n.
- _Key idea_: The sampling distribution describes how statistics behave, not individual data values.

#### Simulations to build intuition

- Simulate taking many samples, calculate the statistic for each, and plot the results.
- Helps visualize center, spread, and shape of the sampling distribution.
- _Tip_: In real life, we take ONE sample. But understanding the sampling distribution helps us make inferences.

### Sampling Distribution of a Sample Proportion

#### p vs. p-hat and their relationship

- **p**: True population proportion (parameter, unknown).
- **\\(\hat{p}\\)**: Sample proportion (statistic, calculated from data).
- \\(\hat{p}\\) varies around p; we use \\(\hat{p}\\) to estimate p.

#### Center, spread, and shape

- **Mean**: \\(\mu_{\hat{p}} = p\\) (unbiased estimator)
- **Standard deviation (standard error)**: \\(\sigma_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}}\\)
- **Shape**: Approximately normal if conditions are met
- _Question_: Population has p = 0.6, sample size n = 100. Find mean and standard error of \\(\hat{p}\\).
- _Answer_: \\(\mu_{\hat{p}} = 0.6\\), \\(\sigma_{\hat{p}} = \sqrt{\frac{0.6(0.4)}{100}} = \sqrt{0.0024} \approx 0.049\\)

#### Conditions: Random, 10% condition, Large Counts

1. **Random**: Random sample or random assignment
2. **10% condition**: \\(n \le 0.10N\\) (sample < 10% of population) when sampling without replacement
3. **Large Counts**: \\(np \ge 10\\) and \\(n(1-p) \ge 10\\)

#### Probabilities involving sample proportions

- Use normal approximation: \\(Z = \frac{\hat{p} - p}{\sqrt{\frac{p(1-p)}{n}}}\\)
- _Question_: p = 0.6, n = 100. Find \\(P(\hat{p} > 0.65)\\).
- _Answer_: \\(Z = \frac{0.65 - 0.6}{0.049} \approx 1.02\\). \\(P(Z > 1.02) \approx 0.154\\)

<details markdown="1" data-auto-footer>
<summary>Example: Sample proportion probability</summary>

In a large population, 30% support a policy. A random sample of 200 is taken. What is the probability that between 25% and 35% of the sample supports the policy?

**Solution**:
- Check conditions: Random (given), 10% (200 likely < 10% of population), Large Counts: \\(200(0.3) = 60 \ge 10\\), \\(200(0.7) = 140 \ge 10\\) ✓
- \\(\mu_{\hat{p}} = 0.3\\), \\(\sigma_{\hat{p}} = \sqrt{\frac{0.3(0.7)}{200}} \approx 0.0324\\)
- \\(Z_1 = \frac{0.25 - 0.3}{0.0324} \approx -1.54\\), \\(Z_2 = \frac{0.35 - 0.3}{0.0324} \approx 1.54\\)
- \\(P(-1.54 < Z < 1.54) \approx 0.876\\)

</details>

### Sampling Distribution of a Sample Mean

#### Sampling distribution of x̄

- **Mean**: \\(\mu_{\bar{x}} = \mu\\) (sample mean is unbiased)
- **Standard deviation (standard error)**: \\(\sigma_{\bar{x}} = \frac{\sigma}{\sqrt{n}}\\)
- As n increases, \\(\sigma_{\bar{x}}\\) decreases (more precise estimates)

#### Standard error of the mean

- **Standard error (SE)**: \\(\frac{\sigma}{\sqrt{n}}\\) or estimate with \\(\frac{s}{\sqrt{n}}\\)
- Measures variability of \\(\bar{x}\\) across samples
- _Question_: Population SD = 20, sample size = 25. Find standard error.
- _Answer_: \\(SE = \frac{20}{\sqrt{25}} = \frac{20}{5} = 4\\)

#### Central Limit Theorem for means

- **CLT**: For large enough n, the sampling distribution of \\(\bar{x}\\) is approximately normal, regardless of population shape.
- **Rule of thumb**: n ≥ 30 usually works, or smaller n if population is approximately normal.
- _Key insight_: Even if population is skewed, \\(\bar{x}\\) becomes approximately normal as n grows.

#### Conditions for using normal approximations

1. **Random**: Random sample or random assignment
2. **10% condition**: \\(n \le 0.10N\\) when sampling without replacement
3. **Normal/Large Sample**: Population is normal OR \\(n \ge 30\\)

<details markdown="1" data-auto-footer>
<summary>Example: Central Limit Theorem</summary>

A population has mean μ = 50, SD σ = 12, and is strongly right-skewed. A random sample of n = 40 is taken.

1. Describe the sampling distribution of \\(\bar{x}\\)
2. Find \\(P(\bar{x} > 53)\\)

**Solution**:
1. \\(\mu_{\bar{x}} = 50\\), \\(\sigma_{\bar{x}} = \frac{12}{\sqrt{40}} \approx 1.897\\). Shape: approximately normal (n = 40 ≥ 30, CLT applies despite skewed population)
2. \\(Z = \frac{53 - 50}{1.897} \approx 1.58\\). \\(P(Z > 1.58) \approx 0.057\\)

</details>

### Sampling Distributions for Differences

#### Sampling distribution of p̂₁ − p̂₂

- **Mean**: \\(\mu_{\hat{p}_1 - \hat{p}_2} = p_1 - p_2\\)
- **Standard error**: \\(\sigma_{\hat{p}_1 - \hat{p}_2} = \sqrt{\frac{p_1(1-p_1)}{n_1} + \frac{p_2(1-p_2)}{n_2}}\\)
- **Shape**: Approximately normal if Large Counts met for BOTH samples
- _Conditions_: Random samples, 10% condition for both, \\(n_1p_1, n_1(1-p_1), n_2p_2, n_2(1-p_2)\\) all ≥ 10

#### Sampling distribution of x̄₁ − x̄₂

- **Mean**: \\(\mu_{\bar{x}_1 - \bar{x}_2} = \mu_1 - \mu_2\\)
- **Standard error**: \\(\sigma_{\bar{x}_1 - \bar{x}_2} = \sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}\\)
- **Shape**: Approximately normal if Normal/Large Sample condition met for both samples

#### Means, variances, and standard errors for differences

- For independent samples, **variances add** (not subtract!)
- \\(\text{Var}(X_1 - X_2) = \text{Var}(X_1) + \text{Var}(X_2)\\)
- \\(SE_{diff} = \sqrt{SE_1^2 + SE_2^2}\\)

<details markdown="1" data-auto-footer>
<summary>Example: Difference of proportions</summary>

In City A, 45% support a measure (n₁ = 150). In City B, 38% support it (n₂ = 200). Assume these are the true population proportions. What is the probability that \\(\hat{p}_1 - \hat{p}_2 > 0.10\\)?

**Solution**:
- \\(\mu_{\hat{p}_1 - \hat{p}_2} = 0.45 - 0.38 = 0.07\\)
- \\(\sigma_{\hat{p}_1 - \hat{p}_2} = \sqrt{\frac{0.45(0.55)}{150} + \frac{0.38(0.62)}{200}} = \sqrt{0.00165 + 0.001178} \approx 0.0532\\)
- \\(Z = \frac{0.10 - 0.07}{0.0532} \approx 0.56\\)
- \\(P(Z > 0.56) \approx 0.288\\)

</details>

### Connecting Sampling Distributions to Inference

#### z vs. t statistics

- **z-statistic**: Use when σ (population SD) is known or working with proportions
- **t-statistic**: Use when σ is unknown and estimate with s (sample SD) for means
- t-distributions have heavier tails; approach normal as df increases

#### Role of standard error in confidence intervals and tests

- **Confidence interval**: \\(\text{statistic} \pm (\text{critical value}) \times SE\\)
- **Test statistic**: \\(\frac{\text{statistic} - \text{parameter}}{SE}\\)
- SE measures how much the statistic varies; smaller SE → more precise inference

#### Interpreting unusual sample results

- If observed statistic is many SEs away from expected value, it's unusual
- This suggests either: rare random chance, or the assumed parameter is wrong
- Basis for hypothesis testing (P-values measure how unusual)

### Practice

**5.1** A population has p = 0.25. A random sample of n = 400 is taken.
1. Check conditions for normal approximation of \\(\hat{p}\\)
2. Find mean and standard error of \\(\hat{p}\\)
3. Find \\(P(\hat{p} < 0.22)\\)

**5.2** A population is normally distributed with μ = 100, σ = 15. Random samples of size n = 9 are taken.
1. Describe the sampling distribution of \\(\bar{x}\\)
2. Find \\(P(\bar{x} > 105)\\)

**5.3** Explain why the Central Limit Theorem is important when the population distribution is unknown or non-normal.

**5.4** Two independent samples: n₁ = 50, \\(\bar{x}_1 = 22\\), s₁ = 4; n₂ = 60, \\(\bar{x}_2 = 18\\), s₂ = 5.
1. Find the mean of \\(\bar{x}_1 - \bar{x}_2\\) if μ₁ - μ₂ = 4
2. Estimate the standard error of \\(\bar{x}_1 - \bar{x}_2\\)

**5.5** Why do we add variances when computing the standard error for a difference of two statistics?

**C5.1** A factory produces bolts with mean length 5 cm and SD 0.1 cm. Bolts are packaged in boxes of 25. What is the probability that the mean length of bolts in a randomly selected box exceeds 5.03 cm? State all conditions and assumptions.

---

### Answers

**5.1**
1. Random (given), 10% (sample < 10% of population assumed), Large Counts: \\(400(0.25) = 100 \ge 10\\), \\(400(0.75) = 300 \ge 10\\) ✓
2. \\(\mu_{\hat{p}} = 0.25\\), \\(\sigma_{\hat{p}} = \sqrt{\frac{0.25(0.75)}{400}} = \sqrt{0.00046875} \approx 0.0217\\)
3. \\(Z = \frac{0.22 - 0.25}{0.0217} \approx -1.38\\). \\(P(Z < -1.38) \approx 0.084\\)

**5.2**
1. \\(\mu_{\bar{x}} = 100\\), \\(\sigma_{\bar{x}} = \frac{15}{\sqrt{9}} = 5\\). Shape: normal (population is normal)
2. \\(Z = \frac{105 - 100}{5} = 1\\). \\(P(Z > 1) \approx 0.159\\)

**5.3**
The CLT allows us to use normal probability calculations for \\(\bar{x}\\) even when the population distribution is non-normal, as long as n is large enough (typically n ≥ 30). This is crucial because in practice we rarely know the population distribution shape.

**5.4**
1. \\(\mu_{\bar{x}_1 - \bar{x}_2} = \mu_1 - \mu_2 = 4\\)
2. \\(SE = \sqrt{\frac{4^2}{50} + \frac{5^2}{60}} = \sqrt{0.32 + 0.417} \approx 0.858\\)

**5.5**
For independent random variables, variances add (not standard deviations). This comes from \\(\text{Var}(X - Y) = \text{Var}(X) + \text{Var}(Y)\\) when X and Y are independent. Since SE is the square root of variance, we compute \\(SE_{diff} = \sqrt{SE_1^2 + SE_2^2}\\).

**C5.1**
- Conditions: Random sample (box randomly selected), 10% (25 likely < 10% of all bolts), Normal (assume bolt lengths approximately normal or use CLT if production is large)
- \\(\mu_{\bar{x}} = 5\\), \\(\sigma_{\bar{x}} = \frac{0.1}{\sqrt{25}} = 0.02\\)
- \\(Z = \frac{5.03 - 5}{0.02} = 1.5\\)
- \\(P(Z > 1.5) \approx 0.067\\) (about 6.7% chance)

---

---

## Unit 6: Inference for Proportions

### Confidence Intervals for One Proportion

#### Point estimate and margin of error

- **Point estimate**: \\(\hat{p} = \frac{x}{n}\\) (sample proportion)
- **Margin of error (ME)**: \\(z^* \times SE\\), where \\(SE = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}\\)
- **Confidence interval**: \\(\hat{p} \pm ME\\)
- _Question_: In a sample of 200, 80 support a policy. Find \\(\hat{p}\\) and SE.
- _Answer_: \\(\hat{p} = \frac{80}{200} = 0.4\\), \\(SE = \sqrt{\frac{0.4(0.6)}{200}} \approx 0.0346\\)

#### Conditions for a one-proportion z-interval

1. **Random**: Random sample or random assignment
2. **10% condition**: \\(n \le 0.10N\\)
3. **Large Counts**: \\(n\hat{p} \ge 10\\) and \\(n(1-\hat{p}) \ge 10\\)

#### Constructing and interpreting confidence intervals

- **Formula**: \\(\hat{p} \pm z^* \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}\\)
- **Common z* values**: 90% → 1.645, 95% → 1.96, 99% → 2.576
- **Interpretation**: "We are [C%] confident that the true proportion of [population] who [context] is between [lower] and [upper]."
- _Tip_: Confidence is about the METHOD, not this specific interval.

#### Interpreting confidence levels correctly

- **Correct**: "If we repeated this process many times, about 95% of intervals would capture the true p."
- **Incorrect**: "There's a 95% chance p is in this interval." (p is fixed, not random!)
- **Incorrect**: "95% of the data is in this interval." (Interval is for p, not individual data!)

#### Determining sample size for a desired margin of error

- **Formula**: \\(n = \left(\frac{z^*}{ME}\right)^2 \hat{p}(1-\hat{p})\\)
- If no estimate available, use \\(\hat{p} = 0.5\\) (most conservative)
- Always round UP to ensure ME is no larger than desired
- _Question_: What n is needed for ME = 0.03 at 95% confidence (use \\(\hat{p} = 0.5\\))?
- _Answer_: \\(n = \left(\frac{1.96}{0.03}\right)^2 (0.5)(0.5) = 1067.1 \rightarrow 1068\\)

<details markdown="1" data-auto-footer>
<summary>Example: One-proportion confidence interval</summary>

In a random sample of 500 voters, 275 support Candidate A. Construct a 95% confidence interval for the proportion of all voters who support A.

**Solution**:
- Conditions: Random (given), 10% (500 < 10% of all voters assumed), Large Counts: \\(500(0.55) = 275 \ge 10\\), \\(500(0.45) = 225 \ge 10\\) ✓
- \\(\hat{p} = \frac{275}{500} = 0.55\\)
- \\(SE = \sqrt{\frac{0.55(0.45)}{500}} \approx 0.0222\\)
- \\(CI = 0.55 \pm 1.96(0.0222) = 0.55 \pm 0.044 = (0.506, 0.594)\\)
- **Interpretation**: We are 95% confident that the true proportion of all voters who support Candidate A is between 50.6% and 59.4%.

</details>

### Significance Tests for One Proportion

#### Null and alternative hypotheses

- **\\(H_0\\)**: \\(p = p_0\\) (null value, often from claim or previous study)
- **\\(H_a\\)**: \\(p < p_0\\), \\(p > p_0\\), or \\(p \ne p_0\\) (research question)
- _Always state hypotheses in terms of the parameter p, not \\(\hat{p}\\)!_

#### Conditions for a one-proportion z-test

1. **Random**: Random sample or random assignment
2. **10% condition**: \\(n \le 0.10N\\)
3. **Large Counts**: \\(np_0 \ge 10\\) and \\(n(1-p_0) \ge 10\\) (use \\(p_0\\) from \\(H_0\\), not \\(\hat{p}\\)!)

#### Test statistic, P-value, and conclusions in context

- **Test statistic**: \\(z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}\\) (use \\(p_0\\) in SE!)
- **P-value**: Probability of observing data as or more extreme than observed, assuming \\(H_0\\) is true
- **Conclusion**: If P-value < α, reject \\(H_0\\). State conclusion in context.

#### One-sided vs. two-sided tests

- **One-sided**: \\(H_a: p > p_0\\) or \\(p < p_0\\) (P-value is one tail)
- **Two-sided**: \\(H_a: p \ne p_0\\) (P-value is two tails)
- Match \\(H_a\\) to the research question!

<details markdown="1" data-auto-footer>
<summary>Example: One-proportion significance test</summary>

A company claims 90% of customers are satisfied. In a random sample of 200 customers, 170 are satisfied. Test at α = 0.05 if the true satisfaction rate is less than 90%.

**Solution**:
- \\(H_0: p = 0.90\\), \\(H_a: p < 0.90\\)
- Conditions: Random (given), 10% (200 < 10% of customers), Large Counts: \\(200(0.9) = 180 \ge 10\\), \\(200(0.1) = 20 \ge 10\\) ✓
- \\(\hat{p} = \frac{170}{200} = 0.85\\)
- \\(z = \frac{0.85 - 0.90}{\sqrt{\frac{0.90(0.10)}{200}}} = \frac{-0.05}{0.0212} \approx -2.36\\)
- P-value: \\(P(Z < -2.36) \approx 0.009\\)
- **Conclusion**: Since P-value (0.009) < α (0.05), we reject \\(H_0\\). There is convincing evidence that the true satisfaction rate is less than 90%.

</details>

### Errors, Power, and Multiple Tests

#### Type I and Type II errors

- **Type I error**: Reject \\(H_0\\) when \\(H_0\\) is true (false positive)
- **Type II error**: Fail to reject \\(H_0\\) when \\(H_0\\) is false (false negative)
- _Cannot make both errors in one test!_

#### Significance level α and its consequences

- **α = P(Type I error)** when \\(H_0\\) is true
- Common: α = 0.05 (5% chance of false positive)
- Lowering α decreases Type I error but increases Type II error

#### Concept of power and factors that affect it

- **Power = P(reject \\(H_0\\) when \\(H_0\\) is false) = 1 - P(Type II error)**
- **Increase power by**: increasing n, increasing α, increasing distance between \\(p_0\\) and true p, decreasing variability

#### Problems of multiple testing and data snooping

- Doing many tests increases chance of Type I error somewhere
- **Data snooping**: Looking at data before choosing hypotheses inflates Type I error
- Solution: Adjust α (e.g., Bonferroni correction) or pre-register hypotheses

### Two-Proportion Inference

#### Conditions for two-sample z-interval for p₁ − p₂

1. **Random**: Independent random samples or random assignment
2. **10% condition**: \\(n_1 \le 0.10N_1\\) and \\(n_2 \le 0.10N_2\\)
3. **Large Counts**: \\(n_1\hat{p}_1, n_1(1-\hat{p}_1), n_2\hat{p}_2, n_2(1-\hat{p}_2)\\) all ≥ 10

#### Interpreting confidence intervals for differences in proportions

- **Formula**: \\((\hat{p}_1 - \hat{p}_2) \pm z^* \sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}\\)
- **Interpretation**: "We are [C%] confident that the true difference in proportions [context] is between [lower] and [upper]."
- If interval contains 0, no significant difference

#### Conditions and mechanics for two-sample z-test

- **Hypotheses**: \\(H_0: p_1 = p_2\\) (or \\(p_1 - p_2 = 0\\)), \\(H_a: p_1 \ne p_2\\) (or <, >)
- **Pooled proportion**: \\(\hat{p}_c = \frac{x_1 + x_2}{n_1 + n_2}\\) (use when assuming \\(H_0\\) is true)
- **Test statistic**: \\(z = \frac{(\hat{p}_1 - \hat{p}_2) - 0}{\sqrt{\hat{p}_c(1-\hat{p}_c)\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}\\)
- **Large Counts**: \\(n_1\hat{p}_c, n_1(1-\hat{p}_c), n_2\hat{p}_c, n_2(1-\hat{p}_c)\\) all ≥ 10

#### Connecting two-sample tests and confidence intervals

- If 95% CI for \\(p_1 - p_2\\) doesn't contain 0, reject \\(H_0: p_1 = p_2\\) at α = 0.05
- Tests and intervals should lead to consistent conclusions

#### Statistical vs. practical significance

- **Statistical significance**: P-value < α (rejects \\(H_0\\))
- **Practical significance**: Effect size is large enough to matter in real life
- With large n, tiny differences can be statistically significant but not practically important

<details markdown="1" data-auto-footer>
<summary>Example: Two-proportion z-test</summary>

Treatment A: 120 out of 200 improve. Treatment B: 100 out of 200 improve. Test if proportions differ at α = 0.05.

**Solution**:
- \\(H_0: p_A = p_B\\), \\(H_a: p_A \ne p_B\\)
- \\(\hat{p}_A = 0.6\\), \\(\hat{p}_B = 0.5\\)
- \\(\hat{p}_c = \frac{120 + 100}{200 + 200} = 0.55\\)
- Conditions: Random (assumed), 10% (assumed), Large Counts: \\(200(0.55) = 110 \ge 10\\), \\(200(0.45) = 90 \ge 10\\) ✓
- \\(z = \frac{0.6 - 0.5}{\sqrt{0.55(0.45)\left(\frac{1}{200} + \frac{1}{200}\right)}} = \frac{0.1}{\sqrt{0.002475}} \approx 2.01\\)
- P-value: \\(2 \times P(Z > 2.01) \approx 2(0.022) = 0.044\\)
- **Conclusion**: Since P-value (0.044) < α (0.05), we reject \\(H_0\\). There is convincing evidence that the true improvement rates differ between treatments.

</details>

### Practice

**6.1** A poll of 800 adults finds 456 support a policy. Construct a 90% confidence interval for the true proportion.

**6.2** A manufacturer claims 95% of products meet quality standards. In a sample of 300, 276 meet standards. Test at α = 0.01 if the true rate is less than 95%.

**6.3** Describe Type I and Type II errors in the context of problem 6.2.

**6.4** What sample size is needed for a margin of error of 0.04 at 95% confidence, assuming no prior estimate?

**6.5** Study 1: \\(n_1 = 150, \hat{p}_1 = 0.40\\). Study 2: \\(n_2 = 200, \hat{p}_2 = 0.30\\).
1. Construct a 95% confidence interval for \\(p_1 - p_2\\)
2. Does the interval suggest a significant difference?

**C6.1** A pharmaceutical company tests a new drug. In a randomized experiment, 80 out of 100 in the treatment group improve, vs. 60 out of 100 in the control group.
1. Test if the treatment is more effective at α = 0.05
2. Calculate the 95% CI for the difference
3. Discuss statistical vs. practical significance

---

### Answers

**6.1**
- \\(\hat{p} = \frac{456}{800} = 0.57\\), \\(z^* = 1.645\\) (90%)
- \\(SE = \sqrt{\frac{0.57(0.43)}{800}} \approx 0.0175\\)
- \\(CI = 0.57 \pm 1.645(0.0175) = 0.57 \pm 0.029 = (0.541, 0.599)\\)
- We are 90% confident that the true proportion of adults who support the policy is between 54.1% and 59.9%.

**6.2**
- \\(H_0: p = 0.95\\), \\(H_a: p < 0.95\\)
- \\(\hat{p} = \frac{276}{300} = 0.92\\)
- Conditions: Random (assumed), 10%, Large Counts: \\(300(0.95) = 285 \ge 10\\), \\(300(0.05) = 15 \ge 10\\) ✓
- \\(z = \frac{0.92 - 0.95}{\sqrt{\frac{0.95(0.05)}{300}}} = \frac{-0.03}{0.0126} \approx -2.38\\)
- P-value: \\(P(Z < -2.38) \approx 0.009\\)
- Since 0.009 < 0.01, reject \\(H_0\\). Evidence that true rate < 95%.

**6.3**
- **Type I**: Conclude rate < 95% when it's actually 95% (unfairly doubt manufacturer)
- **Type II**: Fail to conclude rate < 95% when it's actually < 95% (miss a quality problem)

**6.4**
- \\(n = \left(\frac{1.96}{0.04}\right)^2 (0.5)(0.5) = 600.25 \rightarrow 601\\)

**6.5**
1. \\(SE = \sqrt{\frac{0.4(0.6)}{150} + \frac{0.3(0.7)}{200}} \approx 0.0498\\)
   \\(CI = (0.4 - 0.3) \pm 1.96(0.0498) = 0.10 \pm 0.098 = (0.002, 0.198)\\)
2. Yes, interval doesn't contain 0, suggesting \\(p_1 > p_2\\)

**C6.1**
1. \\(H_0: p_T = p_C\\), \\(H_a: p_T > p_C\\)
   \\(\hat{p}_T = 0.8\\), \\(\hat{p}_C = 0.6\\), \\(\hat{p}_c = 0.7\\)
   \\(z = \frac{0.8 - 0.6}{\sqrt{0.7(0.3)(\frac{1}{100} + \frac{1}{100})}} = \frac{0.2}{0.0648} \approx 3.09\\)
   P-value: \\(P(Z > 3.09) \approx 0.001\\). Reject \\(H_0\\) (treatment more effective).
2. \\(CI = 0.2 \pm 1.96 \sqrt{\frac{0.8(0.2)}{100} + \frac{0.6(0.4)}{100}} = 0.2 \pm 0.108 = (0.092, 0.308)\\)
3. **Statistically significant** (P < 0.05, CI excludes 0). **Practically significant**: 20% improvement is likely meaningful for patients. However, consider costs, side effects, and compare to existing treatments.

---

---

## Unit 7: Inference for Means

### The t-Distribution & One-Sample t Procedures

#### Why we use t instead of z for means

- When σ is **unknown** (almost always!), we estimate it with s
- This adds variability, so we use t-distribution instead of z
- **z**: Known σ (rare in practice)
- **t**: Unknown σ, estimate with s (standard case)

#### Shape and properties of the t-distribution

- Bell-shaped and symmetric (like normal)
- **Heavier tails** than normal (more probability in extremes)
- Different shape for each **degrees of freedom (df = n - 1)**
- As df increases, t approaches normal distribution
- _Question_: Which has heavier tails: t with df=5 or normal?
- _Answer_: t with df=5 (heavier tails reflect extra uncertainty from estimating σ)

#### Conditions for inference about a mean

1. **Random**: Random sample or random assignment
2. **10% condition**: \\(n \le 0.10N\\) (when sampling without replacement)
3. **Normal/Large Sample**: Population approximately normal OR \\(n \ge 30\\) (CLT)
   - For small n, check with dotplot/histogram/boxplot
   - Robust to mild skewness if n ≥ 15, moderate skewness if n ≥ 30

#### One-sample t-interval for a mean

- **Formula**: \\(\bar{x} \pm t^* \frac{s}{\sqrt{n}}\\)
- **df = n - 1**
- Use t-table or calculator to find \\(t^*\\)
- **Interpretation**: "We are [C%] confident that the true mean [context] is between [lower] and [upper]."

#### One-sample t-test for a mean

- **Hypotheses**: \\(H_0: \mu = \mu_0\\), \\(H_a: \mu \ne \mu_0\\) (or <, >)
- **Test statistic**: \\(t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}\\)
- **df = n - 1**
- **P-value**: Use t-distribution with appropriate df

#### Using technology and tables to get P-values

- **Calculator**: Use t-test function (provides P-value directly)
- **Table**: Find critical values; P-value lies between table entries
- _Tip_: Technology gives exact P-values; tables give bounds

<details markdown="1" data-auto-footer>
<summary>Example: One-sample t-interval</summary>

A random sample of 20 students has mean study time \\(\bar{x} = 8.5\\) hours/week, s = 2.3 hours. Construct a 95% confidence interval for the true mean study time.

**Solution**:
- Conditions: Random (given), 10% (20 < 10% of all students), Normal (assume study times approximately normal or check with plot)
- df = 20 - 1 = 19
- \\(t^* = 2.093\\) (from table or calculator, 95%, df=19)
- \\(CI = 8.5 \pm 2.093 \frac{2.3}{\sqrt{20}} = 8.5 \pm 1.08 = (7.42, 9.58)\\)
- **Interpretation**: We are 95% confident that the true mean study time for all students is between 7.42 and 9.58 hours per week.

</details>

### Two-Sample t Procedures

#### Conditions for two-sample t-interval for μ₁ − μ₂

1. **Random**: Independent random samples or random assignment
2. **10% condition**: \\(n_1 \le 0.10N_1\\) and \\(n_2 \le 0.10N_2\\)
3. **Normal/Large Sample**: Both populations approximately normal OR both \\(n_1, n_2 \ge 30\\)

#### Constructing and interpreting two-sample t-intervals

- **Formula**: \\((\bar{x}_1 - \bar{x}_2) \pm t^* \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}\\)
- **df**: Use calculator (conservative: min(\\(n_1 - 1\\), \\(n_2 - 1\\)))
- **Interpretation**: "We are [C%] confident that the true difference in mean [context] is between [lower] and [upper]."

#### Two-sample t-test for difference of means

- **Hypotheses**: \\(H_0: \mu_1 = \mu_2\\) (or \\(\mu_1 - \mu_2 = 0\\)), \\(H_a: \mu_1 \ne \mu_2\\) (or <, >)
- **Test statistic**: \\(t = \frac{(\bar{x}_1 - \bar{x}_2) - 0}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}\\)
- Use calculator for df and P-value

#### Comparing conclusions from tests and intervals

- If 95% CI for \\(\mu_1 - \mu_2\\) doesn't contain 0, reject \\(H_0\\) at α = 0.05
- Confidence intervals provide more information (estimate of effect size)

<details markdown="1" data-auto-footer>
<summary>Example: Two-sample t-test</summary>

Method A: n₁ = 25, \\(\bar{x}_1 = 72\\), s₁ = 8. Method B: n₂ = 30, \\(\bar{x}_2 = 68\\), s₂ = 10. Test if means differ at α = 0.05.

**Solution**:
- \\(H_0: \mu_A = \mu_B\\), \\(H_a: \mu_A \ne \mu_B\\)
- Conditions: Random (assumed), 10% (assumed), Normal/Large Sample (n₁ = 25 borderline, n₂ = 30; assume approximately normal or check plots)
- \\(t = \frac{72 - 68}{\sqrt{\frac{64}{25} + \frac{100}{30}}} = \frac{4}{\sqrt{2.56 + 3.33}} \approx 1.65\\)
- df ≈ 24 (conservative: min(24, 29))
- P-value ≈ 0.11 (two-tailed, from calculator)
- **Conclusion**: Since P-value (0.11) > α (0.05), we fail to reject \\(H_0\\). There is not convincing evidence that the true mean scores differ between methods.

</details>

### Matched Pairs & Special Designs

#### Matched pairs and paired data

- **Paired data**: Two measurements on the same individual/unit (before/after, twin studies, etc.)
- **NOT independent** samples—violates two-sample t conditions!
- Analyze **differences** within pairs, use one-sample t procedures

#### Paired t-interval and paired t-test

- Calculate differences: \\(d_i = x_{1i} - x_{2i}\\)
- Find \\(\bar{d}\\) and \\(s_d\\)
- **Interval**: \\(\bar{d} \pm t^* \frac{s_d}{\sqrt{n}}\\) (df = n - 1, where n = number of pairs)
- **Test**: \\(H_0: \mu_d = 0\\), \\(t = \frac{\bar{d} - 0}{s_d / \sqrt{n}}\\)

#### Design issues: blocking vs. pairing

- **Pairing**: Reduces variability by comparing within matched units
- **Blocking**: Group similar units, randomize within blocks
- Pairing is extreme blocking (block size = 2)

#### Quasi-experiments and their limitations

- **Quasi-experiment**: No random assignment (observational with treatment comparison)
- Cannot establish causation (confounding possible)
- Example: Comparing smokers vs. non-smokers (can't randomly assign smoking!)

<details markdown="1" data-auto-footer>
<summary>Example: Paired t-test</summary>

10 students take a test before and after tutoring. Mean difference (after - before) = 5.2 points, \\(s_d = 3.5\\). Test if tutoring improves scores at α = 0.05.

**Solution**:
- \\(H_0: \mu_d = 0\\), \\(H_a: \mu_d > 0\\) (one-sided: expecting improvement)
- Paired data (same students), so use paired t-test
- \\(t = \frac{5.2 - 0}{3.5 / \sqrt{10}} = \frac{5.2}{1.107} \approx 4.70\\)
- df = 10 - 1 = 9
- P-value ≈ 0.0005 (one-tailed, very small)
- **Conclusion**: Since P-value < 0.05, reject \\(H_0\\). There is convincing evidence that tutoring improves test scores.

</details>

### Standard Error & Interpretation

#### Distinguishing standard deviation from standard error

- **Standard deviation (s)**: Measures variability of **individual data values**
- **Standard error (SE)**: Measures variability of a **statistic** (like \\(\bar{x}\\))
- \\(SE = \frac{s}{\sqrt{n}}\\) (SE decreases as n increases)
- _Common mistake_: Using s instead of \\(\frac{s}{\sqrt{n}}\\) in formulas

#### Interpreting standard error in context

- SE describes how much \\(\bar{x}\\) varies from sample to sample
- Smaller SE → more precise estimate of μ
- _Example_: "The standard error of 2.5 means sample means typically vary by about 2.5 units from the true population mean."

#### Common mistakes with t procedures on the exam

1. Using two-sample t for paired data (or vice versa)
2. Using s instead of \\(\frac{s}{\sqrt{n}}\\)
3. Forgetting to check conditions
4. Interpreting CI incorrectly ("95% of data" vs. "95% confident about μ")
5. Using z instead of t when σ is unknown

### Practice

**7.1** A random sample of 15 trees has mean height 12.3 m, s = 1.8 m. Construct a 90% confidence interval for the true mean height.

**7.2** A company claims average delivery time is 3 days. A sample of 40 deliveries has \\(\bar{x} = 3.4\\) days, s = 0.9 days. Test at α = 0.05 if the true mean is greater than 3 days.

**7.3** Group A: n = 20, \\(\bar{x} = 85\\), s = 12. Group B: n = 25, \\(\bar{x} = 78\\), s = 15. Construct a 95% CI for \\(\mu_A - \mu_B\\).

**7.4** 8 subjects: before mean = 140, after mean = 132, \\(\bar{d} = -8\\), \\(s_d = 5\\). Test if the treatment reduces the measurement at α = 0.01.

**7.5** Explain why SE = \\(\frac{s}{\sqrt{n}}\\) decreases as n increases, and what this means for precision of estimates.

**C7.1** A study compares reaction times for caffeine (n = 30, \\(\bar{x} = 0.25\\) s, s = 0.05 s) vs. placebo (n = 30, \\(\bar{x} = 0.30\\) s, s = 0.06 s).
1. Test if caffeine reduces mean reaction time at α = 0.05
2. Construct a 95% CI for the difference
3. Discuss whether the difference is practically significant for drivers

---

### Answers

**7.1**
- df = 14, \\(t^* \approx 1.761\\) (90%, df=14)
- \\(CI = 12.3 \pm 1.761 \frac{1.8}{\sqrt{15}} = 12.3 \pm 0.82 = (11.48, 13.12)\\) meters
- We are 90% confident the true mean tree height is between 11.48 m and 13.12 m.

**7.2**
- \\(H_0: \mu = 3\\), \\(H_a: \mu > 3\\)
- \\(t = \frac{3.4 - 3}{0.9 / \sqrt{40}} = \frac{0.4}{0.142} \approx 2.81\\)
- df = 39, P-value ≈ 0.004 (one-tailed)
- Reject \\(H_0\\). Evidence that true mean > 3 days.

**7.3**
- \\(SE = \sqrt{\frac{144}{20} + \frac{225}{25}} = \sqrt{7.2 + 9} \approx 4.02\\)
- df ≈ 19 (conservative), \\(t^* \approx 2.093\\)
- \\(CI = 7 \pm 2.093(4.02) = 7 \pm 8.41 = (-1.41, 15.41)\\)
- We are 95% confident the difference in means is between -1.41 and 15.41. (Interval contains 0, so no significant difference at α = 0.05)

**7.4**
- \\(H_0: \mu_d = 0\\), \\(H_a: \mu_d < 0\\) (reduction)
- \\(t = \frac{-8}{5 / \sqrt{8}} = \frac{-8}{1.768} \approx -4.52\\)
- df = 7, P-value ≈ 0.001 (one-tailed)
- Reject \\(H_0\\) (P < 0.01). Evidence of reduction.

**7.5**
As n increases, \\(\sqrt{n}\\) increases, so \\(\frac{s}{\sqrt{n}}\\) decreases. This means larger samples produce more precise estimates of μ (\\(\bar{x}\\) values cluster more tightly around true μ). SE quantifies this precision.

**C7.1**
1. \\(H_0: \mu_C = \mu_P\\), \\(H_a: \mu_C < \mu_P\\)
   \\(t = \frac{0.25 - 0.30}{\sqrt{\frac{0.0025}{30} + \frac{0.0036}{30}}} = \frac{-0.05}{0.0143} \approx -3.50\\)
   P-value ≈ 0.0004. Reject \\(H_0\\). Evidence caffeine reduces reaction time.
2. \\(CI = -0.05 \pm 2.002(0.0143) = -0.05 \pm 0.029 = (-0.079, -0.021)\\) seconds
3. **Statistically significant**: Yes (P < 0.05, CI excludes 0). **Practically significant**: A 0.05 s (50 ms) reduction in reaction time could be meaningful for safety (e.g., at 60 mph, a car travels ~4 feet in 50 ms), but consider individual variation, cost, and side effects of caffeine.

---

---

## Unit 8: Inference for Categorical Data (Chi-Square)

### Chi-Square Basics

#### Chi-square statistic and its distribution

- **Chi-square (χ²)**: Measures how far observed counts deviate from expected counts
- **Formula**: \\(\chi^2 = \sum \frac{(\text{Observed} - \text{Expected})^2}{\text{Expected}}\\)
- Always ≥ 0 (squared differences)
- Right-skewed distribution

#### Degrees of freedom

- **Goodness-of-fit**: df = (number of categories) - 1 - (number of parameters estimated)
- **Two-way table**: df = (rows - 1)(columns - 1)
- df affects shape of χ² distribution

#### Conditions for chi-square procedures

1. **Random**: Random sample or random assignment
2. **10% condition**: When sampling without replacement
3. **Large Counts**: All expected counts ≥ 5 (NOT observed counts!)

### Chi-Square Goodness-of-Fit Test

#### Setting up hypotheses and categories

- **\\(H_0\\)**: Population distribution matches claimed distribution
- **\\(H_a\\)**: Population distribution differs from claimed distribution
- Specify proportions for each category under \\(H_0\\)

#### Computing expected counts

- **Expected count** = (total n) × (proportion under \\(H_0\\))
- Must sum to same total as observed counts
- Check: All expected ≥ 5?

#### Calculating chi-square statistic

- \\(\chi^2 = \sum \frac{(O - E)^2}{E}\\) for each category
- Larger χ² → stronger evidence against \\(H_0\\)

#### Using technology/tables to find P-values

- Use χ² cdf function with calculated χ² and df
- Tables give critical values; compare calculated χ² to table value
- P-value = area to the right of calculated χ²

#### Interpreting results in context

- If P-value < α, reject \\(H_0\\): "There is convincing evidence that the distribution differs from [claimed distribution]."
- If P-value ≥ α, fail to reject \\(H_0\\): "There is not convincing evidence..."

<details markdown="1" data-auto-footer>
<summary>Example: Chi-square goodness-of-fit</summary>

A company claims its candies are 20% red, 30% blue, 30% green, 20% yellow. A random sample of 200 candies: 50 red, 55 blue, 60 green, 35 yellow. Test at α = 0.05.

**Solution**:
- \\(H_0\\): Distribution is 20% red, 30% blue, 30% green, 20% yellow
- \\(H_a\\): Distribution differs
- Expected: Red = 40, Blue = 60, Green = 60, Yellow = 40
- Conditions: Random (given), Large Counts (all expected ≥ 5) ✓
- \\(\chi^2 = \frac{(50-40)^2}{40} + \frac{(55-60)^2}{60} + \frac{(60-60)^2}{60} + \frac{(35-40)^2}{40} = 2.5 + 0.417 + 0 + 0.625 = 3.542\\)
- df = 4 - 1 = 3
- P-value ≈ 0.315 (from calculator)
- **Conclusion**: Since P-value (0.315) > α (0.05), we fail to reject \\(H_0\\). There is not convincing evidence that the color distribution differs from the company's claim.

</details>

### Chi-Square Tests for Two-Way Tables

#### Homogeneity vs. independence

- **Test for homogeneity**: Compare distributions across separate groups (e.g., compare opinion distributions across 3 cities)
- **Test for independence**: Check if two variables are associated in one population (e.g., gender and major choice)
- **Calculations identical**; interpretation differs

#### Hypotheses, conditions, and test statistic

- **Homogeneity \\(H_0\\)**: Distributions are the same across groups
- **Independence \\(H_0\\)**: Variables are independent (no association)
- **Expected count**: \\(\frac{(\text{row total})(\text{column total})}{\text{grand total}}\\)
- \\(\chi^2 = \sum \frac{(O - E)^2}{E}\\) across all cells
- df = (r - 1)(c - 1)

#### Interpreting chi-square output

- Large χ² and small P-value suggest association/difference
- Examine which cells contribute most to χ² (largest \\(\frac{(O-E)^2}{E}\\) values)

#### Connecting chi-square tests to earlier association ideas

- Chi-square tests categorical association (like correlation for quantitative)
- Can follow up with conditional distributions to see patterns

<details markdown="1" data-auto-footer>
<summary>Example: Chi-square test for independence</summary>

Survey of 300 students: Do study habits differ by class year?

|  | Study Daily | Study Before Exams | Total |
|---|---|---|---|
| Freshman | 30 | 70 | 100 |
| Sophomore | 40 | 60 | 100 |
| Junior/Senior | 50 | 50 | 100 |
| **Total** | 120 | 180 | 300 |

Test at α = 0.05.

**Solution**:
- \\(H_0\\): Study habits and class year are independent
- \\(H_a\\): Study habits and class year are not independent
- Expected for Freshman/Daily: \\(\frac{100 \times 120}{300} = 40\\)
- All expected counts: Freshman (40, 60), Sophomore (40, 60), Junior/Senior (40, 60)
- Conditions: Random (assumed), Large Counts (all expected ≥ 5) ✓
- \\(\chi^2 = \frac{(30-40)^2}{40} + \frac{(70-60)^2}{60} + \frac{(40-40)^2}{40} + \frac{(60-60)^2}{60} + \frac{(50-40)^2}{40} + \frac{(50-60)^2}{60}\\)
  \\(= 2.5 + 1.667 + 0 + 0 + 2.5 + 1.667 = 8.334\\)
- df = (3-1)(2-1) = 2
- P-value ≈ 0.015
- **Conclusion**: Since P-value (0.015) < α (0.05), we reject \\(H_0\\). There is convincing evidence that study habits and class year are not independent.

</details>

### Practice

**8.1** A die is rolled 120 times: 1 appears 25 times, 2–6 each appear 19 times. Test if the die is fair at α = 0.05.

**8.2** Explain why we check that **expected counts** (not observed) are ≥ 5 for chi-square procedures.

**8.3** A study compares political affiliation across three regions:

|  | Democrat | Republican | Independent | Total |
|---|---|---|---|---|
| Region A | 40 | 30 | 30 | 100 |
| Region B | 50 | 40 | 10 | 100 |
| **Total** | 90 | 70 | 40 | 200 |

Test for homogeneity at α = 0.05.

**C8.1** A genetics model predicts offspring ratios 9:3:3:1. In an experiment with 160 offspring, observed counts are 95, 28, 27, 10. Test the model at α = 0.01. If rejected, which category deviates most?

---

### Answers

**8.1**
- \\(H_0\\): Die is fair (each face 1/6)
- Expected: each face 20 times
- \\(\chi^2 = \frac{(25-20)^2}{20} + 5 \times \frac{(19-20)^2}{20} = 1.25 + 0.25 = 1.5\\)
- df = 5, P-value ≈ 0.91
- Fail to reject \\(H_0\\). Not enough evidence die is unfair.

**8.2**
The χ² distribution theory requires expected counts ≥ 5 for the approximation to be valid. Small expected counts lead to unreliable P-values.

**8.3**
- \\(H_0\\): Political affiliation distribution is the same across regions
- Expected: Region A (45, 35, 20), Region B (45, 35, 20)
- \\(\chi^2 = \frac{(40-45)^2}{45} + \frac{(30-35)^2}{35} + \frac{(30-20)^2}{20} + \frac{(50-45)^2}{45} + \frac{(40-35)^2}{35} + \frac{(10-20)^2}{20}\\)
  \\(= 0.556 + 0.714 + 5 + 0.556 + 0.714 + 5 = 12.54\\)
- df = (2-1)(3-1) = 2, P-value ≈ 0.002
- Reject \\(H_0\\). Evidence distributions differ across regions.

**C8.1**
- \\(H_0\\): Ratio is 9:3:3:1
- Expected: 90, 30, 30, 10
- \\(\chi^2 = \frac{(95-90)^2}{90} + \frac{(28-30)^2}{30} + \frac{(27-30)^2}{30} + \frac{(10-10)^2}{10} = 0.278 + 0.133 + 0.3 + 0 = 0.711\\)
- df = 3, P-value ≈ 0.87
- Fail to reject \\(H_0\\). Data consistent with 9:3:3:1 model.
- (No category deviates significantly; all contributions small)

---

---

## Unit 9: Inference for Regression

### Conditions for Regression Inference

#### Linearity, independence, normality, equal variance, randomness

**LINE-R**:
1. **Linear**: Relationship between x and y is linear (check scatterplot, residual plot)
2. **Independent**: Observations are independent (random sample/assignment)
3. **Normal**: Residuals are approximately normal (histogram/normal probability plot of residuals)
4. **Equal variance**: Variability of residuals is constant across x values (residual plot shows random scatter, no fan shape)
5. **Random**: Data from random sample or random assignment

#### Checking conditions with residual plots and other graphs

- **Residual plot**: Plot residuals vs. x or \\(\hat{y}\\)
  - Should show random scatter (no pattern)
  - Constant spread (no fan/cone shape)
- **Histogram of residuals**: Should be approximately normal (especially for small n)
- **Normal probability plot**: Points should follow straight line

### t Procedures for the Slope

#### Hypotheses about slope (β)

- **\\(H_0: \beta = 0\\)** (no linear relationship)
- **\\(H_a: \beta \ne 0\\)** (or <, >) (linear relationship exists)
- β is the **true population slope**; b is the sample slope estimate

#### t statistic for slope and associated P-value

- **Test statistic**: \\(t = \frac{b - 0}{SE_b}\\)
- df = n - 2
- **\\(SE_b\\)** (standard error of slope): Measures variability of b across samples
- From computer output: look for "Slope" row, find t-statistic and P-value

#### Confidence interval for slope

- **Formula**: \\(b \pm t^* SE_b\\)
- df = n - 2
- **Interpretation**: "We are [C%] confident that for each unit increase in [x], the true mean [y] increases/decreases by between [lower] and [upper] units."

#### Interpreting slope, intercept, and standard error in context

- **Slope**: Change in mean y per unit increase in x
- **Intercept**: Predicted y when x = 0 (only meaningful if x = 0 is in data range)
- **\\(SE_b\\)**: Typical error in estimating true slope
- **s (residual standard error)**: Typical size of prediction errors

<details markdown="1" data-auto-footer>
<summary>Example: Regression inference</summary>

A random sample of 20 students: x = study hours, y = exam score. Computer output shows:
- \\(\hat{y} = 45 + 5x\\)
- \\(SE_b = 1.2\\)
- t = 4.17, P-value = 0.0006
- df = 18

Test if there's a linear relationship at α = 0.05, and construct a 95% CI for the slope.

**Solution**:
- \\(H_0: \beta = 0\\), \\(H_a: \beta \ne 0\\)
- From output: t = 4.17, P-value = 0.0006
- Since P-value (0.0006) < α (0.05), reject \\(H_0\\). There is convincing evidence of a positive linear relationship between study hours and exam score.
- \\(CI: 5 \pm 2.101(1.2) = 5 \pm 2.52 = (2.48, 7.52)\\) (using \\(t^* \approx 2.101\\) for 95%, df=18)
- **Interpretation**: We are 95% confident that for each additional hour of study, the true mean exam score increases by between 2.48 and 7.52 points.

</details>

### Connecting Regression & Correlation

#### Relationship between slope, correlation, and standard deviations

- \\(b = r \frac{s_y}{s_x}\\)
- Same sign as r
- If r = 0, then b = 0 (no linear relationship)

#### Using regression output as part of a larger inference problem

- Check conditions (LINE-R) before interpreting output
- Look for \\(R^2\\) (proportion of y variance explained by x)
- Identify which output to use for CI vs. test

#### Common AP exam tasks involving regression inference

1. Check conditions from graphs
2. Interpret slope/intercept in context
3. Test \\(H_0: \beta = 0\\)
4. Construct CI for slope
5. Interpret computer output

### Practice

**9.1** For a regression with n = 25, the slope is b = 3.2, \\(SE_b = 0.8\\). Test \\(H_0: \beta = 0\\) vs. \\(H_a: \beta > 0\\) at α = 0.05.

**9.2** Why do we check residual plots rather than the original scatterplot when assessing equal variance?

**9.3** A regression of y = price (\\$1000s) on x = age (years) for 30 cars gives b = -1.5, \\(SE_b = 0.4\\). Construct a 95% CI for the slope and interpret.

**C9.1** A study of n = 50 trees examines diameter (x, cm) vs. height (y, m). Output shows \\(\hat{y} = 2 + 0.15x\\), \\(SE_b = 0.03\\), \\(r^2 = 0.64\\).
1. Interpret the slope
2. Test if \\(\beta > 0\\) at α = 0.01
3. Interpret \\(r^2\\)
4. What additional checks would you perform before trusting this inference?

---

### Answers

**9.1**
- \\(t = \frac{3.2 - 0}{0.8} = 4\\)
- df = 23, P-value ≈ 0.0003 (one-tailed)
- Reject \\(H_0\\). Evidence of positive slope.

**9.2**
The residual plot shows deviations from the fitted line, making it easier to spot patterns in variability. The original scatterplot may obscure unequal variance due to the overall trend.

**9.3**
- \\(t^* \approx 2.045\\) (95%, df=28)
- \\(CI = -1.5 \pm 2.045(0.4) = -1.5 \pm 0.818 = (-2.318, -0.682)\\)
- We are 95% confident that for each additional year of age, the true mean price decreases by between \\$682 and \\$2,318.

**C9.1**
1. For each additional cm in diameter, the predicted mean height increases by 0.15 m.
2. \\(t = \frac{0.15}{0.03} = 5\\), df = 48, P-value < 0.0001 (one-tailed). Reject \\(H_0\\). Strong evidence \\(\beta > 0\\).
3. 64% of the variability in tree height is explained by the linear relationship with diameter.
4. Check LINE-R conditions: residual plot for linearity and equal variance, histogram/normal plot of residuals for normality, confirm random sample.

---

#### Relationship between slope, correlation, and standard deviations

#### Using regression output as part of a larger inference problem

#### Common AP exam tasks involving regression inference

### Practice

---

## Unit 10: AP Exam Preparation & Integrated Practice

### Exam Structure & Strategy

#### Structure of the AP Statistics exam

- **Section I**: 40 Multiple Choice Questions (90 minutes, 50% of score)
  - Part A: No calculator (≈20 questions, 30 min)
  - Part B: Calculator allowed (≈20 questions, 60 min)
- **Section II**: 6 Free Response Questions (90 minutes, 50% of score)
  - 5 shorter questions (≈12 min each)
  - 1 investigative task (≈25 min)
- Covers all units; emphasis on Units 3-9 (inference and design)

#### Calculator and formula sheet tips

- **Formula sheet provided**: Know what's on it (don't memorize formulas that are given!)
- **Calculator skills**: Know how to use statistical functions (1-var stats, 2-var stats, tests, intervals)
- **When not to use calculator**: Conceptual questions, interpreting output, checking conditions
- Practice without calculator for Section I Part A

#### Time management strategies for MCQ and FRQ

- **MCQ**: ≈2.25 min per question; skip hard ones, return later
- **FRQ**: Allocate time appropriately (don't spend 30 min on one question!)
- **Investigative task**: More time, multiple parts, often combines topics
- **Write clearly**: Graders must be able to read your work

### Multiple-Choice Practice Themes

#### Interpreting graphs and numerical summaries

- Read axis labels carefully
- Distinguish between different graph types (histogram, boxplot, scatterplot, etc.)
- Identify outliers, shape, center, spread
- Connect graphs to context

#### Choosing appropriate procedures and checking conditions

- Identify parameter of interest (p, μ, difference, slope?)
- Match procedure to scenario (1-sample, 2-sample, paired, chi-square, regression?)
- Know conditions for each procedure
- Common trap: Paired data treated as two-sample

#### Common trap answers and how to avoid them

- **Correlation vs. causation**: Association doesn't imply causation
- **CI interpretation**: "Confident about parameter" not "probability" or "data"
- **P-value interpretation**: Probability of data assuming \\(H_0\\), not probability of \\(H_0\\)
- **Conditions**: Can't proceed with inference if conditions aren't met
- **Sample vs. population**: Statistics (\\(\bar{x}, \hat{p}\\)) vs. parameters (μ, p)

### Free-Response Practice Themes

#### Four-step process: State–Plan–Do–Conclude

1. **State**: Hypotheses (tests) or parameter (intervals), define parameters in context
2. **Plan**: Name procedure, check conditions explicitly
3. **Do**: Calculate statistic, show work (or describe calculator input/output)
4. **Conclude**: Interpret result in context, link to original question

#### Writing clear conclusions in context

- **Always use context** (not just "reject \\(H_0\\)")
- **Tests**: "There is/is not convincing evidence that [claim in context]"
- **Intervals**: "We are [C%] confident that [parameter in context] is between [values with units]"
- **Avoid**: "Prove", "accept \\(H_0\\)", generic statements

#### Multi-part problems linking several units (e.g., design + inference)

- Part (a): Design a study → Units 3
- Part (b): Collect/describe data → Units 1, 2
- Part (c): Perform inference → Units 4-9
- Part (d): Interpret in broader context
- **Read all parts first** to see connections

### Cumulative Review

#### Mixed practice problems across all units

**Problem 1 (Units 1-2)**: A dataset of exam scores has mean 78, median 82, Q1 = 68, Q3 = 88.
1. Describe the shape
2. Calculate and interpret IQR
3. Identify outliers using 1.5 × IQR rule

**Problem 2 (Unit 3)**: Design an experiment to test if a new teaching method improves test scores compared to traditional methods.

**Problem 3 (Units 4-5)**: Population has p = 0.4. Sample size n = 100.
1. Find \\(P(\hat{p} > 0.45)\\)
2. What happens to this probability as n increases?

**Problem 4 (Unit 6)**: Sample: 60 out of 200 support a proposal. Construct a 95% CI for the true proportion.

**Problem 5 (Unit 7)**: Two groups: Group 1 (n=25, \\(\bar{x}=50\\), s=8), Group 2 (n=30, \\(\bar{x}=46\\), s=10). Test if means differ at α=0.05.

**Problem 6 (Unit 8)**: Survey results:

|  | Agree | Disagree | Total |
|---|---|---|---|
| Urban | 80 | 20 | 100 |
| Rural | 60 | 40 | 100 |
| **Total** | 140 | 60 | 200 |

Test for homogeneity at α=0.05.

**Problem 7 (Unit 9)**: Regression output for predicting weight from height (n=30): b=2.5, \\(SE_b=0.6\\). Test if slope is positive at α=0.05.

---

### Practice Answers

**Problem 1**:
1. Left-skewed (mean < median)
2. IQR = 88 - 68 = 20 points. Middle 50% of scores span 20 points.
3. Lower fence: 68 - 30 = 38, Upper fence: 88 + 30 = 118. Outliers: scores < 38 or > 118.

**Problem 2**:
- **Design**: Randomly assign students to new method (treatment) or traditional method (control). Give same final exam. Compare mean scores.
- **Randomization**: Balances lurking variables across groups.
- **Allows causation**: Random assignment allows causal conclusions if treatment group scores higher.

**Problem 3**:
1. \\(\mu_{\hat{p}} = 0.4\\), \\(\sigma_{\hat{p}} = \sqrt{\frac{0.4(0.6)}{100}} = 0.049\\)
   \\(Z = \frac{0.45-0.4}{0.049} \approx 1.02\\), \\(P(Z>1.02) \approx 0.154\\)
2. As n increases, \\(\sigma_{\hat{p}}\\) decreases → probability decreases (sampling distribution tightens around p)

**Problem 4**:
- \\(\hat{p} = 0.3\\), \\(SE = \sqrt{\frac{0.3(0.7)}{200}} \approx 0.0324\\)
- \\(CI = 0.3 \pm 1.96(0.0324) = 0.3 \pm 0.064 = (0.236, 0.364)\\)
- 95% confident true proportion is between 23.6% and 36.4%

**Problem 5**:
- \\(H_0: \mu_1 = \mu_2\\), \\(H_a: \mu_1 \ne \mu_2\\)
- \\(t = \frac{50-46}{\sqrt{\frac{64}{25}+\frac{100}{30}}} = \frac{4}{2.55} \approx 1.57\\)
- df ≈ 24, P-value ≈ 0.13
- Fail to reject \\(H_0\\). Not enough evidence means differ.

**Problem 6**:
- \\(H_0\\): Opinion distribution same for urban/rural
- Expected: Urban (70, 30), Rural (70, 30)
- \\(\chi^2 = \frac{(80-70)^2}{70} + \frac{(20-30)^2}{30} + \frac{(60-70)^2}{70} + \frac{(40-30)^2}{30} = 1.43 + 3.33 + 1.43 + 3.33 = 9.52\\)
- df=1, P-value ≈ 0.002
- Reject \\(H_0\\). Evidence distributions differ.

**Problem 7**:
- \\(H_0: \beta = 0\\), \\(H_a: \beta > 0\\)
- \\(t = \frac{2.5}{0.6} \approx 4.17\\), df=28
- P-value < 0.001 (one-tailed)
- Reject \\(H_0\\). Evidence of positive slope.

---

## Final Thoughts

You've now covered all 10 units of AP Statistics! 🎉

**Key Takeaways**:
1. **Always check conditions** before performing inference
2. **Context is king**: Every answer should reference the real-world scenario
3. **Know your procedures**: Match the right test/interval to the situation
4. **Practice, practice, practice**: Use past AP exams for realistic preparation
5. **Show your work**: Partial credit is available on FRQs
6. **Read carefully**: Distinguish between sample statistics and population parameters

**Study Tips**:
- Create a procedure flowchart (when to use each test/interval)
- Practice writing complete 4-step solutions
- Review common mistakes and misconceptions
- Time yourself on practice exams
- Study in groups to explain concepts to others

**Good luck on the AP Statistics exam!** 📊✨

#### Interpreting computer output from real data

#### Reflecting on common errors and “red flag” phrases

### Practice
