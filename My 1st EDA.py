import pandas as pd
import numpy as np

pd.set_option("display.precision", 2)
df = pd.read_csv(r"C:\Users\ninol\Desktop\olá\archive\dataset.csv")
# print (df.shape)   // Used for printing number of rows x columns (667,20)
# print(df.columns) // Prints the name of the rows in the dataset
# print(df.info()) //  Gives info about the data typ3 ( (1)bool typ3, (3)are object typ3, and (16)numeric typ3s) here
#                 //   We can also visualize if there's any missing values.
#                //    In this case we can easily see that no values are missing because
#               //     the number of elements in each row equals the value found previously with shape(667).

df["Churn"] = df["Churn"].astype("int64")  # Changing the churn column typ3 from _bool_ to _int64_:

# print (df.describe())                Describe method shows basic statistical characteristics of each numerical feature
#                                      (int64 and float64 types): number of non-missing values,mean, standard deviation,
#                                      range, median, 0.25 and 0.75 quartiles. But we can only see numerical values. In
#                                      order to see bool and objects in the describe method, we gotta insert include:

# print(df.describe(include=["object", "bool"])) // For both boolean and objects types we have to specify.//
#                                                //and use value_counts method.                          //
# print(df["Churn"].value_counts())                                      Looking at "Churn" column. It gives us:0  572
#                                                      (572 loyal customers, 95 churn)                          1  95
#                                                                                      Name: Churn, dtype: int64

# print(df["Churn"].value_counts(normalize=True)) //set the previous values to percentages.(86% loyal;14% churn)//

# /////   THE SORT METHOD  ///////
# A Dataframe can be sorted by the value of one variable (columns) For example, we can sort by Total day calls:
#                         (use ascending=False to sort in descending order):

# print(df.sort_values(by="Total day calls", ascending=False).head())

# We can also sort by multiple columns using:

# print(df.sort_values(by=["Total day calls", "Churn"], ascending=[True, False]).head())


# ////    INDEXING AND RETRIEVING    ////


# To get a single column, you can use a DataFrame['Name'] construction.

#  What is the proportion of churned users in our dataframe?

# df["Churn"].mean().round()      #-->>>> It gives us the value of 0.1424 churn rates (14.24% Is pretty bad for
#                                                                                             a company churn rates)

#  What are average values of numerical features for churned users?


# print(df[df["Churn"] == 1].select_dtypes(include=np.number).mean())


#  How much time (on average) do churned users spend on the phone during daytime?

# print(df[df["Churn"] == 1]["Total day minutes"].mean())       ->>>>> 213.9915

#  What is the maximum length of international calls among loyal users who do not have an international plan?

# print(df[(df["Churn"] == 0) & (df["International plan"] == "No")]["Total intl minutes"].max())    ->>>> 18 minutes

#  We can also index DF's by column name(label) or row name(index) or by the serial number.

# print(df.loc[0:10, "State":"Area code"]) ->>> In this case its printed 0 to 10 index from "state" column to "area code"

# print(df.iloc[0:5, 0:3])        ->>>  In this case we use indexing by number

#  (Loc method is used for indexing by name, while iloc() is used for indexing by number.)

#  If we need 1st line of the DF   ->>>> df[:1] and df[-1:] for last line


# Functions to Cells, Columns and Rows      //////////


# print(df.apply(np.max))  ->>>>>>>>> prints all maximum values in ea column

#  The apply method can be used to perform a function in each row, as below:

#  If we need to select all states starting with S, we can do it like this:

# print(df[df["State"].apply(lambda state: state[0] == "S")].head())

#  The map method can replace values in a column:

d = {"No": False, "Yes": True}
df["International plan"] = df["International plan"].map(d)
# print(df.head())

#  We can also do the same thing with the replace method.

# df = df.replace({"Voice mail plan": d})
# df.head()


# GROUPING ////////////////

#  1st create a variable with the list of columns you wanto be shown, then, use the groupby method with "Churn" as a
# group "caller", that groups them by the Churn column as a new index: ->>>> 3rd we use the describe method as before.

# columns_to_show = ["Total day minutes", "Total eve minutes", "Total night minutes"]

# (print(df.groupby(["Churn"])[columns_to_show].describe(percentiles=[])))

# We're gonna do the same but a little different, we're gonna pass the functions to agg()

# columns_to_show = ["Total day minutes", "Total eve minutes", "Total night minutes"]
# print(df.groupby(["Churn"])[columns_to_show].agg([np.mean, np.std, np.min, np.max]))


# SUMMARY TABLES  //////////////////

# Let's imagine we want to see how observations in our sample is distributed in the context of two variables - "Churn" -
# And "International Plan":

# print(pd.crosstab(df["Churn"], df["International plan"]))  ->>> It gives us the frequency between both Churn and I.P.


# print(pd.crosstab(df["Churn"], df["Voice mail plan"], normalize=True))  ->>>> Frequency between 'VMP' and 'Churn'


# PIVOT TABLES IN PANDAS //////////////

# the pivot_table method takes the following parameters:

# A list of variables to calculate statistics for (values)
# A list of variables to group data by  (index)
# What statistics we need to calculate for groups, ex. sum, mean, maximum, minimum or something else. (aggfunc)

# print(df.pivot_table(
#     ["Total day calls", "Total eve calls", "Total night calls"],
#     ["Area code"],
#     aggfunc="mean",
# ))

# TRANSFORMATIONS IN THE DATAFRAME ///////////////

# 1st Method: Let's create a column named Total calls and paste it into the dataframe:

total_calls = (
df["Total day calls"]
+ df["Total eve calls"]
+ df["Total night calls"]
+ df["Total intl calls"]
 )
df.insert(loc=len(df.columns), column="Total calls", value=total_calls)
# print(df.head())

# 2nd method:

# It is possible to add a column more easily without having to create an intermediate instance:
df["Total charge"]= (
df["Total day charge"]
+df["Total night charge"]
+df["Total eve charge"]
+df["Total intl charge"]
)
#(df.head())

# DELETING COLUMNS OR ROWS: /////////
# In order to delete or drop columns, we should use the drop method and pass the required indexes, and the axis
# parameters.(1 if you wanto delete columns and nothing or 0 if you wanto delete rows)
# The inplace argument tells whether to change the original DataFrame or not. (True replaces the original DF and false
# gives us another DF with dropped columns.)

df.drop(["Total charge","Total calls"], axis=1, inplace=True)

# And here's how to delete rows 1 and 2 :

df.drop([1, 2]).head()

# CHURN PREDICTIONS ////////////

# Lets see how churn is relationed to the international plan feature.

#print(pd.crosstab(df["Churn"], df["International plan"], margins=True))


import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.dpi'] = 150    #->>>>> Makes the visuals clearer

# sns.countplot(x="International plan", hue="Churn", data=df)   ->>>>>>>>> We clearly see that international plans
# plt.show()                                                               have a much higher churn rate. Ask ourselves
#                                                                          why?

# Lets take a look at another important feature:
#pd.crosstab(df["Churn"], df["Customer service calls"], margins=True)
#sns.countplot(x="Customer service calls", hue="Churn", data=df);      #->>>>>>>>>> Here we can see that when 4 customer
#plt.show()                                                            #->>>>>>>>>> calls is reached the number of
                                                                       #->>>>>>>>>> churns increase significantly.








