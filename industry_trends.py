"""Comparing retail sale trends in different industries

This file contains functions and classes to plot the trends in different industries both before and
over the course of the pandemic.

Instructions
===============================
To visualize the graphs, run main.py.

Copyright and Usage Information
===============================

This file is Copyright (c) 2021 Anudev Gill.
"""
import csv
import pandas as pd
import plotly.graph_objects as go
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np


class InvalidDateException(Exception):
    """Exception raised when accessing an invalid date for an industry's retail trade sales.
    """
    def __str__(self) -> str:
        """Return a string representation of this error.
        """
        return 'invalid date accessed'


class IndustryData:
    """A custom data type that represents the monthly retail trade sales for a given industry.

    Instance Attributes:
      - name: the name of the industry

    Representation Invariants:
      - self.name != ''
    """
    name: str

    # Private Instance Attributes:
    #   - _months: the months that data has been recorded for
    #   - _sales: the value of the monthly retail trade sales in Canadian dollars
    _months: list
    _sales: list

    def __init__(self, name: str, filename: str) -> None:
        """Initialize the data for an industry.
        """
        self.name = name

        # Initialize _months and _sales to be empty lists, then call _load_data and clean_data to
        # populate them with the correct values
        self._months = []
        self._sales = []

        self._load_data(filename)
        self._clean_data()

    def _load_data(self, filename: str) -> None:
        """Load the data from filename and store it in the instance attributes.
        """
        with open(filename) as f:
            reader = csv.reader(f, delimiter=',')

            # Load the values from the file to _months and _sales without changing anything. Each
            # file is only 2 lines long, with the first line containing the months and
            # the second line containing the sales corresponding to each month
            months = next(reader)
            self._months.extend(months)

            sales = next(reader)
            self._sales.extend(sales)

    def _clean_data(self) -> None:
        """Clean the data in the instance attributes by converting it to a usable form.
        """
        # Remove the first value from _months and _sales as this is the header for each row
        self._months.pop(0)
        self._sales.pop(0)

        new_sales = []

        for sale in self._sales:
            # Months that are missing retail sale data are represented with the string '..'
            if sale != '..':
                # Some of the sale figures in the data have a letter at the end representing the
                # quality of the data. In this case, this character is excluded when getting the
                # numerical representation of the figure.
                if sale[-1] in 'ABCDEF':
                    value = sale[0:-1]
                else:
                    value = sale

                # Remove commas from the sales figure data before adding its int representation to
                # the new_sales list
                value = value.replace(',', '')
                new_sales.append(int(value))
            # If a given month does not have retail sale data, remove it from the _months list
            else:
                self._months.pop(0)

        # Reassign _sales to new_sales, which contains the cleaned sale data
        self._sales = new_sales

    def get_date(self, date: int) -> tuple[list, list]:
        """Return a tuple of the list of months and the list of sales corresponding to those months
        for all months after date, which is a year-value, inclusive.
        """
        # If data does not exist for the requested date raise an InvalidDateException
        if date < int(self._months[0][-4:]) or date > 2021:
            raise InvalidDateException
        else:
            months_after_date = []
            sales_after_date = []

            for i in range(len(self._months)):
                # Check if the month being accessed is greater than or equal to date by comparing
                # the last 4 characters, which contain the year (e.g. 'January 2014')
                if int(self._months[i][-4:]) >= date:
                    months_after_date.append(self._months[i])
                    sales_after_date.append(self._sales[i])

            return months_after_date, sales_after_date


def plot_industry_data(industry: IndustryData) -> None:
    """Plot the monthly retail trade sales for industry as a line graph.
    """
    # Set the data on the x-axis to be the months and the data on the y-axis to be the retail trade
    # sales. 2004 is used as the starting year since it is the first year that all industries have
    # recorded data for
    x_data, y_data = industry.get_date(2004)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data, y=y_data, name=industry.name))

    fig.update_layout(title=f'Monthly Retail Trade Sales for {industry.name}',
                      xaxis_title='(Month, Year)',
                      yaxis_title='Retail Trade Sales (in Canadian dollars)')

    fig.show()

    # If the above line (e.g. fig.show()) does not work, comment it out, and uncomment the
    # following:

    # fig.write_html('my_figure.html')

    # You will need to manually open the my_figure.html file created above, which will be created
    # in the same directory as this file.


def plot_all_industries(industry_list: list[IndustryData]) -> None:
    """Plot the monthly retail trade sales for each industry in industry_list as a line graph on the
    same axis.
    """
    fig = go.Figure()

    # Create a trace for each industry in industry_list
    for industry in industry_list:
        # Set the data on the x-axis to be the months and the data on the y-axis to be the retail
        # trade sales. 2004 is used as the starting year since it is the first year that all
        # industries have recorded data for
        x_data, y_data = industry.get_date(2004)
        fig.add_trace(go.Scatter(x=x_data, y=y_data, name=industry.name))

    fig.update_layout(title='Monthly Retail Trade Sale Comparison for All Industries',
                      xaxis_title='(Month, Year)',
                      yaxis_title='Retail Trade Sales (in Canadian dollars)')

    fig.show()

    # If the above line (e.g. fig.show()) does not work, comment it out, and uncomment the
    # following:

    # fig.write_html('my_figure.html')

    # You will need to manually open the my_figure.html file created above, which will be created
    # in the same directory as this file.


def industry_trend(industry: IndustryData) -> None:
    """Create and plot a linear regression model for the retail trade sales for industry.
    """
    # Get the months and retail trade sales data lists for industry from 2004 onwards
    data_lists = industry.get_date(2004)

    # Create a list mapping integers to the months data list. This will be used in place of the
    # months list since scikit-learn linear_model requires that both the x- and y-axis data be
    # numeric whereas the months list is a list of strings
    num_of_dates = list(range(0, len(data_lists[0])))

    # Create a 2-dimensional list of the x- and y-axis data and pass it to pd.DataFrame to create
    # a pandas DataFrame
    data = [num_of_dates, data_lists[1]]
    df = pd.DataFrame(data)

    # Get the x- and y-axis data by inverting the DataFrame into a 2-column table and using the
    # headings
    x_data = df.T[0]
    y_data = df.T[1]

    # Reshape the x_data array to use only one feature
    x_data = x_data[:, np.newaxis]

    # Split the data into training and testing sets. The months from February 2004 to December 2014
    # will be used as training data and the months from January 2015 to September 2021 will be
    # used as testing data
    x_data_train = x_data[1:132]
    y_data_train = y_data[1:132]

    x_data_test = x_data[132:]
    y_data_test = y_data[132:]

    # Create a linear regression object called regression
    regression = linear_model.LinearRegression()

    # Train the linear regression object using the training data sets
    regression.fit(x_data_train, y_data_train)

    # Use the linear regression object to create a prediction for the testing x_data dataset
    y_data_prediction = regression.predict(x_data_test)

    # Plot the actual values of the data for the test dataset as a scatter plot
    plt.scatter(x_data_test, y_data_test, color='black')
    # Plot the predicted values/regression for the test dataset as a line
    plt.plot(x_data_test, y_data_prediction, color='blue', linewidth=3)

    # Format the graph
    plt.xticks()
    plt.yticks()

    plt.title('Retail Trade Sales from Jan. 2015 to Sept. 2021 for ' + industry.name)
    plt.xlabel('Number of months since Jan. 2004')
    plt.ylabel('Retail Trade Sales (in Canadian dollars)')

    plt.show()


def all_industry_trends(industry_list: list[IndustryData]) -> None:
    """Create and plot a linear regression model for the retail trade sales for each industry in
    industry_list on the same graph."""
    # Loop through every industry in industry_list and call industry_trend on it to plot the
    # data and the linear regression models for each industry on the same plot
    for industry in industry_list:
        industry_trend(industry)


if __name__ == '__main__':
    ################################################################################
    # BELOW IS THE CODE TO RUN PYTHON-TA ON THIS FILE.
    ################################################################################
    # import doctest
    # doctest.testmod()
    #
    # import python_ta
    # import python_ta.contracts
    #
    # python_ta.contracts.DEBUG_CONTRACTS = False
    # python_ta.contracts.check_all_contracts()
    #
    # python_ta.check_all(config={
    #     'extra-imports': ['csv', 'pandas', 'plotly.graph_objects', 'sklearn', 'matplotlib.pyplot',
    #                       'numpy'],
    #     'allowed-io': ['_load_data'],
    #     'max-line-length': 100,
    #     'max-nested-blocks': 4,
    #     'disable': ['R1705', 'C0200']
    # })

    #################################################################################
    # Industry Trends
    #################################################################################

    # Create IndustryData objects for each industry using the appropriate csv files
    automobile = IndustryData('Automobile Industry', 'automobile.csv')
    liquor = IndustryData('Liquor Industry', 'beer_wine_liquor.csv')
    furniture = IndustryData('Furniture Industry', 'furniture.csv')
    gas_stations = IndustryData('Gas Stations', 'gas_stations.csv')
    health = IndustryData('Health Industry', 'health.csv')
    merchandise = IndustryData('Merchandise Industry', 'merchandise.csv')
    restaurants = IndustryData('Restaurant Industry', 'specialty_food.csv')
    sporting = IndustryData('Sporting Industry', 'sporting_and_hobby.csv')
    grocery_stores = IndustryData('Grocery Stores', 'supermarkets.csv')

    # Create a list of all the industries, which is useful for the below functions
    list_of_industries = [automobile, liquor, furniture, gas_stations, health, merchandise,
                          restaurants, sporting, grocery_stores]

    # Plot the retail trade sales of each industry on the same plot. This will open in a new browser
    # tab
    plot_all_industries(list_of_industries)

    # FURTHER INSTRUCTIONS:

    # Uncomment the below lines (lines 96-104) one-at-a-time to see the various linear regression
    # models for each industry. Make sure that at any given time only one of the lines of code is
    # uncommented (e.g. if line 96 is uncommented then lines 97-104 should be commented out. Each
    # model will open in in a new window.

    industry_trend(automobile)
    # industry_trend(liquor)
    # industry_trend(furniture)
    # industry_trend(gas_stations)
    # industry_trend(health)
    # industry_trend(merchandise)
    # industry_trend(restaurants)
    # industry_trend(sporting)
    # industry_trend(grocery_stores)

    # To plot all the linear regression models on the same graph, make sure the code on lines
    # 96-104 is commented out and uncomment the below code.

    # all_industry_trends(list_of_industries)
