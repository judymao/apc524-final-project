================
Princeton University APC524 Final Project: Developing Trading Strategies in Python
================

Project Description
=============

This project allows users to create portfolios based on various trading strategies. These strategies range from being based in traditional finance (such as the ``Momentum`` strategy) to using machine learning techniques (in the ``MLStrat`` strategy). Users can then backtest the results of the strategies using the methods of the ``Backtest`` class.

Overall, this project allows users to test the effectiveness of various trading strategies, allowing users to trade more successfully in future as well.


Install and Run the Project
=============

To make use of our strategies and backtests, start by cloning the repository, by running the following command:
::
    git clone git@github.com:judymao/apc524-final-project.git

Then, create any Python file and import the desired classes from src.tradestrat. For example, if a momentum-based strategy is desired, the following can be written in the Python file:
::
    from src.tradestrat import Momentum, Backtest

For details on how to use the classes and methods, see ``tutorials.ipynb``.
