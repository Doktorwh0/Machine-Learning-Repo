# Kyle Herbruger
# 4/7/2023

This is my narrative report for my 
EE399 HW1 code.

This program fits two functions to a few
different data sets, and I believe it is
to teach us about how to fit data, as 
well as about overfitting data. In part
i, we started off with fitting a sin
function to a given data set, and then
calculated the least squares error to
see how well it fit.

In part ii we varied the parameters of
the sin function we fit to the data,
and calculated the LSE for each result.
This showed how different parameters
had different impacts on the LSE.
The A and C variables had the greatest
impact, while B and D, which still
greatly impacted the LSE, were almost
insignificant in comparison.

In part iii we trained a 19th degree
polynomial to fit the first 20 data 
points, and then compared its LSE
on the training data to that of the
test data (the last 10 points).
While the training data had a really
good LSE, the test data LSE was worse 
than my mental health.

In part iv we repeated part iii with
the first and last 10 data points as
training data, and the middle 10 as 
the test data. This approach gave
a much better LSE for the test data,
however, it did negatively impact
the LSE for the training data. 

In either case, fitting with the wrong
function can easily cause overfitting
problems.