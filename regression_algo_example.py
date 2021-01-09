"""
REGRESSION PROBLEM
Ex: Real Estate
USING 'ONE OF THE' REGRESSION ALGORITHM 'LINEAR REGRESSION'
"""
'''
                  Bedroom         Kitchen        PoojaRoom       DiningHall
# House-1-data    1                1              1               1                 20
# House-2-data    2                1              1               1                 30
# House-3-data    2                1              1               1                 30
# MyHouse         1                1              1               0                 What is the price????
'''
from sklearn.linear_model import LinearRegression
my_real_estate_model = LinearRegression()
House_Details = [ [1,1,1,1],[2,1,1,1],[2,1,1,1],[2,1,1,1],[2,1,1,1],[2,1,1,1],[2,1,1,1],[2,1,1,1],[2,1,1,1],[2,1,1,1]]
Price_Details = [20,30,30,30,30,30,30,30,30,30]
my_real_estate_model.fit(House_Details,Price_Details)
my_house_price = my_real_estate_model.predict([[2,2,2,2]])
print(f"my_house_price is : {my_house_price} Lakhs")
print("-"*40)
#-----------------------------------


print("How much accurate is this result ?")
print("-"*40)
#--------------------------
from sklearn.metrics import accuracy_score
result = my_real_estate_model.predict(House_Details)
print("result : ",result)
# compare_result = accuracy_score(result,Price_Details)
# compare_result = compare_result * 100
# print(f"Result of house price is {compare_result}% accurate.")
# print("IMPORTANT NOTE :This accuracy calculated using accuracy_score and tested with 10 houses trained data")
# print("Train model with more data to get more accurate result")
# print("-"*40)
#--------------------------


