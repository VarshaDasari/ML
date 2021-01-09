"""
CLASSIFICATION PROBLEM
"""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1) # Creating object / (OR) Creating Machine Learning Model
# knn, is like a fresh doctor, don't know about previous patient details
# We need to Tell PAST Patient data / also called as Training the module
# Patient-1-data    100     110         120     130     D1
# Patient-2-data    110     120         130     140     D2
all_patient_data = [ [100, 110, 120, 130] , [110,120,130,140], [120, 130, 140, 150], [130, 140, 150, 160], [140, 150, 160, 170]]
mapped_patien_disease_type = ["D1","D2","D3", "D4", "D5"]
knn.fit(all_patient_data,mapped_patien_disease_type) # Fit will train the model.
# New patient       101     111         121     131
# Get disease type of new patient
result = knn.predict([ [131,141,151,161] ])
print("New Patient is having disease : ",result)
print("-"*40)
#--------------------------


print("How much accurate is this result ?")
print("Getting Accuracy score on train/test data")
print("-"*40)
#--------------------------
#result = knn.predict([ [101,111,121,131],[101,111,121,131],[101,111,121,131],[101,111,121,131],[101,111,121,131] ])
#result =["D1","D1","D1","D1","D1","D1","D1",]
# result = knn.predict(all_patient_data)
# result should be equal to mapped_patien_disease_type
from sklearn.metrics import accuracy_score
result = knn.predict([[101,111,121,131], [111,121,131,141],[101,111,121,131], [111,121,131,141], [141,151,161,171] ])
# Check result list is equal to mapped_patien_disease_type
print("result : ",result)
print("-"*40)
#--------------------------


compare_result = accuracy_score(result,mapped_patien_disease_type)
# Get in percentage
compare_result = compare_result * 100
print(f"Result of patient test is {compare_result}% accurate.")
print("IMPORTANCT NOTE :This accuracy calculated using accuracy_score and algorithm used is KNeighborsClassifier")
print("-"*40)
#--------------------------


