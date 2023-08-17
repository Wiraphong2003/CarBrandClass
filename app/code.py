import pickle
import json
brand = {
    0:"Audi",
    1:"Hyundai Creta",
    2:"Mahindra Scorpio",
    3:"Rolls Royce",
    4:"Swift",
    5:"Tata Safari",
    6:"Toyota Innova"
}

def predictcar(m,HOG):
    result = m.predict(HOG)
    return brand[result[0]]


# with open("app\hogtest.json", "r") as json_file:
#     data = json.load(json_file)


# hots = data['Hog']
# m = pickle.load(open(r'model/image_modelv2.pk', 'rb'))
# print(predictcar(m,[hots]))

