from pyspark.mllib.recommendation import ALS
from numpy import array
from pyspark import SparkContext

# Load and parse the data
sc = SparkContext("yarn-client", "app")
train_data = sc.textFile("ani91/input/traindata")
ratings = train_data.map(lambda line: array([float(x) for x in line.split(',')]))
test_data = sc.textFile("ani91/input/testdata")
test_ratings = test_data.map(lambda line: array([float(x) for x in line.split(',')]))

# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 20
model = ALS.train(ratings, rank, numIterations)

# Evaluate the model on training data
testdata = test_ratings.map(lambda p: (int(p[0]), int(p[1])))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).reduce(lambda x, y: x + y)/ratesAndPreds.count()
print("Mean Squared Error = " + str(MSE))
