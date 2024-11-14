import Cocoa
import CreateML
import TabularData

let fileURL = URL(fileURLWithPath: "/Users/raymondkim/Desktop/Stuff/Career/Projects/iOS & Swift/Twittermenti/labelled-apple-reviews.csv")

let data = try DataFrame(contentsOfCSVFile: fileURL)

let (trainingData, testingData) = data.stratifiedSplit(on: "class", by: 0.8)

let parameters = MLTextClassifier.ModelParameters(
    validation: .split(strategy: .automatic),
    algorithm: .transferLearning(.bertEmbedding, revision: 1),
    language: .english
)

let sentimentClassifier = try MLTextClassifier(
    trainingData: trainingData,
    textColumn: "text",
    labelColumn: "class",
    parameters: parameters
)

let trainingAccuracy = (1.0 - sentimentClassifier.trainingMetrics.classificationError) * 100

let validationAccuracy = (1.0 - sentimentClassifier.validationMetrics.classificationError) * 100

let evaluationMetrics = sentimentClassifier.evaluation(on: testingData, textColumn: "text", labelColumn: "class")

let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100

let metadata = MLModelMetadata(author: "Ray Kim", shortDescription: "A model trained to classify sentiment on Tweets", version: "1.0")

try sentimentClassifier.write(to: URL(fileURLWithPath: "/Users/raymondkim/Desktop/Stuff/Career/Projects/iOS & Swift/Twittermenti/Twittermenti/TweetSentimentClassifier.mlmodel"))

try sentimentClassifier.prediction(from: "@Apple is a terrible company")
try sentimentClassifier.prediction(from: "I just found the best restaraunt ever @WaffleHouse")
try sentimentClassifier.prediction(from: "I think @Coke is okay")

