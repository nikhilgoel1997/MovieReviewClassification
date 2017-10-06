library(readtext)
library(tm)
library(e1071)
library(dplyr)
library(caret)
#Loading the data set
directory <- system.file("/aclImdb/", package = "readtext")
traindata = readtext(paste0(directory,"/train/*"))
testdata = readtext(paste0(directory,"/test/*"))

#extracting labels from training data as its given in the format [id]_[rating].txt
featuresTrain = traindata[,1]                         #Choosing the filename column
featureVectorTrain = matrix()                          #Creating empty matrix for feature vector
f<-1
for (trainLabel in featuresTrain){
test1 = strsplit(trainLabel,"_")                       #Splitting from "_"
test2 = matrix(unlist(test1),ncol=2,byrow=TRUE)
test3 = test2[,2]                                #Choosing the [rating].txt half
test4 = strsplit(test3,"\\.")                    #Splitting from "."
test5 = matrix(unlist(test4),ncol=2,byrow=TRUE)  
test6 = test5[,1]                                #Choosing [rating]
test6 <- as.numeric(test6)                       #Changing rating to int
if(test6<5){
test6=0} else{                                   #Scaling the rating to 0 - bad or 1 - good
test6=1
}
featureVectorTrain[f]=test6    #Adding each entry to make feature vector
f=f+1
}
traindata[ , "class"] <- featureVectorTrain     #concatenating the feature vector to the train data

#Performing the same process on test data
featuresTest = testdata[,1]
featureVectorTest = matrix()
g<-1
for (testLabel in featuresTest){
test1 = strsplit(testLabel,"_")
test2 = matrix(unlist(test1),ncol=2,byrow=TRUE)
test3 = test2[,2]
test4 = strsplit(test3,"\\.")
test5 = matrix(unlist(test4),ncol=2,byrow=TRUE)
test6 = test5[,1]
test6 <- as.numeric(test6)
if(test6<5){
test6=0} else{
test6=1
}
featureVectorTest[g]=test6
g=g+1
}
testdata[ , "class"] <- featureVectorTest

#Changing class from character to factor
traindata$class <- as.factor(traindata$class)
testdata$class <- as.factor(testdata$class)

#Randomizing the data set
set.seed(1)
traindata <- traindata[sample(nrow(traindata)), ]
traindata <- traindata[sample(nrow(traindata)), ]
traindata <- traindata[sample(nrow(traindata)), ]
traindata <- traindata[sample(nrow(traindata)), ]
traindata <- traindata[sample(nrow(traindata)), ]

testdata <- testdata[sample(nrow(testdata)), ]
testdata <- testdata[sample(nrow(testdata)), ]
testdata <- testdata[sample(nrow(testdata)), ]
testdata <- testdata[sample(nrow(testdata)), ]
testdata <- testdata[sample(nrow(testdata)), ]

#Sampling the data set
trainDataSample <- (traindata[1:1000,c(2,3)])
testDataSample <- (testdata[1:1000,c(2,3)])

#Preprocessing the data
trainVector <- as.vector(trainDataSample$text)
testVector <- as.vector(testDataSample$text)
trainSource <- VectorSource(trainVector)
testSource <- VectorSource(testVector)
trainCorpus <- VCorpus(trainSource)
testCorpus <- VCorpus(testSource)
trainCorpusPreprocessed <- trainCorpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="english")) %>%
  tm_map(stripWhitespace)
trainMatrix <- DocumentTermMatrix(trainCorpusPreprocessed)

testcorpusPreprocessed <- testCorpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)
testMatrix <- DocumentTermMatrix(testcorpusPreprocessed)
#Frequency word dictionary
threeFreq <- findFreqTerms(trainMatrix, 3)
trainMatrixSVM <- DocumentTermMatrix(trainCorpus, control=list(dictionary = threeFreq))
testMatrixSVM <- DocumentTermMatrix(testCorpus, control=list(dictionary = threeFreq))

#SVM
system.time( classifier <- svm(trainMatrixSVM, trainDataSample$class, kernel = "linear",cost=0.01))
system.time( pred <- predict(classifier, testMatrixSVM))

table("Predictions"= pred, "Actual" = testDataSample$class)
conf.mat <- confusionMatrix(pred,testDataSample$class)

