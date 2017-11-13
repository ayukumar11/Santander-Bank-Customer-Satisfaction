install.packages("caret")
library(caret)
install.packages("randomForest")
library(randomForest)

#Reading the Santander Bank data
SantanderTrain= read.csv("train.csv")
View(head(SantanderTrain))
table(SantanderTrain$TARGET)
dim(SantanderTrain)
#Checking missing values, we can use str as well but there are 371 variables in this data
colSums(is.na(SantanderTrain))
#We see that there are no missing values

####using preProcess to eliminate unnecessary variables that do not contribute to the target variable
dim(SantanderTrain)
class(SantanderTrain)

prep= preProcess(SantanderTrain,method = "zv")
train=predict(prep,SantanderTrain)
dim(train)
#We are left with 337 variables

#Target variable
table(train$TARGET)
#We notice that the data is highly unbalanced

#To understand the type of the target variable
str(train$TARGET)
#Convert the Target variable to a factor
train$TARGET<- as.factor(train$TARGET)
#Install the Data Mining with R package
install.packages("DMwR")
library(DMwR)
###SMOTE (Synthetic minority oversampling technique) to balance the dataset
attach(train)
Samplesan= SMOTE(TARGET~.,data = train,perc.over = 200,k=5,perc.under = 200,learner = NULL)
table(Samplesan$TARGET)
dim(Samplesan)
class(Samplesan)
#ALTERNATE Sampling technique
#we can also do a stratified sampling without replacement
#install.packages("sampling")
#library(sampling)
#StratifiedTrain= strata(train,stratanames="TARGET",size=c(3008,3008), method = "srswor",T)
#StrSample<-getdata(train,StratifiedTrain)
#table(StrSample$TARGET)
#nrow(StrSample)
#dim(StrSample) 
#6016 rows and 340 variables
 
# removing prob, id_unit, stratum from StrSample Dataset

#install.packages("dplyr")
#library(dplyr)

#StrSample <- StrSample%>%
 # select(-ID_unit,-Stratum,-Prob,-ID)
#View(head(StrSample))

#Removing Zero Variance columns from the SMOTE sample
prep= preProcess(Samplesan,method = "zv")
SantanderTrain_Sample=predict(prep,Samplesan)
dim(SantanderTrain_Sample)
#We are left with 293 variables from 371
class(SantanderTrain_Sample)

## creating a data.frame will all the numeric variables
str(SantanderTrain_Sample)
View(SantanderTrain_Sample)
num<- 
allnumdf <- SantanderTrain_Sample[,(sapply(SantanderTrain_Sample,is.numeric))]
str(allnumdf)
class(allnumdf)

#Finding Correlation and removing variables that are more than 70% correlated in the stratified sample
correlation <- cor(allnumdf)
View(correlation)
highcorrelation <- findCorrelation(correlation,cutoff=0.7,verbose=T)
str(highcorrelation)

# to remove highly correlated variables from the sample data set 
santander2 <- SantanderTrain_Sample[,-c(highcorrelation)]
ncol(santander2)
# Number of columns after removing 0 variance columns, and highly correlated variables : 81
View(santander2)

################### Important variables selection #######################
#########################################################################

# To get area under the ROC curve for each predictor, filterVarImp can be used
Imp_variable_scoring <- filterVarImp(x = santander2[, -ncol(santander2)], y = santander2$TARGET)
top_50_imp_variables <- tail(Imp_variable_scoring,50) # selected top 50 variables based on the scoring

View(top_50_imp_variables)
write.csv(top_50_imp_variables,"top.csv") #exporting csv for the reference 

########################################################################
###### Splitting data into train and validation data set : 80-20 #######
########################################################################

partition <- createDataPartition(santander2$TARGET, p = 0.8,list = F,times = 1)
train <- Samplesan[partition,]
test <- Samplesan[-partition,]
summary(train$TARGET)
nrow(train)  # Number of Records = 16846
nrow(test)   # Number of Records = 4210

trainnew<- train[,c("saldo_var25",
                    "saldo_var32",
                    "saldo_var37",
                    "delta_imp_aport_var17_1y3",
                    "delta_imp_compra_var44_1y3",
                    "delta_num_aport_var33_1y3",
                    "delta_num_venta_var44_1y3",
                    "imp_aport_var13_ult1",
                    "imp_aport_var33_ult1",
                    "imp_var7_recib_ult1",
                    "imp_reemb_var13_ult1",
                    "imp_var43_emit_ult1",
                    "imp_trasp_var17_in_ult1",
                    "ind_var7_recib_ult1",
                    "ind_var43_recib_ult1",
                    "var21",
                    "num_compra_var44_hace3",
                    "num_ent_var16_ult1",
                    "num_var22_hace2",
                    "num_var22_hace3",
                    "num_var22_ult1",
                    "num_meses_var5_ult3",
                    "num_meses_var17_ult3",
                    "num_meses_var39_vig_ult3",
                    "num_op_var40_comer_ult1",
                    "num_op_var41_efect_ult1",
                    "num_reemb_var17_ult1",
                    "num_reemb_var33_ult1",
                    "num_sal_var16_ult1",
                    "num_var43_recib_ult1",
                    "num_trasp_var11_ult1",
                    "num_trasp_var33_in_hace3",
                    "num_trasp_var33_in_ult1",
                    "num_venta_var44_ult1",
                    "num_var45_hace3",
                    "num_var45_ult1",
                    "saldo_medio_var5_hace2",
                    "saldo_medio_var5_hace3",
                    "saldo_medio_var8_hace2",
                    "saldo_medio_var8_hace3",
                    "saldo_medio_var8_ult1",
                    "saldo_medio_var12_hace2",
                    "saldo_medio_var12_hace3",
                    "saldo_medio_var13_corto_hace3",
                    "saldo_medio_var13_largo_hace3",
                    "saldo_medio_var17_hace2",
                    "saldo_medio_var17_hace3",
                    "saldo_medio_var17_ult1",
                    "saldo_medio_var29_ult3",
                    "var38","TARGET")]

View(trainnew)


install.packages("e1071")
library(e1071)

#######################################################################################################
##### Now, we use Recursive Feature Elimination (RFE) technique to reduce the number of variables #####
#######################################################################################################
Subset <- c(1:8, 10, 15, 20, 25)
control= rfeControl(functions = rfFuncs,method = "cv",number = 3)
results<-rfe(trainnew[,1:50],trainnew[,51],sizes=Subset,rfeControl = control)

plot(results)
#The predictors function can be used to get a text string of variable names that were picked in the final model. 
predictors(results)

#########################################################
################ Random Forest Model ####################
#########################################################

fitControl<- trainControl(method="cv",number= 5,classProbs = T,
                          summaryFunction= twoClassSummary,allowParallel = T)
tunegrid= expand.grid(mtry=c(10:25))
metric="ROC"
str(train$TARGET)
levels(train$TARGET)[levels(train$TARGET)==1] <- "Class_1"
levels(train$TARGET)[levels(train$TARGET)==0] <- "Class_0"
attach(train)
rf_train<- train(TARGET~ num_var45_hace3+ num_meses_var5_ult3          
                 +num_meses_var39_vig_ult3 + var38                        
                 + saldo_medio_var5_hace2       + ind_var43_recib_ult1         
                 + saldo_medio_var12_hace2       +saldo_medio_var5_hace3       
                 + num_var45_ult1                +num_var22_hace2              
                 + num_trasp_var11_ult1          +num_var43_recib_ult1         
                 + saldo_var37                   +imp_aport_var13_ult1         
                 + imp_var43_emit_ult1          +num_var22_hace3              
                 + saldo_medio_var13_corto_hace3 +saldo_medio_var8_ult1        
                 + saldo_medio_var12_hace3       + num_var22_ult1               
                 + num_op_var41_efect_ult1       + num_ent_var16_ult1           
                 + saldo_var25                   + saldo_medio_var8_hace2       
                 + var21  , data= train,
                 method='rf',metric=metric,
                 ntree=100,trControl=fitControl,
                 tuneGrid=tunegrid)


plot(rf_train)
#mtry value of 18 gives the best AUC, 0.90
rf_predict <- predict(rf_train,newdata=test)
confusionMatrix = table(test$TARGET,rf_predict)
confusionMatrix
Accuracy3 <- (confusionMatrix[1,1] + confusionMatrix[2,2])/sum(confusionMatrix)
Accuracy3 
#We get a 84.18% accuracy on the test data
Precision= confusionMatrix[2,2]/sum(confusionMatrix[2,])
Precision
#We get a precision of 72.67%