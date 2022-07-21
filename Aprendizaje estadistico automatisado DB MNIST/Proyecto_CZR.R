#            Aprendizaje Estad\'istico Automatizado
#            Proyecto final: Redes Neuronales (NN).
#            Profesora: Guillermina Eslava G\'omez
#               Alumnos:  Alarc\'on-Gonz\'alez, 
#                         Cerezo-Silva,
#                         Zarco-Romero.
#
library(h2o)
library(randomForest)
library(caret) # para validación cruzada
library(xtable)
h2o.init()
#h2o.shutdown(prompt = FALSE) #por si se acaba la memoria de java.

# Función para exportar las tablas a latex
kabla <- function(data,title=NULL,hold=TRUE,row.names=FALSE){
  library(knitr)
  library(kableExtra)
  if(hold) {
    kable(data, "latex",
          caption = title,
          row.names = row.names,
          align = "c",
          booktabs = T) %>%
      kable_styling(latex_options = c("striped", "hold_position"))
  } else{
    kable(data, "latex",
          caption = title,
          row.names = row.names,
          align = "c",
          booktabs = T) %>%
      kable_styling(latex_options = c("striped"))
  }
}

#Random forest-------------------------------------------------------------------------

#Preliminares: Lectura de las bases de datos.

MNISTtest=read.csv("MNISTtest_9000.csv")
MNISTtrain=read.csv("MNISTtrain_40000.csv")
MNISTvalidate=read.csv("MNISTvalidate_11000.csv")

MNISTtrain$C785 = factor(MNISTtrain$C785)

MNISTtest.h2o=as.h2o(MNISTtest)
MNISTtrain.h2o=as.h2o(MNISTtrain)
MNISTvalidate.h2o=as.h2o(MNISTvalidate)

#Guardaremos los errores.
errores=matrix(NA, ncol=8, nrow=11)
rownames(errores)=c("0","1","2","3","4",
                    "5","6","7","8","9","Global")
colnames(errores)=c("Ap mtries=26","Tst mtries=26",
                    "Ap mtries=28","Tst mtries=28",
                    "Ap mtries=30","Tst mtries=30",
                    "Ap mtries=100","Tst mtries=100")

#I. Calibramos mtries. Modelos están ordenados de acuerdo a mtries.
set.seed(1)
train.rf.2=h2o.randomForest(y=785, training_frame = MNISTtrain.h2o, ntrees=500, mtries = 26)
train.rf.0=h2o.randomForest(y=785, training_frame = MNISTtrain.h2o, ntrees=500, mtries = 28) #sqrt(714)=28
train.rf.3=h2o.randomForest(y=785, training_frame = MNISTtrain.h2o, ntrees=500, mtries = 30)
train.rf.1=h2o.randomForest(y=785, training_frame = MNISTtrain.h2o, ntrees=500, mtries = 100, max_depth = 10)
train.rf.7=h2o.randomForest(y=785, training_frame = MNISTtrain.h2o, ntrees=500, mtries = 714)
#Mejor desempenio con mtries = 28.

#Prueba
set.seed(1)
rf.pred.0=predict(train.rf.0, newdata=MNISTtest.h2o, type="response")
rf.pred.1=predict(train.rf.1, newdata=MNISTtest.h2o, type="response")
rf.pred.2=predict(train.rf.2, newdata=MNISTtest.h2o, type="response")
rf.pred.3=predict(train.rf.3, newdata=MNISTtest.h2o, type="response")

conf.rf.0=table(MNISTtest$C785, as.vector(rf.pred.0[,1]))
conf.rf.1=table(MNISTtest$C785, as.vector(rf.pred.1[,1]))
conf.rf.2=table(MNISTtest$C785, as.vector(rf.pred.2[,1]))
conf.rf.3=table(MNISTtest$C785, as.vector(rf.pred.3[,1]))

#Errores aparentes.
for(i in 1:11){
  errores[i,1]=h2o.confusionMatrix(train.rf.2)[i,11]
  errores[i,3]=h2o.confusionMatrix(train.rf.0)[i,11]
  errores[i,5]=h2o.confusionMatrix(train.rf.3)[i,11]
  errores[i,7]=h2o.confusionMatrix(train.rf.1)[i,11]
}
#Test errors.
for(i in 1:10){
  errores[i,2]=1-conf.rf.2[i,i]/sum(conf.rf.2[i,])
  errores[i,4]=1-conf.rf.0[i,i]/sum(conf.rf.0[i,])
  errores[i,6]=1-conf.rf.3[i,i]/sum(conf.rf.3[i,])
  errores[i,8]=1-conf.rf.1[i,i]/sum(conf.rf.1[i,])
}
#Global test errors.
errores[11,2]=1-sum(diag(conf.rf.2))/sum(conf.rf.2) 
errores[11,4]=1-sum(diag(conf.rf.0))/sum(conf.rf.0) 
errores[11,6]=1-sum(diag(conf.rf.3))/sum(conf.rf.3)
errores[11,8]=1-sum(diag(conf.rf.1))/sum(conf.rf.1)

kabla(round(t(errores*100), digits=2)) #Exportamos a latex.

#II. Calibramos max_depth:

err.max_d=matrix(NA, ncol=6, nrow=11)
rownames(err.max_d)=c("0","1","2","3","4",
                      "5","6","7","8","9","Global")
colnames(err.max_d)=c("Ap m_depth=10","Tst m_depth=10",
                      "Ap m_depth=20","Tst m_depth=20",   #20=default
                      "Ap m_depth=21","Tst m_depth=21")

#Entrenamiento.
train.rf.4=h2o.randomForest(y=785,training_frame = MNISTtrain.h2o, ntrees=500, mtries = 28, max_depth=10) 
train.rf.5=h2o.randomForest(y=785, training_frame = MNISTtrain.h2o, ntrees=500, mtries = 28, max_depth=21) 

#Prueba
set.seed(1)
rf.pred.4=predict(train.rf.4, newdata=MNISTtest.h2o, type="response")
rf.pred.5=predict(train.rf.5, newdata=MNISTtest.h2o, type="response")

conf.rf.4=table(MNISTtest$C785, as.vector(rf.pred.4[,1]))
conf.rf.5=table(MNISTtest$C785, as.vector(rf.pred.5[,1]))

#Errores aparentes.
for(i in 1:11){
  err.max_d[i,1]=h2o.confusionMatrix(train.rf.4)[i,11] #max_depth=10
  err.max_d[i,3]=h2o.confusionMatrix(train.rf.0)[i,11] #max_depth=20
  err.max_d[i,5]=h2o.confusionMatrix(train.rf.5)[i,11] #max_depth=21
}
#Test errors.
for(i in 1:10){
  err.max_d[i,2]=1-conf.rf.4[i,i]/sum(conf.rf.4[i,]) #max_depth=10
  err.max_d[i,4]=1-conf.rf.0[i,i]/sum(conf.rf.0[i,]) #max_depth=20
  err.max_d[i,6]=1-conf.rf.0[i,i]/sum(conf.rf.5[i,]) #max_depth=21
}
#Global test errors.
err.max_d[11,2]=1-sum(diag(conf.rf.4))/sum(conf.rf.4) 
err.max_d[11,4]=1-sum(diag(conf.rf.0))/sum(conf.rf.0) 
err.max_d[11,6]=1-sum(diag(conf.rf.5))/sum(conf.rf.5) 

df_maxdepth=kabla(data.frame(round(t(err.max_d[,c(5,3,1)]*100), digits=2))) #Aparentes max_depth
df_maxdepth=kabla(data.frame(round(t(err.max_d[,c(6,4,2)]*100), digits=2))) #Test max_depth

round(err.max_d*100, digits=3)

#III. Aumentamos el número de árboles:
#Guardaremos los errores.
err_ntrees=matrix(NA, ncol=6, nrow=11)
rownames(err_ntrees)=c("0","1","2","3","4",
                       "5","6","7","8","9","Global")
colnames(err_ntrees)=c("Ap ntrees=500","Tst ntrees=500",
                       "Ap ntrees=800","Tst ntrees=800",
                       "Ap ntrees=1000","Tst ntrees=1000")

train.rf.6=h2o.randomForest(y=785, training_frame = MNISTtrain.h2o, ntrees=800, mtries = 28) 
train.rf.8=h2o.randomForest(y=785, training_frame = MNISTtrain.h2o, ntrees=1000, mtries = 28) 

#Prueba
set.seed(1)
rf.pred.6=predict(train.rf.6, newdata=MNISTtest.h2o, type="response")
rf.pred.8=predict(train.rf.6, newdata=MNISTtest.h2o, type="response")

conf.rf.6=table(MNISTtest$C785, as.vector(rf.pred.6[,1]))
conf.rf.8=table(MNISTtest$C785, as.vector(rf.pred.8[,1]))

#Errores aparentes.
for(i in 1:11){
  err_ntrees[i,1]=h2o.confusionMatrix(train.rf.0)[i,11]
  err_ntrees[i,3]=h2o.confusionMatrix(train.rf.6)[i,11]
  err_ntrees[i,5]=h2o.confusionMatrix(train.rf.8)[i,11]
}
#Test errors.
for(i in 1:10){
  err_ntrees[i,2]=1-conf.rf.0[i,i]/sum(conf.rf.0[i,])
  err_ntrees[i,4]=1-conf.rf.6[i,i]/sum(conf.rf.6[i,])
  err_ntrees[i,6]=1-conf.rf.6[i,i]/sum(conf.rf.8[i,])
}
#Global test errors.
err_ntrees[11,2]=1-sum(diag(conf.rf.0))/sum(conf.rf.0) 
err_ntrees[11,4]=1-sum(diag(conf.rf.6))/sum(conf.rf.6) 
err_ntrees[11,6]=1-sum(diag(conf.rf.8))/sum(conf.rf.8) 

round(err_ntrees*100, digits=3)

#IV. Cambiamos de libreria a randomforest().

train.rf.9=randomForest(C785~.,data=MNISTtrain,ntree=500,mtry=28)
train.rf.10=randomForest(C785~.,data=MNISTtrain,ntree=1000,mtry=28, strata=MNISTtrain$C785)
train.rf.11=randomForest(C785~.,data=MNISTtrain,ntree=800,mtry=28)

round(train.rf.10$confusion[,11]*100, digits=2)
1-sum(diag(train.rf.10$confusion))/sum(train.rf.10$confusion) #0.03355815 error global.

rf.pred.9.train=predict(train.rf.9, newdata=MNISTtrain, type="response")
rf.pred.10.train=predict(train.rf.10, newdata=MNISTtrain, type="response")
rf.pred.11.train=predict(train.rf.11, newdata=MNISTtrain, type="response")

#conf.rf.9.train=table(MNISTtrain$C785, rf.pred.9.train) #ajusta todo bien.
#1-sum(diag(conf.rf.9.train))/sum(conf.rf.9.train) #da cero

kabla(round(t(train.rf.9$confusion[,11])*100, digits=2))

train.rf.9$err.rate[1000]                       #oob errors 0.033625
round(train.rf.10$err.rate[1000]*100, digits=2) #oob errors 0.03345
train.rf.11$err.rate[800] #oob errors 

rf.pred.9.test=predict(train.rf.9, newdata=MNISTtest, type="response")
rf.pred.10.test=predict(train.rf.10, newdata=MNISTtest, type="response")
rf.pred.11.test=predict(train.rf.11, newdata=MNISTtest, type="response")

conf.rf.9.test=table(MNISTtest$C785, rf.pred.9.test)
conf.rf.10.test=table(MNISTtest$C785, rf.pred.10.test)
conf.rf.11.test=table(MNISTtest$C785, rf.pred.11.test)

err.tst=matrix(NA, ncol=3, nrow=11)
rownames(err.tst)=c("0","1","2","3","4",
                    "5","6","7","8","9","Global")
colnames(err.tst)=c("Test ntree=1000", "Test ntree=1000 strata", "Test ntree=800")
err.tst
kabla(cbind(3.36,3.14,3.65))
#Test errors.
for(i in 1:10){
  err.tst[i,1]=1-conf.rf.9.test[i,i]/sum(conf.rf.9.test[i,])
  err.tst[i,2]=1-conf.rf.10.test[i,i]/sum(conf.rf.10.test[i,])
  err.tst[i,3]=1-conf.rf.11.test[i,i]/sum(conf.rf.11.test[i,])
}
#Global test errors.
err.tst[11,1]=1-sum(diag(conf.rf.9.test))/sum(conf.rf.9.test)   #0.03144444
err.tst[11,2]=1-sum(diag(conf.rf.10.test))/sum(conf.rf.10.test) #0.03233333
err.tst[11,3]=1-sum(diag(conf.rf.11.test))/sum(conf.rf.11.test) #0.03211111

kabla(round(t(err.tst)*100,digits=2))

#Validate
validate=predict(train.rf.9, newdata=MNISTvalidate, type="response")
validate=as.vector(validate)
write.csv(validate, file = "Proyecto_Alarcon_RF_pred.csv" )

#Validacion cruzada. 

set.seed(2021)
k=3
err.cv=numeric(k)
folds=createFolds(MNISTtrain$C785, k = k, list = TRUE, returnTrain = F)
for(i in 1:k){
  rf_classifier = randomForest(C785~.,data=MNISTtrain[-folds[[i]],],ntree= 1000,mtry= 28)
  rf.pred.9.cv = predict(rf_classifier,MNISTtrain[folds[[i]],])
  conf.rf.9.cv = table(MNISTtrain[folds[[i]],]$C785,rf.pred.9.cv)
  err.cv[i]=1-sum(diag(conf.rf.9.cv))/sum(conf.rf.9.cv) 
}
round(mean(err.cv)*100,digits=2) #Error CV: 3.65

#-------FIN---------------------------------------------------------------------
