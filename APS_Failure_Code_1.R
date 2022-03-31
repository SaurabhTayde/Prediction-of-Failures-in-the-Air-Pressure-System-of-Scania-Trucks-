#setwd('D:/Saurabh T/R/Project/APS')
setwd('G:/DSP/Project/Scania Trucks')

############### Import the dataset (Skip unnecessaory rows at start):

training_data = read.csv('aps_failure_training_set.csv', skip =20, na.strings = 'na')

test_data = read.csv('aps_failure_test_set.csv', skip = 20, na.strings = 'na')


############### Check the dimensions of both data sets:

dim(training_data)
dim(test_data)


############### Import libraries required:

library(dplyr)
library(caret)

install.packages('caretEnsemble')
library(caretEnsemble)

install.packages('mice')
library(mice)

install.packages('doParallel')
library(doParallel)
library(car)


############### View Data. First look:

glimpse(training_data)
glimpse(test_data)


############### Chcking class imbalance:

summary(training_data$class)
summary(test_data$class)

prop.table(table(training_data$class))  #Class is totally imbalanced.
prop.table(table(test_data$class))      #Class is totally imbalanced.


############## summary :

summary(training_data)
summary(test_data)

colSums(is.na(training_data))  # Too many misssing values
colSums(is.na(test_data))     # Too many misssing values

sum(colSums(is.na(training_data)))  # 850015 
sum(colSums(is.na(test_data)))      # 228680

    #Data has large missing values. So we cannot remove rows with missing values.
    #We will deal with missing values through MICE package


#############  Using MICE package for replacing missing values:

#We don't want to use a technique to impute each of our two sets separately (train and test).
#It has to be a one time imputation using full information.
#In order to do this, we are going to combine both sets, work on them and then separate again.

training_data_bind <- training_data
test_data_bind <- test_data

#create a new column "set" so that we can separate the dataset later

training_data_bind$set <- "TRAIN"
test_data_bind$set <- "TEST"

#merge them into 1 single set
full_dataset = rbind(training_data_bind, test_data_bind)

dim(full_dataset)

#We end up with a single set containing 76,000 samples (16,000 from test set and 60,000 from train set). 
#The number of columns is 172 (171 features + 1 "set" column)


set.seed(123)
imputed_full <- mice(full_dataset, 
                     m=1, 
                     maxit = 5, 
                     method = "mean", 
                     seed = 500)


#Now we store the imputed values:
full_imputed <- complete(imputed_full, 1)

#We then check that we still maintain the same dimensions:

dim(full_imputed)

#Lets check how many missing values are there in new dataset
sum(colSums(is.na(full_imputed)))  # 42203



(na_count_full_imputed <-data.frame(sapply(full_imputed, function(y) sum(length(which(is.na(y)))))))

#notice there are some features that still have missing values. Specifically there are 8 rows.

issue_columns <- subset(imputed_full$loggedEvents, 
                        meth == "constant" | meth == "collinear")

print(issue_columns)


#So we will remove these 8 columns that are having NA values:

#create vector of column names
issue_columns_names <- as.character(issue_columns[, "out"])
issue_columns_names <- issue_columns_names[-2]
print(issue_columns_names)


#We then use the stored vector to remove those columns from the data frame
#and store it as our final imputed data frame:

full_imputed_filtered <- full_imputed[ , !(names(full_imputed) %in% 
                                             issue_columns_names)]

dim(full_imputed_filtered)  #Notice the number of columns reduced from 172 to 164 !!!!


sum(colSums(is.na(full_imputed_filtered)))  #Now we don't have missing values any more



#Finally, it's time to separate our full imputed dataset into train and test sets again,
#and we need to separate it into the exact same samples that it was splitted before.
#In order to do that we just filter the data frame using our set column.


#subset the full_imputed_filtered dataset

training_data_imp <- subset(full_imputed_filtered, set == "TRAIN")
test_data_imp <- subset(full_imputed_filtered, set == "TEST")

#drop the "set" column, we don't need it anymore
training_data_imp$set <- NULL
test_data_imp$set <- NULL

sum(colSums(is.na(training_data_imp))) 

sum(colSums(is.na(test_data_imp))) 


#check dimensions
dim(training_data_imp)
dim(test_data_imp)


#Convert imputed data into csv for future reference
write.csv(training_data_imp, file = "APSTrain.csv")
write.csv(test_data_imp, file = "APSTest.csv")

# As given on the website, we dont have much outliers and hence we wont be removing them

# Also there is no much multicollinearity issue



head(training_data_imp)
head(test_data_imp)

#Now as there is high class imbalance we will downsample the data

'%ni%' <- Negate('%in%')  # define 'not in' func

down_training_data_imp <- downSample(x = training_data_imp[, colnames(training_data_imp) %ni% "class"],
                         y = training_data_imp$class)

table(down_training_data_imp$Class)
dim(down_training_data_imp)


#################### Model Building



logitmod = glm(Class~., family = 'binomial', data = down_training_data_imp)

summary(logitmod)

install.packages('MASS')

library(MASS)

#stepAIC(logitmod)



logitmod2 = glm(formula = Class ~ aa_000 + ab_000 + ac_000 + ad_000 + ae_000 + 
      af_000 + ag_000 + ag_001 + ag_002 + ag_003 + ag_004 + ag_005 + 
      ag_006 + ag_007 + ag_008 + ag_009 + ai_000 + aj_000 + ak_000 + 
      al_000 + am_0 + an_000 + ao_000 + ap_000 + aq_000 + ar_000 + 
      as_000 + at_000 + au_000 + av_000 + ax_000 + ay_000 + ay_001 + 
      ay_002 + ay_003 + ay_004 + ay_005 + ay_006 + ay_007 + ay_008 + 
      ay_009 + az_000 + az_001 + az_002 + az_003 + az_004 + az_005 + 
      az_006 + az_007 + az_008 + az_009 + ba_000 + ba_001 + ba_002 + 
      ba_003 + ba_004 + ba_005 + ba_006 + ba_007 + ba_008 + ba_009 + 
      bb_000 + bc_000 + bd_000 + be_000 + bf_000 + bg_000 + bh_000 + 
      bi_000 + bj_000 + bk_000 + bl_000 + bm_000 + bn_000 + bp_000 + 
      bq_000 + br_000 + bs_000 + bx_000 + bz_000 + ca_000 + cb_000 + 
      cc_000 + ce_000 + cg_000 + ch_000 + ci_000 + cj_000 + ck_000 + 
      cl_000 + cm_000 + cn_000 + cn_001 + cn_002 + cn_003 + cn_004 + 
      cn_005 + cn_006 + cn_007 + cn_008 + cn_009 + cp_000 + cr_000 + 
      cs_000 + cs_001 + cs_002 + cs_003 + cs_004 + cs_005 + cs_006 + 
      cs_007 + cs_008 + cs_009 + ct_000 + cu_000 + cv_000 + cx_000 + 
      cy_000 + cz_000 + da_000 + db_000 + dc_000 + dd_000 + de_000 + 
      df_000 + dg_000 + dh_000 + di_000 + dj_000 + dk_000 + dl_000 + 
      dm_000 + dn_000 + do_000 + dp_000 + dq_000 + dr_000 + ds_000 + 
      dt_000 + du_000 + dv_000 + dx_000 + dy_000 + dz_000 + ea_000 + 
      eb_000 + ec_00 + ed_000 + ee_000 + ee_001 + ee_002 + ee_003 + 
      ee_004 + ee_005 + ee_006 + ee_007 + ee_008 + ee_009 + ef_000 + 
      eg_000, family = "binomial", data = down_training_data_imp)

predictTrain = predict(logitmod, type = 'response')

head(predictTrain)

tapply(predictTrain, training_data_imp$class, mean)
