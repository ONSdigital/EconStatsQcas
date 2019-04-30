from venv.uk.gov.ons.src.baseMethod import baseMethod
import pandas as pd
import numpy as np
myData = pd.read_csv("""\\\\nsdata6\\INGRES_CSAI\\Economic Stats BAU Team\\Transformation\\Developer Folders\\Mike\\testdataforimputation\\initial.csv""")
#print(myData)
class Imputation(baseMethod):
    def printDavid(self):
        print("DAVID!!")

    def __init__(self,*args):
        #TODO
        super().__init__(*args)
        #print("stuff")

    def __str__(self):
       return "This is an imputer"
    #params - target column |imputecolumn|imputationClass|Unique identifier for unit
    #impute function (can call as part of init)

    #Create column names from passed in target and impute columns
    #partitionList is imputationClass + identifier
    #do 'groupby' on partitionList, order by periodColumn

    #Identify interval
        #hardcoded Strings
        #What is the periodicity?
            #if monthly -> add mormalisedPeriod Column
                #fill normalizedPeriod column with the period from periodColumn
                #Call MonthsBetween (we think this will identify and produce a df with all periods data between 2 points)
                #fill nulls with 0(timedifflag n timedifflead
            #if Annual -> add normalisedPeriod Column
                #if one of the first 3 months prevyear is year-1 else, year is year
                #call MonthsBetween(Between this period and the next period I know about)
                #create timedifflag and timefdifflead columns using lagMonDiffA&leadmondiffA(if 12 set to 1, else null)
                #drop lagmondiffa and leedmondiffa and fill nulls with 0 (timedifflag, timedifflead)
            #if quarterly -> add normalisedPeriod Column
                #
                #call monthsBetween
                  #if monthsbetween = periodicity =1 else 0
        #return it
                    #MonthsBetween
                        #create lagNewCol/leadNewCol : containing number of months between period and lead/lag of period(one row before n after, eg if quarterly(201712 itll be 201709 - 201712 - 201803)
                        #essentially, looks for the nearest period in its data before/after current period, then works out the months beween current period and that period.

                    #identify Adjacenttarget
                        #for the target column, give the value of the target column for the preceding and following period.

                    #repartition the data using grouplist and a select of distinct grouplist columns n count(giving us the number of partitions)

                    #build links(*2, lag + lead)
                        #defaults to 1 for us
                        #create new string "imp_"adjacentTgtColumn"_sum"
                        #make a new partitionedcolumnlist, take the defaultlist and add in the normalisedPeriodColumn
                        #so long as targetColumn and adjacenttgtcolumn(lagtarget/leadTarget) isnt null & intervalColumn(lag/lead) = inclusionValue(1)->
                        #groupby partitionedcolumnlist
                        #sum targetcolumn and sum adjacenttargetcolumn
                        #join columns on to input dataframe

                    #doConstruction
                        #create new grouplist containing imputationClass and periodcolumn
                        #so long as targetColumn is not null
                            #group by grouplist
                            #implink is sum of targetcolumn/sum of auxiliarycolumn
                        #join imp_link onto orig dataframe
                        #if targetcol is null && is not to be imputed(timediff !=1) & auxiliaryCol is NOT null
                            #take auxialiarycolumn and multiply by imp_link
                        #else(and auxiliarycol is null)
                            #nullentry(in output col)
            #else copy inputcolumn into outputColumn
                        #create outPutMarker column. with a marker to say what was done(R,C,E)
                            #if targetcol is not null (returned)
                            #else if outputCol is not null (constructed)
                            #else if both are null (error)

                    #filter dataframe, get all entries where targetcol is null
                    #add a col called indexcol
                    #(essentially creating a bunch of rows with kv pairs. The max becomes indexcol(we dont understand this bit atm))


                    #Rolling Imputation<-udf
                    #recieves period and indexCol
                    #for each item in indexCol(we think this represents rows of data for a reference)
                        #If period = currentPeriod AND forwardImpLink>0 AND lagTarget > 0 AND timeDiffLag == 1(if its current period, values non-zero, and to be imputed)
                            #tmpFI = forwardimplink
                            #tmplag = lagTarget
                            #end loop
                        #inSize counter +=1
                        #If forwardimpLink > 0 && timeDifflag = 1 AND (period = currentPeriod OR period = APeriodBeforeCurrent)
                            #if lagTarget>0
                                #tmpFI = forwardimplink
                            #else
                                #tmpfi = tmpfi * forwardImpLink    (tmpfi starts as 1)
                        #if backwardimplink>0 AND timeDiffLead =1 AND tmplead =0 AND (lagTarget = 0 OR lagTarget = Null) AND period = currentPeriod OR period = aperiodaftercurrent
                            #tmpBI = TMPBI*backimplink (tmpbi starts as 1)

                        #if lagtarget>0 and timeDiffLag = 1 AND period = current OR period = a periodbeforecurrent
                            #tmplag = lagtarget
                        #elseIf leadTarget>0 AND timedifflead = 1 and tmplead=0 AND lagtarget = 0/null and period = current or period = aperiodaftercurrent
                            #tmplead = leadtarget
                        #elseif
                            #outputcolumn>0
                                #tmpconstruct = outputcolumn

                    #if(tmpfi>0 AND tmplag>0 -> res = tmplag * tmpFI
                    #elseif (tmpBI >0 AND tmplead>0 -> res = tmplead *tmpBI
                    #elseif (tmpConstruct >0 AND tempFI>1 AND inSize > 1) -> res = tmpconstruct * tmpfi

                    #apply rollingimputation udf to dataframe(where tarket col is null, otherwise make null
                    #create outputcolumn -> if outputvaluemarker is not null and outputvaluemarker(0) is not null(that one is the output value)
                    #create markercolumn -> if outputvaluemarker is not null and outputvaluemarker(1) is not null(the letter that says what happened)
                    #create implink column -> if markerColumn = FI => forwardimplink
                     #                           "   "           BI => backwardimplink
                     #  "           "               "           C => implink

#dataframe, name of periodColumn, name of lag col, name of lead col, group by(sequence/list)
#return dataframe with lagNewCol and leadNewCol


#### group the data by the groupByColumns
#### order by period
#### get one above  one below
#### get period of that row
#### work out difference in months between periods
    def months_between(self,period,otherPeriod):
       # print(period)
       # print(otherPeriod)
        period = int(period)
        otherPeriod = int(otherPeriod)
        if(otherPeriod!=0 and otherPeriod != None and otherPeriod != ''):
            year = 12 * (int(str(period)[:4]) - int(str(otherPeriod)[:4]))
            month = int(str(period)[4:]) - int(str(otherPeriod)[4:])
           # print(year)
           # print(month)
            return abs(year + month)
        else:
            return 0

    def monthsBetween(self,dataframe, periodColumn,groupByColumns,lagNewCol,leadNewCol,count=1):
       # print(dataframe)
        dataframe['previousPeriod'] = dataframe.sort_values(by=[periodColumn]).groupby(groupByColumns)[periodColumn].shift(count).fillna(0)
        dataframe[lagNewCol] = dataframe.apply(lambda x: self.months_between(x[periodColumn],x['previousPeriod']),axis=1)

        dataframe['nextPeriod'] = dataframe.sort_values(by=[periodColumn]).groupby(groupByColumns)[periodColumn].shift(-count).fillna(0)
        dataframe[leadNewCol] = dataframe.apply(lambda x: self.months_between(x[periodColumn], x['nextPeriod']), axis=1)
       # print(dataframe[['responder_id', periodColumn, lagNewCol, leadNewCol, 'nextPeriod', 'previousPeriod']])

        return dataframe





    def identifyInterval(self,dataframe,periodColumn,groupByColumns,periodicity, count=1):
        tempdf = dataframe
        lagMonDiffQ = "lagMonDiffQ"
        leadMonDiffQ = "leadMonDiffQ"
        lagMonDiffA = "lagMonDiffA"
        leadMonDiffA = "leadMonDiffA"
        timeDiffLag = "timeDiffLag"  #in the future this will be built from dataframe columns
        timeDiffLead = "timeDiffLead" #"
        normalisedPeriod= "normalisedPeriod"
        periodicity = periodicity.lower()

        if(periodicity=='m' or periodicity=='monthly'):
            tempdf[normalisedPeriod] = tempdf['period']
            tempdf = self.monthsBetween(tempdf,periodColumn,groupByColumns,timeDiffLag,timeDiffLead)
            tempdf = tempdf.drop(["previousPeriod", "nextPeriod"], inplace=True,axis=1)

        elif(periodicity=='a' or periodicity=='annually' or periodicity == 'yearly'):
            tempdf[normalisedPeriod] = tempdf.apply(lambda x: x['period']-100 if str(x['period'])[4:] in ['01','02','03'] else x['period'], axis = 1)
            tempdf = self.monthsBetween(tempdf, periodColumn, groupByColumns, lagMonDiffA, leadMonDiffA)
            tempdf[timeDiffLag] = tempdf.apply(lambda x: 1 if x[lagMonDiffA] == 12 else 0, axis = 1)
            tempdf[timeDiffLead] = tempdf.apply(lambda x: 1 if x[leadMonDiffA] == 12 else 0, axis = 1)
            #print(tempdf)
            tempdf = tempdf.drop([lagMonDiffA,leadMonDiffA,"previousPeriod","nextPeriod"],inplace=True,axis=1)
        elif(periodicity=='q' or periodicity == "quarterly"):

            def calcQuarter(period):
                period=int(period)
                if(period>10):
                    return int(int(str(period)[4:]) / 3)
                else: return 99


            def calcNomalisedPeriod(period):
                period = int(period)
                year = str(period)[:4]
                #quarter = str(int(str(period)[4:]) / 3)
                quarter=calcQuarter(period)
                out = year + "Q" + str(quarter)
                return out
            tempdf[normalisedPeriod] = tempdf.apply(lambda x: calcNomalisedPeriod(x['period']),axis=1)
            tempdf = self.monthsBetween(tempdf, periodColumn, groupByColumns, lagMonDiffQ, leadMonDiffQ)
            #find qnum of period, qnum - lag
            #print(tempdf)

            #NOTE: We dont think this covers all situations, eg  current 201912 prev 201109 | Continuing on as this seems to match the scala code, but we want to keep an eye on it.
            tempdf['timeDiffLag'] = tempdf.apply(lambda x: 1 if(calcQuarter(x[periodColumn])-calcQuarter(x["previousPeriod"]) == -3 and x[lagMonDiffQ]<=5) else (calcQuarter(x[periodColumn])-calcQuarter(x["previousPeriod"] )),axis=1)

            tempdf['timeDiffLead'] = tempdf.apply(lambda x: 1 if (calcQuarter(x["nextPeriod"]) - calcQuarter(x[periodColumn]) == -3 and x[leadMonDiffQ] <= 5) else (calcQuarter(x["nextPeriod"]) - calcQuarter(x[periodColumn])),axis=1)
            tempdf = tempdf.drop([lagMonDiffQ,leadMonDiffQ],axis=1)
        return tempdf

    def identifyAdjacentTarget(self,dataframe,targetColumn,periodColumn, groupByColumns,count=1):
        lagTarget = "imp_"+targetColumn+"_lag"
        leadTarget = "imp_"+targetColumn+"_lead"
        dataframe[lagTarget] = dataframe.sort_values(by=[periodColumn]).groupby(groupByColumns)[targetColumn].shift(count).fillna(0).astype(np.int64)
        dataframe[leadTarget] = dataframe.sort_values(by=[periodColumn]).groupby(groupByColumns)[targetColumn].shift(-count).fillna(0).astype(np.int64)

       # dataframe[lagTarget] = dataframe[lagTarget].astype(np.int64)
       # dataframe[leadTarget] = dataframe[leadTarget].astype(np.int64)
        #dataframe[leadTarget] = pd.to_numeric(dataframe[leadTarget], downcast='int64')
        return dataframe

    def buildLinks(self,dataframe,groupByColumns,normalisedPeriod,targetColumn,intervalCol,adjacentTargetColumn, newLinkColumn,inclusionValue=1):
        groupByColumns.append(normalisedPeriod)
        dataframe.to_csv("miguel.csv", index=False)
        filtereddataframe = dataframe[(dataframe[intervalCol] == inclusionValue) & (dataframe[targetColumn]!=None) & (dataframe[adjacentTargetColumn] != None)]
        if(filtereddataframe.size >0):
            Aggregateddataframe = filtereddataframe.groupby(groupByColumns,as_index=False).agg({targetColumn:'sum',adjacentTargetColumn:'sum'})
            Aggregateddataframe.to_csv("harry.csv",index=False)
            Aggregateddataframe[newLinkColumn] = Aggregateddataframe.apply(lambda x: x[targetColumn]/x[adjacentTargetColumn] if x[adjacentTargetColumn]!=0 else 0,axis=1)
            Aggregateddataframe = Aggregateddataframe.drop([targetColumn,adjacentTargetColumn],axis=1)
            dataframe = pd.merge(filtereddataframe,Aggregateddataframe,on=groupByColumns)
        else:
            #NOTE: we will need to check for this 0 on the applying of factors so's not to overwrite stuff.
            dataframe[newLinkColumn] = 0
        return dataframe

    #def doConstruction(self,dataframe,groupByColumns, periodColumn, ):


imputer = Imputation()

workingData = imputer.identifyInterval(myData,'period', 'responder_id','q')
workingData = imputer.identifyAdjacentTarget(workingData,"Q601_asphalting_sand","period","responder_id")
#print(workingData)
workingData = imputer.buildLinks(workingData,['responder_id'],'normalisedPeriod','Q601_asphalting_sand','timeDiffLead','imp_Q601_asphalting_sand_lead','FWDLINK')

workingData = imputer.buildLinks(workingData,['responder_id'],'normalisedPeriod','Q601_asphalting_sand','timeDiffLag','imp_Q601_asphalting_sand_lag','BWDLINK')
workingData.to_csv("bob.csv",index=False)
#print(workingData)


#print(imputer.identifyInterval(myData, 'period', 'responder_id','q'))







#monthsBetween(myData,"period",["responder_id"],"lagNewCol","leadNewCol")