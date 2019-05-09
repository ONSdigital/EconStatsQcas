from venv.uk.gov.ons.src.baseMethod import baseMethod
import pandas as pd
import numpy as np
import math
myData = pd.read_csv("""\\\\nsdata6\\INGRES_CSAI\\Economic Stats BAU Team\\Transformation\\Developer Folders\\Mike\\qcastestdata\\testData.csv""")


class Imputation(baseMethod):
    def __init__(self, *args):
        super().__init__(*args)
###we're going to make the mandatory args into the constructor. This'll get around the problem(global args as default params), and make use of mandatory args check
    def __str__(self):
        return "This Is A Construction And Imputation Module."

    backwardImpLink = 'BWDLINK'
    forwardImpLink = 'FWDLINK'
    # indexCol = 'index'
    lagTarget = ''
    leadTarget = ''
    timeDiffLag = ''
    timeDiffLead = ''
    outputValueMarker = ''
    targetSum = ''
    normalisedPeriod = ""
    # monthly = "01"
    # annually = "02"
    # quarterly = "03"
    markerForwardImp = "FI"
    markerBackwardImp = "BI"
    markerConstruct = "C"
    markerReturn = "R"
    markerError = "E"
    impLink = "imputation_link"
    joinType = "left"

    def imputation(self, dataFrame, imputationClass, periodColumn, uniqueIdentifier, targetColumn, outputColumn, markerColumn, auxiliaryColumn, periodicity, periodColumnVal):
        self.lagTarget = "imp_" + targetColumn + "_lag"
        self.leadTarget = "imp_" + targetColumn + "_lead"
        self.timeDiffLag = "imp_" + periodColumn + "_diff_lag"
        self.timeDiffLead = "imp_" + periodColumn + "_diff_lead"
        self.outputValueMarker = outputColumn + "_flag"
        self.targetSum = "imp_" + targetColumn + "_sum"
        self.normalisedPeriod = "imp_" + periodColumn + "_unique"


        #self.normalisedPeriod = periodColumn
        groupByColumns = imputationClass.copy()
        groupByColumns.append(uniqueIdentifier)

        workingDataFrame = self.identifyInterval(dataFrame, periodColumn, groupByColumns, periodicity)
        print(workingDataFrame.columns.values)
        workingDataFrame = self.identifyAdjacentTarget(workingDataFrame, targetColumn, groupByColumns)
        print(workingDataFrame.columns.values)
        print("-------------------------------------------------------------------------------------------------")
        groupByColumns = imputationClass.copy()
        groupByColumns.append(self.normalisedPeriod)
        workingDataFrame = self.buildLinks(workingDataFrame, imputationClass, self.normalisedPeriod, targetColumn, self.timeDiffLag, self.lagTarget, self.forwardImpLink, useROM=False)

        workingDataFrame = self.doConstruction(workingDataFrame, imputationClass, self.normalisedPeriod, targetColumn, auxiliaryColumn, outputColumn, markerColumn)

        # First periodColumn is the entry in the periodColumn, second is the name of the periodColumn.
        workingDataFrame[outputColumn] =  workingDataFrame.apply(lambda x:self.rollingImputation(periodColumnVal,x,periodColumn,outputColumn),axis=1)

    def rollingImputation(self, period, row, periodColumn, outputColumn,
                          forwardImpLink=forwardImpLink, backwardImpLink=backwardImpLink,
                          lagTarget=lagTarget, leadTarget=leadTarget,
                          markerForwardImp=markerForwardImp, markerBackwardImp=markerBackwardImp,
                          timeDiffLag=timeDiffLag, timeDiffLead=timeDiffLead):
        tmpFI = 1
        tmpBI = 1
        tmpLag = 0
        tmpLead = 0
        tmpConstruct = 0
        result = ''

        # Forward imputation from previous period return.
        if row[periodColumn] == period and row[forwardImpLink] > 0 and row[lagTarget] > 0 and row[timeDiffLag] == 1:
            tmpFI = row[forwardImpLink]
            tmpLag = row[lagTarget]
            # In scala, end loop here.

        # Define the link fraction for imputation start.
        if row[forwardImpLink] > 0 and row[timeDiffLag] and row[periodColumn] <= period:
            if row[lagTarget] > 0:
                tmpFI = row[forwardImpLink]
            else:
                # Check this works, we think this is like += . if there is an error around here, this might be why.
                tmpFI *= row[forwardImpLink]

        if row[backwardImpLink] > 0 and row[timeDiffLead] == 1 and tmpLead == 0 and (row[lagTarget] == 0 or row[lagTarget] == None) and row[periodColumn] >= period:
           tmpBI *= row[backwardImpLink]
        # Define the link fraction for imputation(forward&backward) end.

        # Select a linked response to apply the link fraction start.
        if row[lagTarget] > 0 and row[timeDiffLag] == 1 and (row[periodColumn] <= period):
            tmpLag = row[lagTarget]
        elif row[leadTarget]>0 and row[timeDiffLead] and tmpLead == 0 and (row[lagTarget] == 0 or row[lagTarget] == None) and row[periodColumn] >= period:
            tmpLead=row[leadTarget]
        elif row[outputColumn] > 0:
            tmpConstruct = row[outputColumn]
        # Select a linked response to apply the link fraction end.

        # Apply link fraction to related response & mark the imputation type start.
        if tmpFI > 0 and tmpLag > 0:
            result = tmpLag*tmpFI
            out = (result, markerForwardImp)
        elif tmpBI > 0 and tmpLead > 0:
            result = tmpLead * tmpBI
            out = (result, markerBackwardImp)
        elif tmpConstruct > 0 and tmpFI > 1:  # In scala inSize > 1  would be checked here.
            result = tmpConstruct*tmpFI
            out = (result, markerForwardImp)

        return result

    def months_between(self, period, otherPeriod):
        period = int(period)
        otherPeriod = int(otherPeriod)
        if otherPeriod!=0 and otherPeriod != None and otherPeriod != '':
            year = 12 * (int(str(period)[:4]) - int(str(otherPeriod)[:4]))
            month = int(str(period)[4:]) - int(str(otherPeriod)[4:])
            return abs(year + month)
        else:
            return 0

    def monthsBetween(self, dataframe, periodColumn, groupByColumns, lagNewCol, leadNewCol, count=1):
        dataframe['previousPeriod'] = dataframe.sort_values(by=groupByColumns)[periodColumn].shift(count).fillna(0)
        dataframe[lagNewCol] = dataframe.apply(lambda x: self.months_between(x[periodColumn], x['previousPeriod']), axis=1)
        dataframe['nextPeriod'] = dataframe.sort_values(by=groupByColumns)[periodColumn].shift(-count).fillna(0)
        dataframe[leadNewCol] = dataframe.apply(lambda x: self.months_between(x[periodColumn], x['nextPeriod']), axis=1)

        return dataframe

    def identifyInterval(self, dataframe, periodColumn, groupByColumns, periodicity, timeDiffLag=timeDiffLag, timeDiffLead=timeDiffLead, normalisedPeriod=normalisedPeriod):
        print(normalisedPeriod)
        print(timeDiffLag)
        print(timeDiffLead)
        print("AAAARRRRRGGGGGHHHHHHJ")
        tempdf = dataframe
        lagMonDiffQ = "lagMonDiffQ"
        leadMonDiffQ = "leadMonDiffQ"
        lagMonDiffA = "lagMonDiffA"
        leadMonDiffA = "leadMonDiffA"

        periodicity = periodicity.lower()

        if periodicity == 'm' or periodicity == 'monthly':
            tempdf[normalisedPeriod] = tempdf[periodColumn]
            tempdf = self.monthsBetween(tempdf, periodColumn, groupByColumns, timeDiffLag, timeDiffLead)
            tempdf.drop(["previousPeriod", "nextPeriod"], inplace=True,axis=1)

        elif periodicity == 'a' or periodicity == 'annually' or periodicity == 'yearly':
            tempdf[normalisedPeriod] = tempdf.apply(lambda x: x[periodColumn]-100 if str(x[periodColumn])[4:] in ['01','02','03'] else x[periodColumn], axis = 1)
            tempdf = self.monthsBetween(tempdf, periodColumn, groupByColumns, lagMonDiffA, leadMonDiffA)
            tempdf[timeDiffLag] = tempdf.apply(lambda x: 1 if x[lagMonDiffA] == 12 else 0, axis=1)
            tempdf[timeDiffLead] = tempdf.apply(lambda x: 1 if x[leadMonDiffA] == 12 else 0, axis=1)
            tempdf.drop([lagMonDiffA,leadMonDiffA,"previousPeriod","nextPeriod"], inplace=True, axis=1)

        elif periodicity == 'q' or periodicity == "quarterly":
            def calcQuarter(period):
                period = int(period)
                if period > 10:
                    return int(int(str(period)[4:]) / 3)
                else: return 99

            def calcNomalisedPeriod(period):
                period = int(period)
                year = str(period)[:4]
                quarter = calcQuarter(period)
                out = year + "Q" + str(quarter)
                return out
            tempdf[normalisedPeriod] = tempdf.apply(lambda x: calcNomalisedPeriod(x[periodColumn]), axis=1)
            tempdf = self.monthsBetween(tempdf, periodColumn, groupByColumns, lagMonDiffQ, leadMonDiffQ)

            # Find qnum of period, qnum - lag.

            # NOTE: We dont think this covers all situations, eg  current 201912 prev 201109.
            # Continuing on as this seems to match the scala code, but we want to keep an eye on it.
            tempdf['timeDiffLag'] = tempdf.apply(lambda x: 1 if(calcQuarter(x[periodColumn]) - calcQuarter(x["previousPeriod"]) == -3 and x[lagMonDiffQ] <= 5) else (calcQuarter(x[periodColumn]) - calcQuarter(x["previousPeriod"])), axis=1)

            tempdf['timeDiffLead'] = tempdf.apply(lambda x: 1 if (calcQuarter(x["nextPeriod"]) - calcQuarter(x[periodColumn]) == -3 and x[leadMonDiffQ] <= 5) else (calcQuarter(x["nextPeriod"]) - calcQuarter(x[periodColumn])), axis=1)
            tempdf = tempdf.drop([lagMonDiffQ,leadMonDiffQ], axis=1)
        return tempdf

    def identifyAdjacentTarget(self, dataframe, targetColumn, groupByColumns,  lagTarget=lagTarget, leadTarget=leadTarget, count=1,):

        dataframe[lagTarget] = dataframe.sort_values(by=groupByColumns)[targetColumn].shift(count).fillna(0).astype(np.int64)
        dataframe[leadTarget] = dataframe.sort_values(by=groupByColumns)[targetColumn].shift(-count).fillna(0).astype(np.int64)
        return dataframe

    def buildLinks(self, dataframe, groupByColumns, normalisedPeriod, targetColumn, intervalCol, adjacentTargetColumn, newLinkColumn, inclusionValue=1, useROM=True):
        if(useROM==True):
            groupByColumns.append(normalisedPeriod)

            # Note: this section is a refactored version of the dtrades scala code. But for qcas the links are always 1.
            # It works in that it produces factors and doesn't fall over
            # BUT cant be tested against data because our data is qcas, and qcas doesnt calculate the links.
            filtereddataframe = dataframe[(dataframe[intervalCol] == inclusionValue) & (dataframe[targetColumn]!=None) & (dataframe[adjacentTargetColumn] != None)]
            if filtereddataframe.size >0:
                Aggregateddataframe = filtereddataframe.groupby(groupByColumns,as_index=False).agg({targetColumn:'sum',adjacentTargetColumn:'sum'})

                Aggregateddataframe[newLinkColumn] = Aggregateddataframe.apply(lambda x: x[targetColumn]/x[adjacentTargetColumn] if x[adjacentTargetColumn] != 0 else 0, axis=1)
                Aggregateddataframe = Aggregateddataframe.drop([targetColumn,adjacentTargetColumn], axis=1)
                dataframe = pd.merge(dataframe, Aggregateddataframe, on=groupByColumns)
            else:
                # NOTE: we will need to check for this 0 on the applying of factors so's not to overwrite stuff.
                dataframe[newLinkColumn] = 0
        else:
            dataframe[newLinkColumn] = 1
        return dataframe

    def doConstruction(self, dataframe, groupByColumns, periodColumn, targetColumn, auxColumn, outputColumn, outMarkerCol, timeDiffLag=timeDiffLag, impLink=impLink, markerReturn=markerReturn, markerConstruct=markerConstruct, markerError=markerError, joinType=joinType):

        groupByColumns.append(periodColumn)
        workingdataframe = dataframe[(dataframe[targetColumn] != None)]

        workingdataframe = workingdataframe.groupby(groupByColumns, as_index=False).agg({targetColumn: 'sum', auxColumn: 'sum'})
        workingdataframe[impLink] = workingdataframe[targetColumn]/workingdataframe[auxColumn]

        mergeCols = groupByColumns.copy()
        mergeCols.append(impLink)
        dataframe = pd.merge(dataframe, workingdataframe[mergeCols],how=joinType, left_on=groupByColumns,right_on=groupByColumns)

        dataframe[outputColumn] = dataframe.apply(lambda x: x[auxColumn]*x[impLink] if((math.isnan(x[targetColumn])) and (x[timeDiffLag] != 1) and (not math.isnan(x[auxColumn]))) else None if((math.isnan(x[targetColumn])) and (math.isnan(x[auxColumn]))) else x[targetColumn], axis=1)
        # (ABOVE)if targetCol is NULL, timeDiffLag!=1 and auxColumn is NOT NULL    =  outputCol =auxCol/imp_link
        # elseif targetcol is NULL and auxCol is NULL = outputCol = NULL
        # else outputCol = targetCol

        dataframe[outMarkerCol] = dataframe.apply(lambda x: markerReturn if (not math.isnan(x[targetColumn])) else markerConstruct if (not math.isnan(x[outputColumn])) else markerError if((math.isnan(x[targetColumn])) and (math.isnan(x[auxColumn]))) else None,axis=1)
        # (ABOVE)if targetCol is not null   = R
        # else if outputCol is not null = C
        # else if targetCol is NULL and auxCol is NULL = E
        # else = Null
        return dataframe

imputer = Imputation()
imputer.imputation(myData,['classification','cell_no','question'],'period','responder_id','adjusted_value','adjusted_value','MarkerCol','selection_data','q',201809)

#workingData = imputer.identifyInterval(myData,'period', ['classification','cell_no','question','responder_id'],'q')

#workingData = imputer.identifyAdjacentTarget(workingData,'adjusted_value',['classification','cell_no','question', 'responder_id'])

#workingData = imputer.buildLinks(workingData,['responder_id'],'normalisedPeriod','adjusted_value','timeDiffLead','imp_adjusted_value_lead','FWDLINK',useROM=False)

#workingData = imputer.buildLinks(workingData,['responder_id'],'normalisedPeriod','adjusted_value','timeDiffLag','imp_adjusted_value_lag','BWDLINK',useROM=False)

#workingData = imputer.doConstruction(workingData,['classification','cell_no','question'],'normalisedPeriod','adjusted_value','selection_data','Constructed_Value','MarkerCol')

# workingData['result'] = workingData.apply(lambda x:imputer.rollingImputation(201812,x),axis=1)

# workingData.to_csv("bob.csv",index=False)