from uk.gov.ons.src.baseMethod import baseMethod
import pandas as pd
import numpy as np
import math
myData = pd.read_csv("""C:\\Users\\Off.Network.User4\\Desktop\data\\testData.csv""")


class Imputation(baseMethod):
    """
    Description: Imputation Class controls the imputation of data.
    Extends baseMethod.

    Params(constructor):
            :param dataFrame: Dataframe     - The input dataset(with things to be imputed.
            :param imputationClass: List    - Contains the columns to group on(for qcas: classification,cell_no,question).
            :param periodColumn: String     - The name of the column that contains period(for qcas it is 'period').
            :param uniqueIdentifier: String - The name of the column that contains a unique id for a reference(for qcas it is responder_id).
            :param targetColumn: String     - The name of the column that contains the value for imputation(for qcas it is adjusted_value).
            :param outputColumn: String     - The name of the column that will contain the output to imputation.
            :param markerColumn: String     - The name of the column that will contain the imputation marker representing how it was imputed.
                                                                                                     (FI - Forward Imputed,
                                                                                                      BI - Backward Imputed,
                                                                                                       R - Returned,
                                                                                                       C - Constructed,
                                                                                                       E - Error)
            :param auxiliaryColumn: String - The name of the column containing the auxiliary variable used in construction(for qcas this is selection_emp)
            :param periodicity: String     - Indication of whether survey is quarterly, annual, or monthly
            :param periodColumnVal: String - Current period(yyyymm)

    returns: workingDataframe: Dataframe - The input dataset with imputed/constructed values where necessary.
    """
    # We're going to make the mandatory args into the constructor.
    # This'll get around the problem(global args as default params),
    # and make use of mandatory args check.
   # imputer = Imputation(myData, ['classification', 'cell_no', 'question'], 'period', 'responder_id', 'adjusted_value',
           #              'adjusted_values', 'MarkerCol', 'selection_data', 'q', 201809)
    def __init__(self, dataFrame, imputationClass, periodColumn, uniqueIdentifier, targetColumn, outputColumn, markerColumn, auxiliaryColumn, periodicity, periodColumnVal):
        """
        Description: Constructor for imputation class, set up global variables and performs 'mandatoryArgsCheck' to ensure that params are not null.
        :param dataFrame: Dataframe     - The input dataset(with things to be imputed.
        :param imputationClass: List    - Contains the columns to group on(for qcas: classification,cell_no,question).
        :param periodColumn: String     - The name of the column that contains period(for qcas it is 'period').
        :param uniqueIdentifier: String - The name of the column that contains a unique id for a reference(for qcas it is responder_id).
        :param targetColumn: String     - The name of the column that contains the value for imputation(for qcas it is adjusted_value).
        :param outputColumn: String     - The name of the column that will contain the output to imputation.
        :param markerColumn: String     - The name of the column that will contain the imputation marker representing how it was imputed.
                                                                                                 (FI - Forward Imputed,
                                                                                                  BI - Backward Imputed,
                                                                                                   R - Returned,
                                                                                                   C - Constructed,
                                                                                                   E - Error)
        :param auxiliaryColumn: String - The name of the column containing the auxiliary variable used in construction(for qcas this is selection_emp)
        :param periodicity: String     - Indication of whether survey is quarterly, annual, or monthly
        :param periodColumnVal: String - Current period(yyyymm)
        """

        #Dataframe removed from super init because cannot compare with == None.
        super().__init__(imputationClass, periodColumn, uniqueIdentifier, targetColumn, outputColumn, markerColumn, auxiliaryColumn, periodicity, periodColumnVal)

        # Passed In.
        self.dataFrame = dataFrame
        self.imputationClass = imputationClass
        self.periodColumn = periodColumn
        self.uniqueIdentifier = uniqueIdentifier
        self.targetColumn = targetColumn
        self.outputColumn = outputColumn
        self.markerColumn = markerColumn
        self.auxiliaryColumn = auxiliaryColumn
        self.periodicity = periodicity
        self.periodColumnVal = periodColumnVal

        # Main Variables
        self.lagTarget = "imp_" + targetColumn + "_lag"
        self.leadTarget = "imp_" + targetColumn + "_lead"
        self.timeDiffLag = "imp_" + periodColumn + "_diff_lag"
        self.timeDiffLead = "imp_" + periodColumn + "_diff_lead"
        self.targetSum = "imp_" + targetColumn + "_sum"
        self.normalisedPeriod = "imp_" + periodColumn + "_unique"

    def __str__(self):
        return "This Is A Construction And Imputation Module."

    backwardImpLink = 'BWDLINK'
    forwardImpLink = 'FWDLINK'
    lagTarget = ''
    leadTarget = ''
    timeDiffLag = ''
    timeDiffLead = ''
    targetSum = ''
    normalisedPeriod = ''
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

    def imputation(self):
        """
        Description: This function controls the flow of imputation, calling the subfunctions before eventually returning the imputed dataframe

        :return: workingDataFrame: Dataframe - Input dataframe with imputed/constructed values added
        """

        groupByColumns = self.imputationClass.copy()
        groupByColumns.append(self.uniqueIdentifier)

        workingDataFrame = self.identifyInterval(self.dataFrame, self.periodColumn, groupByColumns, self.periodicity, self.timeDiffLag, self.timeDiffLead, self.normalisedPeriod)

        workingDataFrame = self.identifyAdjacentTarget(workingDataFrame, self.targetColumn, groupByColumns, self.lagTarget, self.leadTarget)

        groupByColumns = self.imputationClass.copy()
        groupByColumns.append(self.normalisedPeriod)
        #Below->these might be the wrong way around(lag/lead)
        workingDataFrame = self.buildLinks(workingDataFrame, self.imputationClass, self.normalisedPeriod, self.targetColumn, self.timeDiffLag, self.lagTarget, self.forwardImpLink, useROM=False)
        workingDataFrame = self.buildLinks(workingDataFrame, self.imputationClass, self.normalisedPeriod, self.targetColumn, self.timeDiffLead, self.leadTarget, self.backwardImpLink, useROM=False)

        workingDataFrame = self.doConstruction(workingDataFrame, self.imputationClass, self.normalisedPeriod, self.targetColumn, self.auxiliaryColumn, self.outputColumn, self.markerColumn, self.timeDiffLag, self.impLink, self.markerReturn, self.markerConstruct, self.markerError, self.joinType)

        # First periodColumn is the entry in the periodColumn, second is the name of the periodColumn.
        #(workingDataFrame[self.outputColumn],workingDataFrame[self.markerColumn]) = workingDataFrame.apply(lambda x: self.rollingImputation(self.periodColumnVal, x, self.periodColumn, self.outputColumn, self.forwardImpLink, self.backwardImpLink, self.lagTarget, self.leadTarget, self.markerForwardImp,self.markerBackwardImp, self.timeDiffLag, self.timeDiffLead), axis=1)
        workingDataFrame[[self.outputColumn,self.markerColumn]] = workingDataFrame.apply(
            lambda x: self.rollingImputation(self.periodColumnVal, x, self.periodColumn, self.outputColumn,
                                             self.forwardImpLink, self.backwardImpLink, self.lagTarget, self.leadTarget,
                                             self.markerForwardImp, self.markerBackwardImp, self.timeDiffLag,
                                             self.timeDiffLead, self.markerColumn), axis=1)

        workingDataFrame[self.impLink] = workingDataFrame.apply(lambda x: x[self.forwardImpLink] if x[self.markerColumn] == self.markerForwardImp else x[self.backwardImpLink] if x[self.markerColumn] == self.markerBackwardImp else x[self.impLink] if x[self.markerColumn] == self.markerConstruct else None,axis=1)

        workingDataFrame.drop([self.timeDiffLag, self.timeDiffLead, self.lagTarget, self.leadTarget, self.forwardImpLink, self.backwardImpLink, self.normalisedPeriod], inplace=True, axis=1)
        return workingDataFrame

    def rollingImputation(self, period, row, periodColumn, outputColumn, forwardImpLink, backwardImpLink, lagTarget, leadTarget, markerForwardImp, markerBackwardImp, timeDiffLag, timeDiffLead, markerCol):
        """

        :param period: String - Current period (yyyymm)
        :param row: Row - One row of data from the input dataframe
        :param periodColumn: String - The name of the column that contains period(for qcas it is 'period').
        :param outputColumn: String - The name of the column that will contain the output to imputation.
        :param forwardImpLink: String - The name of the column that holds the forward imputation link (for qcas: FWDLINK)
        :param backwardImpLink: String - The name of the column that holds the forward imputation link (for qcas: BWDLINK)
        :param lagTarget: String - The name of a column that will contain the value of the target column for the period previous to current one.
        :param leadTarget: String - The name of a column that will contain the value of the target column for the period after the current one.
        :param markerForwardImp: String - Marker for forward imputed rows (FI)
        :param markerBackwardImp: String - Marker for backward imputed rows (FI)
        :param timeDiffLag: String - The period difference between the current period target and the lagTarget(taking into account periodicity)
        :param timeDiffLead: String - The period difference between the current period target and the leadTarget(taking into account periodicity)
        :param markerCol:

        :return:
        """
        tmpFI = 1
        tmpBI = 1
        tmpLag = 0
        tmpLead = 0
        tmpConstruct = 0

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
        out=row[outputColumn]
        marker=row[markerCol]
        # Apply link fraction to related response & mark the imputation type start.
        if tmpFI > 0 and tmpLag > 0:
            result = tmpLag*tmpFI
            (out,marker) = (result, markerForwardImp)
        elif tmpBI > 0 and tmpLead > 0:
            result = tmpLead * tmpBI
            (out, marker) = (result, markerBackwardImp)
        elif tmpConstruct > 0 and tmpFI > 1:  # In scala inSize > 1  would be checked here.
            result = tmpConstruct*tmpFI
            (out, marker) = (result, markerForwardImp)

        return pd.Series([out,marker])

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

    def identifyInterval(self, dataframe, periodColumn, groupByColumns, periodicity, timeDiffLag, timeDiffLead, normalisedPeriod):
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
            tempdf[timeDiffLag] = tempdf.apply(lambda x: 1 if(calcQuarter(x[periodColumn]) - calcQuarter(x["previousPeriod"]) == -3 and x[lagMonDiffQ] <= 5) else (calcQuarter(x[periodColumn]) - calcQuarter(x["previousPeriod"])), axis=1)

            tempdf[timeDiffLead] = tempdf.apply(lambda x: 1 if (calcQuarter(x["nextPeriod"]) - calcQuarter(x[periodColumn]) == -3 and x[leadMonDiffQ] <= 5) else (calcQuarter(x["nextPeriod"]) - calcQuarter(x[periodColumn])), axis=1)
            tempdf = tempdf.drop([lagMonDiffQ,leadMonDiffQ,"previousPeriod","nextPeriod"], axis=1)
        return tempdf

    def identifyAdjacentTarget(self, dataframe, targetColumn, groupByColumns,  lagTarget, leadTarget, count=1):

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

    def doConstruction(self, dataframe, groupByColumns, periodColumn, targetColumn, auxColumn, outputColumn, outMarkerCol, timeDiffLag, impLink, markerReturn, markerConstruct, markerError, joinType):

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

imputer=Imputation(myData,['classification','cell_no','question'],'period','responder_id','adjusted_value','adjusted_values','MarkerCol','selection_data','q',201809)
workingData = imputer.imputation()

workingData.to_csv("bob.csv",index=False)