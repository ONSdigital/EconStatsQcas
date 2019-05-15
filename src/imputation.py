from uk.gov.ons.src.baseMethod import baseMethod
import pandas as pd
import numpy as np
import math
#myData = pd.read_csv("C:\\Users\\Off.Network.User4\\Desktop\data\\testData.csv")
myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_bi_bi_r.json""")
#myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_bi_bi_r_fi_fi_r.json""")
#myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_r_fi_fi_Quarterly.json""")
#myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_r_fi_fi_Annually.json""")
#myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_bi_r_fi_r_fi.json""")
#myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_c_fi.json""")
#myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_c_fi_fi.json""")
#myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_c_fi_NotSelected_r.json""")
#myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_r_NotSelected_c.json""")
#myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_construct.json""")
#myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_e.json""")
#myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_r_fi.json""")
#myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_r_r_r.json""")
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
            :param impFactorColumn:String  - The name of the column that indicates whether a hardcoded factor is to be used.(If this column has a value use that, else impute as per dtrades)

    returns: workingDataframe: Dataframe - The input dataset with imputed/constructed values where necessary.
    """

    def __init__(self, dataFrame, imputationClass, periodColumn, uniqueIdentifier, targetColumn, outputColumn, markerColumn, auxiliaryColumn, periodicity, periodColumnVal, impFactorColumn):
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
        self.impFactorColumn = impFactorColumn

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
    impFactorColumn = ''

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

        workingDataFrame = self.buildLinks(workingDataFrame, groupByColumns, self.targetColumn, self.timeDiffLag, self.lagTarget, self.forwardImpLink, self.impFactorColumn, self.normalisedPeriod)

        workingDataFrame = self.buildLinks(workingDataFrame, groupByColumns, self.targetColumn, self.timeDiffLead, self.leadTarget, self.backwardImpLink, self.impFactorColumn, self.normalisedPeriod)

        workingDataFrame = self.doConstruction(workingDataFrame, self.imputationClass, self.normalisedPeriod, self.targetColumn, self.auxiliaryColumn, self.outputColumn, self.markerColumn, self.timeDiffLag, self.impLink, self.markerReturn, self.markerConstruct, self.markerError, self.joinType)

        outputDataFrame = workingDataFrame[workingDataFrame[self.targetColumn].isnull()]
        if outputDataFrame.size >0:
            outputDataFrame[[self.outputColumn,self.markerColumn]] = outputDataFrame.apply(
                lambda x: self.rollingImputation(self.periodColumnVal, x, self.normalisedPeriod, self.outputColumn,
                                                 self.forwardImpLink, self.backwardImpLink, self.lagTarget, self.leadTarget,
                                                 self.markerForwardImp, self.markerBackwardImp, self.timeDiffLag,
                                                 self.timeDiffLead, self.markerColumn,workingDataFrame,self.uniqueIdentifier), axis=1)

            workingDataFrame= workingDataFrame.dropna(subset = [self.targetColumn])

            workingDataFrame=workingDataFrame.append(outputDataFrame)


        workingDataFrame[self.impLink] = workingDataFrame.apply(lambda x: x[self.forwardImpLink] if x[self.markerColumn] == self.markerForwardImp else x[self.backwardImpLink] if x[self.markerColumn] == self.markerBackwardImp else x[self.impLink] if x[self.markerColumn] == self.markerConstruct else None, axis=1)

        workingDataFrame.drop([self.timeDiffLag, self.timeDiffLead, self.lagTarget, self.leadTarget, self.forwardImpLink, self.backwardImpLink, self.normalisedPeriod], inplace=True, axis=1)
        return workingDataFrame

    def rollingImputation(self, period, dfrow, periodColumn, outputColumn, forwardImpLink, backwardImpLink, lagTarget, leadTarget, markerForwardImp, markerBackwardImp, timeDiffLag, timeDiffLead, markerColumn,workingDataFrame,uniqueIdentifier):
        """
        Description: This method works on a single row (through an apply method). First it works out which type of imputation needs to take place, then it will do the imputaiton

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
        :param markerColumn: String     - The name of the column that will contain the imputation marker representing how it was imputed.
                                                                                                 (FI - Forward Imputed,
                                                                                                  BI - Backward Imputed,
                                                                                                   R - Returned,
                                                                                                   C - Constructed,
                                                                                                   E - Error)

        :return: out: Series(Float)     - Imputed/constructed value for row
                 marker: Series(String) - marker determining type of imputation                  (FI - Forward Imputed,
                                                                                                  BI - Backward Imputed)
        """
        tmpFI = 1
        tmpBI = 1
        tmpLag = 0
        tmpLead = 0
        tmpConstruct = 0

        #This slice needs to include groupby columns
        slicedDF = workingDataFrame[workingDataFrame[uniqueIdentifier]==dfrow[uniqueIdentifier]]
        refPeriods = slicedDF[periodColumn].drop_duplicates().tolist()
        Periods = workingDataFrame[periodColumn].drop_duplicates()

        i=0
        blank=False
        outlist=[]

        for x in Periods:
            if i == len(refPeriods):break
            if dfrow[periodColumn] == x:
                blank = True
            if x == refPeriods[i]:
                outlist.append(x)
                i += 1
            else:
                if not blank:
                    outlist = []
                else:
                    break

        slicedDF = slicedDF[(slicedDF[periodColumn].isin(outlist))]
        insize=0

        for x in slicedDF.head().iterrows():
            row=x[1]

            # Forward imputation from previous period return.
            if row[periodColumn] == dfrow[periodColumn] and row[forwardImpLink] > 0 and row[lagTarget] > 0 and row[timeDiffLag] == 1:
                tmpFI = row[forwardImpLink]
                tmpLag = row[lagTarget]
                break
                # In scala, end loop here.
            insize+=1
            # Define the link fraction for imputation start.

            if row[forwardImpLink] > 0 and row[timeDiffLag] ==1 and row[periodColumn] <= dfrow[periodColumn]:

                if row[lagTarget] > 0:
                    tmpFI = row[forwardImpLink]
                else:
                    tmpFI *= row[forwardImpLink]

            if row[backwardImpLink] > 0 and row[timeDiffLead] == 1 and tmpLead == 0 and (row[lagTarget] == 0 or row[lagTarget] == None or math.isnan(row[lagTarget])) and row[periodColumn] >= dfrow[periodColumn]:
               tmpBI *= row[backwardImpLink]
            # Define the link fraction for imputation(forward&backward) end.
            # Select a linked response to apply the link fraction start.
            if row[lagTarget] > 0 and row[timeDiffLag] == 1 and (row[periodColumn] <= dfrow[periodColumn]):
                tmpLag = row[lagTarget]
            elif row[leadTarget]>0 and row[timeDiffLead] ==1 and tmpLead == 0 and (row[lagTarget] == 0 or row[lagTarget] == None or math.isnan(row[lagTarget])) and row[periodColumn] >= dfrow[periodColumn]:
                tmpLead=row[leadTarget]
            elif row[outputColumn] > 0 or not math.isnan(row[outputColumn]):
                tmpConstruct = row[outputColumn]

        # Select a linked response to apply the link fraction end.
        out=dfrow[outputColumn]
        marker=dfrow[markerColumn]
        # Apply link fraction to related response & mark the imputation type start.
        if tmpFI > 0 and tmpLag > 0:
            result = tmpLag*tmpFI
            (out,marker) = (result, markerForwardImp)
        elif tmpBI > 0 and tmpLead > 0:
            result = tmpLead * tmpBI
            (out, marker) = (result, markerBackwardImp)
        elif tmpConstruct > 0 and tmpFI > 1 and insize>1:
            result = tmpConstruct*tmpFI
            (out, marker) = (result, markerForwardImp)

        return pd.Series([out,marker])

    def months_between(self, period, otherPeriod):
        """
        Description: Calculates the number of months between 2 given periods
        :param period: String - Current period (yyyymm)
        :param otherPeriod: String - previous or next period (yyyymm)
        :return: :Int - The number of months between the two given periods.
        """
        if otherPeriod!=0 and otherPeriod != None and str(otherPeriod) != '' and str(otherPeriod) != 'nan':
            if ('Q' in period):
                year = 12 * (int(str(period)[:4]) - int(str(otherPeriod)[:4]))
                quarter = int(str(period)[5:]) - int(str(otherPeriod)[5:])
                month = quarter * 3
                return abs(year + month)
            otherPeriod = int(otherPeriod)

            if(len(str(otherPeriod))==4):
                return abs((int(period) - int(otherPeriod))*12)

            year = 12 * (int(str(period)[:4]) - int(str(otherPeriod)[:4]))
            month = int(str(period)[4:]) - int(str(otherPeriod)[4:])

            return abs(year + month)
        else:
            return 0

    def monthsBetween(self, dataframe, periodColumn, groupByColumns, lagNewCol, leadNewCol, periodicity, count=1):
        """
        Description: This method works out the next/previous period for a row.
        :param dataFrame: Dataframe     - The input dataset(with things to be imputed.
        :param periodColumn: String - The name of the column that contains period(for qcas it is 'period').
        :param groupByColumns:
        :param lagNewCol:
        :param leadNewCol:
        :param periodicity
        :param count:
        :return:
        """


        def calculateAdjacentPeriod(row,periodicity,dataframe):
            currentPeriod = row[periodColumn]
            lastOut = None
            nextOut = None

            if periodicity == "01":
                currentMonth = str(currentPeriod)[4:]
                currentYear = str(currentPeriod)[:4]
                nextMonth = int(currentMonth)+int(periodicity)
                nextYear=int(currentYear)

                if(nextMonth>12):
                    nextYear+=1
                    nextMonth-=12

                if(nextMonth<10):
                    nextMonth = "0"+str(nextMonth)

                nextPeriod = str(nextYear)+str(nextMonth)

                lastMonth = int(currentMonth)-int(periodicity)
                lastYear=int(currentYear)
                if (lastMonth <1):
                    lastYear -=1
                    lastMonth += 12
                if (lastMonth < 10):
                    lastMonth = "0" + str(lastMonth)

                lastPeriod = str(lastYear)+str(lastMonth)

            elif periodicity == "02":

                nextPeriod = str(int(currentPeriod) + 1)
                lastPeriod = str(int(currentPeriod) - 1)

            else:#quarterly
                currentMonth = str(currentPeriod)[5:]
                currentYear = str(currentPeriod)[:4]
                nextMonth = int(currentMonth) + 1
                nextYear = int(currentYear)

                if (nextMonth > 4):
                    nextYear += 1
                    nextMonth -= 4

                nextPeriod = str(nextYear) + "Q" + str(nextMonth)

                lastMonth = int(currentMonth) - 1
                lastYear = int(currentYear)
                if (lastMonth < 1):
                    lastYear -= 1
                    lastMonth += 4

                lastPeriod = str(lastYear) + "Q" + str(lastMonth)

            rowDF = pd.DataFrame([row])
            filteredDataframe = dataframe[dataframe[periodColumn] == str(nextPeriod)]
            nextPeriodRows = pd.merge(filteredDataframe, rowDF, on=groupByColumns)

            if (nextPeriodRows.size > 0):
                nextOut = nextPeriod

            rowDF= pd.DataFrame([row])
            filteredDataframe = dataframe[dataframe[periodColumn] == str(lastPeriod)]
            lastPeriodRows = pd.merge(filteredDataframe,rowDF,on=groupByColumns)

            if(lastPeriodRows.size > 0):
                lastOut = lastPeriod

            return pd.Series([nextOut,lastOut])

        dataframe[['nextPeriod','previousPeriod']] = dataframe.apply(lambda x: calculateAdjacentPeriod(x,periodicity,dataframe),axis=1)
        dataframe[lagNewCol] = dataframe.apply(lambda x: self.months_between(x[periodColumn], x['previousPeriod']), axis=1)
        dataframe[leadNewCol] = dataframe.apply(lambda x: self.months_between(x[periodColumn], x['nextPeriod']), axis=1)

        return dataframe

    def identifyInterval(self, dataframe, periodColumn, groupByColumns, periodicity, timeDiffLag, timeDiffLead, normalisedPeriod):
        tempdf = dataframe
        lagMonDiffQ = "lagMonDiffQ"
        leadMonDiffQ = "leadMonDiffQ"
        lagMonDiffA = "lagMonDiffA"
        leadMonDiffA = "leadMonDiffA"

        if periodicity == '01':
            tempdf[normalisedPeriod] = tempdf.apply(lambda x:str(x[periodColumn]), axis=1)
            tempdf = self.monthsBetween(tempdf, normalisedPeriod, groupByColumns, timeDiffLag, timeDiffLead, periodicity)

        elif periodicity == '02':
            tempdf[normalisedPeriod] = tempdf.apply(lambda x: str(int(str(x[periodColumn])[:4])-1) if str(x[periodColumn])[4:] in ['01','02','03'] else str(x[periodColumn])[:4], axis = 1)
            tempdf = self.monthsBetween(tempdf, normalisedPeriod, groupByColumns, lagMonDiffA, leadMonDiffA, periodicity)

            tempdf[timeDiffLag] = tempdf.apply(lambda x: 1 if x[lagMonDiffA] == 12 else 0, axis=1)
            tempdf[timeDiffLead] = tempdf.apply(lambda x: 1 if x[leadMonDiffA] == 12 else 0, axis=1)
            tempdf.drop([lagMonDiffA,leadMonDiffA], inplace=True, axis=1)

        elif periodicity == '03':

            def calcNomalisedPeriod(period):
                def calcQuarter(period):
                    period = int(period)
                    if period > 10:
                        return math.ceil(int(str(period)[4:]) / 3)
                    else:
                        return 99
                period = int(period)
                year = str(period)[:4]
                quarter = calcQuarter(period)
                out = year + "Q" + str(quarter)
                return out

            tempdf[normalisedPeriod] = tempdf.apply(lambda x: calcNomalisedPeriod(x[periodColumn]), axis=1)
            tempdf = self.monthsBetween(tempdf, normalisedPeriod, groupByColumns, lagMonDiffQ, leadMonDiffQ, periodicity)
            tempdf[timeDiffLag] = tempdf.apply(lambda x: 1 if (x["previousPeriod"]!= None and int(x[normalisedPeriod][5:]) - int(x["previousPeriod"][5:]) == -3 and x[lagMonDiffQ] <= 5)  else (int(x[normalisedPeriod][5:]) - int(x["previousPeriod"][5:])) if x["previousPeriod"]!= None else 0, axis=1)
            tempdf[timeDiffLead] = tempdf.apply(lambda x: 1 if (x["nextPeriod"]!= None and int(x["nextPeriod"][5:]) - int(x[normalisedPeriod][5:]) == -3 and x[leadMonDiffQ] <= 5) else (int(x["nextPeriod"][5:]) - int(x[normalisedPeriod][5:])) if x["nextPeriod"]!= None else 0, axis=1)

            tempdf = tempdf.drop([lagMonDiffQ,leadMonDiffQ], axis=1)

        return tempdf

    def identifyAdjacentTarget(self, dataframe, targetColumn, groupByColumns,  lagTarget, leadTarget, count=1):

        def adjacentTargetLambda(row, dataframe,laglead):
            out = np.float("nan")
            if(row[laglead]!=None and str(row[laglead])!='nan'):
                sliceOfDataFrame = dataframe[(dataframe[self.normalisedPeriod]==row[laglead])& (dataframe[self.uniqueIdentifier]==row[self.uniqueIdentifier])]
                if(sliceOfDataFrame.size>0):
                    out= sliceOfDataFrame[self.targetColumn].tolist()[0]
            return out

        dataframe[lagTarget] = dataframe.apply(lambda x: adjacentTargetLambda(x,dataframe,'previousPeriod'),axis=1)
        dataframe[leadTarget] = dataframe.apply(lambda x: adjacentTargetLambda(x, dataframe, 'nextPeriod'), axis=1)

        return dataframe

    def buildLinks(self, dataframe, groupByColumns, targetColumn, intervalCol, adjacentTargetColumn, newLinkColumn,impFactorColumn, normalisedPeriod, inclusionValue=1):

        periodList = dataframe[dataframe[targetColumn].isnull()][normalisedPeriod].drop_duplicates().tolist()
        interimDF = dataframe[dataframe[normalisedPeriod]=="Blank"]
        interimDF[newLinkColumn] = interimDF[normalisedPeriod]

        for period in periodList:
            filtereddataframe = dataframe[(dataframe[intervalCol] == inclusionValue) & (dataframe[targetColumn].notnull()) & (dataframe[adjacentTargetColumn].notnull()) & (dataframe[normalisedPeriod] == str(period))]

            if filtereddataframe.size >0:
                Aggregateddataframe = filtereddataframe.groupby(groupByColumns,as_index=False).agg({targetColumn:'sum',adjacentTargetColumn:'sum'})

                Aggregateddataframe[newLinkColumn] = Aggregateddataframe.apply(lambda x: x[targetColumn]/x[adjacentTargetColumn] if x[adjacentTargetColumn] != 0 else 0, axis=1)
                interimDF=interimDF.append(Aggregateddataframe)

        dataSelect = groupByColumns.copy()
        dataSelect.append(newLinkColumn)
        interimDF = interimDF[dataSelect]
        dataframe = pd.merge(dataframe, interimDF, on=groupByColumns, how="left")

        if impFactorColumn in dataframe.columns.values:
            dataframe[newLinkColumn] = dataframe.apply(lambda x: x[impFactorColumn] if not math.isnan(x[impFactorColumn]) else x[newLinkColumn], axis=1)

        return dataframe

    def doConstruction(self, dataframe, groupByColumns, periodColumn, targetColumn, auxColumn, outputColumn, outMarkerCol, timeDiffLag, impLink, markerReturn, markerConstruct, markerError, joinType):
        groupByColumns.append(periodColumn)
        workingdataframe = dataframe[dataframe[targetColumn].notnull()]
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


#imputer = Imputation(myData,['classification','cell_no','question'],'period','responder_id','adjusted_value','adjusted_values','MarkerCol','selection_data','q',201809,'imp_factor')
imputer = Imputation(myData,['strata'],'time','ref','value','values','MarkerCol','frozen_value','01',201104,'imp_factor')
workingData = imputer.imputation()
workingData.to_csv("bob.csv",index=False)