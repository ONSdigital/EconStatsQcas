from uk.gov.ons.src.baseMethod import baseMethod
import pandas as pd
import numpy as np
import math
#myData = pd.read_csv("C:\\Users\\Off.Network.User4\\Desktop\data\\testData.csv")
#myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_bi_bi_r.json""")
#myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_bi_bi_r_fi_fi_r.json""")
myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_r_fi_fi_Quarterly.json""")
myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_r_fi_fi_Annually.json""")
myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_bi_r_fi_r_fi.json""")
myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_c_fi.json""")
myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_c_fi_fi.json""")
myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_c_fi_NotSelected_r.json""")
#myData = pd.read_json("""C:\\Users\\Off.Network.User4\\Desktop\\dtradestest\\dtrades test data\\dtrades test data\\in\\Imputation_r_NotSelected_c.json""")
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
    # We're going to make the mandatory args into the constructor.
    # This'll get around the problem(global args as default params),
    # and make use of mandatory args check.
   # imputer = Imputation(myData, ['classification', 'cell_no', 'question'], 'period', 'responder_id', 'adjusted_value',
           #              'adjusted_values', 'MarkerCol', 'selection_data', 'q', 201809)
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
        #workingDataFrame = self.identifyInterval(self.dataFrame, self.periodColumn, groupByColumns, self.periodicity, self.timeDiffLead,self.timeDiffLag, self.normalisedPeriod)
        workingDataFrame.to_csv("1.csv")
        workingDataFrame = self.identifyAdjacentTarget(workingDataFrame, self.targetColumn, groupByColumns, self.lagTarget, self.leadTarget)
        #workingDataFrame = self.identifyAdjacentTarget(workingDataFrame, self.targetColumn, groupByColumns, self.leadTarget,
         #                                              self.lagTarget)
        workingDataFrame.to_csv("2.csv")
        groupByColumns = self.imputationClass.copy()
        groupByColumns.append(self.normalisedPeriod)
        #groupByColumns.append(self.periodColumn)

        workingDataFrame = self.buildLinks(workingDataFrame, groupByColumns, self.targetColumn, self.timeDiffLag, self.lagTarget, self.forwardImpLink, self.impFactorColumn, self.normalisedPeriod)
        workingDataFrame.to_csv("3.csv")
        workingDataFrame = self.buildLinks(workingDataFrame, groupByColumns, self.targetColumn, self.timeDiffLead, self.leadTarget, self.backwardImpLink, self.impFactorColumn, self.normalisedPeriod)
        workingDataFrame.to_csv("4.csv")
        workingDataFrame = self.doConstruction(workingDataFrame, self.imputationClass, self.normalisedPeriod, self.targetColumn, self.auxiliaryColumn, self.outputColumn, self.markerColumn, self.timeDiffLag, self.impLink, self.markerReturn, self.markerConstruct, self.markerError, self.joinType)
        workingDataFrame.to_csv("5.csv")


        #Now, for each ref passed in
        #pass in df, filtered to only contain passed in refs


        # First periodColumn is the entry in the periodColumn, second is the name of the periodColumn.
        #workingDataFrame[[self.outputColumn,self.markerColumn]] = workingDataFrame[workingDataFrame[self.targetColumn].isnull()].apply(
          #  lambda x: self.rollingImputation(self.periodColumnVal, x, self.periodColumn, self.outputColumn,
             #                                self.forwardImpLink, self.backwardImpLink, self.lagTarget, self.leadTarget,
                #                             self.markerForwardImp, self.markerBackwardImp, self.timeDiffLag,
                  #                           self.timeDiffLead, self.markerColumn,workingDataFrame,self.uniqueIdentifier), axis=1)
        outputDataFrame = workingDataFrame[workingDataFrame[self.targetColumn].isnull()]
        outputDataFrame[[self.outputColumn,self.markerColumn]] = outputDataFrame.apply(
            lambda x: self.rollingImputation(self.periodColumnVal, x, self.periodColumn, self.outputColumn,
                                             self.forwardImpLink, self.backwardImpLink, self.lagTarget, self.leadTarget,
                                             self.markerForwardImp, self.markerBackwardImp, self.timeDiffLag,
                                             self.timeDiffLead, self.markerColumn,workingDataFrame,self.uniqueIdentifier), axis=1)

        workingDataFrame= workingDataFrame.dropna(subset = [self.targetColumn])

        workingDataFrame=workingDataFrame.append(outputDataFrame)


        # Some data is coming through with no imp link because it doesn't hit any conditions aka bad data we think.
        # Just double check good data doesn't get blanks.
        workingDataFrame[self.impLink] = workingDataFrame.apply(lambda x: x[self.forwardImpLink] if x[self.markerColumn] == self.markerForwardImp else x[self.backwardImpLink] if x[self.markerColumn] == self.markerBackwardImp else x[self.impLink] if x[self.markerColumn] == self.markerConstruct else None, axis=1)

        workingDataFrame.drop([self.timeDiffLag, self.timeDiffLead, self.lagTarget, self.leadTarget, self.forwardImpLink, self.backwardImpLink, self.normalisedPeriod], inplace=True, axis=1)
        return workingDataFrame

    #def rollingImputation(self, period, row, periodColumn, outputColumn, forwardImpLink, backwardImpLink, lagTarget, leadTarget, markerForwardImp, markerBackwardImp, timeDiffLag, timeDiffLead, markerColumn):


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
                    # Check this works, we think this is like += . if there is an error around here, this might be why.
                    tmpFI *= row[forwardImpLink]

            if row[backwardImpLink] > 0 and row[timeDiffLead] == 1 and tmpLead == 0 and (row[lagTarget] == 0 or row[lagTarget] == None) and row[periodColumn] >= dfrow[periodColumn]:
               tmpBI *= row[backwardImpLink]
            # Define the link fraction for imputation(forward&backward) end.
            # Select a linked response to apply the link fraction start.
            if row[lagTarget] > 0 and row[timeDiffLag] == 1 and (row[periodColumn] <= dfrow[periodColumn]):
                tmpLag = row[lagTarget]
            elif row[leadTarget]>0 and row[timeDiffLead] ==1 and tmpLead == 0 and (row[lagTarget] == 0 or row[lagTarget] == None) and row[periodColumn] >= dfrow[periodColumn]:
                tmpLead=row[leadTarget]
            elif row[outputColumn] > 0:
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
        elif tmpConstruct > 0 and tmpFI > 1 and insize>1:  # In scala inSize > 1  would be checked here.
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
        print(period)
        print(otherPeriod)
        #pass periodicity- if not annual or monthly -> instead of current behaviour-check if abs diff is 1 ->we know that it is the next quarter




        if otherPeriod!=0 and otherPeriod != None and str(otherPeriod) != '' and str(otherPeriod) != 'nan':
            if ('Q' in period):
                year = 12 * (int(str(period)[:4]) - int(str(otherPeriod)[:4]))
                quarter = int(str(period)[5:]) - int(str(otherPeriod)[5:])
                month = quarter * 3
                return abs(year + month)
            otherPeriod = int(otherPeriod)
            print(otherPeriod)
            print(period)
            #annual periods are 4 digits, this causes a fail here.
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

                nextPeriod = int(currentPeriod) + 1
                lastPeriod = int(currentPeriod) - 1

            else:#quarterly
                currentMonth = str(currentPeriod)[5:]
                currentYear = str(currentPeriod)[:4]
                nextMonth = int(currentMonth) + 1
                nextYear = int(currentYear)
                #instead of adding periodicity, add 1, if next mo >4, year changed, and -4 from month
                if (nextMonth > 4):
                    nextYear += 1
                    nextMonth -= 4


                #NOTE: we'll need to adjust this bit to get Q1 Q2 Q3  etc, currently we have potential for Q12
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
          #  lastPeriodRows = dataframe[dataframe[periodColumn]==lastPeriod and dataframe[groupByColumns] ==row[groupByColumns]]
            if(lastPeriodRows.size > 0):
                lastOut = lastPeriod

            return pd.Series([nextOut,lastOut])
        #Given row with current period & periodicity
        #use periodicity to calculate next/prev
        #search df for next/prev, if so use it, if not return 0

        dataframe.to_csv("monBet.csv")

        dataframe[['nextPeriod','previousPeriod']] = dataframe.apply(lambda x: calculateAdjacentPeriod(x,periodicity,dataframe),axis=1)
        dataframe.to_csv("OPeriod.csv")
        #dataframe['previousPeriod'] = dataframe.sort_values(by=groupByColumns)[periodColumn].shift(count).fillna(0)
        dataframe[lagNewCol] = dataframe.apply(lambda x: self.months_between(x[periodColumn], x['previousPeriod']), axis=1)
        #dataframe['nextPeriod'] = dataframe.sort_values(by=groupByColumns)[periodColumn].shift(-count).fillna(0)
        dataframe[leadNewCol] = dataframe.apply(lambda x: self.months_between(x[periodColumn], x['nextPeriod']), axis=1)
        #dataframe[leadNewCol] = dataframe.apply(lambda x: self.months_between(x[periodColumn], x['previousPeriod']), axis=1)
        #dataframe['nextPeriod'] = dataframe.sort_values(by=groupByColumns)[periodColumn].shift(-count).fillna(0)
        #dataframe[lagNewCol] = dataframe.apply(lambda x: self.months_between(x[periodColumn], x['nextPeriod']), axis=1)

        return dataframe

    def identifyInterval(self, dataframe, periodColumn, groupByColumns, periodicity, timeDiffLag, timeDiffLead, normalisedPeriod):
        tempdf = dataframe
        print(tempdf)

        lagMonDiffQ = "lagMonDiffQ"
        leadMonDiffQ = "leadMonDiffQ"
        lagMonDiffA = "lagMonDiffA"
        leadMonDiffA = "leadMonDiffA"

        periodicity = periodicity.lower()
        print(periodColumn)
        if periodicity == '01':
            tempdf[normalisedPeriod] = tempdf.apply(lambda x:str(x[periodColumn]), axis=1)
            tempdf = self.monthsBetween(tempdf, normalisedPeriod, groupByColumns, timeDiffLag, timeDiffLead, periodicity)
          #  tempdf.drop(["previousPeriod", "nextPeriod"], inplace=True,axis=1)

        elif periodicity == '02' or periodicity == 'annually' or periodicity == 'yearly':
            tempdf[normalisedPeriod] = tempdf.apply(lambda x: str(int(str(x[periodColumn])[:4])-1) if str(x[periodColumn])[4:] in ['01','02','03'] else str(x[periodColumn])[:4], axis = 1)
            tempdf = self.monthsBetween(tempdf, normalisedPeriod, groupByColumns, lagMonDiffA, leadMonDiffA, periodicity)
            tempdf.to_csv("PreMon.csv")
            tempdf[timeDiffLag] = tempdf.apply(lambda x: 1 if x[lagMonDiffA] == 12 else 0, axis=1)
            tempdf[timeDiffLead] = tempdf.apply(lambda x: 1 if x[leadMonDiffA] == 12 else 0, axis=1)
            tempdf.to_csv("Mon.csv")
            tempdf.drop([lagMonDiffA,leadMonDiffA,"previousPeriod","nextPeriod"], inplace=True, axis=1)

        elif periodicity == '03' or periodicity == "quarterly":


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

            # Find qnum of period, qnum - lag.

            # NOTE: We dont think this covers all situations, eg  current 201912 prev 201109.
            # Continuing on as this seems to match the scala code, but we want to keep an eye on it.

            #IF NORMALISEDpERIOD AND OTHERPERIOD != BLANK, OUT IS BELOW ELSE: 0
            tempdf[timeDiffLag] = tempdf.apply(lambda x: 1 if (x["previousPeriod"]!= None and int(x[normalisedPeriod][5:]) - int(x["previousPeriod"][5:]) == -3 and x[lagMonDiffQ] <= 5)  else (int(x[normalisedPeriod][5:]) - int(x["previousPeriod"][5:])) if x["previousPeriod"]!= None else 0, axis=1)


            #tempdf[timeDiffLag] = tempdf.apply(lambda x: 1 if(calcQuarter(x[periodColumn]) - calcQuarter(x["previousPeriod"]) == -3 and x[lagMonDiffQ] <= 5) else (calcQuarter(x[periodColumn]) - calcQuarter(x["previousPeriod"])), axis=1)
            tempdf[timeDiffLead] = tempdf.apply(lambda x: 1 if (x["nextPeriod"]!= None and int(x["nextPeriod"][5:]) - int(x[normalisedPeriod][5:]) == -3 and x[leadMonDiffQ] <= 5) else (int(x["nextPeriod"][5:]) - int(x[normalisedPeriod][5:])) if x["nextPeriod"]!= None else 0, axis=1)

            #tempdf[timeDiffLead] = tempdf.apply(lambda x: 1 if (calcQuarter(x["nextPeriod"]) - calcQuarter(x[periodColumn]) == -3 and x[leadMonDiffQ] <= 5) else (calcQuarter(x["nextPeriod"]) - calcQuarter(x[periodColumn])), axis=1)
            tempdf = tempdf.drop([lagMonDiffQ,leadMonDiffQ,"previousPeriod","nextPeriod"], axis=1)

        return tempdf

    def identifyAdjacentTarget(self, dataframe, targetColumn, groupByColumns,  lagTarget, leadTarget, count=1):
        #Within groupby:
            #get first and last period for groupby criteria
            #ensure that every period between first and last exists:
                #if it doesnt, add row with ref, groupbystuff, and 0's


        dataframe.to_csv("withdatetime.csv")


        #get slice containing only current ref. Order by period, now do shift

        #get me a row where period = nextPeriod and groupcols = groupcols
        #dataframe[lagTarget] = dataframe.sort_values(by=groupByColumns)[targetColumn].shift(count).fillna(0).astype(np.int64)
        #dataframe[leadTarget] = dataframe.sort_values(by=groupByColumns)[targetColumn].shift(-count).fillna(0).astype(np.int64)

        #Given a row that is ref, groupyby, period
        #run function to calculate previous period and next period
        #lagtarget = targetcolumn where ref & groupbycolumns = ref & groupbycols AND period = lastperiod
        #dont calculate, keep.
        def adjacentTargetLambda(row, dataframe,laglead):
            out = np.float("nan")
            #print(row[laglead])
            if(row[laglead]!=None):
                sliceOfDataFrame = dataframe[(dataframe[self.normalisedPeriod]==row[laglead])& (dataframe[self.uniqueIdentifier]==row[self.uniqueIdentifier])]
                # & (dataframe[self.uniqueIdentifier]==row[self.uniqueIdentifier]) & (dataframe[groupByColumns] == row[groupByColumns])
                if(sliceOfDataFrame.size>0):
                    out= sliceOfDataFrame[self.targetColumn].tolist()[0]
            print(out)
            print(type(out))
            return out

        #lambda: func(row,dataframe)
        #row['lagtarget'] = dataframe[[normalisedP == rowLastp, ref=ref,groupbys=groupbys]][targetColumn]
        #dataframe[lagTarget] = dataframe.groupby(groupByColumns)[targetColumn].shift(count).fillna(0).astype(np.int64) # backwards

        dataframe[lagTarget] = dataframe.apply(lambda x: adjacentTargetLambda(x,dataframe,'previousPeriod'),axis=1)

        dataframe[leadTarget] = dataframe.groupby(groupByColumns)[targetColumn].shift(-count).fillna(0).astype(np.int64) # forwards
        dataframe.to_csv("adjacentTarget.csv")
        return dataframe

    def buildLinks(self, dataframe, groupByColumns, targetColumn, intervalCol, adjacentTargetColumn, newLinkColumn,impFactorColumn, normalisedPeriod, inclusionValue=1):

        # Note: this section is a refactored version of the dtrades scala code. But for qcas the links are always 1.
        # It works in that it produces factors and doesn't fall over
        # BUT cant be tested against data because our data is qcas, and qcas doesnt calculate the links.

        periodList = dataframe[dataframe[targetColumn].isnull()][normalisedPeriod].drop_duplicates().tolist()
        interimDF = dataframe[dataframe[normalisedPeriod]=="Blank"]
        interimDF[newLinkColumn] = interimDF[normalisedPeriod]
        dataframe.to_csv("Inter.csv")
        print(periodList)
        for period in periodList:
            filtereddataframe = dataframe[(dataframe[intervalCol] == inclusionValue) & (dataframe[targetColumn].notnull()) & (dataframe[adjacentTargetColumn].notnull()) & (dataframe[normalisedPeriod] == str(period))]
            filtereddataframe.to_csv(str(period) + "Inter.csv")
            if filtereddataframe.size >0:
                Aggregateddataframe = filtereddataframe.groupby(groupByColumns,as_index=False).agg({targetColumn:'sum',adjacentTargetColumn:'sum'})

                Aggregateddataframe[newLinkColumn] = Aggregateddataframe.apply(lambda x: x[targetColumn]/x[adjacentTargetColumn] if x[adjacentTargetColumn] != 0 else 0, axis=1)
                interimDF=interimDF.append(Aggregateddataframe)
                interimDF.to_csv(str(period)+"InterEnd.csv")

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