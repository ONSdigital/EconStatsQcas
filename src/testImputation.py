import pytest
import venv.uk.gov.ons.src.imputation as imputation
import pandas as pd
from pandas.util.testing import assert_frame_equal
imputation=imputation.Imputation()

def test_NullParams():
    #Test that null values cause exception

    with pytest.raises(Exception):
        bob = None
        boris = imputation.imputation(bob)

def test_Months_Between():
    #Test that given 2 periods, function will return number of months between
    assert(imputation.months_between(201712,201812) == 12)

    #Test that if period 0, return 0
    assert(imputation.months_between(201712,0) == 0)

def test_monthsbetween():
    #given a dataframe of #period	responder_id
                          #  201806	123456789
                          #  201803	123456789
    #test that outpyut is x

    testData = pd.read_csv(
        """\\\\nsdata6\\INGRES_CSAI\\Economic Stats BAU Team\\Transformation\\Developer Folders\\Mike\\testdataforimputation\\initial3.csv""")
    mydata = imputation.monthsBetween(testData,"period","responder_id","lagNewCol","leadNewCol")
    mydata.drop(["nextPeriod", "previousPeriod"],inplace=True,axis=1)
    myTestOutput = pd.DataFrame(data={"period":[201806,201803],"responder_id":[123456789,123456789],"lagNewCol":[0,3],"leadNewCol":[3,0]})

    assert_frame_equal(mydata, myTestOutput)


def test_identifyInterval():
    #correctly identify quarterly interval
    #testData = pd.DataFrame(data={"period":[201806,201803],"responder_id":[123456789,123456789]})
    testData = pd.DataFrame(data={"period": [201803,201806], "responder_id": [123456789, 123456789]})
    testout = imputation.identifyInterval(testData,"period","responder_id","q")
    testout.to_csv("tteestout.csv")
    testout.drop(["normalisedPeriod", "previousPeriod","nextPeriod"],axis=1,inplace=True)
    assert("timeDiffLag" in testout.columns.values)
    myTestOutput = pd.DataFrame(data={"period":[201803,201806],"responder_id":[123456789,123456789],"timeDiffLag":[-98,1],"timeDiffLead":[1,97]})
    print(myTestOutput)
    print(testout)
    assert_frame_equal(testout, myTestOutput)

    #identify quarterly interview when crossing year boundary
    testData = pd.DataFrame(data={"period":[201712,201803],"responder_id":[123456789,123456789]})
    testout = imputation.identifyInterval(testData,"period","responder_id","q")
    testout.drop(["normalisedPeriod", "previousPeriod","nextPeriod"],axis=1,inplace=True)
    assert("timeDiffLag" in testout.columns.values)
    myTestOutput = pd.DataFrame(data={"period":[201712,201803],"responder_id":[123456789,123456789],"timeDiffLag":[-95,1],"timeDiffLead":[1,98]})


    assert_frame_equal(testout, myTestOutput)
    print("AAAAERGH")
    # correctly identify yearly interval.
    testData = pd.DataFrame(data={"period":[201712,201812],"responder_id":[123456789,123456789]})
    testout = imputation.identifyInterval(testData, "period", "responder_id", "a")

    testout.drop(["normalisedPeriod"], axis=1,inplace=True)
    assert ("timeDiffLag" in testout.columns.values)
    myTestOutput = pd.DataFrame(data={"period": [201712, 201812], "responder_id": [123456789, 123456789], "timeDiffLag": [0, 1],"timeDiffLead": [1, 0]})
    assert_frame_equal(testout, myTestOutput)


    #correctly identify monthly interval
    testData = pd.DataFrame(data={"period": [201812, 201811], "responder_id": [123456789, 123456789]})
    testout = imputation.identifyInterval(testData, "period", "responder_id", "m")
    testout.drop(["normalisedPeriod"], axis=1, inplace=True)
    assert ("timeDiffLag" in testout.columns.values)
    myTestOutput = pd.DataFrame(
        data={"period": [201812, 201811], "responder_id": [123456789, 123456789], "timeDiffLag": [0, 1],
              "timeDiffLead": [1, 0]})
    assert_frame_equal(testout, myTestOutput)

    #test for same month
    testData = pd.DataFrame(data={"period": [201812, 201812], "responder_id": [123456789, 123456789]})
    testout = imputation.identifyInterval(testData, "period", "responder_id", "m")
    testout.drop(["normalisedPeriod"], axis=1, inplace=True)
    assert ("timeDiffLag" in testout.columns.values)
    myTestOutput = pd.DataFrame(
        data={"period": [201812, 201812], "responder_id": [123456789, 123456789], "timeDiffLag": [0, 0],
              "timeDiffLead": [0, 0]})
    assert_frame_equal(testout, myTestOutput)

    #test crossing year boundary
    testData = pd.DataFrame(data={"period": [201812, 201901], "responder_id": [123456789, 123456789]})
    testout = imputation.identifyInterval(testData, "period", "responder_id", "m")
    testout.drop(["normalisedPeriod"], axis=1, inplace=True)
    assert ("timeDiffLag" in testout.columns.values)
    myTestOutput = pd.DataFrame(
        data={"period": [201812, 201901], "responder_id": [123456789, 123456789], "timeDiffLag": [0, 1],
              "timeDiffLead": [1, 0]})
    assert_frame_equal(testout, myTestOutput)

def test_identifyAdjacentTarget():
    testData = pd.DataFrame(data={"period": [201812, 201901,201902], "responder_id": [123456789, 123456789, 123456789],"target":[1,2,3]})
    testout = imputation.identifyAdjacentTarget(testData,"target","period","responder_id")
    myTestOutput = pd.DataFrame(data={"period": [201812, 201901,201902], "responder_id": [123456789, 123456789, 123456789],"target":[1,2,3],"imp_target_lag":[0,1,2],"imp_target_lead":[2,3,0]})

    assert_frame_equal(testout, myTestOutput)

def test_buildLinks():
    #Test forward imputation
    testData = pd.DataFrame(data={"period": [201812, 201812, 201903], "responder_id": [123456789, 123456789, 123456789],"target": [1, 2, 0],"normalisedPeriod":[201812,201812,201903],"imp_target_lead":[2,0,0],"timeDiffLead": [1,1,0]})
    testout = imputation.buildLinks(testData,["responder_id"],'normalisedPeriod','target','timeDiffLead','imp_target_lead','FWDLINK')
    myTestOutput = pd.DataFrame(data={"period": [201812,201812], "responder_id": [123456789, 123456789],"target": [1, 2],"normalisedPeriod":[201812,201812],"imp_target_lead":[2,0],"timeDiffLead": [1,1],"FWDLINK":[1.5,1.5]})
    assert_frame_equal(testout, myTestOutput)

    #test backward imputation
   # groupByColumns, normalisedPeriod, targetColumn, intervalCol, adjacentTargetColumn, newLinkColumn
    testData = pd.DataFrame(data={"period": [201812, 201812, 201903], "responder_id": [123456789, 123456789, 123456789],
                                  "target": [1, 2, 0], "normalisedPeriod": [201812, 201812, 201903],
                                  "imp_target_lag": [2, 0, 0], "timeDiffLag": [1, 1, 0]})
    testout = imputation.buildLinks(testData, ["responder_id"], 'normalisedPeriod', 'target', 'timeDiffLag',
                                    'imp_target_lag', 'BWDLINK')
    myTestOutput = pd.DataFrame(
        data={"period": [201812, 201812], "responder_id": [123456789, 123456789], "target": [1, 2],
              "normalisedPeriod": [201812, 201812], "imp_target_lag": [2, 0], "timeDiffLag": [1, 1],
              "BWDLINK": [1.5, 1.5]})
    assert_frame_equal(testout, myTestOutput)

def test_doConstruction():
    import numpy as np
    testData = pd.DataFrame(data={"classification":[11000,11000],"cell_no":[115,115],"responder_id":[123,321],"question_no":[689,689],"adjusted_value":[np.float("nan"),6],"selection_data":[563,563],"period":[201809,201809],"timeDiffLag":[-1,-1],"timeDiffLead":[1,1]})
    testout = imputation.doConstruction(testData,['classification',"cell_no","question_no"],"period","adjusted_value","selection_data",'Constructed_Value','MarkerCol')

    myTestOutput = pd.DataFrame(data={"classification": [11000, 11000], "cell_no": [115, 115], "responder_id": [123, 321],
                                  "question_no": [689, 689], "adjusted_value": [np.float("nan"),6.0], "selection_data": [563, 563],
                                  "period": [201809, 201809],"Constructed_Value":[3.0,6.0],"MarkerCol":["C","R"]})
    testout.drop(["timeDiffLag", "timeDiffLead", "imputation_link"], axis=1, inplace=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(testout)
        print("--------------")
        print(myTestOutput)

    #drop adjusted value because test fails on comparing NaN's
    assert_frame_equal(testout, myTestOutput)

    #dataframe, groupByColumns, periodColumn, targetColumn, auxColumn, outputColumn, outMarkerCol



#1 crossing year boundary, one doesnt cross year boundary, one has months far apart
#test same month(should get 0's)
#test_doConstruction()