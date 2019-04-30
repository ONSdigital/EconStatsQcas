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
    myTestOutput = pd.DataFrame(data={"period":[201806,201803],"responder_id":[123456789,123456789],"lagNewCol":[3,0],"leadNewCol":[0,3]})

    assert_frame_equal(mydata, myTestOutput)


def test_identifyInterval():
    #correctly identify quarterly interval
    testData = pd.DataFrame(data={"period":[201806,201803],"responder_id":[123456789,123456789]})
    testout = imputation.identifyInterval(testData,"period","responder_id","q")
    testout.drop(["normalisedPeriod", "previousPeriod","lagMonDiffQ","nextPeriod","leadMonDiffQ"],axis=1,inplace=True)
    assert("timeDiffLag" in testout.columns.values)
    myTestOutput = pd.DataFrame(data={"period":[201806,201803],"responder_id":[123456789,123456789],"timeDiffLag":[1,-98],"timeDiffLead":[97,1]})
    assert_frame_equal(testout, myTestOutput)

    #identify quarterly interview when crossing year boundary
    testData = pd.DataFrame(data={"period":[201712,201803],"responder_id":[123456789,123456789]})
    testout = imputation.identifyInterval(testData,"period","responder_id","q")
    testout.drop(["normalisedPeriod", "previousPeriod","lagMonDiffQ","nextPeriod","leadMonDiffQ"],axis=1,inplace=True)
    assert("timeDiffLag" in testout.columns.values)
    myTestOutput = pd.DataFrame(data={"period":[201712,201803],"responder_id":[123456789,123456789],"timeDiffLag":[-95,1],"timeDiffLead":[1,98]})
    print(testout)
    print(myTestOutput)

    assert_frame_equal(testout, myTestOutput)

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
        data={"period": [201812, 201811], "responder_id": [123456789, 123456789], "timeDiffLag": [1, 0],
              "timeDiffLead": [0, 1]})
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

    print(testout)
    print(myTestOutput)
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


#1 crossing year boundary, one doesnt cross year boundary, one has months far apart
#test same month(should get 0's)
test_buildLinks()