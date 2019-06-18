
import uk.gov.ons.src.imputation as imputation
import pandas as pd
from pandas.util.testing import assert_frame_equal


def test_months_between():
    # Test that given 2 periods, function will return number of months between
    assert imputation.months_between(201712, 201812) == 12

    # Test that if period 0, return 0
    assert imputation.months_between(201712, 0) == 0


def test_monthsbetween():
    # given a dataframe of #period	responder_id
    #  201806	123456789
    #  201803	123456789
    # test that outpyut is x

    test_data = pd.DataFrame(
        data={
            "period": ["201806", "201805", "201804"],
            "responder_id": [123456789, 123456789, 123456789],
            "normalised_period": ["201806", "201805", "201804"],
        }
    )
    mydata = imputation.months_calculation(
        test_data,
        "period",
        "responder_id",
        "lagNewCol",
        "leadNewCol",
        "01",
        "previous_period_column",
        "next_period_column",
    )
    mydata.drop(["next_period_column", "previous_period_column"], inplace=True, axis=1)

    my_test_output = pd.DataFrame(
        data={
            "period": ["201806", "201805", "201804"],
            "responder_id": [123456789, 123456789, 123456789],
            "normalised_period": ["201806", "201805", "201804"],
            "lagNewCol": [1, 1, 0],
            "leadNewCol": [0, 1, 1],
        }
    )

    assert_frame_equal(mydata, my_test_output)


def test_identify_interval():
    # correctly identify quarterly interval
    # test_data = pd.DataFrame(data={"period":[201806,201803],"responder_id":[123456789,123456789]})
    test_data = pd.DataFrame(
        data={"period": [201803, 201806], "responder_id": [123456789, 123456789]}
    )
    test_out = imputation.identify_interval(
        test_data,
        "period",
        "responder_id",
        "03",
        "time_diff_lag",
        "time_diff_lead",
        "normalised_period",
        "previous_period_column",
        "next_period_column",
    )

    test_out.drop(
        ["normalised_period", "previous_period_column", "next_period_column"],
        axis=1,
        inplace=True,
    )
    assert "time_diff_lag" in test_out.columns.values
    my_test_output = pd.DataFrame(
        data={
            "period": [201803, 201806],
            "responder_id": [123456789, 123456789],
            "time_diff_lag": [0, 1],
            "time_diff_lead": [1, 0],
        }
    )

    assert_frame_equal(test_out, my_test_output)

    # identify quarterly interview when crossing year boundary
    test_data = pd.DataFrame(
        data={"period": [201712, 201803], "responder_id": [123456789, 123456789]}
    )
    test_out = imputation.identify_interval(
        test_data,
        "period",
        "responder_id",
        "03",
        "time_diff_lag",
        "time_diff_lead",
        "normalised_period",
        "previous_period_column",
        "next_period_column",
    )
    test_out.drop(
        ["normalised_period", "previous_period_column", "next_period_column"],
        axis=1,
        inplace=True,
    )
    assert "time_diff_lag" in test_out.columns.values
    my_test_output = pd.DataFrame(
        data={
            "period": [201712, 201803],
            "responder_id": [123456789, 123456789],
            "time_diff_lag": [0, 1],
            "time_diff_lead": [1, 0],
        }
    )

    assert_frame_equal(test_out, my_test_output)
    print("AAAAERGH")
    # correctly identify yearly interval.
    test_data = pd.DataFrame(
        data={"period": [201712, 201812], "responder_id": [123456789, 123456789]}
    )
    test_out = imputation.identify_interval(
        test_data,
        "period",
        "responder_id",
        "02",
        "time_diff_lag",
        "time_diff_lead",
        "normalised_period",
        "previous_period_column",
        "next_period_column",
    )

    test_out.drop(
        ["normalised_period", "previous_period_column", "next_period_column"],
        axis=1,
        inplace=True,
    )
    assert "time_diff_lag" in test_out.columns.values
    my_test_output = pd.DataFrame(
        data={
            "period": [201712, 201812],
            "responder_id": [123456789, 123456789],
            "time_diff_lag": [0, 1],
            "time_diff_lead": [1, 0],
        }
    )
    assert_frame_equal(test_out, my_test_output)

    # correctly identify monthly interval
    test_data = pd.DataFrame(
        data={"period": [201812, 201811], "responder_id": [123456789, 123456789]}
    )
    test_out = imputation.identify_interval(
        test_data,
        "period",
        "responder_id",
        "01",
        "time_diff_lag",
        "time_diff_lead",
        "normalised_period",
        "previous_period_column",
        "next_period_column",
    )
    test_out.drop(
        ["normalised_period", "previous_period_column", "next_period_column"],
        axis=1,
        inplace=True,
    )
    assert "time_diff_lag" in test_out.columns.values
    my_test_output = pd.DataFrame(
        data={
            "period": [201812, 201811],
            "responder_id": [123456789, 123456789],
            "time_diff_lag": [1, 0],
            "time_diff_lead": [0, 1],
        }
    )
    assert_frame_equal(test_out, my_test_output)

    # test for same month
    test_data = pd.DataFrame(
        data={"period": [201812, 201812], "responder_id": [123456789, 123456789]}
    )
    test_out = imputation.identify_interval(
        test_data,
        "period",
        "responder_id",
        "01",
        "time_diff_lag",
        "time_diff_lead",
        "normalised_period",
        "previous_period_column",
        "next_period_column",
    )
    test_out.drop(
        ["normalised_period", "previous_period_column", "next_period_column"],
        axis=1,
        inplace=True,
    )
    assert "time_diff_lag" in test_out.columns.values
    my_test_output = pd.DataFrame(
        data={
            "period": [201812, 201812],
            "responder_id": [123456789, 123456789],
            "time_diff_lag": [0, 0],
            "time_diff_lead": [0, 0],
        }
    )
    assert_frame_equal(test_out, my_test_output)

    # test crossing year boundary
    test_data = pd.DataFrame(
        data={"period": [201812, 201901], "responder_id": [123456789, 123456789]}
    )
    test_out = imputation.identify_interval(
        test_data,
        "period",
        "responder_id",
        "01",
        "time_diff_lag",
        "time_diff_lead",
        "normalised_period",
        "previous_period_column",
        "next_period_column",
    )
    test_out.drop(
        ["normalised_period", "previous_period_column", "next_period_column"],
        axis=1,
        inplace=True,
    )
    assert "time_diff_lag" in test_out.columns.values
    my_test_output = pd.DataFrame(
        data={
            "period": [201812, 201901],
            "responder_id": [123456789, 123456789],
            "time_diff_lag": [0, 1],
            "time_diff_lead": [1, 0],
        }
    )
    assert_frame_equal(test_out, my_test_output)


def test_identify_adjacent_target():
    import numpy as np

    test_data = pd.DataFrame(
        data={
            "period": [201812, 201901, 201902],
            "responder_id": [123456789, 123456789, 123456789],
            "target": [1, 2, 3],
            "previous_period_column": [None, 201812, 201901],
            "normalised_period": [201812, 201901, 201902],
            "imp_period_unique": [201812, 201901, 201902],
            "next_period_column": [201901, 201902, None],
        }
    )

    testout = imputation.identify_adjacent_target(
        test_data,
        "target",
        "imp_target_lag",
        "imp_target_lead",
        "previous_period_column",
        "next_period_column", "normalised_period", "responder_id"
    ).drop(["normalised_period"], axis=1)
    my_test_output = pd.DataFrame(
        data={
            "period": [201812, 201901, 201902],
            "responder_id": [123456789, 123456789, 123456789],
            "target": [1, 2, 3],
            "previous_period_column": [None, 201812, 201901],
            "imp_period_unique": [201812, 201901, 201902],
            "next_period_column": [201901, 201902, None],
            "imp_target_lag": [np.float("nan"), 1.0, 2.0],
            "imp_target_lead": [2.0, 3.0, np.float("nan")],
        }
    )

    assert_frame_equal(testout, my_test_output)


def test_build_links():
    # Test forward imputation
    test_data = pd.DataFrame(
        data={
            "period": [201812, 201901, 201902, 201901],
            "responder_id": [123456789, 123456789, 123456789, 123456789],
            "target": [None, None, None, 8.0],
            "normalisedPeriod": ["201812", "201901", "201901", "201901"],
            "imp_target_lead": [8.0, 8.0, 8.0, 8.0],
            "timeDiffLead": [1, 1, 1, 1],
        }
    )
    testout = imputation.build_links(
        test_data,
        ["responder_id"],
        "target",
        "timeDiffLead",
        "imp_target_lead",
        "FWDLINK",
        "imp_factor_column",
        "normalisedPeriod",
        "left",
    )
    my_test_output = pd.DataFrame(
        data={
            "period": [201812, 201901, 201902, 201901],
            "responder_id": [123456789, 123456789, 123456789, 123456789],
            "target": [None, None, None, 8.0],
            "normalisedPeriod": ["201812", "201901", "201901", "201901"],
            "imp_target_lead": [8.0, 8.0, 8.0, 8.0],
            "timeDiffLead": [1, 1, 1, 1],
            "FWDLINK": [1.0, 1.0, 1.0, 1.0],
        }
    )

    assert_frame_equal(testout, my_test_output)

    # test backward imputation
    # groupByColumns, normalisedPeriod, targetColumn, intervalCol, adjacentTargetColumn, newLinkColumn
    test_data = pd.DataFrame(
        data={
            "period": [201812, 201901, 201902, 201901],
            "responder_id": [123456789, 123456789, 123456789, 123456789],
            "target": [None, None, None, 8.0],
            "normalisedPeriod": ["201812", "201901", "201901", "201901"],
            "imp_target_lag": [8.0, 8.0, 8.0, 8.0],
            "timeDiffLag": [1, 1, 1, 1],
        }
    )

    testout = imputation.build_links(
        test_data,
        ["responder_id"],
        "target",
        "timeDiffLag",
        "imp_target_lag",
        "BWDLINK",
        "imp_factor_column",
        "normalisedPeriod",
        "left",
    )

    my_test_output = pd.DataFrame(
        data={
            "period": [201812, 201901, 201902, 201901],
            "responder_id": [123456789, 123456789, 123456789, 123456789],
            "target": [None, None, None, 8.0],
            "normalisedPeriod": ["201812", "201901", "201901", "201901"],
            "imp_target_lag": [8.0, 8.0, 8.0, 8.0],
            "timeDiffLag": [1, 1, 1, 1],
            "BWDLINK": [1.0, 1.0, 1.0, 1.0],
        }
    )
    assert_frame_equal(testout, my_test_output)


def test_do_construction():

    import numpy as np

    test_data = pd.DataFrame(
        data={
            "classification": [11000, 11000],
            "cell_no": [115, 115],
            "responder_id": [123, 321],
            "question_no": [689, 689],
            "adjusted_value": [np.float("nan"), 6],
            "selection_data": [563, 563],
            "period": [201809, 201809],
            "timeDiffLag": [-1, -1],
            "timeDiffLead": [1, 1],
        }
    )
    testout = imputation.do_construction(
        test_data,
        ["classification", "cell_no", "question_no"],
        "period",
        "adjusted_value",
        "selection_data",
        "Constructed_Value",
        "MarkerCol",
        "timeDiffLag",
        "imputation_link",
        "R",
        "C",
        "E",
        "left",
    )

    my_test_output = pd.DataFrame(
        data={
            "classification": [11000, 11000],
            "cell_no": [115, 115],
            "responder_id": [123, 321],
            "question_no": [689, 689],
            "adjusted_value": [np.float("nan"), 6.0],
            "selection_data": [563, 563],
            "period": [201809, 201809],
            "Constructed_Value": [6.0, 6.0],
            "MarkerCol": ["C", "R"],
        }
    )
    testout.drop(
        ["timeDiffLag", "timeDiffLead", "imputation_link"], axis=1, inplace=True
    )

    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None
    ):  # more options can be specified also
        print(testout)
        print("--------------")
        print(my_test_output)

    # drop adjusted value because test fails on comparing NaN's
    assert_frame_equal(testout, my_test_output)

    # dataframe, groupByColumns, periodColumn, targetColumn, auxColumn, outputColumn, outMarkerCol


# 1 crossing year boundary, one doesnt cross year boundary, one has months far apart
# test same month(should get 0's)
# test_doConstruction()

test_identify_interval()
test_identify_adjacent_target()
test_months_between()
test_monthsbetween()
test_build_links()
test_do_construction()
