# EconStatsQcas

This is the d-trades-esque imputation module. Based heavily on dtrades generic imputation code, but altered to also work for qcas.

Data Requirements:
At a minimum this process needs:<br>
**target column**: The column that should hold a returned value(the one that'll be imputed)<br>
**period column**: A column that gives an indication of period within the data<br>
**groupby_columns**: Column(s) that represent the way that the data is grouped(strata etc)<br>
**unique identifier column**: A column that holds a unique id for a reference(responder_id, etc)<br>
**auxiliary data column**: A column that holds aux data, that is data to be used in construction when there is no previous data to impute from.<br><br>

Data Outputs:
This system will output the original dataset with the addition of 2 columns:<br>
**output column** - The column that should hold the output to imputation(can be separate to target column, or the same column)<br>
**marker column** - The column that will contain the imputation marker representing how it was imputed.<br>
                                                                                             (FI - Forward Imputed,<br>
                                                                                              BI - Backward Imputed,<br>
                                                                                               R - Returned,<br>
                                                                                               C - Constructed,<br>
                                                                                               E - Error)<br><br>



Main Methods:

    """ IDENTIFY INTERVAL
    Description: This method controls the identification of adjacent periods taking into account periodicity. Returning the time difference between current and next/previous period
    :param data: dataframe - The input dataframe
    :param period_column: String - The name of the column that holds the period
    :param groupby_columns: List[String] - The names of the columns to use to group the data
    :param periodicity: String - The periodicity of the survey we are imputing for: 01 = monthly, 02 = annually, 03 = quarterly
    :param time_diff_lag: String - The name of the column containing the period difference between current and previous period
    :param time_diff_lead: String - The name of the column containing the period difference between current and next period
    :param normalised_period: - The name of the column containing the 'normalised period' (for monthly it is the period, annual it is the year, quarterly it is yearQn
    :param previous_period_column: String - The name of the column that holds the previous period for a record
    :param next_period_column: String - The name of the column that holds the next period for a record
    :return: tempdf: Dataframe - The input data with the addition of 3 columns(normalised_period, time_diff_lead, time_diff_lag
    """

    """ IDENTIFY ADJACENT TARGET
    Description: This function finds the value of the target column for previous/next period rows
    :param data: dataframe - The input data
    :param target_column: String - The name of the column holding the target(the column that is to be imputed)
    :param lag_target: String - The name of the column holding the value of the target column for the previous period
    :param lead_target: String - The name of the column holding the value of the target column for the next period
    :param previous_period_column: String - The name of the column that holds the previous period for a record
    :param next_period_column: String - The name of the column that holds the next period for a record
    :param normalised_period: - The name of the column containing the 'normalised period' (for monthly it is the period, annual it is the year, quarterly it is yearQn
    :param unique_identifier: String - The name of the column containing the unique id for a reference(in dtrades this is ref)
    :return: data: dataframe - The input data with the addition of 2 columns(lag_target and lead_target)
    """

    """ BUILD LINKS
    Description: This method produces imputation links for a class(determined by groupby columns) based upon data for responders of that class.
    :param data: dataframe - The input dataframe
    :param groupby_columns: List[String] - The names of the columns to use to group the data
    :param target_column: String     - The name of the column that contains the value for imputation(for qcas it is adjusted_value).
    :param interval_column: String - The name of the column that represents the time difference between current period and next/previous
    :param adjacent_target_column: String - The name of the column that contains the value of the target column for next/previous row
    :param new_link_column: String - The name of the column that will hold the imputation link(forwards or backwards)
    :param imp_factor_column: String - The name of the column that, if exists, will override the links. The imputation link will be set to the content of this column
    :param normalised_period: String - The name of the column containing the 'normalised period' (for monthly it is the period, annual it is the year, quarterly it is yearQn
    :param join_type: String - The type of join used to attach the links onto the dataframe(hardcoded to 'left')
    :param inclusion_value: Int - The value that determines what time diff we are looking for(should be 1 in most cases)
    :return: data: dataframe - The input dataframe with the addition of the new_link_column that holds an imputation link
    """

     """ DO CONSTRUCTION
    Description: This method calculates and applies a construction ratio to rows of the input data(overridden in rolling imputation if there is data to impute from)
    :param data:  Dataframe - The input dataset
    :param groupby_columns: List[String] - The names of the columns to use to group the data
    :param period_column: String - The name of the column that holds the period
    :param target_column: String     - The name of the column that contains the value for imputation(for qcas it is adjusted_value).
    :param auxiliary_column: String - The name of the column containing the auxiliary variable used in construction(for qcas this is selection_emp)
    :param output_column: String     - The name of the column that will contain the output to imputation.
    :param marker_column: String     - The name of the column that will contain the imputation marker representing how it was imputed.
                                                                                             (FI - Forward Imputed,
                                                                                              BI - Backward Imputed,
                                                                                               R - Returned,
                                                                                               C - Constructed,
                                                                                               E - Error)
    :param time_diff_lag: String - The name of the column that holds the time difference between current and previous rows
    :param imp_link: String - The name of the column holding the imputation link for a row
    :param marker_return: String - Marker for data that is returned ('R')
    :param marker_construct: String - Marker for data that is constructed ('C')
    :param marker_error: String - Marker for data that is in error ('E')
    :param join_type: String - The type of join used to attach the links onto the dataframe(hardcoded to 'left')
    :return: data: dataframe - The input dataframe with the addition of 3 new columns(imp_link, marker_column, output_column)
    """

    """ ROLLING IMPUTATION
    Description: This method works on a single row (through an apply method). First it works out which type of imputation needs to take place, then it will do the imputation.

    :param current_row: Row - One row of data from the input data
    :param period_column: String - The name of the column that contains period(for qcas it is 'period').
    :param output_column: String - The name of the column that will contain the output to imputation.
    :param forward_imp_link: String - The name of the column that holds the forward imputation link (for qcas: FWDLINK)
    :param backward_imp_link: String - The name of the column that holds the forward imputation link (for qcas: BWDLINK)
    :param lag_target: String - The name of a column that will contain the value of the target column for the period previous to current one.
    :param lead_target: String - The name of a column that will contain the value of the target column for the period after the current one.
    :param marker_forward_imp: String - Marker for forward imputed rows (FI)
    :param marker_backward_imp: String - Marker for backward imputed rows (FI)
    :param time_diff_lag: String - The period difference between the current period target and the lag_target(taking into account periodicity)
    :param time_diff_lead: String - The period difference between the current period target and the lead_target(taking into account periodicity)
    :param marker_column: String     - The name of the column that will contain the imputation marker representing how it was imputed.
                                                                                             (FI - Forward Imputed,
                                                                                              BI - Backward Imputed,
                                                                                               R - Returned,
                                                                                               C - Constructed,
                                                                                               E - Error)
    :param working_data: Dataframe - The full dataset(required in order to impute based on rows from different periods)
    :param unique_identifier: String - The name of the column containing the unique id for a reference(in dtrades this is ref)

    :return: out: Series(Float)     - Imputed/constructed value for row
             marker: Series(String) - marker determining type of imputation                  (FI - Forward Imputed,
                                                                                              BI - Backward Imputed)
    """

