#from uk.gov.ons.src.baseMethod import baseMethod
import pandas as pd
import numpy as np
import math

class Imputation():
    """
    Description: Imputation Class controls the imputation of data.
    Extends baseMethod.

    Params(constructor):
            :param data: data     - The input dataset with things to be imputed.
            :param imputation_class: List    - Contains the columns to group on(for qcas: classification,cell_no,question).
            :param period_column: String     - The name of the column that contains period(for qcas it is 'period').
            :param unique_identifier: String - The name of the column that contains a unique id for a reference(for qcas it is responder_id).
            :param target_column: String     - The name of the column that contains the value for imputation(for qcas it is adjusted_value).
            :param output_column: String     - The name of the column that will contain the output to imputation.
            :param marker_column: String     - The name of the column that will contain the imputation marker representing how it was imputed.
                                                                                                     (FI - Forward Imputed,
                                                                                                      BI - Backward Imputed,
                                                                                                       R - Returned,
                                                                                                       C - Constructed,
                                                                                                       E - Error)
            :param auxiliary_column: String - The name of the column containing the auxiliary variable used in construction(for qcas this is selection_emp)
            :param periodicity: String     - Indication of whether survey is quarterly, annual, or monthly
            :param imp_factor_column:String  - The name of the column that indicates whether a hardcoded factor is to be used.(If this column has a value use that, else impute as per dtrades)

    returns: working_data: data - The input dataset with imputed/constructed values where necessary.
    """

    backward_imp_link = 'BWDLINK'
    forward_imp_link = 'FWDLINK'
    lag_target = ''
    lead_target = ''
    time_diff_lag = ''
    time_diff_lead = ''
    target_sum = ''
    normalised_period = ''
    monthly = "01"
    annually = "02"
    quarterly = "03"
    marker_forward_imp = "FI"
    marker_backward_imp = "BI"
    marker_construct = "C"
    marker_return = "R"
    marker_error = "E"
    imp_link = "imputation_link"
    join_type = "left"
    imp_factor_column = ''
    previous_period_column = "previous_period"
    next_period_column = "next_period"

    def __init__(self, data, imputation_class, period_column, unique_identifier, target_column, output_column, marker_column, auxiliary_column, periodicity, imp_factor_column):
        """
        Description: Constructor for imputation class, set up global variables and performs 'mandatoryArgsCheck' to ensure that params are not null.
        :param data: data     - The input dataset(with things to be imputed.
        :param imputation_class: List    - Contains the columns to group on(for qcas: classification,cell_no,question).
        :param period_column: String     - The name of the column that contains period(for qcas it is 'period').
        :param unique_identifier: String - The name of the column that contains a unique id for a reference(for qcas it is responder_id).
        :param target_column: String     - The name of the column that contains the value for imputation(for qcas it is adjusted_value).
        :param output_column: String     - The name of the column that will contain the output to imputation.
        :param marker_column: String     - The name of the column that will contain the imputation marker representing how it was imputed.
                                                                                                 (FI - Forward Imputed,
                                                                                                  BI - Backward Imputed,
                                                                                                   R - Returned,
                                                                                                   C - Constructed,
                                                                                                   E - Error)
        :param auxiliary_column: String - The name of the column containing the auxiliary variable used in construction(for qcas this is selection_emp)
        :param periodicity: String     - Indication of whether survey is quarterly, annual, or monthly
        :param imp_factor_column:String  - The name of the column that indicates whether a hardcoded factor is to be used.(If this column has a value use that, else impute as per dtrades)

        """
        # Passed In.
        self.data = data
        self.imputation_class = imputation_class
        self.period_column = period_column
        self.unique_identifier = unique_identifier
        self.target_column = target_column
        self.output_column = output_column
        self.marker_column = marker_column
        self.auxiliary_column = auxiliary_column
        self.periodicity = periodicity
        self.imp_factor_column = imp_factor_column

        # Main Variables
        self.lag_target = "imp_" + target_column + "_lag"
        self.lead_target = "imp_" + target_column + "_lead"
        self.time_diff_lag = "imp_" + period_column + "_diff_lag"
        self.time_diff_lead = "imp_" + period_column + "_diff_lead"
        self.target_sum = "imp_" + target_column + "_sum"
        self.normalised_period = "imp_" + period_column + "_unique"

    def __str__(self):
        return "This Is A Construction And Imputation Module."

    def imputation(self):
        """
        Description: This function controls the flow of imputation, calling the subfunctions before eventually returning the imputed data

        :return: working_data: data - Input data with imputed/constructed values added
        """

        groupby_columns = self.imputation_class.copy()
        groupby_columns.append(self.unique_identifier)

        working_data = self.identify_interval(self.data, self.period_column, groupby_columns, self.periodicity, self.time_diff_lag, self.time_diff_lead, self.normalised_period, self.previous_period_column, self.next_period_column)
        working_data = self.identify_adjacent_target(working_data, self.target_column, self.lag_target, self.lead_target, self.previous_period_column, self.next_period_column)

        groupby_columns = self.imputation_class.copy()
        groupby_columns.append(self.normalised_period)

        working_data = self.build_links(working_data, groupby_columns, self.target_column, self.time_diff_lag, self.lag_target, self.forward_imp_link, self.imp_factor_column, self.normalised_period, self.join_type)

        working_data = self.build_links(working_data, groupby_columns, self.target_column, self.time_diff_lead, self.lead_target, self.backward_imp_link, self.imp_factor_column, self.normalised_period, self.join_type)

        working_data = self.do_construction(working_data, self.imputation_class, self.normalised_period, self.target_column, self.auxiliary_column, self.output_column, self.marker_column, self.time_diff_lag, self.imp_link, self.marker_return, self.marker_construct, self.marker_error, self.join_type)

        imputation_data = working_data[working_data[self.target_column].isnull()]
        if imputation_data.size > 0:
            imputation_data[[self.output_column, self.marker_column]] = imputation_data.apply(
                lambda x: self.rolling_imputation(x, self.normalised_period, self.output_column,
                                                  self.forward_imp_link, self.backward_imp_link, self.lag_target, self.lead_target,
                                                  self.marker_forward_imp, self.marker_backward_imp, self.time_diff_lag,
                                                  self.time_diff_lead, self.marker_column, working_data, self.unique_identifier), axis=1)

            working_data = working_data.dropna(subset=[self.target_column])

            working_data = working_data.append(imputation_data)

        working_data[self.imp_link] = working_data.apply(lambda x: x[self.forward_imp_link] if x[self.marker_column] == self.marker_forward_imp else x[self.backward_imp_link] if x[self.marker_column] == self.marker_backward_imp else x[self.imp_link] if x[self.marker_column] == self.marker_construct else None, axis=1)

        working_data.drop([self.time_diff_lag, self.time_diff_lead, self.lag_target, self.lead_target, self.forward_imp_link, self.backward_imp_link, self.normalised_period], inplace=True, axis=1)
        return working_data

    def rolling_imputation(self, current_row, period_column, output_column, forward_imp_link, backward_imp_link, lag_target, lead_target, marker_forward_imp, marker_backward_imp, time_diff_lag, time_diff_lead, marker_column, working_data, unique_identifier):
        """
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
        temp_forward_imp = 1
        temp_backward_imp = 1
        temp_lag = 0
        temp_lead = 0
        temp_construction = 0

        # This slice needs to include groupby columns.
        sliced_data = working_data[working_data[unique_identifier] == current_row[unique_identifier]]
        periods_for_ref = sliced_data[period_column].drop_duplicates().tolist()
        periods = working_data[period_column].drop_duplicates()

        i = 0
        blank = False
        outlist = []

        for x in periods:
            if i == len(periods_for_ref):
                break
            if current_row[period_column] == x:
                blank = True
            if x == periods_for_ref[i]:
                outlist.append(x)
                i += 1
            else:
                if not blank:
                    outlist = []
                else:
                    break

        sliced_data = sliced_data[(sliced_data[period_column].isin(outlist))]
        insize = 0

        for x in sliced_data.head().iterrows():
            row = x[1]

            # Forward imputation from previous period return.
            if row[period_column] == current_row[period_column] and row[forward_imp_link] > 0 and row[lag_target] > 0 and row[time_diff_lag] == 1:
                temp_forward_imp = row[forward_imp_link]
                temp_lag = row[lag_target]
                break
                # In scala, end loop here.
            insize += 1
            # Define the link fraction for imputation start.

            if row[forward_imp_link] > 0 and row[time_diff_lag] == 1 and row[period_column] <= current_row[period_column]:

                if row[lag_target] > 0:
                    temp_forward_imp = row[forward_imp_link]
                else:
                    temp_forward_imp *= row[forward_imp_link]

            if row[backward_imp_link] > 0 and row[time_diff_lead] == 1 and temp_lead == 0 and (row[lag_target] == 0 or row[lag_target] == None or math.isnan(row[lag_target])) and row[period_column] >= current_row[period_column]:
                temp_backward_imp *= row[backward_imp_link]
            # Define the link fraction for imputation(forward&backward) end.
            # Select a linked response to apply the link fraction start.
            if row[lag_target] > 0 and row[time_diff_lag] == 1 and (row[period_column] <= current_row[period_column]):
                temp_lag = row[lag_target]
            elif row[lead_target] > 0 and row[time_diff_lead] == 1 and temp_lead == 0 and (row[lag_target] == 0 or row[lag_target] == None or math.isnan(row[lag_target])) and row[period_column] >= current_row[period_column]:
                temp_lead = row[lead_target]
            elif row[output_column] > 0 or not math.isnan(row[output_column]):
                temp_construction = row[output_column]

        # Select a linked response to apply the link fraction end.
        out = current_row[output_column]
        marker = current_row[marker_column]
        # Apply link fraction to related response & mark the imputation type start.
        if temp_forward_imp > 0 and temp_lag > 0:
            result = temp_lag * temp_forward_imp
            (out, marker) = (result, marker_forward_imp)
        elif temp_backward_imp > 0 and temp_lead > 0:
            result = temp_lead * temp_backward_imp
            (out, marker) = (result, marker_backward_imp)
        elif temp_construction > 0 and temp_forward_imp > 1 and insize > 1:
            result = temp_construction * temp_forward_imp
            (out, marker) = (result, marker_forward_imp)

        return pd.Series([out, marker])

    def months_between(self, period, other_period):
        """
        Description: Calculates the number of months between 2 given periods
        :param period: String - Current period (yyyymm)
        :param other_period: String - previous or next period (yyyymm)
        :return: :Int - The number of months between the two given periods.
        """
        if other_period != 0 and other_period != None and str(other_period) != '' and str(other_period) != 'nan':
            if ('Q' in str(period)):
                year = 12 * (int(str(period)[:4]) - int(str(other_period)[:4]))
                quarter = int(str(period)[5:]) - int(str(other_period)[5:])
                month = quarter * 3
                return abs(year + month)
            other_period = int(other_period)

            if (len(str(other_period)) == 4):
                return abs((int(period) - int(other_period)) * 12)

            year = 12 * (int(str(period)[:4]) - int(str(other_period)[:4]))
            month = int(str(period)[4:]) - int(str(other_period)[4:])

            return abs(year + month)
        else:
            return 0

    def months_calculation(self, data, period_column, groupby_columns, lag_new_col, lead_new_col, periodicity, previous_period_column, next_period_column):
        """
        Description: This method works out the next/previous period for a row.
        :param data: dataframe     - The input dataset(with things to be imputed.
        :param period_column: String - The name of the column that contains period(for qcas it is 'period').
        :param groupby_columns: List[String] - The names of the columns to use to group the data
        :param lag_new_col: String - The name of the column which will hold the number of periods difference between current and previous data(should be 1 if there is a previous period record)
        :param lead_new_col: String - The name of the column which will hold the number of periods difference between current and next data(should be 1 if there is a next period record)
        :param periodicity: String - The periodicity of the survey we are imputing for: 01 = monthly, 02 = annually, 03 = quarterly
        :param previous_period_column: String - The name of the column that holds the previous period for a record
        :param next_period_column: String - The name of the column that holds the next period for a record

        :return: data: dataframe - The input dataset with 4 new columns(previous_period_column,next_period_column, lag_new_col, lead_new_col
        """

        def calculate_adjacent_periods(row, periodicity, data):
            """
            Description: This method uses periodicity to calculate what should be the adjacent periods for a row, Then uses a filter to confirm whether these periods exist for a record.
            :param row: row - A single row of the input dataframe
            :param periodicity: String - The periodicity of the survey we are imputing for: 01 = monthly, 02 = annually, 03 = quarterly
            :param data: dataframe - The input dataframe
            :return: next_out: String - The next period for the current row
                     last_out: String - The previous period of the current row
            """
            current_period = row[period_column]
            last_out = None
            next_out = None

            if periodicity == self.monthly:

                current_month = str(current_period)[4:]
                current_year = str(current_period)[:4]
                next_month = int(current_month) + int(periodicity)
                next_year = int(current_year)

                if (next_month > 12):
                    next_year += 1
                    next_month -= 12

                if (next_month < 10):
                    next_month = "0" + str(next_month)

                next_period = str(next_year) + str(next_month)

                last_month = int(current_month) - int(periodicity)
                last_year = int(current_year)
                if (last_month < 1):
                    last_year -= 1
                    last_month += 12
                if (last_month < 10):
                    last_month = "0" + str(last_month)

                last_period = str(last_year) + str(last_month)

            elif periodicity == self.annually:

                next_period = str(int(current_period) + 1)
                last_period = str(int(current_period) - 1)

            else:  # quarterly
                current_month = str(current_period)[5:]
                current_year = str(current_period)[:4]
                next_month = int(current_month) + 1
                next_year = int(current_year)

                if (next_month > 4):
                    next_year += 1
                    next_month -= 4

                next_period = str(next_year) + "Q" + str(next_month)

                last_month = int(current_month) - 1
                last_year = int(current_year)
                if (last_month < 1):
                    last_year -= 1
                    last_month += 4

                last_period = str(last_year) + "Q" + str(last_month)

            row_dataframe = pd.DataFrame([row])
            filtered_dataframe = data[data[period_column] == str(next_period)]
            nextPeriodRows = pd.merge(filtered_dataframe, row_dataframe, on=groupby_columns)

            if (nextPeriodRows.size > 0):
                next_out = next_period

            row_dataframe = pd.DataFrame([row])
            filtered_dataframe = data[data[period_column] == str(last_period)]
            last_period_rows = pd.merge(filtered_dataframe, row_dataframe, on=groupby_columns)

            if (last_period_rows.size > 0):
                last_out = last_period

            return pd.Series([next_out, last_out])

        data[[next_period_column, previous_period_column]] = data.apply(lambda x: calculate_adjacent_periods(x, periodicity, data), axis=1)

        data[lag_new_col] = data.apply(lambda x: self.months_between(x[period_column], x[previous_period_column]), axis=1)
        data[lead_new_col] = data.apply(lambda x: self.months_between(x[period_column], x[next_period_column]), axis=1)

        return data

    def identify_interval(self, data, period_column, groupby_columns, periodicity, time_diff_lag, time_diff_lead, normalised_period, previous_period_column, next_period_column):
        """
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
        tempdf = data
        lag_months_diff_quarterly = "lag_months_diff_quarterly"
        lead_months_diff_quarterly = "lead_months_diff_quarterly"
        lag_months_diff_annually = "lag_months_diff_annually"
        lead_months_diff_annually = "lead_months_diff_annually"

        if periodicity == self.monthly:
            tempdf[normalised_period] = tempdf.apply(lambda x: str(x[period_column]), axis=1)
            tempdf = self.months_calculation(tempdf, normalised_period, groupby_columns, time_diff_lag, time_diff_lead, periodicity, previous_period_column, next_period_column)

        elif periodicity == self.annually:
            tempdf[normalised_period] = tempdf.apply(lambda x: str(int(str(x[period_column])[:4]) - 1) if str(x[period_column])[4:] in ['01', '02', '03'] else str(x[period_column])[:4], axis=1)
            tempdf = self.months_calculation(tempdf, normalised_period, groupby_columns, lag_months_diff_annually, lead_months_diff_annually, periodicity, previous_period_column, next_period_column)

            tempdf[time_diff_lag] = tempdf.apply(lambda x: 1 if x[lag_months_diff_annually] == 12 else 0, axis=1)
            tempdf[time_diff_lead] = tempdf.apply(lambda x: 1 if x[lead_months_diff_annually] == 12 else 0, axis=1)
            tempdf.drop([lag_months_diff_annually, lead_months_diff_annually], inplace=True, axis=1)

        elif periodicity == self.quarterly:

            def calc_normalised_period(period):

                period = int(period)
                year = str(period)[:4]
                quarter = 7
                if period > 10:
                    quarter = math.ceil(int(str(period)[4:]) / 3)
                else:
                    quarter = 99

                out = year + "Q" + str(quarter)
                return out

            tempdf[normalised_period] = tempdf.apply(lambda x: calc_normalised_period(x[period_column]), axis=1)
            tempdf = self.months_calculation(tempdf, normalised_period, groupby_columns, lag_months_diff_quarterly, lead_months_diff_quarterly, periodicity, previous_period_column, next_period_column)
            tempdf[time_diff_lag] = tempdf.apply(lambda x: 1 if (x[previous_period_column] != None and int(x[normalised_period][5:]) - int(x[previous_period_column][5:]) == -3 and x[lag_months_diff_quarterly] <= 5) else (int(x[normalised_period][5:]) - int(x[previous_period_column][5:])) if x[previous_period_column] != None else 0, axis=1)
            tempdf[time_diff_lead] = tempdf.apply(lambda x: 1 if (x[next_period_column] != None and int(x[next_period_column][5:]) - int(x[normalised_period][5:]) == -3 and x[lead_months_diff_quarterly] <= 5) else (int(x[next_period_column][5:]) - int(x[normalised_period][5:])) if x[next_period_column] != None else 0, axis=1)

            tempdf = tempdf.drop([lag_months_diff_quarterly, lead_months_diff_quarterly], axis=1)

        return tempdf

    def identify_adjacent_target(self, data, target_column, lag_target, lead_target, previous_period_column, next_period_column):
        """
        Description: This function finds the value of the target column for previous/next period rows
        :param data: dataframe - The input data
        :param target_column: String - The name of the column holding the target(the column that is to be imputed)
        :param lag_target: String - The name of the column holding the value of the target column for the previous period
        :param lead_target: String - The name of the column holding the value of the target column for the next period
        :param previous_period_column: String - The name of the column that holds the previous period for a record
        :param next_period_column: String - The name of the column that holds the next period for a record
        :return: data: dataframe - The input data with the addition of 2 columns(lag_target and lead_target)
        """
        def adjacent_target_calculation(row, data, adjacentPeriod):
            """
            Description: The main method of identify_adjacent_target. Takes a row and collects the adjacent target value from next/previous period row
            :param row: row - A single row from the input data
            :param data: dataframe - The input dataframe
            :param adjacentPeriod: String - The name of the column containing the adjacent period(next or previous)
            :return: out: float - The value of the target column for the adjacent period
            """
            out = np.float("nan")
            if (row[adjacentPeriod] != None and str(row[adjacentPeriod]) != 'nan'):
                data_slice = data[(data[self.normalised_period] == row[adjacentPeriod]) & (data[self.unique_identifier] == row[self.unique_identifier])]
                if (data_slice.size > 0):
                    out = data_slice[target_column].tolist()[0]
            return out

        data[lag_target] = data.apply(lambda x: adjacent_target_calculation(x, data, previous_period_column), axis=1)
        data[lead_target] = data.apply(lambda x: adjacent_target_calculation(x, data, next_period_column), axis=1)

        return data

    def build_links(self, data, groupby_columns, target_column, interval_column, adjacent_target_column, new_link_column, imp_factor_column, normalised_period, join_type, inclusion_value=1):
        """
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
        period_list = data[data[target_column].isnull()][normalised_period].drop_duplicates().tolist()
        interim_data = data[data[normalised_period] == "Blank"]
        interim_data[new_link_column] = interim_data[normalised_period]
#newest

        for period in period_list:

            filtered_dataframe = data[(data[interval_column] == inclusion_value) & (data[target_column].notnull()) & (data[adjacent_target_column].notnull()) & (data[normalised_period] == str(period))]

            if filtered_dataframe.size > 0:
                aggregated_data = filtered_dataframe.groupby(groupby_columns, as_index=False).agg({target_column: 'sum', adjacent_target_column: 'sum'})
                aggregated_data[new_link_column] = aggregated_data.apply(lambda x: x[target_column] / x[adjacent_target_column] if x[adjacent_target_column] != 0 else 0, axis=1)
                interim_data = interim_data.append(aggregated_data)

        columns_to_select = groupby_columns.copy()
        columns_to_select.append(new_link_column)
        interim_data = interim_data[columns_to_select]
        data = pd.merge(data, interim_data, on=groupby_columns, how=join_type)

        if imp_factor_column in data.columns.values:
            data[new_link_column] = data.apply(lambda x: x[imp_factor_column] if not math.isnan(x[imp_factor_column]) else x[new_link_column], axis=1)

        return data


    def do_construction(self, data, groupby_columns, period_column, target_column, auxiliary_column, output_column, marker_column, time_diff_lag, imp_link, marker_return, marker_construct, marker_error, join_type):
        """
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

        groupby_columns.append(period_column)
        working_data = data[data[target_column].notnull()]
        working_data = working_data.groupby(groupby_columns, as_index=False).agg({target_column: 'sum', auxiliary_column: 'sum'})
        working_data[imp_link] = working_data[target_column] / working_data[auxiliary_column]
        merge_columns = groupby_columns.copy()
        merge_columns.append(imp_link)
        data = pd.merge(data, working_data[merge_columns], how=join_type, left_on=groupby_columns, right_on=groupby_columns)
        data[output_column] = data.apply(lambda x: x[auxiliary_column] * x[imp_link] if ((math.isnan(x[target_column])) and (x[time_diff_lag] != 1) and (not math.isnan(x[auxiliary_column]))) else None if ((math.isnan(x[target_column])) and (math.isnan(x[auxiliary_column]))) else x[target_column], axis=1)
        # (ABOVE)if targetCol is NULL, time_diff_lag!=1 and auxiliary_column is NOT NULL    =  outputCol =auxCol/imp_link
        # elseif targetcol is NULL and auxCol is NULL = outputCol = NULL
        # else outputCol = targetCol
        data[marker_column] = data.apply(lambda x: marker_return if (not math.isnan(x[target_column])) else marker_construct if (not math.isnan(x[output_column])) else marker_error if ((math.isnan(x[target_column])) and (math.isnan(x[auxiliary_column]))) else None, axis=1)
        # (ABOVE)if targetCol is not null   = R
        # else if outputCol is not null = C
        # else if targetCol is NULL and auxCol is NULL = E
        # else = Null
        return data


# imputer = Imputation(input_data,['classification','cell_no','question'],'period','responder_id','adjusted_value','adjusted_values','MarkerCol','selection_data','q',201809,'imp_factor')
#imputer = Imputation(input_data, ['strata'], 'time', 'ref', 'value', 'values', 'MarkerCol', 'frozen_value', '01', 'imp_factor')
#workingData = imputer.imputation()
