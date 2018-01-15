"""
Evaluation script for CAMELYON17.
"""

import sklearn.metrics
import pandas as ps
import argparse


# ----------------------------------------------------------------------------------------------------

def calculate_kappa(ground_truth, submission):
    """
    Calculate inter-annotator agreement with quadratic weighted Kappa.

    Args:
        ground_truth (pandas.DataFrame): List of labels assigned by the organizers.
        submission (pandas.DataFrame): List of labels assigned by participant.

    Returns:
        float: Kappa score.
    """

    # The accepted stages are pN0, pN0(i+), pN1mi, pN1, pN2 as described on the website. During parsing all strings converted to lowercase.
    #
    stage_list = ['pN0', 'pN0(i+)', 'pN1mi', 'pN1', 'pN2']

    # Extract the patient pN stages from the tables for evaluation.
    #
    ground_truth_map = {df_row[0]: df_row[1] for _, df_row in ground_truth.iterrows() if
                        str(df_row[0]).lower().endswith('.zip')}
    submission_map = {df_row[0]: df_row[1] for _, df_row in submission.iterrows() if
                      str(df_row[0]).lower().endswith('.zip')}

    # Reorganize data into lists with the same patient order and check consistency.
    #
    ground_truth_stage_list = []
    submission_stage_list = []
    for patient_id, ground_truth_stage in ground_truth_map.iteritems():
        # Check consistency: all stages must be from the official stage list and there must be a submission for each patient in the ground truth.
        #
        if ground_truth_stage not in stage_list:
            raise ValueError('Unknown stage in ground truth: {stage}'.format(stage=ground_truth_stage))
        if patient_id not in submission_map:
            raise ValueError('Patient missing from submission: {patient}'.format(patient=patient_id))
        if submission_map[patient_id] not in stage_list:
            raise ValueError('Unknown stage in submission: {stage}'.format(stage=submission_map[patient_id]))

        # Add the pair to the lists.
        #
        ground_truth_stage_list.append(ground_truth_stage)
        submission_stage_list.append(submission_map[patient_id])

    # Return the Kappa score.
    #
    return sklearn.metrics.cohen_kappa_score(y1=ground_truth_stage_list, y2=submission_stage_list, labels=stage_list,
                                             weights='quadratic')


# ----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.

    Returns:
        (str, str): The parsed ground truth and submission CSV file paths.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Calculate inter-annotator agreement.')

    argument_parser.add_argument('-g', '--ground_truth', required=True, type=str, help='ground truth CSV path')
    argument_parser.add_argument('-s', '--submission', required=True, type=str, help='submission CSV path')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_ground_truth_path = arguments['ground_truth']
    parsed_submission_path = arguments['submission']

    return argument_parser.description, parsed_ground_truth_path, parsed_submission_path


# ----------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # Parse parameters.
    #
    description, ground_truth_path, submission_path = collect_arguments()
    print(description)
    print('Ground truth: {path}'.format(path=ground_truth_path))
    print('Submission: {path}'.format(path=submission_path))

    # Load tables to Pandas data frames.
    #
    ground_truth_df = ps.read_csv(ground_truth_path)
    submission_df = ps.read_csv(submission_path)

    # Calculate kappa score.
    #
    try:
        kappa_score = calculate_kappa(ground_truth=ground_truth_df, submission=submission_df)
    except Exception as exception:
        print(exception)
    else:
        print('Score: {score}'.format(score=kappa_score))
