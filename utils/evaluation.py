import numpy as np
import pandas as pd




def evaluate_model(model, test_file : str = '../data/test.csv', submission_file : str = None, row_id_cols : str = 'row_id',
                   target_col : str = 'microbusiness_density', verbose : bool = True):
    """
    Evaluate the model on the test data and save the submission file
    """
    # load the test data
    test = pd.read_csv(test_file)

    # get the rows ids
    ids = test[row_id_cols].values

    # get the test data features
    x_test = model.preprocess(test)

    # compute the target
    y_pred = model.predict(x_test)

    if submission_file is None:
        y_true = test[target_col].values
        score = compute_smape(y_true, y_pred)

        if verbose:
            print(f'score SMAPE: {score:.3f}')

        return score

    else:
        # create a submission file
        submission = pd.DataFrame({'id': ids, 'target': y_pred})

        # save the submission file
        submission.to_csv(submission_file, index=False)

        # print the submission file
        if verbose:
            print(submission.head())



# calculate the smape
def compute_smape(y_true, y_pred):

    if not (isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)):
        y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 2






if __name__ == '__main__':


    # list of true values
    y_true = [1, 2, 3, 4, 5]

    # list of predicted values
    y_pred = [5, 4, 1, 2, 3]

    # calculate smape
    smape = eval_smape(y_true, y_pred)

    # print smape
    print('SMAPE: %.3f' % smape)