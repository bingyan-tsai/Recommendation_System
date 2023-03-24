
import pandas as pd
from tqdm import tqdm


def group_dataset(threshold_c1, threshold_user):
    print('>> start loading & grouping data')
    acg = pd.read_csv(r"C:\Users\Jason\Desktop\巴哈姆特\data\acg.csv")
    acg_review = acg[acg[' action'] == 'review']

    # aggregate 'acg_review' by c1
    agg_reviews_c1 = acg_review.groupby(' c1').agg(mean_rating=(' score', 'mean'),
                                                   number_of_ratings=(' score', 'count')).reset_index()

    agg_reviews_c1 = agg_reviews_c1.sort_values(by=['number_of_ratings', 'mean_rating'], ascending=[False, False])

    # aggregate dataset 'acg_review' by user_id
    agg_reviews_user = acg_review.groupby('userid').agg(mean_rating=(' score', 'mean'),
                                                        number_of_ratings=(' score', 'count')).reset_index()

    agg_reviews_user = agg_reviews_user.sort_values(by=['number_of_ratings', 'mean_rating'], ascending=[False, False])

    # extract c1 with more than n rates
    agg_reviews_c1_n = agg_reviews_c1[agg_reviews_c1['number_of_ratings'] > threshold_c1]
    print(f'>> there are {agg_reviews_c1_n.shape[0]} c1 with more than {threshold_c1} rating')
    # extract users who rate more than n times
    agg_reviews_user_n = agg_reviews_user[agg_reviews_user['number_of_ratings'] > threshold_user]
    print(f'>> there are {agg_reviews_user_n.shape[0]} user rate more than {threshold_user} times')

    return acg_review, agg_reviews_c1_n, agg_reviews_user_n


def fetch_user_item_matrix(acg_review, agg_reviews_c1_n, agg_reviews_user_n, normalize=True):
    print('>> start fetching user-item matrix')
    # extract c1 that has been rated more than n times
    acg_review_n = pd.merge(acg_review, agg_reviews_c1_n[[' c1']], on=' c1', how='inner')

    # acg_review_n contains user's Count(rate)>50 & c1's Count(rate)>100
    acg_review_n = pd.merge(acg_review_n, agg_reviews_user_n[['userid']], on='userid', how='inner')

    # create user-item matrix (user-c1 matrix)
    ui_matrix = acg_review_n.pivot_table(index='userid', columns=' c1', values=' score')

    while normalize:
        ui_matrix = ui_matrix.subtract(ui_matrix.mean(axis=1), axis='rows')
        break

    print(f'>> shape of user-item matrix: {ui_matrix.shape}')

    return ui_matrix


def fetch_user_c1_dict(df, group_by='userid'):
    """
    user_c1_dict = {'User_id': 'List of c1'}
    """
    user_c1_dict = {}

    print('>> start grouping data')
    g = df.groupby(group_by)

    # c1_of_user is the df that shows all c1 for user i
    for user, c1_of_user in tqdm(g):
        user_c1_dict[user] = list(set(c1_of_user[' c1'].values))

    print('>> successfully create user_c1_dict')

    return user_c1_dict
