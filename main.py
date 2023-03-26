
import warnings
from apriori import run_apriori
from preprocess import group_dataset, fetch_user_item_matrix
from collaborative_filtering import fetch_similar_users, matrix_transformation, generate_recommendation


def recommendation_system(method='apriori'):
    """
    apriori: generate prediction based on association rules
    collaborative_filtering: generate prediction based on similar user's preference(c1)
    """
    if method == 'apriori':
        # min_support, min_confidence
        run_apriori(0.01, 0.5)
    elif method == 'collaborative_filtering':
        # num_c1, num_user, picked_user
        run_co_filtering(100, 50, 20)


def run_co_filtering(n_c1, n_user, picked_user, n_sim_user=10, threshold=0.01):
    """
    n_c1: number of c1
    n_user: number of user
    picked_user: who to generate recommendation
    n_sim_user: number of picked_user's similar user
    threshold: control the cosine similarity of similar user
    """
    warnings.filterwarnings("ignore")
    acg_review, agg_reviews_c1_n, agg_reviews_user_n = group_dataset(n_c1, n_user)
    ui_matrix = fetch_user_item_matrix(acg_review, agg_reviews_c1_n, agg_reviews_user_n)
    similar_users = fetch_similar_users(ui_matrix, picked_user, n_sim_user, threshold)
    sim_user_c1, sim_user_index, map_dict = matrix_transformation(ui_matrix, picked_user, similar_users)
    rank_c1_score = generate_recommendation(similar_users, sim_user_c1, sim_user_index, map_dict)

    return rank_c1_score


if __name__ == '__main__':
    print('>> pre-flight')
    recommendation_system(method='collaborative_filtering')
    print('>> end-flight')
