
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def fetch_similar_users(ui_matrix, user_id, n, threshold=0.3):
    """
    1. user_id is the picked user
    2. n is the number of similar user
    3. use threshold to filter user with cosine similarity
    """
    print(f'>> start fetching similar users of {user_id}')
    # cosine_similarity does not take missing values
    df = pd.DataFrame(cosine_similarity(ui_matrix.fillna(0))).drop(index=user_id)
    similar_users = df[df[user_id] > threshold][user_id].sort_values(ascending=False)[:n]

    print(f'>> the similar users for user {user_id} are:\n{similar_users}')
    return similar_users


def matrix_transformation(ui_matrix, picked_user, similar_users):
    print('>> start matrix transformation')
    # extract all c1 that be seen by picked_user
    picked_user_watched = ui_matrix.iloc[picked_user].dropna(how='all')

    # c1 that similar users watched (remove c1 with none)
    sim_user_index = [ui_matrix.iloc[i].name for i in similar_users.keys()]
    sim_user_c1 = ui_matrix[ui_matrix.index.isin(sim_user_index)].dropna(axis=1, how='all')

    # remove the watched c1 from the c1 list(避免推薦已看過的作品)
    sim_user_c1.drop(picked_user_watched.keys(), axis=1, inplace=True, errors='ignore')

    # create a dict to map the user_id and their index
    map_dict = {}
    for i in list(similar_users.keys()):
        map_dict[ui_matrix.iloc[i].name] = i

    return sim_user_c1, sim_user_index, map_dict


def generate_recommendation(similar_users, sim_user_c1, sim_user_index, map_dict):
    c1_score = {}
    print('>> start generate c1 list for recommendation')
    # 對各作品(c1)建立迴圈計算推薦程度
    for i, c1 in enumerate(sim_user_c1.columns):
        # fetch ratings for each c1
        c1_rating = sim_user_c1[c1]

        # 各作品(c1)推薦分數
        total = 0
        # 各作品(c1)推薦分數的計算次數(被多少相似user看過)
        count = 0

        for u in sim_user_index:
            # if c1 has rating(not nan)
            if not pd.isna(c1_rating[u]):
                # 相似user與推薦對象的相似度 * 相似user對該c1的評分
                score = similar_users[map_dict[u]] * c1_rating[u]
                total += score
                count += 1
                # fetch the average score for each c1
        c1_score[i] = total / count

    c1_score = pd.DataFrame(c1_score.items(), columns=['c1', 'c1_score'])
    rank_c1_score = c1_score.sort_values(by='c1_score', ascending=False)
    print('-------------------')
    print(rank_c1_score.head(5))
    print('-------------------')
    return rank_c1_score
