
import os
import pandas as pd
from apyori import apriori
from preprocess import fetch_user_c1_dict


def run_apriori(min_support=0.01, min_confidence=0.5, max_length=3):
    current_path = os.getcwd()
    file_path = os.path.join(current_path, 'result\\apriori_result.xlsx')

    if os.path.isfile(file_path):
        print('>> apriori result already exists!')
    else:
        print('>> start running apriori algorithm')
        # use dataset 'acg' to generate apriori result
        data_path = os.path.join(current_path, 'data\\acg.csv')
        acg = pd.read_csv(data_path)
        acg_gather = acg[acg[' action'] == 'gather'].drop([' c2', ' action', ' score', ' ctime'], axis=1)

        user_c1_dict = fetch_user_c1_dict(acg_gather)

        print('>> creating input for apriori')
        g = acg_gather.groupby('userid')
        apriori_input = g.apply(lambda x: list(set(x[" c1"].values))).tolist()

        # for user, c1_of_user in tqdm(g):
        #     apriori_input.append(list(set(c1_of_user[" c1"].values)))

        print('>> start running apriori, this method usually takes a few minutes')
        association_rules = apriori(apriori_input,
                                    min_support=min_support,
                                    min_confidence=min_confidence,
                                    max_length=max_length)

        association_results = list(association_rules)

        sup = [item[1] for item in association_results]
        con = [item[2][0][2] for item in association_results]
        lift = [item[2][0][3] for item in association_results]

        x = []
        y = []

        for item in association_results:
            pair = item[0]
            items = [x for x in pair]
            x.append(items[0])
            y.append(items[1])

        df = {
            'x': x,
            'y': y,
            'support': sup,
            'confidence': con,
            'lift': lift,
        }

        data = pd.DataFrame(df).sort_values(by='confidence', ascending=False)
        data.to_excel('result\\apriori_result.xlsx', index=False)

        return user_c1_dict
