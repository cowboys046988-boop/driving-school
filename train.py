from sklearn.tree import DecisionTreeClassifier
from training import load_examples, save_model

MAX_DEPTH = 6
MIN_SAMPLES_LEAF = 3

def train():
    rows = load_examples()
    feature_columns = ['left','front_left','front','front_right','right']
    X = [[row[column]for column in feature_columns]for row in rows]
    Y = [row['steering']for row in rows]

    model = DecisionTreeClassifier(

        max_depth = MAX_DEPTH,
        min_samples_leaf = MIN_SAMPLES_LEAF,
        random_state =1101,
    )
    model.fit(X,Y)

    save_model(model, rows)
    print(f'Trained on {len(rows)}examples.')
    return model

if __name__ == '__main__':
    try:
        train()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
