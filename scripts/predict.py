import os.path
from pandas import DataFrame
from preprocess import data_generator


test_data_dir = '../data/test'

nb_test_samples = 10000


def predict(model, result_dir):
    test_gen = data_generator(test_data_dir, shuffle=False)

    proba = model.predict_generator(test_gen, nb_test_samples)
    proba_df = DataFrame(proba, index=test_gen.filenames)

    proba_df.to_csv(os.path.join(result_dir, 'proba.csv'))
    proba_df.idxmax(axis=1).to_csv(os.path.join(result_dir, 'pred.csv'))


if __name__ == "__main__":
    import sys
    from model import create_model

    weights_path = sys.argv[1]

    model = create_model()
    model.load_weights(weights_path)

    predict(model, '../results')
