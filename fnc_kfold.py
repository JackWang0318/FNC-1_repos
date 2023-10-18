import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version
import nltk


def generate_features(stances,dataset,name):
    '''
    这个函数根据输入的文本和数据集，生成用于机器学习模型的特征矩阵和标签。
    特征包括词重叠特征、反驳特征、极性特征、手工特征等。
    函数的输出是特征矩阵 X 和标签向量 y。
    '''
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X,y

if __name__ == "__main__":
    #download nltk packages 
    nltk.download('punkt')
    nltk.download('wordnet')

    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    '''
    将数据集分成多个"fold"或"折叠"的目的是为了进行交叉验证(Cross-Validation)。
    交叉验证是一种评估模型性能和泛化能力的技术，它有助于解决以下问题：
    模型性能评估：通过将数据集划分成多个子集（折叠），可以多次训练和测试模型，从而获得对模型性能的更稳定和全面的评估。
    减少过拟合风险：交叉验证有助于降低模型过拟合的风险。如果只使用单一的训练集和测试集，模型可能过度适应特定的训练数据，无法泛化到未见过的数据。
    超参数调优：通过交叉验证，可以比较不同参数配置下模型的性能，并选择最佳的参数设置，以提高模型的泛化能力。
    数据利用率：有效地利用了所有可用数据，因为每个子集都会作为训练集和测试集的一部分，确保数据充分参与模型评估。
    每个折叠都可以轮流作为测试集, 而其他K-1个折叠作为训练集, 以便多次训练和测试模型。这就是K折交叉验证(K-Fold Cross-Validation)的基本思想。
    通过交叉验证，可以获得多个性能评估的结果，通常是均值或其他统计指标，以更好地了解模型的性能。
    这有助于准确评估模型的性能，发现模型的弱点，并提高模型的预测能力。
    '''


    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))


    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        clf.fit(X_train, y_train)

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf



    #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")

    #Run on competition dataset
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual,predicted)
