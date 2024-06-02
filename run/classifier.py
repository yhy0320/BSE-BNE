import time

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from run.field import *
from utils.utils_log import logger

classifiers_map = {
    "Nearest_Neighbors": {
        'n_neighbors': 3,
    },
    "Linear_SVM": {
        'C': 0.025,
    },
    "RBF_SVM": {
        'gamma': 2,
        'C': 1,
    },
    "Decision_Tree": {
        'max_depth': 5,
    },
    "Random_Forest": {
        'max_depth': 5,
        'n_estimators': 10,
        'max_features': 1,
    },
    "Neural_Net": {
        'alpha': 1,
        'max_iter': 1000,
    },
    "AdaBoost": {
        'learning_rate': 1.0,
    },
    "Naive_Bayes": {
        'var_smoothing': 1e-9,
    },
}


def construct_classifier(conf: dict):
    """
    根据配置生成分类器
    """
    logger.info(f'输入配置: {conf}')
    algo = conf['algorithm']

    # 设置默认值
    for k, v in classifiers_map[algo].items():
        if k not in conf:
            conf[k] = v

    if algo == Classify_Nearest_Neighbors:
        clf = KNeighborsClassifier(n_neighbors=conf['n_neighbors'])
        return clf

    if algo == Classify_Linear_SVM:
        clf = SVC(kernel="linear", C=conf['C'], random_state=conf['seed'])
        return clf

    if algo == Classify_RBF_SVM:
        clf = SVC(gamma=conf['gamma'], C=conf['C'], random_state=conf['seed'])
        return clf

    if algo == Classify_Gaussian_Process:
        clf = GaussianProcessClassifier(conf['kernel'] * RBF(conf['kernel']), random_state=conf['seed'])
        return clf

    if algo == Classify_Decision_Tree:
        clf = DecisionTreeClassifier(max_depth=conf['max_depth'], random_state=conf['seed'])
        return clf

    if algo == Classify_Random_Forest:
        clf = RandomForestClassifier(
            max_depth=conf['max_depth'],
            n_estimators=conf['n_estimators'],
            max_features=conf['max_features'],
            random_state=conf['seed']
        )
        return clf

    if algo == Classify_Neural_Net:
        clf = MLPClassifier(alpha=conf['alpha'], max_iter=conf['max_iter'], random_state=conf['seed'])
        return clf

    if algo == Classify_AdaBoost:
        clf = AdaBoostClassifier(learning_rate=conf['learning_rate'], random_state=conf['seed'])
        return clf

    if algo == Classify_Naive_Bayes:
        clf = GaussianNB(var_smoothing=conf['var_smoothing'])
        return clf

    if algo == Classify_QDA:
        clf = QuadraticDiscriminantAnalysis(tol=conf['tol'])
        return clf

    assert False, "这啥算法啊"


def run_classify(ds, label, conf: dict) -> (float, float):
    """
    运行分类算法
    """
    logger.info(f'运行分类算法，算法为 {conf["algorithm"]}')
    clf = construct_classifier(conf)
    clf = make_pipeline(StandardScaler(), clf)

    x_train, x_test, y_train, y_test = train_test_split(
        ds, label, random_state=conf['seed'], test_size=0.1
    )

    start = time.time()
    clf.fit(x_train, y_train)
    cost_time = time.time() - start
    logger.info(f'算法执行时间为: {cost_time}')

    score = clf.score(x_test, y_test)
    logger.info(f'算法模型得分为: {score}')

    if conf['algorithm'] == Classify_Decision_Tree:
        pass

    return {'score': score}, cost_time


def print_algorithm_const():
    """
    打印可以运行的分类算法的生成常量
    格式：Classify_<name> = <name>
    """
    out = ""

    for k in classifiers_map.keys():
        out += f"Classify_{k} = '{k}'\n"

    print(out)
