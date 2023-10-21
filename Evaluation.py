from sklearn.metrics import classification_report


def evaluate_f1(y_true, y_pred):
    """
    Метод для расчёта avg f-1 score по датасету
    :param y_pred - предсказанные моделью метки классов
    :param y_true - реальные метки классов
    :return значение f-1 score по всем классам цветков
    """
    return classification_report(y_true, y_pred,
                                 target_names=['buttercup', 'coreopsis', 'daffodil', 'dandelion', 'sunflower'],
                                 output_dict=True)['macro avg']['f1-score']
