import motif
import detensor
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import logging
logger = logging.getLogger("Motif")
logger.setLevel(logging.INFO)
motif.PrivatesLoader.dev = True
motif.PrivatesLoader.myfunc = {
    1: {"get_data": "http://localhost:7000/get_data"}, # Site S
    2: {"get_data": "http://localhost:7001/get_data"}, # Site A
    3: {"get_data": "http://localhost:7002/get_data"}, # Site B
    4: {"get_data": "http://localhost:7003/get_data"}, # Site C
}

# 用diabetes数据集测试
def test_by_diabetes(A_Site, B_Site, C_Site, D_Site, S_Site):
    
    print('diabetes test:')
    # 使用 detensor 库训练+预测
    get_a_data = motif.private_interface('get_data', A_Site)
    get_b_data = motif.private_interface('get_data', B_Site)
    get_c_data = motif.private_interface('get_data', C_Site)

    x_train_0 = motif.RemoteCallAt(A_Site)(np.array)(get_a_data(name='diabetes_x0_train.csv'), dtype='float64')
    x_train_1 = motif.RemoteCallAt(B_Site)(np.array)(get_b_data(name='diabetes_x1_train.csv'), dtype='float64')
    Y_train = motif.RemoteCallAt(C_Site)(np.array)(get_c_data(name='diabetes_y_train.csv'), dtype='float64')
    X_train = [x_train_0, x_train_1]

    model = detensor.linear_model.LinearRegression()
    model.fit(detensor.detensor(X_train, axis=1), detensor.detensor([Y_train], axis=0), D_Site)

    x_test_0 = motif.RemoteCallAt(A_Site)(np.array)(get_a_data(name='diabetes_x0_test.csv'), dtype='float64')
    x_test_1 = motif.RemoteCallAt(B_Site)(np.array)(get_b_data(name='diabetes_x1_test.csv'), dtype='float64')
    y_test = motif.RemoteCallAt(C_Site)(np.array)(get_c_data(name='diabetes_y_test.csv'), dtype='float64')
    X_test = [x_test_0, x_test_1]
    Ans = model.predict(detensor.detensor(X_test, axis=1), D_Site)
    res = Ans[0].at(S_Site) + Ans[1].at(S_Site) + Ans[2].at(S_Site)
    
    mse = mean_squared_error(res.at_public().evaluate(), y_test.at_public().evaluate())
    mae = mean_absolute_error(res.at_public().evaluate(), y_test.at_public().evaluate())
    print(f'detensor: mean squared error = {mse}')
    print(f'detensor: mean absolute error = {mae}')
    
    # 使用 sklearn 库训练+预测
    def sklearn_ans(x_train, y_train, x_test, y_test):
        model2 = LinearRegression()
        model2.fit(x_train, y_train)
        W2_res = np.vstack([model2.coef_.T, model2.intercept_.reshape(1, 1)])

        res2 = model2.predict(x_test)

        mse = mean_squared_error(res2, y_test)
        mae = mean_absolute_error(res2, y_test)
        return mse, mae

    get_s_data = motif.private_interface('get_data', S_Site)
    x_train = motif.RemoteCallAt(S_Site)(np.array)(get_s_data(name='diabetes_x_train.csv'), dtype='float64')
    y_train = motif.RemoteCallAt(S_Site)(np.array)(get_s_data(name='diabetes_y_train.csv'), dtype='float64')
    x_test = motif.RemoteCallAt(S_Site)(np.array)(get_s_data(name='diabetes_x_test.csv'), dtype='float64')
    y_test = motif.RemoteCallAt(S_Site)(np.array)(get_s_data(name='diabetes_y_test.csv'), dtype='float64')
    ans = motif.RemoteCallAt(S_Site)(sklearn_ans)(x_train, y_train, x_test, y_test).at_public().evaluate()
    mse2 = ans[0]
    mae2 = ans[1]
    print(f'sklearn: mean squared error = {mse2}')
    print(f'sklearn: mean absolute error = {mae2}')
    
def main():
    S_Site = motif.Site(1)
    A_Site = motif.Site(2)
    B_Site = motif.Site(3)
    C_Site = motif.Site(4)
    D_Site = motif.Site(5)
    
    test_by_diabetes(A_Site, B_Site, C_Site, D_Site, S_Site)

if __name__ == '__main__':
    res = main()