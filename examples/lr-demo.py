import motif
import numpy as np
import detensor as dt

# 测试接口
motif.PrivatesLoader.dev = True
motif.PrivatesLoader.myfunc = {
    2:{"get_data": "http://localhost:7001/get_data"},
    3:{"get_data": "http://localhost:7002/get_data"},
    4:{"get_data": "http://localhost:7003/get_data"}
}
def main():
    S_Site=motif.Site(1)
    A_Site=motif.Site(2)
    B_Site=motif.Site(3)
    C_Site=motif.Site(4)
    D_Site=motif.Site(5)

    get_x_train_0 = motif.private_interface("get_data", A_Site)
    get_x_train_1 = motif.private_interface("get_data", B_Site)
    get_y_train = motif.private_interface("get_data", C_Site)

    x_train_0 = motif.RemoteCallAt(A_Site)(np.array)(get_x_train_0(), dtype="float64")
    x_train_1 = motif.RemoteCallAt(B_Site)(np.array)(get_x_train_1(), dtype="float64")
    x_train = dt.detensor([x_train_0, x_train_1], axis=1)

    y_train = motif.RemoteCallAt(C_Site)(np.array)(get_y_train(), dtype="float64")

    model = dt.linear_model.LinearRegression()
    model.fit(x_train ,y_train, D_Site, S_Site)
    print(model.W.evaluate())

if __name__=='__main__':
    main()