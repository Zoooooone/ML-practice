import numpy as np
import matplotlib.pyplot as plt
import typing


class Net:
    def __init__(self, input_size, hidden_size, output_size):
        self.w_input_hidden = np.random.uniform(size=(hidden_size, input_size))
        self.w_hidden_output = np.random.uniform(size=(output_size, hidden_size))

    def ReLU(self, vec: np.ndarray) -> np.ndarray:
        return np.maximum(np.zeros(vec.shape), vec)

    def d_ReLU(self, vec: np.ndarray) -> np.ndarray:
        return np.where(vec < 0, 0, 1)

    def forward(self, X: np.ndarray) -> None:
        # u_j
        self.hidden_in = np.dot(self.w_input_hidden, X)
        # y_j
        self.hidden_out = self.ReLU(self.hidden_in)
        # u_k
        self.output_in = np.dot(self.w_hidden_output, self.hidden_out)
        # y_k
        self.output_out = self.ReLU(self.output_in)

    def back_propagation(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01) -> None:
        # 计算误差, 以及反向传播推导式中的delta_k和delta_j
        loss = self.output_out - y
        delta_output = loss * self.d_ReLU(self.output_in)
        delta_hidden = np.dot(self.w_hidden_output.T, delta_output) * self.d_ReLU(self.hidden_in)

        # 更新权重
        self.w_hidden_output -= lr * np.dot(delta_output, self.hidden_out.T)
        self.w_input_hidden -= lr * np.dot(delta_hidden, X.T)

    def train(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 10000) -> None:
        self.losses = []

        for epoch in range(epochs):
            loss = 0
            '''
            if epoch % (epochs // 10) == 0:
                print('\n')
            '''

            for i in range(X.shape[0]):
                x = X[i].reshape(X[i].shape[0], 1)
                y_i = y[i].reshape(y[i].shape[0], 1)
                loss += np.mean((self.predict(x) - y_i) ** 2) / X.shape[0]
                self.forward(x)
                self.back_propagation(x, y_i, lr)

                '''
                if epoch % (epochs // 10) == 0:
                    print(f"epoch = {epoch}, predict = {self.output_out}, loss = {np.mean((self.output_out - y_i) ** 2)}")
                '''
            self.losses.append(loss)

        # fig1 = plt.figure(figsize=(10, 8))
        plt.plot(np.arange(epochs // 10, epochs), self.losses[epochs // 10:])
        plt.title("Training process")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        # fig1.savefig("machine_learning/ANN/Training_process.png")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.forward(X)
        return self.output_out


if __name__ == '__main__':
    X = np.array([[0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 1, 1, 0, 0, 0,
                   0, 0, 1, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 1, 1, 1, 1, 1, 0],  # 1

                  [0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 1, 1, 0, 0, 0,
                   0, 0, 1, 1, 1, 0, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 1, 1, 1, 1, 1, 0],  # 1

                  [0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 1, 1, 0, 0, 0,
                   0, 0, 1, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 1, 1, 0, 0, 0,
                   0, 0, 1, 1, 1, 1, 1, 0],  # 1

                  [0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 1, 1, 0, 0, 0,
                   0, 0, 1, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 1, 1, 1, 1, 1, 1, 0],  # 1

                  [0, 0, 0, 1, 1, 1, 0, 0,
                   0, 0, 1, 0, 0, 0, 1, 0,
                   0, 1, 0, 0, 0, 1, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 0, 0,
                   0, 0, 1, 0, 0, 0, 0, 0,
                   0, 1, 0, 0, 0, 0, 0, 0,
                   0, 1, 1, 1, 1, 1, 1, 0],  # 2

                  [0, 0, 0, 1, 1, 1, 0, 0,
                   0, 0, 1, 0, 0, 0, 1, 0,
                   0, 1, 0, 0, 0, 1, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 0, 0,
                   0, 0, 1, 0, 0, 0, 0, 0,
                   0, 1, 0, 0, 0, 0, 0, 0,
                   0, 1, 1, 1, 1, 1, 1, 1],  # 2

                  [0, 0, 0, 1, 1, 1, 0, 0,
                   0, 0, 1, 0, 0, 1, 1, 0,
                   0, 1, 0, 0, 0, 1, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 0, 0,
                   0, 0, 1, 0, 0, 0, 0, 0,
                   0, 1, 0, 0, 0, 0, 0, 0,
                   0, 1, 1, 1, 1, 1, 1, 1],  # 2

                  [0, 0, 1, 1, 1, 1, 0, 0,
                   0, 1, 1, 0, 0, 0, 1, 0,
                   0, 1, 0, 0, 0, 1, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 0, 0,
                   0, 0, 1, 0, 0, 0, 0, 0,
                   0, 1, 1, 0, 0, 0, 0, 0,
                   0, 1, 1, 1, 1, 1, 1, 1],  # 2

                  [0, 0, 0, 1, 1, 1, 0, 0,
                   0, 0, 1, 0, 0, 0, 1, 0,
                   0, 0, 1, 0, 0, 1, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 0, 1, 0, 0,
                   0, 0, 0, 0, 0, 0, 1, 0,
                   0, 0, 1, 0, 0, 0, 1, 0,
                   0, 0, 0, 1, 1, 1, 0, 0],  # 3

                  [0, 0, 0, 1, 1, 1, 0, 0,
                   0, 0, 1, 0, 0, 0, 1, 0,
                   0, 0, 1, 0, 0, 1, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 0, 1, 0, 0,
                   0, 0, 1, 0, 0, 0, 1, 0,
                   0, 0, 1, 0, 0, 0, 1, 0,
                   0, 0, 0, 1, 1, 1, 1, 0],  # 3

                  [0, 0, 0, 1, 1, 1, 0, 0,
                   0, 0, 1, 0, 0, 0, 1, 0,
                   0, 0, 1, 0, 0, 1, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 1, 1, 0, 0,
                   0, 0, 0, 0, 0, 0, 1, 0,
                   0, 0, 1, 0, 0, 0, 1, 0,
                   0, 0, 1, 1, 1, 1, 1, 0],  # 3

                  [0, 0, 0, 1, 1, 1, 0, 0,
                   0, 0, 1, 0, 0, 0, 1, 0,
                   0, 0, 1, 0, 0, 1, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 0, 1, 0, 0,
                   0, 0, 0, 0, 0, 0, 1, 0,
                   0, 0, 1, 0, 0, 0, 1, 0,
                   0, 0, 1, 1, 1, 1, 1, 0]])  # 3

    fig2, axes = plt.subplots(nrows=4, ncols=X.shape[0] // 2, figsize=(10, 8))
    for i in range(X.shape[0]):
        x = X[i].reshape((8, 8))
        row_id = i // (X.shape[0] // 2)
        col_id = i - row_id * X.shape[0]
        axes[row_id, col_id].imshow(x)
        axes[2, col_id].imshow(np.zeros((8, 8)))

    y = np.array([[1], [1], [1], [1], [2], [2], [2], [2], [3], [3], [3], [3]])
    NN = Net(64, 10, 1)
    NN.train(X, y, lr=0.001, epochs=30000)

    test = np.array([[0, 0, 0, 0, 1, 1, 0, 0,
                      0, 0, 0, 1, 1, 0, 0, 0,
                      0, 0, 1, 0, 1, 0, 0, 0,
                      0, 1, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 1, 1, 1, 0, 0,
                      0, 1, 1, 0, 1, 1, 1, 1],  # 1

                     [0, 0, 1, 1, 1, 1, 0, 0,
                      0, 1, 1, 0, 0, 1, 0, 0,
                      0, 1, 0, 0, 0, 1, 0, 0,
                      1, 0, 0, 0, 1, 1, 0, 0,
                      0, 0, 0, 1, 1, 0, 0, 0,
                      0, 0, 1, 1, 0, 0, 0, 0,
                      0, 1, 0, 0, 0, 0, 0, 1,
                      1, 1, 1, 1, 0, 1, 1, 0],  # 2

                     [0, 0, 0, 1, 1, 1, 0, 0,
                      0, 0, 1, 0, 0, 1, 1, 0,
                      0, 0, 0, 0, 0, 1, 0, 0,
                      0, 0, 0, 0, 1, 1, 0, 0,
                      0, 0, 0, 0, 0, 1, 0, 0,
                      0, 0, 1, 0, 0, 0, 1, 0,
                      1, 0, 0, 0, 0, 1, 1, 0,
                      1, 1, 1, 1, 1, 1, 1, 1]])  # 3

    for i in range(test.shape[0]):
        x = test[i].reshape(8, 8)
        axes[3, i].imshow(x)
        axes[3, i + 3].imshow(np.zeros((8, 8)))
        print(NN.predict(x.reshape(64, 1)))

    # fig2.savefig("machine_learning/ANN/input_num.png")
