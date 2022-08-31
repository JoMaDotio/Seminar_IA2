import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor

np.random.seed(42)
m = 300
r = 0.5
ruido = r * np.random.randn(m,1)
x = 6 * np.random.rand(m,1) - 3
y = 0.5 * x ** 2 + x + 2 + ruido


"""
Class Tree params:
    max_depth=num
    min_samples_leaf=num, range (0, .5] lower the value the tree start to memorize
                                 
Class KNN
    K mayor la K sub entrena, menor la k sobre entrena
    
Class SUR
    gamma='scale'
    kernel='rbf'
    c entre mas grande sea se sobre entrena, entre mas pequeña se sub entrena
    
Kernel Ridge
    alpha = mas pequeño sub entrena, mas grande sobreentrena


"""
names = ['decission_tree', 'KNN', 'SVR', 'Kernel_riedge', 'MAMAlona']
models = [DecisionTreeRegressor(), KNeighborsRegressor(n_neighbors=5, n_jobs=-1), SVR(gamma='scale', C=1, epsilon=0.1, kernel='rbf'), KernelRidge(alpha=0.0, kernel='rbf'), MLPRegressor(hidden_layer_sizes=(120, 60, 10))]

x_train, x_test, y_train, y_test = train_test_split(x, y)


def test_model(model: any, name: str) -> None:
    
    model.fit(x_train, y_train)
    
    print(f'Score {name}: {model.score(x_train, y_train)}')
    print(f'Score {name}: {model.score(x_test, y_test)}')

    x_new = np.linspace(-3, 3, 200).reshape(-1,1)
    y_new = model.predict(x_new)
    
    
    plt.figure()
    plt.plot(x_new, y_new, '-k', linewidth=3)
    plt.plot(x_train, y_train, 'b.')
    plt.plot(x_test, y_test, 'r.')
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.axis([-3,3,0,10])
    #plt.show()
    plt.savefig(name + '.png')
    

def main():
    for i in range (len(models)):
        test_model(models[i], names[i])

        
if __name__ == '__main__':
    main()
    

    