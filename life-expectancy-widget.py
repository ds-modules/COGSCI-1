from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

le = pd.read_csv('data/life-expectancy.csv')
le = le.dropna()

train, test = train_test_split(le, test_size = 0.1, random_state = 42)
X_train = train.drop(['Country', 'Year', 'Status', 'Life expectancy '], axis = 1)
Y_train = train[['Life expectancy ']]

X_test = test.drop(['Country', 'Year', 'Status', 'Life expectancy '], axis = 1)
Y_test = test[['Life expectancy ']]

def predict_life_expectancy(variables):
    X_train = train[list(variables)]
    X_test = test[list(variables)]
        
    Y_train = train[['Life expectancy ']]
    Y_test = test[['Life expectancy ']]

    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    print('Overall Model Accuracy: ', model.score(X_test, Y_test))
    
    plt.figure(figsize=(15,5))
    
    if len(variables) == 1:
        plt.bar(variables, model.score(X_test, Y_test))
        print('Model trained on: ', variables[0])    
    if len(variables) > 1:
        print('Model trained on: ', variables) 
        scores = []
        for v in list(variables):
            X_train = train[[v]]
            X_test = test[[v]]

            Y_train = train[['Life expectancy ']]
            Y_test = test[['Life expectancy ']]

            model = LinearRegression()
            model.fit(X_train, Y_train)
            scores.append(model.score(X_test, Y_test))
        
        plt.bar(list(variables), scores)
    plt.ylim(0, 1)
    plt.xlabel('Feature')
    plt.xticks(rotation=90)
    plt.ylabel('Single Feature Model Accuracy')
    plt.title('Life Expectancy Prediction Accuracy');
        
var = widgets.SelectMultiple(
    options=list(le.columns)[4:],
    value=['Adult Mortality'],
    rows=10,
    description='Variables',
    disabled=False
)