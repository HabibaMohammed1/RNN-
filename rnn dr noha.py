random_seed = 42

def simple_random():
    global random_seed
    a, c, m = 1664525, 1013904223, 2**32
    random_seed = (a * random_seed + c) % m
    return random_seed / m

def random_uniform(low, high):
    return low + (high - low) * simple_random()

def exp(x, terms=20):
    result = 1.0
    num = 1.0
    den = 1.0
    for i in range(1, terms):
        num *= x
        den *= i
        result += num / den
    return result

def log(x, terms=20):
    if x <= 0:
        raise ValueError("log undefined for x <= 0")
    y = (x - 1) / (x + 1)
    result = 0.0
    for n in range(terms):
        result += (1 / (2 * n + 1)) * (y ** (2 * n + 1))
    return 2 * result

def tanh(x):
    e_pos = exp(x)
    e_neg = exp(-x)
    return (e_pos - e_neg) / (e_pos + e_neg)

def softmax(x):
    max_x = max(x)
    exps = [exp(xi - max_x) for xi in x]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def cross_entropy(y_hat, y_true):
    return -sum(y_true[i] * log(y_hat[i]) for i in range(len(y_true)))

# دوال مصفوفات يدوية
def zeros(shape):
    rows, cols = shape
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

def matmul(a, b):
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    if cols_a != rows_b:
        raise ValueError("Matrices can't be multiplied!")
    result = zeros((rows_a, cols_b))
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result

def add(a, b):
    rows, cols = len(a), len(a[0])
    result = zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            result[i][j] = a[i][j] + b[i][j]
    return result

def transpose_matrix(matrix):
    rows, cols = len(matrix), len(matrix[0])
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]

# كلاس الشبكة العصبية البسيطة
class SimpleRNN:
    def __init__(self, words, hidden_size=10, learning_rate=0.01):
        self.words = words
        self.vocab_size = len(words)
        self.hidden_size = hidden_size
        self.lr = learning_rate

        self.word_to_idx = {word: idx for idx, word in enumerate(words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        self.Wxh = self.random_matrix(hidden_size, self.vocab_size)
        self.Whh = self.random_matrix(hidden_size, hidden_size)
        self.Why = self.random_matrix(self.vocab_size, hidden_size)
        self.bh = zeros((hidden_size, 1))
        self.by = zeros((self.vocab_size, 1))

    def random_matrix(self, rows, cols):
        return [[random_uniform(-0.1, 0.1) for _ in range(cols)] for _ in range(rows)]

    def one_hot(self, index):
        vec = [[0.0] for _ in range(self.vocab_size)]
        if 0 <= index < self.vocab_size:
            vec[index][0] = 1.0
        return vec

    def forward(self, X):
        h = [[0.0] for _ in range(self.hidden_size)]
        hs = [h]
        outputs = []

        for x in X:
            xh = matmul(self.Wxh, x)
            hh = matmul(self.Whh, h)
            h = [[tanh(val[0])] for val in add(add(xh, hh), self.bh)]
            hs.append(h)
            y_lin = add(matmul(self.Why, h), self.by)
            y_pred = softmax([val[0] for val in y_lin])
            outputs.append(y_pred)

        return outputs[-1], hs

    def backward(self, X, Y, y_pred, hs):
        dWxh = zeros((self.hidden_size, self.vocab_size))
        dWhh = zeros((self.hidden_size, self.hidden_size))
        dWhy = zeros((self.vocab_size, self.hidden_size))
        dbh = zeros((self.hidden_size, 1))
        dby = zeros((self.vocab_size, 1))

        dy = [[y_pred[i] - Y[i][0]] for i in range(self.vocab_size)]
        dWhy = add(dWhy, matmul(dy, transpose_matrix(hs[-1])))
        dby = add(dby, dy)
        dh_next = zeros((self.hidden_size, 1))

        for t in reversed(range(len(X))):
            dh = add(matmul(transpose_matrix(self.Why), dy), dh_next)
            dtanh = [[(1 - (hs[t+1][i][0] ** 2)) * dh[i][0]] for i in range(self.hidden_size)]

            dbh = add(dbh, dtanh)
            dWxh = add(dWxh, matmul(dtanh, transpose_matrix(X[t])))
            dWhh = add(dWhh, matmul(dtanh, transpose_matrix(hs[t])))

            dh_next = matmul(transpose_matrix(self.Whh), dtanh)

        return dWxh, dWhh, dWhy, dbh, dby

    def update_params(self, dWxh, dWhh, dWhy, dbh, dby):
        self.Wxh = add(self.Wxh, [[-self.lr * grad for grad in row] for row in dWxh])
        self.Whh = add(self.Whh, [[-self.lr * grad for grad in row] for row in dWhh])
        self.Why = add(self.Why, [[-self.lr * grad for grad in row] for row in dWhy])
        self.bh = add(self.bh, [[-self.lr * grad[0]] for grad in dbh])
        self.by = add(self.by, [[-self.lr * grad[0]] for grad in dby])

    def train(self, epochs=500):
        X = [self.one_hot(self.word_to_idx[w]) for w in self.words[:-1]]
        Y = self.one_hot(self.word_to_idx[self.words[-1]])

        for epoch in range(epochs):
            y_pred, hs = self.forward(X)
            loss = cross_entropy(y_pred, [Y[i][0] for i in range(self.vocab_size)])

            dWxh, dWhh, dWhy, dbh, dby = self.backward(X, Y, y_pred, hs)
            self.update_params(dWxh, dWhh, dWhy, dbh, dby)

            if epoch % 50 == 0:
                print(f"Epoch {epoch} Loss: {loss}")

    def predict(self):
        X = [self.one_hot(self.word_to_idx[w]) for w in self.words[:-1]]
        y_pred, _ = self.forward(X)
        predicted_idx = y_pred.index(max(y_pred))
        print("\nتوقع الكلمة الرابعة بناءً على الثلاث كلمات:")
        print("الكلمة المتوقعة:", self.idx_to_word[predicted_idx])



print("ادخل 4 كلمات مفصولة بمسافة:")
user_input = input()
words = user_input.strip().split()

if len(words) != 4:
    print("يجب إدخال 4 كلمات فقط!")
    exit()

rnn = SimpleRNN(words)
rnn.train(epochs=500)
rnn.predict()
