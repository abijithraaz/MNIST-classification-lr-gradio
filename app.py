import plotly.express as px
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import gradio as gr


# Load data from https://www.openml.org/d/554
X, y = fetch_openml(
    "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
)

print("Data loaded")
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))


scaler = StandardScaler()


def dataset_display(digit, count_per_digit, binary_image):
    if digit not in range(10):
        # return a figure displaying an error message
        return px.imshow(
            np.zeros((28, 28)),
            labels=dict(x="Pixel columns", y="Pixel rows"),
            title=f"Digit {digit} is not in the data",
        )

    binary_value = True if binary_image == 1 else False
    digit_idxs = np.where(y == str(digit))[0]
    random_idxs = np.random.choice(digit_idxs, size=count_per_digit, replace=False)

    fig = px.imshow(
        np.array([X[i].reshape(28, 28) for i in random_idxs]),
        labels=dict(x="Pixel columns", y="Pixel rows"),
        title=f"Examples of Digit {digit} in Data",
        facet_col=0,
        facet_col_wrap=5,
        binary_string=binary_value,
    )

    return fig


def predict(img):
    try:
        img = img.reshape(1, -1)
    except:
        return "Show Your Drawing Skills"

    try:
        img = scaler.transform(img)
        prediction = clf.predict(img)
        return prediction[0]
    except:
        return "Train the model first"


def train_model(train_sample=5000, c=0.1, tol=0.1, solver="sage", penalty="l1"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_sample, test_size=10000
    )

    penalty_dict = {
        "l2": ["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"],
        "l1": ["liblinear", "saga"],
        "elasticnet": ["saga"],
    }

    if solver not in penalty_dict[penalty]:
        return (
            "Solver not supported for the selected penalty",
            "Change the Combination",
            None,
        )

    global clf
    global scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(C=c, penalty=penalty, solver=solver, tol=tol)
    clf.fit(X_train, y_train)
    sparsity = np.mean(clf.coef_ == 0) * 100
    score = clf.score(X_test, y_test)

    coef = clf.coef_.copy()
    scale = np.abs(coef).max()

    fig = px.imshow(
        np.array([coef[i].reshape(28, 28) for i in range(10)]),
        labels=dict(x="Pixel columns", y="Pixel rows"),
        title=f"Classification vector for each digit",
        range_color=[-scale, scale],
        facet_col=0,
        facet_col_wrap=5,
        facet_col_spacing=0.01,
        color_continuous_scale="RdBu",
        zmin=-scale,
        zmax=scale,
    )

    return score, sparsity, fig


with gr.Blocks() as demo:
    gr.Markdown("# MNIST classification using multinomial logistic + L1 ")
    gr.Markdown(
        """This interactive demo is based on the [MNIST classification using multinomial logistic + L1](https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html#sphx-glr-auto-examples-linear-model-plot-sparse-logistic-regression-mnist-py) example from the popular [scikit-learn](https://scikit-learn.org/stable/)  library, which is a widely-used library for machine learning in Python. The primary goal of this demo is to showcase the use of logistic regression in classifying handwritten digits from the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset, which is a well-known benchmark dataset in computer vision. The dataset is loaded from [OpenML](https://www.openml.org/d/554), which is an open platform for machine learning research that provides easy access to a large number of datasets.
The model is trained using the scikit-learn library, which provides a range of tools for machine learning, including classification, regression, and clustering algorithms, as well as tools for data preprocessing and model evaluation. The demo calculates the score and sparsity metrics using test data, which provides insight into the model's performance and sparsity, respectively. The score metric indicates how well the model is performing, while the sparsity metric provides information about the number of non-zero coefficients in the model, which can be useful for interpreting the model and reducing its complexity.
    """
    )

    with gr.Tab("Explore the Data"):
        gr.Markdown("## ")
        with gr.Row():
            digit = gr.Slider(0, 9, label="Select the Digit", value=5, step=1)
            count_per_digit = gr.Slider(
                1, 10, label="Number of Images", value=10, step=1
            )
            binary_image = gr.Slider(0, 1, label="Binary Image", value=0, step=1)

        gen_btn = gr.Button("Show Me ")
        gen_btn.click(
            dataset_display,
            inputs=[digit, count_per_digit, binary_image],
            outputs=gr.Plot(),
        )

    with gr.Tab("Trian Your Model"):
        gr.Markdown("# Play with the parameters to see how the model changes")

        gr.Markdown("## Solver and penalty")

        gr.Markdown(
            """
        Penalty | Solver
        -------|---------------
        l1 | liblinear, saga 
        l2  | lbfgs, newton-cg, newton-cholesky, sag, saga             
        elasticnet | saga
        """
        )

        with gr.Row():
            train_sample = gr.Slider(
                1000, 60000, label="Train Sample", value=5000, step=1
            )

            c = gr.Slider(0.1, 1, label="C", value=0.1, step=0.1)
            tol = gr.Slider(
                0.1, 1, label="Tolerance for stopping criteria.", value=0.1, step=0.1
            )
            max_iter = gr.Slider(100, 1000, label="Max Iter", value=100, step=1)

            penalty = gr.Dropdown(
                ["l1", "l2", "elasticnet"], label="Penalty", value="l1"
            )
            solver = gr.Dropdown(
                ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
                label="Solver",
                value="saga",
            )

        train_btn = gr.Button("Train")
        train_btn.click(
            train_model,
            inputs=[train_sample, c, tol, solver, penalty],
            outputs=[
                gr.Textbox(label="Score"),
                gr.Textbox(label="Sparsity"),
                gr.Plot(),
            ],
        )

    with gr.Tab("Predict the Digit"):
        gr.Markdown("## Draw a digit and see the model's prediction")
        inputs = gr.Sketchpad(brush_radius=1.0)
        outputs = gr.Textbox(label="Predicted Label", lines=1)
        skecth_btn = gr.Button("Classify the Sketch")
        skecth_btn.click(predict, inputs, outputs)


demo.launch()
