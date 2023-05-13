import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
plt.switch_backend('Agg')
import seaborn as sns; sns.set()
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import mlflow



def plot_confusion_matrix(trueY, predictions, figsize=(10, 4), log_to_mlflow=False,
                          title="Confusion Matrix (Normalised & Non-Normalised)"):
    # Confusion matrix
    f, axs = plt.subplots(1, 2, figsize=figsize)
    f.suptitle(title, fontsize=16)
    ConfusionMatrixDisplay.from_predictions(trueY, predictions,
                                            display_labels=['Bad', 'Good'],
                                            normalize="true",
                                            ax=axs[0])
    axs[0].grid()
    ConfusionMatrixDisplay.from_predictions(trueY, predictions,
                                            display_labels=['Bad', 'Good'],
                                            normalize=None,
                                            ax=axs[1])
    axs[1].grid()

    # Log plot with MLflow
    if log_to_mlflow:
        mlflow.log_figure(f, 'plots/confusion_matrix.png')


def plot_ROC_AUC_curve(model, testX, testY, name="LightGBM", figsize=(5, 4), 
                       log_to_mlflow=False, title="ROC AUC Curve on Test Data"):
    # Plot ROC AUC curve on the test set
    f, axs = plt.subplots(1, 1, figsize=figsize)
    f.suptitle(title, fontsize=16)
    RocCurveDisplay.from_estimator(model, testX, testY, 
                                  name=name,
                                  ax=axs,
    )
    # Log plot with MLflow
    if log_to_mlflow:
        mlflow.log_figure(f, 'plots/ROC_AUC.png')


def get_results(df, score):
    """ Function for extracting train and test scores. """
    score = score.lower()
    return pd.DataFrame({f'train_{score}': df[f'train_{score}'],
                         f'test_{score}': df[f'test_{score}']})


def add_median_labels(ax, fmt='.2f'):
    """ Function for plotting the median on a boxplot.
    """
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])


def plot_cv_scores(scores, figsize=(12, 5), log_to_mlflow=False, 
                   title="CV Metrics"):
    """ Function for plotting AUC, Specificity, Recall and Accuracy using boxplots.
    """
    # Convert the scores to DataFrames
    scores = pd.DataFrame(scores)

    # Plot the train and test results for the two models. Use equal y-axis
    f, axs = plt.subplots(1, 4, figsize=figsize, sharey='row')
    f.suptitle(title, fontsize=16)

    for i, name in enumerate(['AUC', 'Specificity', 'Recall', 'Accuracy']):
        sns.boxplot(data=get_results(scores, name).values, ax=axs[i])
        axs[i].set_title(name, size=18)
        axs[i].set_xticks([0, 1], ['Train', 'Val'], size=12)
        add_median_labels(axs[i])

    # Log plot with MLflow
    if log_to_mlflow:
        mlflow.log_figure(f, 'plots/cv_metrics.png')


