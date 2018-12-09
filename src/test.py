from sklearn import metrics
import config as cfg

INPUT_PATH = cfg.DATASET_PATH + "/" 
test_data = pd.read_csv("test.csv")

test_size = len(train_data)
test_samples = np.zeros(test_size, dtype=[('input', float, (256, 256)), ('output', float, 1)])

for index, row in test_data.iterrows():
    test_samples[index] = imread(INPUT_PATH + "test/" + row['Image_Index']) / 255, row[8]

test_x = np.reshape(test_samples['input'], (len(test_samples), 256, 256, 1))

# Load the best model
model = load_model("weights.hdf5")

# Predicating with test data
preds = model.predict(test_x)

# Calculating the error on test data
test_mse = mean_squared_error(test_samples['output'], preds)
print("Test MSE: %f" % (test_mse))
# model.summary()

fpr, tpr, thresholds = metrics.roc_curve(test_samples['output'], preds)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for chest diease classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()
