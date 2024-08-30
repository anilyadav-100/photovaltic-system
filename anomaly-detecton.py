reconstructed_data = model.predict(X_test)
reconstruction_error = np.mean(np.square(X_test - reconstructed_data), axis=1)


threshold to classify anomalies
threshold = 0.1  




anomalies = X_test[reconstruction_error > threshold]
# Plot Anomalies
if len(anomalies) > 0:
    plt.figure(figsize=(8, 6))
    plt.scatter(anomalies[:, 0], anomalies[:, 1], alpha=0.5, color='red', label='Anomalies')
    plt.scatter(X_test[:, 0], X_test[:, 1], alpha=0.5, color='blue', label='Normal')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Anomaly Detection")
    plt.legend()
    plt.grid()
    plt.show()
// we can adjust threshold value according to our need.
