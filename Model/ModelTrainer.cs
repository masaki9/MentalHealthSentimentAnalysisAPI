using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using System.Diagnostics;

namespace MentalHealthSentimentAnalysisAPI.Model;

/// <summary>
/// Class to train, evaluate, and make predictions using a machine learning model for mental health sentiment analysis.
/// </summary>
public class ModelTrainer
{
    /// <summary>
    /// Load data from the specified path and split it into training and testing sets.
    /// </summary>
    /// <param name="mlContext">The MLContext instance.</param>
    /// <returns>A TrainTestData object containing the training and testing data.</returns>
    public TrainTestData LoadData(MLContext mlContext, string dataPath)
    {
        // Load data
        var dataView = mlContext.Data.LoadFromTextFile<MentalHealthData>(dataPath, separatorChar: ',', hasHeader: true);

        // Split the data into training and testing sets (80% training, 20% testing)
        var splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

        return splitDataView;
    }

    /// <summary>
    /// Build and train the model using the training data.
    /// </summary>
    /// <param name="mlContext">The MLContext instance.</param>
    /// <param name="splitTrainSet">The training data.</param>
    /// <returns>The trained model.</returns>
    public ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
    {
        // Build pipeline
        var estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label") // Convert the label to a key type
            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(MentalHealthData.Statement))) // Convert text to features
            .AppendCacheCheckpoint(mlContext) // Cache the data for faster training
            .Append(mlContext.MulticlassClassification.Trainers.NaiveBayes(labelColumnName: "Label", featureColumnName: "Features")) // Use Naive Bayes for multiclass classification
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel")); // Convert the predicted label back to its original value

        Console.WriteLine("=============== Create and Train the Model ===============");
        // Start timer
        var watch = Stopwatch.StartNew();

        // Train the model
        var model = estimator.Fit(splitTrainSet);

        // End timer and print the elasped time
        watch.Stop();
        var elapsedMs = watch.ElapsedMilliseconds;
        Console.WriteLine($"Training time: {elapsedMs} ms");
        Console.WriteLine("=============== End of training ===============");
        Console.WriteLine();

        return model;
    }

    /// <summary>
    /// Evaluate the model using the test data.
    /// </summary>
    /// <param name="mlContext">The MLContext instance.</param>
    /// <param name="model">The trained model.</param>
    /// <param name="splitTestSet">The test data.</param>
    /// <returns>None</returns>
    public void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
    {
        var cachedTestData = mlContext.Data.Cache(splitTestSet); // Cache the test set for faster evaluation

        Console.WriteLine(">>> Starting Evaluate: Transforming test set...");
        var watch = Stopwatch.StartNew();
        var predictions = model.Transform(cachedTestData); // Transform the test set
        watch.Stop();
        Console.WriteLine($">>> Transform took: {watch.ElapsedMilliseconds} ms");

        Console.WriteLine(">>> Now computing metrics...");
        watch.Restart();
        var metrics = mlContext.MulticlassClassification.Evaluate(
                predictions,
                labelColumnName: "Label",
                scoreColumnName: "Score",
                predictedLabelColumnName: "PredictedLabel"
            ); // Evaluate the model using the test set
        watch.Stop();
        Console.WriteLine($">>> Evaluate(...) took: {watch.ElapsedMilliseconds} ms");

        // Print evaluation metrics
        Console.WriteLine("=============== Evaluating Model ===============");
        Console.WriteLine($"Log Loss: {metrics.LogLoss}");
        Console.WriteLine($"Log Loss Reduction: {metrics.LogLossReduction}");
        Console.WriteLine($"Macro Average Accuracy: {metrics.MacroAccuracy}");
        Console.WriteLine($"Micro Average Accuracy: {metrics.MicroAccuracy}");
        Console.WriteLine($"Top K Accuracy: {metrics.TopKAccuracy}");
        Console.WriteLine("=============== End of evaluation ===============");
    }

    /// <summary>
    /// Make a prediction using the trained model.
    /// </summary>
    /// <param name="mlContext">The MLContext instance.</param>
    /// <param name="model">The trained model.</param>
    /// <param name="statement">The input statement for prediction.</param>
    /// <returns>None</returns>
    public void GetPredictionForMentalHealth(MLContext mlContext, ITransformer model, string statement)
    {
        // Create a prediction engine
        var predictionFunction = mlContext.Model.CreatePredictionEngine<MentalHealthData, MentalHealthPrediction>(model);

        // Create a new instance of the input data
        var inputData = new MentalHealthData
        {
            Statement = statement
        };

        // Make a prediction
        var prediction = predictionFunction.Predict(inputData);

        // Print the prediction results
        Console.WriteLine($"Statement: {inputData.Statement}");
        Console.WriteLine($"Predicted Status: {prediction.Prediction}");
        Console.WriteLine($"Scores: {prediction.Score}");
    }

    /// <summary>
    /// Run the model trainer.
    /// </summary>
    /// <param name="args">Command line arguments.</param>
    /// <returns>None</returns>
    public void Run(string[] args)
    {
        var dataPath = Path.Combine(Environment.CurrentDirectory, "..", "Data", "MentalHealthData.csv");
        var mlContext = new MLContext(); // Create a new MLContext
        var splitDataView = LoadData(mlContext, dataPath); // Load data and split into training and testing sets
        var model = BuildAndTrainModel(mlContext, splitDataView.TrainSet); // Build and train the model
        Evaluate(mlContext, model, splitDataView.TestSet); // Evaluate the model
        GetPredictionForMentalHealth(mlContext, model, "I feel anxious and overwhelmed."); // Make a prediction

        // Save the trained model
        string modelPath = Path.Combine(Environment.CurrentDirectory, "..", "Data", "MentalHealthModel.zip");
        mlContext.Model.Save(model, splitDataView.TrainSet.Schema, modelPath);
        Console.WriteLine($"Model saved to {modelPath}");
    }

    /// <summary>
    /// Entry point for the application.
    /// </summary>
    /// <param name="args">Command line arguments.</param>
    /// <returns>None</returns>
    public static void Main(string[] args)
    {
        var trainer = new ModelTrainer();
        trainer.Run(args);
    }
}
