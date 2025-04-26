using Microsoft.ML;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.Transforms;
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
    /// Build the training pipeline for the model.
    /// </summary>
    /// <param name="mlContext">The MLContext instance.</param>
    /// <param name="ngramLength">The length of the n-grams to be used.</param>
    /// <param name="useAllLengths">Whether to store all n-gram lengths up to ngramLength, or only ngramLength.</param>
    /// <param name="maximumNgramsCount">The maximum number of n-grams to be used for each level of n-grams.</param>
    /// <param name="removeStopWords">Whether to remove stop words.</param>
    /// <returns>A training pipeline estimator.</returns>
    public IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext, int ngramLength, bool useAllLengths,
                                                    int[] maximumNgramsCount, bool removeStopWords)
    {
        var estimator = mlContext.Transforms.DropColumns(nameof(MentalHealthData.Id)); // Drop the Id column

        var labelMap = mlContext.Transforms.Conversion.MapValueToKey("Label"); // Convert the label to a key type

        // Configure text featurization options
        var tfOptions = new TextFeaturizingEstimator.Options
        {
            // Text normalization options
            CaseMode = TextNormalizingEstimator.CaseMode.Lower,
            KeepPunctuations = false,
            KeepNumbers = true,

            // Stopwords
            StopWordsRemoverOptions = removeStopWords
                ? new StopWordsRemovingEstimator.Options()
                : null,

            // Ngram feature extractor for words
            WordFeatureExtractor = new WordBagEstimator.Options
            {
                NgramLength = ngramLength,
                UseAllLengths = useAllLengths,
                MaximumNgramsCount = maximumNgramsCount
            },

            //  Turn off Ngram feature extractor to use for characters
            CharFeatureExtractor = null
        };

        // Create text featurizing estimator with the specified options
        var featurizer = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", options: tfOptions, inputColumnNames: nameof(MentalHealthData.Statement));

        return estimator
            .Append(labelMap)
            .Append(featurizer)
            .AppendCacheCheckpoint(mlContext)
            .Append(mlContext.MulticlassClassification.Trainers
                 .NaiveBayes(labelColumnName: "Label",
                             featureColumnName: "Features"))
            .Append(mlContext.Transforms
                 .Conversion.MapKeyToValue("PredictedLabel"));
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
        var mlContext = new MLContext();
        var splitDataView = LoadData(mlContext, dataPath); // Load data and split into training and testing sets

        Console.WriteLine("=============== Training the model ===============");
        var watch = Stopwatch.StartNew();
        // var trainingModel = BuildTrainingPipeline(mlContext).Fit(splitDataView.TrainSet); // Fit training pipeline

        // 2) build & train with your custom n-gram settings
        Console.WriteLine("=== Training with 2-grams and max 10k terms ===");
        var trainingModel = BuildTrainingPipeline(
            mlContext,
            ngramLength: 2,
            useAllLengths: true,
            maximumNgramsCount: new int[] { 10000 },
            removeStopWords: true).Fit(splitDataView.TrainSet);

        watch.Stop();
        var elapsedMs = watch.ElapsedMilliseconds;
        Console.WriteLine($"Training time: {elapsedMs} ms");
        Console.WriteLine("=============== End of training ===============");
        Console.WriteLine();

        Evaluate(mlContext, trainingModel, splitDataView.TestSet); // Evaluate the model
        GetPredictionForMentalHealth(mlContext, trainingModel, "I feel anxious and overwhelmed."); // Make a prediction

        string modelPath = Path.Combine(Environment.CurrentDirectory, "..", "Data", "MentalHealthModel.zip");
        mlContext.Model.Save(trainingModel, splitDataView.TrainSet.Schema, modelPath);
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
