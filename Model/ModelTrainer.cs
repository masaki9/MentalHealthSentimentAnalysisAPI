using Microsoft.ML;
using Microsoft.ML.Transforms.Text;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;

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
    /// Build the text featurization part of the pipeline.
    /// </summary>
    /// <param name="mlContext">The MLContext instance.</param>
    /// <param name="ngramLength">The length of the n-grams to be used.</param>
    /// <param name="useAllLengths">Whether to store all n-gram lengths up to ngramLength, or only ngramLength.</param>
    /// <param name="maximumNgramsCount">The maximum number of n-grams to be used for each level of n-grams.</param>
    /// <param name="removeStopWords">Whether to remove stop words.</param>
    /// <returns>A text featurization estimator.</returns>
    public IEstimator<ITransformer> BuildTextFeaturizer(MLContext mlContext, int ngramLength, bool useAllLengths, int[] maximumNgramsCount, bool removeStopWords)
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

        return estimator.Append(labelMap).Append(featurizer);
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
        var mlContext = new MLContext();
        var dataPath = Path.Combine(Environment.CurrentDirectory, "..", "Data", "MentalHealthData.csv");
        var splitDataView = LoadData(mlContext, dataPath);
        var featurizer = BuildTextFeaturizer(mlContext, ngramLength: 2, useAllLengths: true,
                                             maximumNgramsCount: new int[] { 10000 }, removeStopWords: true);

        // Different configurations for SDCA
        var sdcaConfigs = new[]
        {
            (Name: "SDCA-default",  Opts: new SdcaMaximumEntropyMulticlassTrainer.Options {
                LabelColumnName           = "Label",
                FeatureColumnName         = "Features"
            }),
            (Name: "SDCA-l2=0.1",    Opts: new SdcaMaximumEntropyMulticlassTrainer.Options {
                LabelColumnName           = "Label",
                FeatureColumnName         = "Features",
                L2Regularization          = 0.1f
            }),
            (Name: "SDCA-iters=50",  Opts: new SdcaMaximumEntropyMulticlassTrainer.Options {
                LabelColumnName           = "Label",
                FeatureColumnName         = "Features",
                MaximumNumberOfIterations =  50
            }),
            (Name: "SDCA-tol=1e-2",  Opts: new SdcaMaximumEntropyMulticlassTrainer.Options {
                LabelColumnName           = "Label",
                FeatureColumnName         = "Features",
                ConvergenceTolerance      = 1e-2f
            }),
        };

        var cachedTest = mlContext.Data.Cache(splitDataView.TestSet); // Cache the test set for faster evaluation

        ITransformer? bestModel = null;
        var bestName = "";
        var bestMacro = 0.0;

        // Evaluate each configuration
        foreach (var (name, opts) in sdcaConfigs)
        {
            Console.WriteLine($"\n=== {name} ===");
            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(opts);
            var pipeline = featurizer.AppendCacheCheckpoint(mlContext).Append(trainer).Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            var model = pipeline.Fit(splitDataView.TrainSet);
            var preds = model.Transform(cachedTest);
            var metrics = mlContext.MulticlassClassification.Evaluate(
                                preds,
                                labelColumnName: "Label",
                                scoreColumnName: "Score",
                                predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine($"{name} : MicroAcc={metrics.MicroAccuracy:F3}, MacroAcc={metrics.MacroAccuracy:F3}, LogLoss={metrics.LogLoss:F2}");

            if (metrics.MacroAccuracy > bestMacro)
            {
                bestMacro = metrics.MacroAccuracy;
                bestModel = model;
                bestName = name;
            }
        }

        // Save the best model
        var modelPath = Path.Combine(Environment.CurrentDirectory, "..", "Data", "MentalHealthModel.zip");
        Console.WriteLine($"\n>>> Saving best model '{bestName}' " + $"(MacroAcc={bestMacro:F3}) to {modelPath}...");
        mlContext.Model.Save(bestModel!, splitDataView.TrainSet.Schema, modelPath);
        Console.WriteLine(">>> Model saved!");
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
